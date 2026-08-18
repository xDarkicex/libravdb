package libravdb

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/xDarkicex/lexer"
	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// virtualGraphSemijoinState is the query-local result of a graph subquery
// used as an IN predicate. The candidate list is kept in result order for
// deterministic output when the outer query has no ORDER BY; candidateSet is
// the membership structure used by the predicate evaluator.
type virtualGraphSemijoinState struct {
	subqueryID      int32
	collection      string
	candidateIDs    []string
	candidateSet    map[string]struct{}
	subqueryHasNull bool
	snapshotLSN     uint64
}

type virtualGraphSemijoinContextKey struct{}

func withVirtualGraphSemijoin(ctx context.Context, state virtualGraphSemijoinState) context.Context {
	return context.WithValue(ctx, virtualGraphSemijoinContextKey{}, state)
}

func virtualGraphSemijoinFromContext(ctx context.Context) (virtualGraphSemijoinState, bool) {
	if ctx == nil {
		return virtualGraphSemijoinState{}, false
	}
	state, ok := ctx.Value(virtualGraphSemijoinContextKey{}).(virtualGraphSemijoinState)
	return state, ok
}

// tryExecuteGraphSemijoin recognizes the recommendation-query shape:
//
//	SELECT p.id, p.metadata
//	FROM people p
//	WHERE p.id IN (
//	  SELECT src.id
//	  FROM people src
//	  JOIN MATCH (src)-[]->(shared)
//	  JOIN MATCH (origin)-[]->(shared)
//	  WHERE origin.id = $1
//	)
//
// The graph subquery is planned and executed once. Its source IDs are then
// used as a semijoin candidate set for the outer relation. Unsupported shapes
// return handled=false so the existing general virtual evaluator remains the
// compatibility fallback.
func (db *Database) tryExecuteGraphSemijoin(ctx context.Context, src []byte, doc *parser.QueryDoc, rootStmt *parser.SelectStmt, params *optimizer.ParameterSet, legacy QueryParams) (*SearchResults, bool, error) {
	if db == nil || doc == nil || rootStmt == nil || len(rootStmt.Joins) != 0 || rootStmt.FromTable.Kind != parser.NodeKindTableExpr {
		return nil, false, nil
	}
	if rootStmt.WhereExpr.Kind == parser.NodeKindUnknown {
		return nil, false, nil
	}
	outerTable := &doc.TableExprs[rootStmt.FromTable.ID]
	if outerTable.IsDerived || outerTable.IsFunction || outerTable.TemporalRange {
		return nil, false, nil
	}

	inRef, ok := findConjunctiveGraphSemijoin(doc, rootStmt.WhereExpr)
	if !ok || inRef.ID < 0 || int(inRef.ID) >= len(doc.InExprs) {
		return nil, false, nil
	}
	in := &doc.InExprs[inRef.ID]
	if in.Not || !in.HasSubquery || in.Subquery.Kind != parser.NodeKindSubqueryExpr || in.Subquery.ID < 0 || int(in.Subquery.ID) >= len(doc.SubqueryExprs) {
		return nil, false, nil
	}
	if in.Expr.Kind != parser.NodeKindIdentifier || in.Expr.ID < 0 || int(in.Expr.ID) >= len(doc.Identifiers) {
		return nil, false, nil
	}
	outerID := doc.Identifiers[in.Expr.ID]
	if !strings.EqualFold(sourceSpan(src, outerID.Start, outerID.End), "id") {
		return nil, false, nil
	}
	outerAlias := sourceSpan(src, outerTable.Alias, outerTable.AliasEnd)
	if outerAlias != "" && outerID.QualEnd > outerID.QualStart && !strings.EqualFold(sourceSpan(src, outerID.QualStart, outerID.QualEnd), outerAlias) {
		return nil, false, nil
	}

	sq := &doc.SubqueryExprs[in.Subquery.ID]
	if sq.Stmt.Kind != parser.NodeKindSelectStmt || sq.Stmt.ID < 0 || int(sq.Stmt.ID) >= len(doc.SelectStmts) {
		return nil, false, nil
	}
	innerStmt := &doc.SelectStmts[sq.Stmt.ID]
	collection, supported := graphSemijoinSubqueryShape(src, doc, innerStmt)
	if !supported || !strings.EqualFold(collection, sourceSpan(src, outerTable.Start, outerTable.End)) {
		return nil, false, nil
	}
	if innerStmt.SourceStart >= innerStmt.SourceEnd || innerStmt.SourceEnd > uint32(len(src)) {
		return nil, true, fmt.Errorf("graph semijoin has invalid subquery source span")
	}

	snapshotLSN, err := db.graphSemijoinSnapshotLSN(ctx, src, doc, outerTable, params)
	if err != nil {
		return nil, true, err
	}
	innerSQL := strings.TrimSpace(string(src[innerStmt.SourceStart:innerStmt.SourceEnd]))
	innerResults, err := db.executeGraphSemijoinSubquery(ctx, innerSQL, params, snapshotLSN)
	if err != nil {
		return nil, true, fmt.Errorf("execute graph semijoin subquery: %w", err)
	}

	state := virtualGraphSemijoinState{
		subqueryID:   in.Subquery.ID,
		collection:   collection,
		candidateSet: make(map[string]struct{}, len(innerResults.Results)),
		snapshotLSN:  snapshotLSN,
	}
	for _, row := range innerResults.Results {
		value, found := graphSemijoinResultValue(row, innerResults.Columns)
		if !found || value == nil {
			state.subqueryHasNull = true
			continue
		}
		id := recordMetaToString(value)
		if _, exists := state.candidateSet[id]; exists {
			continue
		}
		state.candidateSet[id] = struct{}{}
		state.candidateIDs = append(state.candidateIDs, id)
	}

	semijoinCtx := withVirtualGraphSemijoin(ctx, state)
	rows, columns, err := db.evaluateVirtualSelectRows(semijoinCtx, src, doc, rootStmt, nil, params, legacy)
	if err != nil {
		return nil, true, err
	}
	return finishVirtualRows(db, doc, src, rootStmt, rows, columns, params), true, nil
}

// findConjunctiveGraphSemijoin accepts an IN predicate only when it is a
// top-level conjunct. Restricting the optimization to AND-connected filters
// is important: `id IN (...) OR name = ...` cannot be reduced to the IN
// candidate set without changing SQL semantics.
func findConjunctiveGraphSemijoin(doc *parser.QueryDoc, ref parser.NodeRef) (parser.NodeRef, bool) {
	if doc == nil {
		return parser.NodeRef{}, false
	}
	switch ref.Kind {
	case parser.NodeKindInExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.InExprs) {
			return parser.NodeRef{}, false
		}
		in := doc.InExprs[ref.ID]
		if in.HasSubquery {
			return ref, true
		}
		return parser.NodeRef{}, false
	case parser.NodeKindBinaryExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
			return parser.NodeRef{}, false
		}
		be := doc.BinaryExprs[ref.ID]
		return findConjunctiveGraphSemijoinBinary(doc, ref, be)
	default:
		return parser.NodeRef{}, false
	}
}

func findConjunctiveGraphSemijoinBinary(doc *parser.QueryDoc, ref parser.NodeRef, be parser.BinaryExpr) (parser.NodeRef, bool) {
	// KindAnd is stable in the shared lexer and is intentionally imported by
	// cte_execute.go. Keeping this small helper separate makes the shape check
	// easy to audit: only AND can safely push a candidate set into a scan.
	if be.Operator != uint8(lexer.KindAnd) {
		return parser.NodeRef{}, false
	}
	left, leftOK := findConjunctiveGraphSemijoin(doc, be.Left)
	right, rightOK := findConjunctiveGraphSemijoin(doc, be.Right)
	if leftOK && rightOK {
		// Two independent semijoins would require an intersection of candidate
		// sets. Keep this first implementation deliberately single-set.
		return parser.NodeRef{}, false
	}
	if leftOK {
		return left, true
	}
	return right, rightOK
}

func graphSemijoinSubqueryShape(src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt) (string, bool) {
	if doc == nil || stmt == nil || stmt.FromTable.Kind != parser.NodeKindTableExpr || len(stmt.Joins) != 2 || stmt.CTEsCount != 0 || len(stmt.GroupBy) != 0 || stmt.HavingExpr.Kind != parser.NodeKindUnknown {
		return "", false
	}
	from := &doc.TableExprs[stmt.FromTable.ID]
	if from.IsDerived || from.IsFunction {
		return "", false
	}
	if stmt.ProjectionsCount < 1 || stmt.ProjectionsStart < 0 || stmt.ProjectionsStart+stmt.ProjectionsCount > int32(len(doc.Projections)) {
		return "", false
	}
	projection := &doc.Projections[stmt.ProjectionsStart]
	if projection.Star || projection.Expr.Kind != parser.NodeKindIdentifier || projection.Expr.ID < 0 || int(projection.Expr.ID) >= len(doc.Identifiers) {
		return "", false
	}
	projected := doc.Identifiers[projection.Expr.ID]
	if !strings.EqualFold(sourceSpan(src, projected.Start, projected.End), "id") {
		return "", false
	}

	anchor0, terminal0, ok0 := graphSemijoinMatchAliases(src, doc, stmt.Joins[0].MatchPath)
	anchor1, terminal1, ok1 := graphSemijoinMatchAliases(src, doc, stmt.Joins[1].MatchPath)
	if !ok0 || !ok1 || stmt.Joins[0].Type != parser.JoinInner || stmt.Joins[1].Type != parser.JoinInner {
		return "", false
	}
	if anchor0 == "" || anchor1 == "" || terminal0 == "" || terminal1 == "" || strings.EqualFold(anchor0, anchor1) || !strings.EqualFold(terminal0, terminal1) {
		return "", false
	}
	if projected.QualEnd > projected.QualStart && !strings.EqualFold(sourceSpan(src, projected.QualStart, projected.QualEnd), anchor0) {
		return "", false
	}
	return sourceSpan(src, from.Start, from.End), true
}

func graphSemijoinMatchAliases(src []byte, doc *parser.QueryDoc, ref parser.NodeRef) (anchor, terminal string, ok bool) {
	if ref.Kind != parser.NodeKindMatchPath || ref.ID < 0 || int(ref.ID) >= len(doc.MatchPaths) {
		return "", "", false
	}
	path := &doc.MatchPaths[ref.ID]
	for i := int32(0); i < path.PathNodesCount; i++ {
		index := path.PathNodesStart + i
		if index < 0 || int(index) >= len(doc.Nodes) {
			return "", "", false
		}
		node := doc.Nodes[index]
		if node.Kind != parser.NodeKindVertex || node.ID < 0 || int(node.ID) >= len(doc.Vertexes) {
			continue
		}
		vertex := doc.Vertexes[node.ID]
		alias := sourceSpan(src, vertex.Alias, vertex.AliasEnd)
		if alias == "" {
			return "", "", false
		}
		if anchor == "" {
			anchor = alias
		}
		terminal = alias
	}
	return anchor, terminal, anchor != "" && terminal != ""
}

func graphSemijoinResultValue(row *SearchResult, columns []string) (interface{}, bool) {
	if row == nil {
		return nil, false
	}
	if len(columns) > 0 && row.Metadata != nil {
		if value, ok := row.Metadata[columns[0]]; ok {
			return value, true
		}
	}
	if row.Metadata != nil {
		if value, ok := row.Metadata["id"]; ok {
			return value, true
		}
	}
	if row.ID != "" {
		if separator := strings.IndexByte(row.ID, '|'); separator >= 0 {
			return row.ID[:separator], true
		}
		return row.ID, true
	}
	return nil, false
}

func (db *Database) executeGraphSemijoinSubquery(ctx context.Context, sql string, params *optimizer.ParameterSet, snapshotLSN uint64) (*SearchResults, error) {
	src := []byte(sql)
	doc := &parser.QueryDoc{}
	if err := parser.Parse(src, doc); err != nil {
		return nil, fmt.Errorf("parse graph semijoin subquery: %w", err)
	}
	db.mu.RLock()
	cat := db.catalog
	generation := db.catalogGeneration.Load()
	db.mu.RUnlock()
	if cat == nil {
		return nil, fmt.Errorf("catalog not initialized")
	}
	cacheable, parameterSlots := graphSemijoinPlanCacheEligible(src, doc)
	cacheKey := "__graph_semijoin__:" + normalizeSQLPlanKey(sql)
	if cacheable && db.sqlPlanCache != nil {
		if cached, ok := db.sqlPlanCache.get(cacheKey, generation, cat, params, src); ok {
			if tracker := sqlTrackerFromContext(ctx); tracker != nil {
				tracker.planCacheHits++
			}
			if snapshotLSN != 0 {
				return newExecutor(db).ExecuteAtLSN(ctx, cached, snapshotLSN)
			}
			return newExecutor(db).Execute(ctx, cached)
		}
		if tracker := sqlTrackerFromContext(ctx); tracker != nil {
			tracker.planCacheMisses++
		}
	}
	if err := catalog.NewBinder(cat, src).Bind(doc); err != nil {
		return nil, fmt.Errorf("bind graph semijoin subquery: %w", err)
	}
	plan, err := optimizer.NewOptimizer(cat).OptimizeWithBoundParams(doc, src, params)
	if err != nil {
		return nil, fmt.Errorf("optimize graph semijoin subquery: %w", err)
	}
	if cacheable && db.sqlPlanCache != nil {
		db.sqlPlanCache.put(cacheKey, generation, cat, plan, parameterSlots)
	}
	if snapshotLSN != 0 {
		return newExecutor(db).ExecuteAtLSN(ctx, plan, snapshotLSN)
	}
	return newExecutor(db).Execute(ctx, plan)
}

func graphSemijoinPlanCacheEligible(src []byte, doc *parser.QueryDoc) (bool, []sqlPlanParameterSlot) {
	if doc == nil || len(doc.SelectStmts) != 1 {
		return false, nil
	}
	stmt := &doc.SelectStmts[0]
	if stmt.FromTable.Kind != parser.NodeKindTableExpr || len(stmt.Joins) != 2 {
		return false, nil
	}
	for i := range stmt.Joins {
		if stmt.Joins[i].MatchPath.Kind != parser.NodeKindMatchPath {
			return false, nil
		}
	}
	slots, ok := sqlPlanParameterSlots(src, doc, stmt)
	return ok, slots
}

func (db *Database) graphSemijoinSnapshotLSN(ctx context.Context, src []byte, doc *parser.QueryDoc, table *parser.TableExpr, params *optimizer.ParameterSet) (uint64, error) {
	if table == nil {
		return 0, nil
	}
	if table.TemporalLSN {
		return parseTemporalLSN(src, table.LSNStart, table.LSNEnd, params)
	}
	if table.Temporal {
		when, err := parseTemporalRangeTime(src, table.TimestampStart, table.TimestampEnd, params)
		if err != nil {
			return 0, fmt.Errorf("AS OF TIMESTAMP: %w", err)
		}
		snapshot, err := db.SnapshotAt(ctx, when)
		if err != nil {
			return 0, fmt.Errorf("AS OF TIMESTAMP: %w", err)
		}
		lsn := snapshot.LSN
		snapshot.Close()
		return lsn, nil
	}
	return 0, nil
}

func (db *Database) virtualGraphSemijoinSourceRows(ctx context.Context, src []byte, table *parser.TableExpr, state virtualGraphSemijoinState, outer *virtualSQLRow) ([]virtualSQLRow, error) {
	if outer != nil {
		return nil, fmt.Errorf("graph semijoin source cannot be correlated")
	}
	collection := sourceSpan(src, table.Start, table.End)
	col, err := db.GetCollection(collection)
	if err != nil {
		return nil, err
	}

	byID := make(map[string]Record, len(state.candidateIDs))
	if epochFromContext(ctx) != nil || transactionFromContext(ctx) != nil {
		records, listErr := recordsVisibleInContext(ctx, col)
		if listErr != nil {
			return nil, listErr
		}
		for _, record := range records {
			if _, wanted := state.candidateSet[record.ID]; wanted {
				byID[record.ID] = record
			}
		}
	} else {
		for _, id := range state.candidateIDs {
			var record Record
			if state.snapshotLSN != 0 {
				historical, getErr := col.GetAtLSN(ctx, id, state.snapshotLSN)
				if getErr != nil {
					return nil, getErr
				}
				if historical == nil {
					continue
				}
				record = *historical
			} else {
				current, getErr := col.Get(ctx, id)
				if getErr != nil {
					if errors.Is(getErr, ErrRecordNotFound) {
						continue
					}
					return nil, getErr
				}
				record = current
			}
			byID[id] = record
			trackSQLIndexHit(ctx, 1)
		}
	}

	alias := sourceSpan(src, table.Alias, table.AliasEnd)
	if alias == "" {
		alias = collection
	}
	rows := make([]virtualSQLRow, 0, len(byID))
	for _, id := range state.candidateIDs {
		record, ok := byID[id]
		if !ok {
			continue
		}
		values := cloneMetadata(record.Metadata)
		if values == nil {
			values = make(map[string]interface{})
		}
		values["id"] = record.ID
		row := virtualSQLRow{ID: record.ID, Values: values}
		qualifyVirtualRow(&row, alias)
		rows = append(rows, row)
	}
	return rows, nil
}
