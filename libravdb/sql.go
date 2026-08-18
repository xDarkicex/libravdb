package libravdb

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// QueryParams holds typed query parameter values.
type QueryParams map[string]interface{}

// Query parses, binds, optimizes, and executes a SQL/PGQ query.
func (db *Database) Query(ctx context.Context, sql string) (*SearchResults, error) {
	return db.QueryWithParams(ctx, sql, nil)
}

// QueryWithParams executes SQL with named parameters ($param or @param).
func (db *Database) QueryWithParams(ctx context.Context, sql string, params QueryParams) (*SearchResults, error) {
	return db.queryWithContext(ctx, sql, params)
}

func (db *Database) queryWithContext(ctx context.Context, sql string, params QueryParams) (*SearchResults, error) {
	return db.queryWithBoundParams(ctx, sql, optimizer.NewParameterSet(params), params)
}

// QueryWithBoundParams is the native typed execution entry point used by
// protocol adapters. The SQL source remains unchanged; decoded values are
// resolved by AST parameter spans inside the optimizer.
func (db *Database) QueryWithBoundParams(ctx context.Context, sql string, params *optimizer.ParameterSet) (*SearchResults, error) {
	return db.queryWithBoundParams(ctx, sql, params, nil)
}

// QueryWithSessionConfig executes a query with connection-local settings.
// Settings affect execution only and are never persisted in catalog/WAL state.
func (db *Database) QueryWithSessionConfig(ctx context.Context, sql string, params QueryParams, config *SessionConfig) (*SearchResults, error) {
	return db.queryWithSessionConfig(ctx, sql, params, config)
}

// QueryWithBoundParamsAndSessionConfig is the typed-parameter counterpart
// used by protocol adapters that also carry per-connection SQL settings.
func (db *Database) QueryWithBoundParamsAndSessionConfig(ctx context.Context, sql string, params *optimizer.ParameterSet, config *SessionConfig) (*SearchResults, error) {
	return db.queryWithBoundParamsAndConfig(ctx, sql, params, nil, config)
}

func (db *Database) queryWithSessionConfig(ctx context.Context, sql string, params QueryParams, config *SessionConfig) (*SearchResults, error) {
	return db.queryWithBoundParamsAndConfig(ctx, sql, optimizer.NewParameterSet(params), params, config)
}

func (db *Database) queryWithBoundParams(ctx context.Context, sql string, boundParams *optimizer.ParameterSet, legacyParams QueryParams) (*SearchResults, error) {
	return db.queryWithBoundParamsAndConfig(ctx, sql, boundParams, legacyParams, nil)
}

func (db *Database) queryWithBoundParamsAndConfig(ctx context.Context, sql string, boundParams *optimizer.ParameterSet, legacyParams QueryParams, sessionConfig *SessionConfig) (results *SearchResults, err error) {
	if ctx == nil {
		ctx = context.Background()
	}
	tracker := sqlTrackerFromContext(ctx)
	root := tracker == nil
	started := time.Now()
	defer func() {
		if err != nil {
			err = normalizeSQLError(err)
		}
	}()
	if root {
		tracker = &sqlQueryTracker{}
		ctx = context.WithValue(ctx, sqlQueryTrackerContextKey{}, tracker)
		defer func() {
			db.recordSQLQuery(time.Since(started), results, err, tracker)
		}()
	}
	return db.queryWithBoundParamsAndConfigInternal(ctx, sql, boundParams, legacyParams, sessionConfig, tracker)
}

func (db *Database) queryWithBoundParamsAndConfigInternal(ctx context.Context, sql string, boundParams *optimizer.ParameterSet, legacyParams QueryParams, sessionConfig *SessionConfig, tracker *sqlQueryTracker) (*SearchResults, error) {
	// Keep native SQL consistent with pgwire for PostgreSQL catalog names.
	// The lexer accepts bare system relations (pg_indexes, pg_class, ...), but
	// does not represent schema-qualified table expressions. Strip only the
	// pg_catalog qualifier outside quoted SQL text before parsing.
	sql = rewriteNativePgCatalogPrefix(sql)
	src := []byte(sql)

	// 1 & 2. Lex & Parse
	doc := &parser.QueryDoc{}
	if err := parser.Parse(src, doc); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}
	if doc.Explain {
		return db.executeSQLExplain(ctx, src, doc, boundParams, legacyParams, sessionConfig, tracker)
	}
	// Expose the existing durable commit LSN as a scalar SQL function. This
	// route is intentionally before catalog binding because the function has no
	// FROM relation and must work on an otherwise empty database.
	if results, handled, err := db.executeLatestCommitLSNQuery(ctx, src, doc); handled {
		return results, err
	}
	if results, handled, err := db.executeSQLStatsQuery(ctx, src, doc); handled {
		return results, err
	}

	// 3. Bind OIDs (Modifies doc in place)
	db.mu.RLock()
	cat := db.catalog
	generation := db.catalogGeneration.Load()
	db.mu.RUnlock()

	if cat == nil {
		return nil, fmt.Errorf("catalog not initialized")
	}

	// COMPUTE LEIDEN is a standalone statement that bypasses the catalog/optimizer.
	// It requires an active epoch transaction. Skip if the COMPUTE LEIDEN is
	// contained within a CTE (WITH ... AS).
	standaloneLeiden := false
	for i := range doc.ComputeLeidenStmts {
		cteReferenced := false
		for _, cte := range doc.CTEs {
			if cte.Body.Kind == parser.NodeKindComputeLeidenStmt && cte.Body.ID == int32(i) {
				cteReferenced = true
				break
			}
		}
		if !cteReferenced {
			standaloneLeiden = true
			break
		}
	}
	if standaloneLeiden && len(doc.ComputeLeidenStmts) > 0 {
		return db.executeComputeLeiden(ctx, src, doc)
	}
	if len(doc.SessionSettingStmts) > 0 {
		return nil, fmt.Errorf("SET/RESET requires a session; use Database.NewSQLSession or pgwire")
	}

	// CTE SELECT: preserve the existing Leiden virtual relation path, while
	// ordinary SELECT CTEs execute through an in-memory virtual relation.  No
	// temporary collection or catalog/WAL mutation is involved.
	if root := rootSelectIndex(doc); root >= 0 && root < len(doc.SelectStmts) && doc.SelectStmts[root].CTEsCount > 0 {
		cte := doc.CTEs[doc.SelectStmts[root].CTEsStart]
		if cte.Body.Kind == parser.NodeKindComputeLeidenStmt {
			return db.executeLeidenCTE(ctx, src, doc, boundParams)
		}
		return db.executeGenericCTE(ctx, src, doc, boundParams, legacyParams, sessionConfig)
	}
	// VERSIONS OF ... BETWEEN TIMESTAMP ... is a virtual temporal relation;
	// evaluate it with the query-local row engine so its historical tuples can
	// participate in normal WHERE/ORDER/OFFSET/LIMIT projection semantics.
	if root := rootSelectIndex(doc); root >= 0 && root < len(doc.SelectStmts) && selectHasTemporalRange(doc, &doc.SelectStmts[root]) {
		return db.executeSubquerySelect(ctx, src, doc, boundParams, legacyParams)
	}
	// JSON extraction and containment are evaluated by the query-local row
	// engine so projected JSON values and nested predicates retain their
	// document shape instead of being flattened into scalar catalog bytes.
	if root := rootSelectIndex(doc); root >= 0 && root < len(doc.SelectStmts) && virtualSelectHasJSON(src, doc, &doc.SelectStmts[root]) {
		return db.executeSubquerySelect(ctx, src, doc, boundParams, legacyParams)
	}
	// Window functions require a post-filter partition/order pass before the
	// outer ORDER BY/LIMIT. Keep them in the query-local virtual evaluator so
	// the physical relational plan cannot discard window scope.
	if root := rootSelectIndex(doc); root >= 0 && root < len(doc.SelectStmts) && virtualSelectHasWindow(doc, &doc.SelectStmts[root]) {
		return db.executeSubquerySelect(ctx, src, doc, boundParams, legacyParams)
	}
	// ARRAY_AGG and STRING_AGG are ordinary PostgreSQL aggregate names but are
	// intentionally parsed as FunctionExpr nodes (they are not lexer keywords).
	// Route them through the query-local relation evaluator so grouped and
	// nullable inputs retain their row values and the physical scalar planner
	// cannot flatten the resulting array/string.
	if root := rootSelectIndex(doc); root >= 0 && root < len(doc.SelectStmts) && virtualSelectHasCollectionAggregate(src, doc, &doc.SelectStmts[root]) {
		return db.executeSubquerySelect(ctx, src, doc, boundParams, legacyParams)
	}
	// Ordered-set aggregates (PERCENTILE_CONT, PERCENTILE_DISC, MODE) use
	// WITHIN GROUP ordering and are evaluated by the query-local relation path;
	// the physical aggregate planner only handles ordinary aggregates.
	if root := rootSelectIndex(doc); root >= 0 && root < len(doc.SelectStmts) && virtualSelectHasOrderedSetAggregate(doc, &doc.SelectStmts[root]) {
		return db.executeSubquerySelect(ctx, src, doc, boundParams, legacyParams)
	}
	// Direct aggregates over bound parameters, such as MIN($threshold), need
	// the row-aware virtual aggregate evaluator. The physical aggregate planner
	// only sees catalog columns and otherwise drops MIN/MAX or treats SUM's
	// parameter operand as a zero-valued scalar.
	if root := rootSelectIndex(doc); root >= 0 && root < len(doc.SelectStmts) && virtualSelectHasParameterizedAggregate(src, doc, &doc.SelectStmts[root]) {
		return db.executeSubquerySelect(ctx, src, doc, boundParams, legacyParams)
	}
	// Aggregate-derived scalar expressions such as SUM(alpha) / SUM(beta)
	// require the grouped virtual evaluator so each aggregate operand is
	// materialized before the enclosing arithmetic expression is evaluated.
	if root := rootSelectIndex(doc); root >= 0 && root < len(doc.SelectStmts) && virtualSelectHasNestedAggregateProjection(doc, &doc.SelectStmts[root]) {
		return db.executeSubquerySelect(ctx, src, doc, boundParams, legacyParams)
	}
	// CASE and general casts are scalar SQL expressions.  They must be
	// evaluated after the visible row has been materialized; the physical
	// relational planner deliberately only projects catalog columns and would
	// otherwise drop these expression nodes.  Keep this route query-local so
	// staged/temporal rows and typed parameters retain their normal semantics.
	if root := rootSelectIndex(doc); root >= 0 && root < len(doc.SelectStmts) && virtualSelectHasScalarExpressions(src, doc, &doc.SelectStmts[root]) {
		return db.executeSubquerySelect(ctx, src, doc, boundParams, legacyParams)
	}
	// Derived tables are query-local virtual relations. Execute them through
	// the same AST evaluator as correlated subqueries before catalog binding;
	// a parenthesized SELECT has no physical catalog identity.
	if root := rootSelectIndex(doc); root >= 0 && root < len(doc.SelectStmts) && selectHasDerivedRelation(doc, &doc.SelectStmts[root]) {
		return db.executeSubquerySelect(ctx, src, doc, boundParams, legacyParams)
	}
	// Uncorrelated IN (SELECT ...) and EXISTS (SELECT ...) predicates are
	// evaluated as virtual membership filters before the ordinary binder. This
	// keeps subquery rows out of the catalog while retaining snapshot/epoch
	// visibility through the recursive query path.
	if len(doc.SubqueryExprs) > 0 && len(doc.SelectStmts) > 0 {
		return db.executeSubquerySelect(ctx, src, doc, boundParams, legacyParams)
	}
	// JSON predicates in UPDATE WHERE clauses need the same row-aware
	// evaluator used by JSON SELECTs. The physical UPDATE plan still owns the
	// assignment lowering and transaction, but row selection must evaluate the
	// expression against the decoded document rather than discard it as an
	// unsupported catalog-column predicate.
	if len(doc.UpdateStmts) > 0 && updateHasVirtualJSONPredicate(src, doc) {
		return db.executeVirtualJSONUpdate(ctx, src, doc, boundParams, legacyParams)
	}

	// Empty statement list is a valid no-op.
	if len(doc.SelectStmts) == 0 && len(doc.InsertStmts) == 0 &&
		len(doc.InsertGraphEdgeStmts) == 0 &&
		len(doc.UpdateStmts) == 0 && len(doc.DeleteStmts) == 0 &&
		len(doc.CreateTableStmts) == 0 && len(doc.CreateEdgeTypeStmts) == 0 && len(doc.DropTableStmts) == 0 &&
		len(doc.CreateIndexStmts) == 0 && len(doc.DropIndexStmts) == 0 &&
		len(doc.AlterTableStmts) == 0 {
		if len(doc.TransactionStmts) > 0 {
			if epoch := epochFromContext(ctx); epoch != nil {
				return db.handleEpochTransactionStmts(ctx, epoch, doc.TransactionStmts)
			}
		}
		return &SearchResults{}, nil
	}

	cacheable, parameterSlots := sqlPlanCacheEligible(src, doc)
	cacheKey := normalizeSQLPlanKey(sql)
	if cacheable && db.sqlPlanCache != nil {
		if cached, ok := db.sqlPlanCache.get(cacheKey, generation, cat, boundParams, src); ok {
			if tracker != nil {
				tracker.planCacheHits++
			}
			return newExecutor(db).Execute(ctx, cached)
		}
		if tracker != nil {
			tracker.planCacheMisses++
		}
	}

	binder := catalog.NewBinder(cat, src)
	if err := binder.Bind(doc); err != nil {
		return nil, fmt.Errorf("bind error: %w", err)
	}

	// 4. Optimize (AST -> Physical Plan)
	opt := optimizer.NewOptimizer(cat)
	var plan *optimizer.PhysicalPlan
	var err error
	if boundParams != nil {
		plan, err = opt.OptimizeWithBoundParams(doc, src, boundParams)
	} else {
		plan, err = opt.OptimizeWithParams(doc, src, legacyParams)
	}
	if err != nil {
		return nil, fmt.Errorf("optimize error: %w", err)
	}
	if cacheable && (plan.Kind == optimizer.QueryKindRelational || plan.Kind == optimizer.QueryKindAggregate) && db.sqlPlanCache != nil {
		db.sqlPlanCache.put(cacheKey, generation, cat, plan, parameterSlots)
	}

	// 5. Execute Physical Plan
	return newExecutor(db).Execute(ctx, plan)
}

func rewriteNativePgCatalogPrefix(query string) string {
	const prefix = "pg_catalog."
	var rewritten strings.Builder
	rewritten.Grow(len(query))
	inSingleQuote := false
	inDoubleQuote := false
	replaced := false
	for i := 0; i < len(query); {
		if inSingleQuote {
			rewritten.WriteByte(query[i])
			if query[i] == '\'' {
				if i+1 < len(query) && query[i+1] == '\'' {
					rewritten.WriteByte(query[i+1])
					i += 2
					continue
				}
				inSingleQuote = false
			}
			i++
			continue
		}
		if inDoubleQuote {
			rewritten.WriteByte(query[i])
			if query[i] == '"' {
				if i+1 < len(query) && query[i+1] == '"' {
					rewritten.WriteByte(query[i+1])
					i += 2
					continue
				}
				inDoubleQuote = false
			}
			i++
			continue
		}

		switch query[i] {
		case '\'':
			inSingleQuote = true
			rewritten.WriteByte(query[i])
			i++
			continue
		case '"':
			inDoubleQuote = true
			rewritten.WriteByte(query[i])
			i++
			continue
		}
		if i+len(prefix) <= len(query) && strings.EqualFold(query[i:i+len(prefix)], prefix) &&
			(i == 0 || !isSQLIdentifierByte(query[i-1])) {
			replaced = true
			i += len(prefix)
			continue
		}
		rewritten.WriteByte(query[i])
		i++
	}
	if !replaced {
		return query
	}
	return rewritten.String()
}

func isSQLIdentifierByte(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') ||
		(b >= '0' && b <= '9') || b == '_' || b == '$'
}

func updateHasVirtualJSONPredicate(src []byte, doc *parser.QueryDoc) bool {
	if doc == nil {
		return false
	}
	for i := range doc.UpdateStmts {
		where := doc.UpdateStmts[i].WhereExpr
		if where.Kind != parser.NodeKindUnknown && nodeHasJSON(src, doc, where) {
			return true
		}
	}
	return false
}

// executeVirtualJSONUpdate resolves the target rows with the query-local
// expression evaluator, then delegates all assignments and writes to the
// normal Executor transaction path. No update is staged until every matching
// row has been selected and the assignment expressions can be evaluated.
func (db *Database) executeVirtualJSONUpdate(ctx context.Context, src []byte, doc *parser.QueryDoc, boundParams *optimizer.ParameterSet, legacyParams QueryParams) (*SearchResults, error) {
	db.mu.RLock()
	cat := db.catalog
	db.mu.RUnlock()
	if cat == nil {
		return nil, fmt.Errorf("catalog not initialized")
	}
	binder := catalog.NewBinder(cat, src)
	if err := binder.Bind(doc); err != nil {
		return nil, fmt.Errorf("bind error: %w", err)
	}
	opt := optimizer.NewOptimizer(cat)
	var (
		plan *optimizer.PhysicalPlan
		err  error
	)
	if boundParams != nil {
		plan, err = opt.OptimizeWithBoundParams(doc, src, boundParams)
	} else {
		plan, err = opt.OptimizeWithParams(doc, src, legacyParams)
	}
	if err != nil {
		return nil, fmt.Errorf("optimize error: %w", err)
	}
	if plan == nil || plan.Kind != optimizer.QueryKindUpdate || len(doc.UpdateStmts) == 0 {
		return nil, fmt.Errorf("UPDATE JSON predicate did not produce an UPDATE plan")
	}
	col, err := db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	records, err := recordsVisibleInContext(ctx, col)
	if err != nil {
		return nil, err
	}
	stmt := &doc.UpdateStmts[0]
	rows := &SearchResults{Results: make([]*SearchResult, 0, len(records))}
	for _, record := range records {
		values := cloneMetadata(record.Metadata)
		if values == nil {
			values = make(map[string]interface{})
		}
		values["id"] = record.ID
		matched, evalErr := db.evalVirtualExpr(ctx, src, doc, stmt.WhereExpr,
			virtualSQLRow{ID: record.ID, Values: values}, boundParams, legacyParams)
		if evalErr != nil {
			return nil, fmt.Errorf("UPDATE WHERE row %q: %w", record.ID, evalErr)
		}
		if !matched {
			continue
		}
		rows.Results = append(rows.Results, &SearchResult{
			ID: record.ID, Vector: cloneVector(record.Vector), Metadata: cloneMetadata(record.Metadata),
		})
	}
	rows.Total = len(rows.Results)
	return newExecutor(db).executeUpdateRows(ctx, plan, rows)
}

// executeComputeLeiden runs the full COMPUTE LEIDEN pipeline:
// parse → lower → resolve collection → bind → execute → materialize → SearchResults.
func (db *Database) executeComputeLeiden(ctx context.Context, src []byte, doc *parser.QueryDoc) (*SearchResults, error) {
	epoch := epochFromContext(ctx)
	if epoch == nil {
		return nil, fmt.Errorf("COMPUTE LEIDEN requires an active epoch transaction")
	}

	if len(doc.ComputeLeidenStmts) != 1 {
		return nil, fmt.Errorf("COMPUTE LEIDEN: expected 1 statement, got %d", len(doc.ComputeLeidenStmts))
	}

	// Lower AST → logical plan.
	plan, err := LowerComputeLeidenPlan(src, doc, 0)
	if err != nil {
		return nil, fmt.Errorf("lower COMPUTE LEIDEN plan: %w", err)
	}

	// Resolve collection: find the collection whose graph has labeled nodes
	// matching the seed label. For the standalone COMPUTE LEIDEN grammar,
	// the collection is not specified in the SQL text.
	if plan.Collection == "" {
		coll, err := db.resolveLeidenCollection(ctx, plan.SeedLabel)
		if err != nil {
			return nil, fmt.Errorf("resolve COMPUTE LEIDEN collection: %w", err)
		}
		plan.Collection = coll
	}

	// Bind against the active epoch.
	bound, err := epoch.BindLeidenMatchPlan(ctx, plan, "")
	if err != nil {
		return nil, fmt.Errorf("bind COMPUTE LEIDEN plan: %w", err)
	}

	// Execute.
	result, err := epoch.ExecuteBoundLeidenMatchPlan(ctx, bound)
	if err != nil {
		return nil, fmt.Errorf("execute COMPUTE LEIDEN: %w", err)
	}

	// Convert LeidenRelation rows to SearchResults.
	return leidenRelationToSearchResults(result.Relation), nil
}

// leidenRelationToSearchResults converts a LeidenRelation into the standard
// SearchResults shape. Each community row becomes one result with node_id and
// community_id in metadata, plus the propagated Leiden diagnostics.
func leidenRelationToSearchResults(rel *LeidenRelation) *SearchResults {
	if rel == nil {
		return &SearchResults{}
	}
	results := &SearchResults{
		Results: make([]*SearchResult, len(rel.Rows)),
		Total:   len(rel.Rows),
		// COMPUTE LEIDEN result columns in exact order:
		//   node_id, community_id, collection, record_id, truncated, scope, modularity
		Columns: []string{
			"node_id", "community_id", "collection", "record_id",
			"truncated", "scope", "modularity",
		},
	}
	for i, row := range rel.Rows {
		results.Results[i] = &SearchResult{
			ID:    fmt.Sprintf("%d", row.NodeID),
			Score: 1.0,
			Metadata: map[string]interface{}{
				"node_id":      row.NodeID,
				"community_id": row.CommunityID,
				"collection":   row.Collection,
				"record_id":    row.RecordID,
				"truncated":    rel.Truncated,
				"scope":        string(rel.Scope),
				"modularity":   rel.Modularity,
			},
		}
	}
	return results
}

// executeLeidenCTE executes a CTE SELECT: binds the CTE, executes the Leiden
// plan, builds the local_clusters virtual relation, and performs the JOIN
// against the outer FROM table.
func (db *Database) executeLeidenCTE(ctx context.Context, src []byte, doc *parser.QueryDoc, params *optimizer.ParameterSet) (*SearchResults, error) {
	epoch := epochFromContext(ctx)
	if epoch == nil {
		return nil, fmt.Errorf("COMPUTE LEIDEN CTE requires an active epoch transaction")
	}

	// Bind the CTE.
	bound, err := epoch.BindLeidenCTE(ctx, src, doc, 0)
	if err != nil {
		return nil, fmt.Errorf("bind CTE: %w", err)
	}

	// Execute the full pipeline: Leiden → JOIN → SearchResults.
	return epoch.ExecuteLeidenCTE(ctx, src, doc, bound, 0, params)
}

// handleEpochTransactionStmts processes transaction statements inside an epoch.
func (db *Database) handleEpochTransactionStmts(ctx context.Context, epoch *EpochTx, stmts []parser.TransactionStmt) (*SearchResults, error) {
	for _, stmt := range stmts {
		switch stmt.Kind {
		case parser.TransactionBeginEpoch:
		case parser.TransactionCommit:
			if err := epoch.Commit(ctx); err != nil {
				return nil, fmt.Errorf("COMMIT: %w", err)
			}
		case parser.TransactionRollback:
			if err := epoch.Rollback(ctx); err != nil {
				return nil, fmt.Errorf("ROLLBACK: %w", err)
			}
		case parser.TransactionSavepoint:
			if err := epoch.Savepoint(stmt.SavepointName); err != nil {
				return nil, fmt.Errorf("SAVEPOINT: %w", err)
			}
		case parser.TransactionRollbackToSavepoint:
			if err := epoch.RollbackTo(stmt.SavepointName); err != nil {
				return nil, fmt.Errorf("ROLLBACK TO SAVEPOINT: %w", err)
			}
		case parser.TransactionReleaseSavepoint:
			if err := epoch.ReleaseSavepoint(stmt.SavepointName); err != nil {
				return nil, fmt.Errorf("RELEASE SAVEPOINT: %w", err)
			}
		}
	}
	return &SearchResults{}, nil
}
