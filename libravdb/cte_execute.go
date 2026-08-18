package libravdb

// This file contains the first relational virtual-relation executor.  A
// generic SELECT CTE is evaluated into an owned in-memory row set and joined
// directly; no temporary collection, catalog entry, graph node, or WAL record
// is created. The implementation deliberately covers the SQL shapes that can
// be represented by the current SoA parser: multiple declaration-ordered CTEs,
// projections, scalar predicates, ordering, OFFSET/LIMIT, IN subqueries, and
// uncorrelated EXISTS.

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/xDarkicex/lexer"
	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/optimizer"
)

type virtualSQLRow struct {
	ID     string
	Values map[string]interface{}
}

type virtualCTEEnv map[string][]virtualSQLRow

type virtualCTEContextKey struct{}

type virtualCTEState struct {
	env    virtualCTEEnv
	config *SessionConfig
}

func withVirtualCTEs(ctx context.Context, env virtualCTEEnv, config *SessionConfig) context.Context {
	return context.WithValue(ctx, virtualCTEContextKey{}, virtualCTEState{env: env, config: config})
}

func virtualCTEsFromContext(ctx context.Context) virtualCTEEnv {
	if ctx == nil {
		return nil
	}
	state, _ := ctx.Value(virtualCTEContextKey{}).(virtualCTEState)
	return state.env
}

func virtualSessionConfigFromContext(ctx context.Context) *SessionConfig {
	if ctx == nil {
		return nil
	}
	state, _ := ctx.Value(virtualCTEContextKey{}).(virtualCTEState)
	return state.config
}

func virtualCTEName(name string) string { return strings.ToLower(strings.TrimSpace(name)) }

func rootSelectIndex(doc *parser.QueryDoc) int {
	for i := len(doc.Nodes) - 1; i >= 0; i-- {
		if doc.Nodes[i].Kind == parser.NodeKindSelectStmt {
			return int(doc.Nodes[i].ID)
		}
	}
	if len(doc.SelectStmts) == 0 {
		return -1
	}
	return len(doc.SelectStmts) - 1
}

func selectHasDerivedRelation(doc *parser.QueryDoc, stmt *parser.SelectStmt) bool {
	if stmt == nil {
		return false
	}
	if stmt.FromTable.Kind == parser.NodeKindTableExpr && stmt.FromTable.ID >= 0 && int(stmt.FromTable.ID) < len(doc.TableExprs) && doc.TableExprs[stmt.FromTable.ID].IsDerived {
		return true
	}
	for i := range stmt.Joins {
		if stmt.Joins[i].Derived.Kind == parser.NodeKindTableExpr {
			return true
		}
	}
	return false
}

func (db *Database) executeGenericCTE(ctx context.Context, src []byte, doc *parser.QueryDoc, params *optimizer.ParameterSet, legacy QueryParams, config *SessionConfig) (*SearchResults, error) {
	root := rootSelectIndex(doc)
	if root < 0 || root >= len(doc.SelectStmts) {
		return nil, fmt.Errorf("generic CTE has no outer SELECT")
	}
	outer := &doc.SelectStmts[root]
	if outer.CTEsCount <= 0 || outer.CTEsStart < 0 || outer.CTEsStart+outer.CTEsCount > int32(len(doc.CTEs)) {
		return nil, fmt.Errorf("invalid CTE range")
	}
	env := make(virtualCTEEnv, outer.CTEsCount)
	cteCtx := withVirtualCTEs(ctx, env, config)
	for i := int32(0); i < outer.CTEsCount; i++ {
		cte := &doc.CTEs[outer.CTEsStart+i]
		name := sourceSpan(src, cte.NameStart, cte.NameEnd)
		if cte.Body.Kind != parser.NodeKindSelectStmt || cte.Body.ID < 0 || int(cte.Body.ID) >= len(doc.SelectStmts) {
			return nil, fmt.Errorf("CTE %q is not a relational SELECT", name)
		}
		bodyStmt := &doc.SelectStmts[cte.Body.ID]
		var rows []virtualSQLRow
		var err error
		if cte.Recursive {
			rows, err = db.evaluateRecursiveCTE(cteCtx, src, doc, bodyStmt, virtualCTEName(name), params, legacy)
		} else {
			var columns []string
			rows, columns, err = db.evaluateVirtualSelectRows(cteCtx, src, doc, bodyStmt, nil, params, legacy)
			if err == nil {
				rows, err = db.applyVirtualSelectClauses(cteCtx, src, doc, bodyStmt, rows, columns, params, legacy)
			}
		}
		if err != nil {
			return nil, fmt.Errorf("execute CTE %q: %w", name, err)
		}
		env[virtualCTEName(name)] = rows
	}
	rows, columns, err := db.evaluateVirtualSelectRows(cteCtx, src, doc, outer, nil, params, legacy)
	if err != nil {
		return nil, err
	}
	return finishVirtualRows(db, doc, src, outer, rows, columns, params), nil
}

const maxRecursiveCTEIterations = 10000

func (db *Database) evaluateRecursiveCTE(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt, name string, params *optimizer.ParameterSet, legacy QueryParams) ([]virtualSQLRow, error) {
	if stmt.SetOp != parser.SetOpUnion || stmt.UnionNext.Kind != parser.NodeKindSelectStmt {
		return nil, fmt.Errorf("recursive CTE %q requires UNION or UNION ALL", name)
	}
	anchor, _, err := db.evaluateVirtualSelectRows(ctx, src, doc, stmt, nil, params, legacy)
	if err != nil {
		return nil, fmt.Errorf("recursive CTE %q anchor: %w", name, err)
	}
	env := virtualCTEsFromContext(ctx)
	if env == nil {
		return nil, fmt.Errorf("recursive CTE %q has no evaluation environment", name)
	}
	accumulated := make([]virtualSQLRow, 0, len(anchor))
	working := append([]virtualSQLRow(nil), anchor...)
	seen := make(map[string]struct{}, len(anchor))
	for _, row := range anchor {
		key := virtualRowKey(row)
		if !stmt.SetOpAll {
			if _, exists := seen[key]; exists {
				continue
			}
			seen[key] = struct{}{}
		}
		accumulated = append(accumulated, row)
	}
	branch := &doc.SelectStmts[stmt.UnionNext.ID]
	for iteration := 0; ; iteration++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		maxIterations := maxRecursiveCTEIterations
		if config := virtualSessionConfigFromContext(ctx); config != nil && config.MaxRecursionDepth > 0 && uint32(maxIterations) > config.MaxRecursionDepth {
			maxIterations = int(config.MaxRecursionDepth)
		}
		if iteration >= maxIterations {
			return nil, fmt.Errorf("recursive CTE %q exceeded %d iterations", name, maxIterations)
		}
		// SQL recursive evaluation feeds only the previous iteration's working
		// set into the recursive term; the accumulated result is output state,
		// not the next join input. This prevents tree edges from being replayed
		// once for every ancestor while preserving UNION ALL multiplicity.
		env[name] = working
		delta, _, err := db.evaluateVirtualSelectRows(ctx, src, doc, branch, nil, params, legacy)
		if err != nil {
			return nil, fmt.Errorf("recursive CTE %q step %d: %w", name, iteration+1, err)
		}
		if len(delta) == 0 {
			break
		}
		nextWorking := make([]virtualSQLRow, 0, len(delta))
		added := 0
		for _, row := range delta {
			if !stmt.SetOpAll {
				key := virtualRowKey(row)
				if _, exists := seen[key]; exists {
					continue
				}
				seen[key] = struct{}{}
			}
			accumulated = append(accumulated, row)
			nextWorking = append(nextWorking, row)
			added++
		}
		working = nextWorking
		if !stmt.SetOpAll && added == 0 {
			break
		}
	}
	return accumulated, nil
}

func virtualRowKey(row virtualSQLRow) string {
	keys := make([]string, 0, len(row.Values))
	for key := range row.Values {
		if strings.Contains(key, ".") {
			continue
		}
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, key := range keys {
		value := fmt.Sprintf("%T:%v", row.Values[key], row.Values[key])
		b.WriteString(strconv.Itoa(len(key)))
		b.WriteByte(':')
		b.WriteString(key)
		b.WriteString(strconv.Itoa(len(value)))
		b.WriteByte(':')
		b.WriteString(value)
		b.WriteByte('|')
	}
	return b.String()
}

// executeSubquerySelect evaluates SELECT statements containing subqueries
// against epoch-visible virtual rows. Unlike the old membership-only path,
// this keeps the current outer row in scope, so correlated IN/EXISTS and
// scalar subqueries use the same AST without rewriting SQL text.
func (db *Database) executeSubquerySelect(ctx context.Context, src []byte, doc *parser.QueryDoc, params *optimizer.ParameterSet, legacy QueryParams) (*SearchResults, error) {
	root := rootSelectIndex(doc)
	if root < 0 || root >= len(doc.SelectStmts) {
		return nil, fmt.Errorf("subquery query has no outer SELECT")
	}
	stmt := &doc.SelectStmts[root]
	if result, handled, err := db.tryExecuteGraphSemijoin(ctx, src, doc, stmt, params, legacy); handled {
		return result, err
	}
	rows, columns, err := db.evaluateVirtualSelectRows(ctx, src, doc, stmt, nil, params, legacy)
	if err != nil {
		return nil, err
	}
	return finishVirtualRows(db, doc, src, stmt, rows, columns, params), nil
}

func firstVirtualValue(row virtualSQLRow, columns []string) (interface{}, bool) {
	if len(columns) > 0 {
		value, ok := row.Values[columns[0]]
		if ok {
			return value, true
		}
	}
	for _, value := range row.Values {
		return value, true
	}
	return nil, false
}

// evaluateVirtualSelectRows evaluates a relational SELECT into owned rows.
// It is intentionally separate from the catalog executor: derived tables and
// correlated subqueries are query-local virtual relations, so they must not
// create collections, graph nodes, or WAL records. Physical source rows still
// come from recordsVisibleInContext, preserving epoch and snapshot visibility.
func (db *Database) evaluateVirtualSelectRows(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt, outer *virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) ([]virtualSQLRow, []string, error) {
	if stmt == nil {
		return nil, nil, fmt.Errorf("nil virtual SELECT")
	}

	rows, usedIndex, err := db.virtualIndexedSourceRows(ctx, src, doc, stmt, outer, params, legacy)
	if !usedIndex {
		rows, err = db.virtualSourceRows(ctx, src, doc, stmt.FromTable, outer, params, legacy)
	}
	if err != nil {
		return nil, nil, err
	}

	for i := range stmt.Joins {
		join := &stmt.Joins[i]
		// JOIN MATCH has no relational right-hand table in the AST. Its right
		// side is produced from the left row's graph node, so it follows the
		// same per-left-row evaluation path as a correlated derived relation.
		graphMatch := join.MatchPath.Kind == parser.NodeKindMatchPath
		correlatedDerived := graphMatch || join.Derived.Kind == parser.NodeKindTableExpr || join.IsFunction
		if correlatedDerived && (join.Type == parser.JoinRight || join.Type == parser.JoinFull) {
			if graphMatch {
				return nil, nil, fmt.Errorf("graph JOIN MATCH supports INNER, LEFT, and CROSS JOIN only")
			}
			return nil, nil, fmt.Errorf("correlated derived tables support INNER, LEFT, and CROSS JOIN only")
		}
		var right []virtualSQLRow
		if !correlatedDerived {
			var err error
			right, err = db.virtualJoinRows(ctx, src, doc, stmt, join, outer, params, legacy)
			if err != nil {
				return nil, nil, err
			}
		}
		next := make([]virtualSQLRow, 0)
		matchedRight := make([]bool, len(right))
		for _, left := range rows {
			currentRight := right
			if correlatedDerived {
				var err error
				currentRight, err = db.virtualJoinRows(ctx, src, doc, stmt, join, &left, params, legacy)
				if err != nil {
					return nil, nil, err
				}
			}
			matched := false
			for j := range currentRight {
				merged := mergeVirtualRows(left, currentRight[j])
				// A graph MATCH relation has already applied its join condition
				// while traversing from the left row. Unlike a relational derived
				// table, it does not require an additional ON expression.
				ok := join.Type == parser.JoinCross || graphMatch
				if !ok && join.OnExpr.Kind != parser.NodeKindUnknown {
					var evalErr error
					ok, evalErr = db.evalVirtualExpr(ctx, src, doc, join.OnExpr, merged, params, legacy)
					if evalErr != nil {
						return nil, nil, evalErr
					}
				}
				if !ok {
					continue
				}
				matched = true
				if !correlatedDerived {
					matchedRight[j] = true
				}
				next = append(next, merged)
			}
			if !matched && (join.Type == parser.JoinLeft || join.Type == parser.JoinFull) {
				next = append(next, left)
			}
		}
		if !correlatedDerived && (join.Type == parser.JoinRight || join.Type == parser.JoinFull) {
			for j := range right {
				if matchedRight[j] {
					continue
				}
				next = append(next, right[j])
			}
		}
		rows = next
	}

	filtered := make([]virtualSQLRow, 0, len(rows))
	for _, row := range rows {
		if stmt.WhereExpr.Kind != parser.NodeKindUnknown {
			ok, err := db.evalVirtualExpr(ctx, src, doc, stmt.WhereExpr, row, params, legacy)
			if err != nil {
				return nil, nil, err
			}
			if !ok {
				continue
			}
		}
		filtered = append(filtered, row)
	}

	projected, columns, err := db.projectVirtualRows(ctx, src, doc, stmt, filtered, params, legacy)
	if err != nil {
		return nil, nil, err
	}
	return projected, columns, nil
}

func (db *Database) virtualIndexedSourceRows(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt, outer *virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) ([]virtualSQLRow, bool, error) {
	if outer != nil || stmt == nil || stmt.FromTable.Kind != parser.NodeKindTableExpr || stmt.FromTable.ID < 0 || int(stmt.FromTable.ID) >= len(doc.TableExprs) {
		return nil, false, nil
	}
	t := &doc.TableExprs[stmt.FromTable.ID]
	if t.IsDerived || t.Temporal || t.TemporalLSN || t.TemporalRange {
		return nil, false, nil
	}
	table := sourceSpan(src, t.Start, t.End)
	col, err := db.GetCollection(table)
	if err != nil {
		return nil, false, err
	}
	if column, operator, valueRef, ok := virtualJSONContainmentPredicate(src, doc, stmt.WhereExpr); ok {
		value, valueOK, err := db.virtualExprValue(ctx, src, doc, valueRef, virtualSQLRow{}, params, legacy)
		if err != nil {
			return nil, true, err
		}
		if !valueOK {
			return []virtualSQLRow{}, true, nil
		}
		records, used, err := col.lookupJSONContainment(ctx, column, operator, value)
		if err != nil || !used {
			return nil, used, err
		}
		alias := sourceSpan(src, t.Alias, t.AliasEnd)
		if alias == "" {
			alias = table
		}
		rows := make([]virtualSQLRow, 0, len(records))
		for _, record := range records {
			values := cloneMetadata(record.Metadata)
			if values == nil {
				values = make(map[string]interface{})
			}
			values["id"] = record.ID
			row := virtualSQLRow{ID: record.ID, Values: values}
			qualifyVirtualRow(&row, alias)
			rows = append(rows, row)
		}
		return rows, true, nil
	}
	column, path, textResult, valueRef, ok := virtualJSONIndexPredicate(src, doc, stmt.WhereExpr)
	if !ok {
		return nil, false, nil
	}
	value, valueOK, err := db.virtualExprValue(ctx, src, doc, valueRef, virtualSQLRow{}, params, legacy)
	if err != nil {
		return nil, true, err
	}
	if !valueOK {
		return []virtualSQLRow{}, true, nil
	}
	var records []Record
	var used bool
	if epochFromContext(ctx) != nil || transactionFromContext(ctx) != nil {
		records, used, err = col.lookupVisibleJSONOverlay(ctx, column, path, textResult, value)
	} else {
		records, used, err = col.lookupIndexedJSON(ctx, column, path, textResult, value)
	}
	if err != nil || !used {
		return nil, used, err
	}
	alias := sourceSpan(src, t.Alias, t.AliasEnd)
	if alias == "" {
		alias = table
	}
	rows := make([]virtualSQLRow, 0, len(records))
	for _, record := range records {
		values := cloneMetadata(record.Metadata)
		if values == nil {
			values = make(map[string]interface{})
		}
		values["id"] = record.ID
		row := virtualSQLRow{ID: record.ID, Values: values}
		qualifyVirtualRow(&row, alias)
		rows = append(rows, row)
	}
	return rows, true, nil
}

func virtualJSONContainmentPredicate(src []byte, doc *parser.QueryDoc, ref parser.NodeRef) (string, lexer.Kind, parser.NodeRef, bool) {
	if doc == nil || ref.Kind != parser.NodeKindBinaryExpr || ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
		return "", 0, parser.NodeRef{}, false
	}
	be := doc.BinaryExprs[ref.ID]
	operator := lexer.Kind(be.Operator)
	if operator != lexer.KindJSONContains && operator != lexer.KindJSONContainedBy && operator != lexer.KindJSONExists {
		return "", 0, parser.NodeRef{}, false
	}
	if be.Left.Kind != parser.NodeKindIdentifier || be.Left.ID < 0 || int(be.Left.ID) >= len(doc.Identifiers) {
		return "", 0, parser.NodeRef{}, false
	}
	id := doc.Identifiers[be.Left.ID]
	return sourceSpan(src, id.Start, id.End), operator, be.Right, true
}

func (db *Database) virtualSourceRows(ctx context.Context, src []byte, doc *parser.QueryDoc, ref parser.NodeRef, outer *virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) ([]virtualSQLRow, error) {
	if ref.Kind == parser.NodeKindUnknown {
		values := make(map[string]interface{})
		if outer != nil {
			values = cloneMetadata(outer.Values)
		}
		return []virtualSQLRow{{Values: values}}, nil
	}
	if ref.Kind != parser.NodeKindTableExpr || ref.ID < 0 || int(ref.ID) >= len(doc.TableExprs) {
		return nil, fmt.Errorf("virtual SELECT requires a relational table source")
	}
	t := &doc.TableExprs[ref.ID]
	if t.TemporalRange {
		return db.temporalRangeRows(ctx, src, doc, t, params)
	}
	if t.IsFunction {
		if t.Function.Kind != parser.NodeKindFunctionExpr || t.Function.ID < 0 || int(t.Function.ID) >= len(doc.FunctionExprs) {
			return nil, fmt.Errorf("invalid table function reference")
		}
		fn := doc.FunctionExprs[t.Function.ID]
		args := make([]interface{}, 0, fn.ArgsCount)
		for i := int32(0); i < fn.ArgsCount; i++ {
			if fn.ArgsStart+i < 0 || int(fn.ArgsStart+i) >= len(doc.FunctionArgs) {
				return nil, fmt.Errorf("invalid table function argument reference")
			}
			functionRow := virtualSQLRow{}
			if outer != nil {
				functionRow = *outer
			}
			value, ok, err := db.virtualExprValue(ctx, src, doc, doc.FunctionArgs[fn.ArgsStart+i], functionRow, params, legacy)
			if err != nil {
				return nil, err
			}
			if !ok {
				return nil, fmt.Errorf("table function argument is unavailable")
			}
			args = append(args, value)
		}
		if strings.EqualFold(sourceSpan(src, fn.NameStart, fn.NameEnd), "GRAPH_SEMIJOIN") {
			rows, err := db.virtualGraphSemijoinRelationRows(ctx, args)
			if err != nil {
				return nil, err
			}
			alias := sourceSpan(src, t.Alias, t.AliasEnd)
			if alias != "" {
				for i := range rows {
					qualifyVirtualRow(&rows[i], alias)
				}
			}
			return rows, nil
		}
		items, handled, err := evaluateJSONArrayExpansion(sourceSpan(src, fn.NameStart, fn.NameEnd), args)
		if err != nil {
			return nil, err
		}
		if !handled {
			return nil, fmt.Errorf("unsupported JSON table function %q", sourceSpan(src, fn.NameStart, fn.NameEnd))
		}
		alias := sourceSpan(src, t.Alias, t.AliasEnd)
		rows := make([]virtualSQLRow, 0, len(items))
		for _, item := range items {
			values := map[string]interface{}{alias: item, "value": item}
			if object, ok := item.(map[string]interface{}); ok {
				for key, value := range object {
					values[key] = value
				}
			}
			row := virtualSQLRow{Values: values}
			qualifyVirtualRow(&row, alias)
			rows = append(rows, row)
		}
		return rows, nil
	}
	if t.IsDerived {
		if t.Derived.Kind != parser.NodeKindSelectStmt || t.Derived.ID < 0 || int(t.Derived.ID) >= len(doc.SelectStmts) {
			return nil, fmt.Errorf("invalid derived table SELECT")
		}
		// Preserve the enclosing row scope. This is what makes a derived table
		// behave like a correlated/lateral relation when it references aliases
		// from one or more outer SELECT levels.
		inner, _, err := db.evaluateVirtualSelectRows(ctx, src, doc, &doc.SelectStmts[t.Derived.ID], outer, params, legacy)
		if err != nil {
			return nil, fmt.Errorf("execute derived table: %w", err)
		}
		alias := sourceSpan(src, t.Alias, t.AliasEnd)
		if alias == "" {
			return nil, fmt.Errorf("derived table requires an alias")
		}
		for i := range inner {
			qualifyVirtualRow(&inner[i], alias)
		}
		return inner, nil
	}
	table := sourceSpan(src, t.Start, t.End)
	if env := virtualCTEsFromContext(ctx); env != nil {
		if cteRows, ok := env[virtualCTEName(table)]; ok {
			alias := sourceSpan(src, t.Alias, t.AliasEnd)
			if alias == "" {
				alias = table
			}
			rows := make([]virtualSQLRow, 0, len(cteRows))
			for _, cteRow := range cteRows {
				row := virtualSQLRow{ID: cteRow.ID, Values: cloneMetadata(cteRow.Values)}
				qualifyVirtualRow(&row, alias)
				if outer != nil {
					row = overlayVirtualRow(*outer, row)
				}
				rows = append(rows, row)
			}
			return rows, nil
		}
	}
	col, err := db.GetCollection(table)
	if err != nil {
		return nil, err
	}
	if state, ok := virtualGraphSemijoinFromContext(ctx); ok && outer == nil && strings.EqualFold(state.collection, table) {
		return db.virtualGraphSemijoinSourceRows(ctx, src, t, state, outer)
	}
	if t.Temporal || t.TemporalLSN {
		return db.virtualTemporalSourceRows(ctx, src, t, col, params, outer)
	}
	records, err := recordsVisibleInContext(ctx, col)
	if err != nil {
		return nil, err
	}
	alias := sourceSpan(src, t.Alias, t.AliasEnd)
	if alias == "" {
		alias = table
	}
	rows := make([]virtualSQLRow, 0, len(records))
	for _, record := range records {
		values := cloneMetadata(record.Metadata)
		if values == nil {
			values = make(map[string]interface{})
		}
		values["id"] = record.ID
		row := virtualSQLRow{ID: record.ID, Values: values}
		qualifyVirtualRow(&row, alias)
		if outer != nil {
			row = overlayVirtualRow(*outer, row)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

// virtualTemporalSourceRows materializes a table source at its AS OF
// TIMESTAMP or AS OF LSN snapshot. This is intentionally query-local so a temporal source
// inside a CTE can be bounded before its rows feed an outer aggregate.
func (db *Database) virtualTemporalSourceRows(ctx context.Context, src []byte, table *parser.TableExpr, col *Collection, params *optimizer.ParameterSet, outer *virtualSQLRow) ([]virtualSQLRow, error) {
	var snapshot *TemporalSnapshot
	var err error
	if table.TemporalLSN {
		lsn, lsnErr := parseTemporalLSN(src, table.LSNStart, table.LSNEnd, params)
		if lsnErr != nil {
			return nil, fmt.Errorf("AS OF LSN: %w", lsnErr)
		}
		snapshot, err = db.SnapshotAtLSN(ctx, lsn)
		if err != nil {
			return nil, fmt.Errorf("AS OF LSN %d: %w", lsn, err)
		}
	} else {
		when, timeErr := parseTemporalRangeTime(src, table.TimestampStart, table.TimestampEnd, params)
		if timeErr != nil {
			return nil, fmt.Errorf("AS OF TIMESTAMP: %w", timeErr)
		}
		snapshot, err = db.SnapshotAt(ctx, when)
		if err != nil {
			return nil, fmt.Errorf("AS OF TIMESTAMP: %w", err)
		}
	}
	defer snapshot.Close()
	alias := sourceSpan(src, table.Alias, table.AliasEnd)
	if alias == "" {
		alias = sourceSpan(src, table.Start, table.End)
	}
	rows := make([]virtualSQLRow, 0)
	err = col.ListVisibleAtLSN(ctx, snapshot.LSN, func(record *Record) bool {
		values := cloneMetadata(record.Metadata)
		if values == nil {
			values = make(map[string]interface{})
		}
		values["id"] = record.ID
		row := virtualSQLRow{ID: record.ID, Values: values}
		qualifyVirtualRow(&row, alias)
		if outer != nil {
			row = overlayVirtualRow(*outer, row)
		}
		rows = append(rows, row)
		return ctx.Err() == nil
	})
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

// virtualJSONIndexPredicate recognizes the narrow, indexable form
// `payload#>>'{a,b}' = $1` (and its #> counterpart). The full evaluator still
// runs afterward, so an index can only remove candidates and never change
// SQL semantics.
func virtualJSONIndexPredicate(src []byte, doc *parser.QueryDoc, ref parser.NodeRef) (column, path string, textResult bool, value parser.NodeRef, ok bool) {
	if doc == nil || ref.Kind != parser.NodeKindBinaryExpr || ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
		return "", "", false, parser.NodeRef{}, false
	}
	comparison := doc.BinaryExprs[ref.ID]
	if lexer.Kind(comparison.Operator) != lexer.KindEquals {
		return "", "", false, parser.NodeRef{}, false
	}
	left, right := comparison.Left, comparison.Right
	if left.Kind != parser.NodeKindBinaryExpr {
		left, right = right, left
	}
	if left.Kind != parser.NodeKindBinaryExpr || left.ID < 0 || int(left.ID) >= len(doc.BinaryExprs) {
		return "", "", false, parser.NodeRef{}, false
	}
	jsonExpr := doc.BinaryExprs[left.ID]
	if lexer.Kind(jsonExpr.Operator) != lexer.KindJSONPath && lexer.Kind(jsonExpr.Operator) != lexer.KindJSONPathText {
		return "", "", false, parser.NodeRef{}, false
	}
	if jsonExpr.Left.Kind != parser.NodeKindIdentifier || jsonExpr.Left.ID < 0 || int(jsonExpr.Left.ID) >= len(doc.Identifiers) {
		return "", "", false, parser.NodeRef{}, false
	}
	if jsonExpr.Right.Kind != parser.NodeKindString || jsonExpr.Right.ID < 0 || int(jsonExpr.Right.ID) >= len(doc.Strings) {
		return "", "", false, parser.NodeRef{}, false
	}
	identifier := doc.Identifiers[jsonExpr.Left.ID]
	literal := doc.Strings[jsonExpr.Right.ID]
	if literal.End <= literal.Start+1 || literal.End > uint32(len(src)) {
		return "", "", false, parser.NodeRef{}, false
	}
	column = sourceSpan(src, identifier.Start, identifier.End)
	path = sourceSpan(src, literal.Start+1, literal.End-1)
	return column, path, lexer.Kind(jsonExpr.Operator) == lexer.KindJSONPathText, right, true
}

func (db *Database) virtualJoinRows(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt, join *parser.JoinClause, outer *virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) ([]virtualSQLRow, error) {
	if join.MatchPath.Kind == parser.NodeKindMatchPath {
		return db.virtualGraphJoinRows(ctx, src, doc, stmt, join, outer, params)
	}
	if join.IsFunction {
		ref := parser.NodeRef{Kind: parser.NodeKindTableExpr, ID: int32(len(doc.TableExprs))}
		doc.TableExprs = append(doc.TableExprs, parser.TableExpr{ID: ref.ID, Function: join.Function, IsFunction: true, Alias: join.Alias, AliasEnd: join.AliasEnd})
		rows, err := db.virtualSourceRows(ctx, src, doc, ref, outer, params, legacy)
		doc.TableExprs = doc.TableExprs[:len(doc.TableExprs)-1]
		return rows, err
	}
	if join.Derived.Kind == parser.NodeKindTableExpr {
		return db.virtualSourceRows(ctx, src, doc, join.Derived, outer, params, legacy)
	}
	if join.TableEnd <= join.TableStart {
		return nil, fmt.Errorf("virtual JOIN requires a table or derived relation")
	}
	ref := parser.NodeRef{Kind: parser.NodeKindTableExpr, ID: int32(len(doc.TableExprs))}
	doc.TableExprs = append(doc.TableExprs, parser.TableExpr{ID: ref.ID, Start: join.TableStart, End: join.TableEnd, Alias: join.Alias, AliasEnd: join.AliasEnd})
	rows, err := db.virtualSourceRows(ctx, src, doc, ref, outer, params, legacy)
	doc.TableExprs = doc.TableExprs[:len(doc.TableExprs)-1]
	return rows, err
}

// virtualGraphJoinRows evaluates one JOIN MATCH stage for one left row.  A
// graph join is a row-dependent relation: the path's first vertex identifies
// the left row and the final vertex becomes the qualified right-side row.
// Keeping this in the virtual executor is what lets graph results participate
// in IN/EXISTS/derived subqueries without creating a temporary collection.
func (db *Database) virtualGraphJoinRows(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt, join *parser.JoinClause, left *virtualSQLRow, params *optimizer.ParameterSet) ([]virtualSQLRow, error) {
	if stmt == nil || left == nil {
		return nil, fmt.Errorf("JOIN MATCH requires a left row")
	}
	if stmt.FromTable.Kind != parser.NodeKindTableExpr || stmt.FromTable.ID < 0 || int(stmt.FromTable.ID) >= len(doc.TableExprs) {
		return nil, fmt.Errorf("JOIN MATCH subquery requires a relational table source")
	}
	from := &doc.TableExprs[stmt.FromTable.ID]
	if from.IsDerived || from.IsFunction {
		return nil, fmt.Errorf("JOIN MATCH subquery requires a base collection source")
	}
	table := sourceSpan(src, from.Start, from.End)
	col, err := db.GetCollection(table)
	if err != nil {
		return nil, err
	}
	g := col.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("JOIN MATCH left collection %q has no graph", table)
	}
	if join.MatchPath.ID < 0 || int(join.MatchPath.ID) >= len(doc.MatchPaths) {
		return nil, fmt.Errorf("invalid JOIN MATCH path reference")
	}
	matchPath := &doc.MatchPaths[join.MatchPath.ID]
	anchorAlias := virtualMatchAnchor(src, doc, matchPath)
	anchorRecordID := left.ID
	if anchorAlias != "" {
		if value, ok := left.Values[anchorAlias+".id"]; ok && value != nil {
			anchorRecordID = recordMetaToString(value)
		}
	}
	if anchorRecordID == "" {
		return nil, nil
	}
	db.mu.RLock()
	cat := db.catalog
	db.mu.RUnlock()
	optimizerEdges, maxHops, err := optimizer.NewOptimizer(cat).ExtractMatchPath(doc, src, matchPath, params)
	if err != nil {
		return nil, err
	}
	if len(optimizerEdges) == 0 {
		return nil, fmt.Errorf("JOIN MATCH path has no edges")
	}
	edges := make([]EdgePlan, len(optimizerEdges))
	for i, edge := range optimizerEdges {
		edges[i] = graphEdgePlanForTraversal(edge)
	}

	nodeID, err := newExecutor(db).lookupNodeIDInContext(ctx, table, anchorRecordID)
	if err != nil {
		return nil, nil
	}
	records, err := recordsVisibleInContext(ctx, col)
	if err != nil {
		return nil, err
	}
	recordsByNode := make(map[uint64]Record, len(records))
	for i := range records {
		id, lookupErr := newExecutor(db).lookupNodeIDInContext(ctx, table, records[i].ID)
		if lookupErr == nil {
			recordsByNode[id] = records[i]
		}
	}

	bitset, err := g.GetBitset()
	if err != nil {
		return nil, err
	}
	defer g.PutBitset(bitset)
	frontier, err := g.GetFrontierBuf()
	if err != nil {
		return nil, err
	}
	defer g.PutFrontierBuf(frontier)

	terminalAlias, terminalLabel := virtualMatchTerminal(src, doc, matchPath)
	labelNodes := map[uint64]struct{}(nil)
	if terminalLabel != "" {
		labelNodes = make(map[uint64]struct{})
		for _, id := range g.GetLabelNodes(terminalLabel) {
			labelNodes[id] = struct{}{}
		}
	}
	terminalIDs := make([]uint64, 0)
	seen := make(map[uint64]struct{})
	if err := g.BFSPattern(nodeID, edges, maxHops, func(id uint64, band int, step int) bool {
		lastBand := len(edges) - 1
		if band != lastBand || step < edges[band].Min {
			return true
		}
		// An unquantified edge is exactly one hop; do not emit the seed.
		if id == nodeID && band == 0 && step == 0 && edges[0].Min > 0 {
			return true
		}
		if len(labelNodes) > 0 {
			if _, ok := labelNodes[id]; !ok {
				return true
			}
		}
		if _, ok := seen[id]; !ok {
			seen[id] = struct{}{}
			terminalIDs = append(terminalIDs, id)
		}
		return true
	}, bitset, frontier); err != nil {
		return nil, err
	}

	rows := make([]virtualSQLRow, 0, len(terminalIDs))
	for _, id := range terminalIDs {
		record, ok := recordsByNode[id]
		if !ok {
			continue
		}
		values := cloneMetadata(record.Metadata)
		if values == nil {
			values = make(map[string]interface{})
		}
		values["id"] = record.ID
		row := virtualSQLRow{ID: record.ID, Values: values}
		if terminalAlias == "" {
			terminalAlias = table
		}
		qualifyVirtualRow(&row, terminalAlias)
		rows = append(rows, row)
	}
	return rows, nil
}

func virtualMatchTerminal(src []byte, doc *parser.QueryDoc, path *parser.MatchPath) (alias, label string) {
	if path == nil {
		return "", ""
	}
	for i := path.PathNodesCount - 1; i >= 0; i-- {
		index := path.PathNodesStart + i
		if index < 0 || int(index) >= len(doc.Nodes) {
			continue
		}
		ref := doc.Nodes[index]
		if ref.Kind != parser.NodeKindVertex || ref.ID < 0 || int(ref.ID) >= len(doc.Vertexes) {
			continue
		}
		vertex := &doc.Vertexes[ref.ID]
		return sourceSpan(src, vertex.Alias, vertex.AliasEnd), sourceSpan(src, vertex.LabelStart, vertex.LabelEnd)
	}
	return "", ""
}

func virtualMatchAnchor(src []byte, doc *parser.QueryDoc, path *parser.MatchPath) string {
	if path == nil {
		return ""
	}
	for i := int32(0); i < path.PathNodesCount; i++ {
		index := path.PathNodesStart + i
		if index < 0 || int(index) >= len(doc.Nodes) {
			continue
		}
		ref := doc.Nodes[index]
		if ref.Kind != parser.NodeKindVertex || ref.ID < 0 || int(ref.ID) >= len(doc.Vertexes) {
			continue
		}
		vertex := &doc.Vertexes[ref.ID]
		return sourceSpan(src, vertex.Alias, vertex.AliasEnd)
	}
	return ""
}

func qualifyVirtualRow(row *virtualSQLRow, alias string) {
	if row == nil || alias == "" {
		return
	}
	for key, value := range row.Values {
		if strings.Contains(key, ".") {
			continue
		}
		row.Values[alias+"."+key] = value
	}
}

func mergeVirtualRows(left, right virtualSQLRow) virtualSQLRow {
	values := cloneMetadata(left.Values)
	if values == nil {
		values = make(map[string]interface{})
	}
	for key, value := range right.Values {
		if _, exists := values[key]; !exists {
			values[key] = value
		}
	}
	id := left.ID
	if id == "" {
		id = right.ID
	}
	return virtualSQLRow{ID: id, Values: values}
}

// overlayVirtualRow adds an inner/subquery row to an outer scope. Unqualified
// names must resolve to the innermost relation, while qualified outer names
// remain available for correlated predicates such as d.author_id = a.id.
func overlayVirtualRow(outer, inner virtualSQLRow) virtualSQLRow {
	values := cloneMetadata(outer.Values)
	if values == nil {
		values = make(map[string]interface{})
	}
	for key, value := range inner.Values {
		if !strings.Contains(key, ".") {
			values[key] = value
			continue
		}
		if _, exists := values[key]; !exists {
			values[key] = value
		}
	}
	id := inner.ID
	if id == "" {
		id = outer.ID
	}
	return virtualSQLRow{ID: id, Values: values}
}

func (db *Database) projectVirtualRows(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) ([]virtualSQLRow, []string, error) {
	if stmt.ProjectionsCount == 0 {
		return rows, nil, nil
	}
	if virtualSelectHasWindow(doc, stmt) {
		if virtualSelectHasAggregate(doc, stmt) || virtualSelectHasCollectionAggregate(src, doc, stmt) {
			var projected []virtualSQLRow
			var err error
			if len(stmt.GroupBy) > 0 {
				projected, _, err = db.projectVirtualGroupedAggregateRows(ctx, src, doc, stmt, rows, params, legacy)
			} else {
				projected, _, err = db.projectVirtualAggregateRows(ctx, src, doc, stmt, rows, params, legacy)
			}
			if err != nil {
				return nil, nil, err
			}
			return db.projectVirtualWindowRows(ctx, src, doc, stmt, projected, params, legacy)
		}
		return db.projectVirtualWindowRows(ctx, src, doc, stmt, rows, params, legacy)
	}
	if virtualSelectHasAggregate(doc, stmt) || virtualSelectHasCollectionAggregate(src, doc, stmt) {
		if len(stmt.GroupBy) > 0 {
			return db.projectVirtualGroupedAggregateRows(ctx, src, doc, stmt, rows, params, legacy)
		}
		return db.projectVirtualAggregateRows(ctx, src, doc, stmt, rows, params, legacy)
	}
	columns := make([]string, 0, stmt.ProjectionsCount)
	for j := int32(0); j < stmt.ProjectionsCount; j++ {
		projection := &doc.Projections[stmt.ProjectionsStart+j]
		if projection.Star {
			if len(rows) > 0 {
				for key := range rows[0].Values {
					if !strings.Contains(key, ".") {
						columns = append(columns, key)
					}
				}
				sort.Strings(columns)
			}
			continue
		}
		name := ""
		switch projection.Expr.Kind {
		case parser.NodeKindIdentifier:
			id := &doc.Identifiers[projection.Expr.ID]
			name = sourceSpan(src, id.Start, id.End)
		case parser.NodeKindSubqueryExpr:
			name = "scalar_subquery"
		case parser.NodeKindBinaryExpr:
			if projection.AliasEnd > projection.Alias {
				name = sourceSpan(src, projection.Alias, projection.AliasEnd)
			} else {
				name = "json_expression"
			}
		case parser.NodeKindFunctionExpr:
			if projection.Expr.ID < 0 || int(projection.Expr.ID) >= len(doc.FunctionExprs) {
				return nil, nil, fmt.Errorf("invalid function expression reference")
			}
			fn := doc.FunctionExprs[projection.Expr.ID]
			name = sourceSpan(src, fn.NameStart, fn.NameEnd)
		case parser.NodeKindCaseExpr:
			name = "case"
		case parser.NodeKindCastExpr:
			name = "cast"
		default:
			return nil, nil, fmt.Errorf("virtual SELECT projection supports identifiers, CASE, casts, JSON expressions, functions, scalar subqueries, and *")
		}
		if projection.AliasEnd > projection.Alias {
			name = sourceSpan(src, projection.Alias, projection.AliasEnd)
		}
		columns = append(columns, name)
	}
	out := make([]virtualSQLRow, 0, len(rows))
	for _, row := range rows {
		values := make(map[string]interface{})
		for j := int32(0); j < stmt.ProjectionsCount; j++ {
			projection := &doc.Projections[stmt.ProjectionsStart+j]
			if projection.Star {
				for key, value := range row.Values {
					if !strings.Contains(key, ".") {
						values[key] = value
					}
				}
				continue
			}
			name := columns[j]
			var value interface{}
			var ok bool
			var err error
			switch projection.Expr.Kind {
			case parser.NodeKindIdentifier:
				value, ok = virtualIdentifierValue(src, &doc.Identifiers[projection.Expr.ID], row)
			case parser.NodeKindSubqueryExpr:
				value, ok, err = db.virtualSubqueryValue(ctx, src, doc, projection.Expr, row, params, legacy)
			case parser.NodeKindBinaryExpr:
				value, ok, err = db.virtualExprValue(ctx, src, doc, projection.Expr, row, params, legacy)
			case parser.NodeKindFunctionExpr:
				value, ok, err = db.virtualExprValue(ctx, src, doc, projection.Expr, row, params, legacy)
			case parser.NodeKindCaseExpr, parser.NodeKindCastExpr:
				value, ok, err = db.virtualExprValue(ctx, src, doc, projection.Expr, row, params, legacy)
			}
			if err != nil {
				return nil, nil, err
			}
			if ok {
				values[name] = value
			} else {
				values[name] = nil
			}
		}
		out = append(out, virtualSQLRow{ID: row.ID, Values: values})
	}
	return out, columns, nil
}

// virtualSelectHasAggregate reports whether a SELECT contains an aggregate
// projection.  Aggregate scalar subqueries are evaluated by the virtual
// relation path, so they must not be sent through the physical planner (which
// would lose the correlated outer-row scope).
func virtualSelectHasAggregate(doc *parser.QueryDoc, stmt *parser.SelectStmt) bool {
	if doc == nil || stmt == nil {
		return false
	}
	for j := int32(0); j < stmt.ProjectionsCount; j++ {
		if stmt.ProjectionsStart+j < 0 || int(stmt.ProjectionsStart+j) >= len(doc.Projections) {
			continue
		}
		projection := doc.Projections[stmt.ProjectionsStart+j]
		refs := make(map[int32]struct{})
		collectVirtualAggregateRefs(doc, projection.Expr, refs)
		for id := range refs {
			if id >= 0 && int(id) < len(doc.AggregateExprs) && !doc.AggregateExprs[id].HasWindow {
				return true
			}
		}
	}
	return false
}

func virtualSelectHasNestedAggregateProjection(doc *parser.QueryDoc, stmt *parser.SelectStmt) bool {
	if doc == nil || stmt == nil {
		return false
	}
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		projection := doc.Projections[stmt.ProjectionsStart+i]
		if projection.Expr.Kind == parser.NodeKindAggregateExpr {
			continue
		}
		refs := make(map[int32]struct{})
		collectVirtualAggregateRefs(doc, projection.Expr, refs)
		for id := range refs {
			if id >= 0 && int(id) < len(doc.AggregateExprs) && !doc.AggregateExprs[id].HasWindow {
				return true
			}
		}
	}
	return false
}

// virtualSelectHasParameterizedAggregate identifies ordinary aggregate
// projections whose input expression depends on a bound SQL parameter. The
// physical aggregate planner is column-oriented; route this narrow shape to
// the virtual evaluator, which evaluates the parameter once per visible row
// and preserves MIN/MAX/SUM NULL and cardinality semantics.
func virtualSelectHasParameterizedAggregate(src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt) bool {
	if doc == nil || stmt == nil {
		return false
	}
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		index := stmt.ProjectionsStart + i
		if index < 0 || int(index) >= len(doc.Projections) {
			continue
		}
		projection := doc.Projections[index]
		if projection.Expr.Kind != parser.NodeKindAggregateExpr || projection.Expr.ID < 0 || int(projection.Expr.ID) >= len(doc.AggregateExprs) {
			continue
		}
		ae := doc.AggregateExprs[projection.Expr.ID]
		if ae.HasWindow {
			continue
		}
		if virtualExprContainsParameter(src, doc, ae.Expr) {
			return true
		}
	}
	return false
}

func virtualExprContainsParameter(src []byte, doc *parser.QueryDoc, ref parser.NodeRef) bool {
	if doc == nil || ref.Kind == parser.NodeKindUnknown {
		return false
	}
	switch ref.Kind {
	case parser.NodeKindIdentifier:
		if ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
			return false
		}
		id := doc.Identifiers[ref.ID]
		return id.Start < uint32(len(src)) && id.End > id.Start && (src[id.Start] == '$' || src[id.Start] == '@')
	case parser.NodeKindBinaryExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
			return false
		}
		be := doc.BinaryExprs[ref.ID]
		return virtualExprContainsParameter(src, doc, be.Left) || virtualExprContainsParameter(src, doc, be.Right)
	case parser.NodeKindUnaryExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.UnaryExprs) {
			return false
		}
		return virtualExprContainsParameter(src, doc, doc.UnaryExprs[ref.ID].Expr)
	case parser.NodeKindCastExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.CastExprs) {
			return false
		}
		return virtualExprContainsParameter(src, doc, doc.CastExprs[ref.ID].Expr)
	case parser.NodeKindFunctionExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.FunctionExprs) {
			return false
		}
		fn := doc.FunctionExprs[ref.ID]
		for i := int32(0); i < fn.ArgsCount; i++ {
			if fn.ArgsStart+i >= 0 && int(fn.ArgsStart+i) < len(doc.FunctionArgs) && virtualExprContainsParameter(src, doc, doc.FunctionArgs[fn.ArgsStart+i]) {
				return true
			}
		}
	}
	return false
}

func virtualSelectHasCollectionAggregate(src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt) bool {
	if doc == nil || stmt == nil {
		return false
	}
	for j := int32(0); j < stmt.ProjectionsCount; j++ {
		if stmt.ProjectionsStart+j < 0 || int(stmt.ProjectionsStart+j) >= len(doc.Projections) {
			continue
		}
		projection := doc.Projections[stmt.ProjectionsStart+j]
		if virtualFunctionNameIsCollectionAggregate(src, doc, projection.Expr) {
			return true
		}
	}
	return false
}

func virtualFunctionNameIsCollectionAggregate(src []byte, doc *parser.QueryDoc, ref parser.NodeRef) bool {
	if doc == nil || ref.Kind != parser.NodeKindFunctionExpr || ref.ID < 0 || int(ref.ID) >= len(doc.FunctionExprs) {
		return false
	}
	fn := doc.FunctionExprs[ref.ID]
	if fn.HasWindow || fn.NameEnd > uint32(len(src)) || fn.NameStart > fn.NameEnd {
		return false
	}
	name := sourceSpan(src, fn.NameStart, fn.NameEnd)
	return strings.EqualFold(name, "array_agg") || strings.EqualFold(name, "string_agg")
}

func virtualSelectHasJSON(src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt) bool {
	if doc == nil || stmt == nil {
		return false
	}
	if stmt.FromTable.Kind == parser.NodeKindTableExpr && stmt.FromTable.ID >= 0 && int(stmt.FromTable.ID) < len(doc.TableExprs) && doc.TableExprs[stmt.FromTable.ID].IsFunction {
		return true
	}
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		if stmt.ProjectionsStart+i >= 0 && int(stmt.ProjectionsStart+i) < len(doc.Projections) &&
			nodeHasJSON(src, doc, doc.Projections[stmt.ProjectionsStart+i].Expr) {
			return true
		}
	}
	if stmt.WhereExpr.Kind != parser.NodeKindUnknown && nodeHasJSON(src, doc, stmt.WhereExpr) {
		return true
	}
	for i := range stmt.Joins {
		if stmt.Joins[i].IsFunction {
			return true
		}
		if stmt.Joins[i].OnExpr.Kind != parser.NodeKindUnknown && nodeHasJSON(src, doc, stmt.Joins[i].OnExpr) {
			return true
		}
	}
	if stmt.OrderBy.Kind != parser.NodeKindUnknown && nodeHasJSON(src, doc, stmt.OrderBy) {
		return true
	}
	return false
}

// virtualSelectHasScalarExpressions identifies SELECT shapes whose result or
// predicate contains a CASE or an explicit SQL cast.  These expressions are
// represented in the parser's expression arena, but are intentionally not
// lowered into the catalog-column projection list by the physical planner.
func virtualSelectHasScalarExpressions(src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt) bool {
	if doc == nil || stmt == nil {
		return false
	}
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		if stmt.ProjectionsStart+i >= 0 && int(stmt.ProjectionsStart+i) < len(doc.Projections) && ((nodeHasScalarExpression(doc, doc.Projections[stmt.ProjectionsStart+i].Expr) && !nodeHasVectorDistanceExpression(doc, doc.Projections[stmt.ProjectionsStart+i].Expr)) || nodeHasScalarFunction(src, doc, doc.Projections[stmt.ProjectionsStart+i].Expr)) {
			return true
		}
	}
	if (nodeHasScalarExpression(doc, stmt.WhereExpr) && !nodeHasVectorDistanceExpression(doc, stmt.WhereExpr)) || (nodeHasScalarExpression(doc, stmt.HavingExpr) && !nodeHasVectorDistanceExpression(doc, stmt.HavingExpr)) || (nodeHasScalarExpression(doc, stmt.OrderBy) && !nodeHasVectorDistanceExpression(doc, stmt.OrderBy)) || nodeHasScalarFunction(src, doc, stmt.WhereExpr) || nodeHasScalarFunction(src, doc, stmt.HavingExpr) || nodeHasScalarFunction(src, doc, stmt.OrderBy) {
		return true
	}
	for i := range stmt.Joins {
		if (nodeHasScalarExpression(doc, stmt.Joins[i].OnExpr) && !nodeHasVectorDistanceExpression(doc, stmt.Joins[i].OnExpr)) || nodeHasScalarFunction(src, doc, stmt.Joins[i].OnExpr) {
			return true
		}
	}
	return false
}

// nodeHasVectorDistanceExpression identifies a pgvector distance operator
// anywhere in an expression tree. An explicit ::vector cast is part of that
// operator's typed operand and must not force the whole SELECT into the
// scalar virtual-row evaluator; the physical vector executor owns both the
// distance and the stored-vector projection.
func nodeHasVectorDistanceExpression(doc *parser.QueryDoc, ref parser.NodeRef) bool {
	if doc == nil || ref.Kind == parser.NodeKindUnknown {
		return false
	}
	switch ref.Kind {
	case parser.NodeKindBinaryExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
			return false
		}
		be := doc.BinaryExprs[ref.ID]
		switch lexer.Kind(be.Operator) {
		case lexer.KindL2Dist, lexer.KindIPDist, lexer.KindCosineDist:
			return true
		}
		return nodeHasVectorDistanceExpression(doc, be.Left) || nodeHasVectorDistanceExpression(doc, be.Right)
	case parser.NodeKindCastExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.CastExprs) {
			return false
		}
		return nodeHasVectorDistanceExpression(doc, doc.CastExprs[ref.ID].Expr)
	case parser.NodeKindUnaryExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.UnaryExprs) {
			return false
		}
		return nodeHasVectorDistanceExpression(doc, doc.UnaryExprs[ref.ID].Expr)
	}
	return false
}

func nodeHasScalarFunction(src []byte, doc *parser.QueryDoc, ref parser.NodeRef) bool {
	if doc == nil || ref.Kind == parser.NodeKindUnknown {
		return false
	}
	if ref.Kind == parser.NodeKindFunctionExpr {
		if ref.ID < 0 || int(ref.ID) >= len(doc.FunctionExprs) {
			return false
		}
		fn := doc.FunctionExprs[ref.ID]
		name := sourceSpan(src, fn.NameStart, fn.NameEnd)
		if strings.EqualFold(name, "now") || strings.EqualFold(name, "nullif") {
			return true
		}
		for i := int32(0); i < fn.ArgsCount; i++ {
			if fn.ArgsStart+i >= 0 && int(fn.ArgsStart+i) < len(doc.FunctionArgs) && nodeHasScalarFunction(src, doc, doc.FunctionArgs[fn.ArgsStart+i]) {
				return true
			}
		}
		return false
	}
	switch ref.Kind {
	case parser.NodeKindBinaryExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.BinaryExprs) {
			be := doc.BinaryExprs[ref.ID]
			return nodeHasScalarFunction(src, doc, be.Left) || nodeHasScalarFunction(src, doc, be.Right)
		}
	case parser.NodeKindUnaryExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.UnaryExprs) {
			return nodeHasScalarFunction(src, doc, doc.UnaryExprs[ref.ID].Expr)
		}
	case parser.NodeKindCaseExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.CaseExprs) {
			ce := doc.CaseExprs[ref.ID]
			for i := int32(0); i < ce.WhensCount; i++ {
				when := doc.CaseWhens[ce.WhensStart+i]
				if nodeHasScalarFunction(src, doc, when.Condition) || nodeHasScalarFunction(src, doc, when.Value) {
					return true
				}
			}
			return ce.HasElse && nodeHasScalarFunction(src, doc, ce.Else)
		}
	case parser.NodeKindCastExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.CastExprs) {
			return nodeHasScalarFunction(src, doc, doc.CastExprs[ref.ID].Expr)
		}
	}
	return false
}

func nodeHasScalarExpression(doc *parser.QueryDoc, ref parser.NodeRef) bool {
	if doc == nil || ref.Kind == parser.NodeKindUnknown {
		return false
	}
	switch ref.Kind {
	case parser.NodeKindCaseExpr, parser.NodeKindCastExpr:
		return true
	case parser.NodeKindBinaryExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
			return false
		}
		be := doc.BinaryExprs[ref.ID]
		return nodeHasScalarExpression(doc, be.Left) || nodeHasScalarExpression(doc, be.Right)
	case parser.NodeKindUnaryExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.UnaryExprs) {
			return false
		}
		return nodeHasScalarExpression(doc, doc.UnaryExprs[ref.ID].Expr)
	case parser.NodeKindBetweenExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.BetweenExprs) {
			return false
		}
		b := doc.BetweenExprs[ref.ID]
		return nodeHasScalarExpression(doc, b.Expr) || nodeHasScalarExpression(doc, b.Lower) || nodeHasScalarExpression(doc, b.Upper)
	case parser.NodeKindInExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.InExprs) {
			return false
		}
		in := doc.InExprs[ref.ID]
		if nodeHasScalarExpression(doc, in.Expr) {
			return true
		}
		for i := int32(0); i < in.ListCount; i++ {
			if in.ListStart+i >= 0 && int(in.ListStart+i) < len(doc.Nodes) && nodeHasScalarExpression(doc, doc.Nodes[in.ListStart+i]) {
				return true
			}
		}
		return false
	case parser.NodeKindFunctionExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.FunctionExprs) {
			return false
		}
		fn := doc.FunctionExprs[ref.ID]
		for i := int32(0); i < fn.ArgsCount; i++ {
			if fn.ArgsStart+i >= 0 && int(fn.ArgsStart+i) < len(doc.FunctionArgs) && nodeHasScalarExpression(doc, doc.FunctionArgs[fn.ArgsStart+i]) {
				return true
			}
		}
	case parser.NodeKindSubqueryExpr:
		return false
	}
	return false
}

func nodeHasJSON(src []byte, doc *parser.QueryDoc, ref parser.NodeRef) bool {
	if doc == nil {
		return false
	}
	switch ref.Kind {
	case parser.NodeKindBinaryExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
			return false
		}
		be := doc.BinaryExprs[ref.ID]
		if lexer.Kind(be.Operator) == lexer.KindConcat {
			// CONCAT is also the PostgreSQL JSONB merge operator. Route it
			// through the virtual evaluator; that evaluator preserves ordinary
			// text concatenation when the operands are not JSON trees.
			return true
		}
		if isJSONOperator(be.Operator) || isJSONPathPredicateOperator(be.Operator) && (be.Operator != uint8(lexer.KindFTSMatch) || jsonPathOperand(src, doc, be.Right)) {
			return true
		}
		return nodeHasJSON(src, doc, be.Left) || nodeHasJSON(src, doc, be.Right)
	case parser.NodeKindUnaryExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.UnaryExprs) {
			return false
		}
		return nodeHasJSON(src, doc, doc.UnaryExprs[ref.ID].Expr)
	case parser.NodeKindCaseExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.CaseExprs) {
			return false
		}
		ce := doc.CaseExprs[ref.ID]
		for i := int32(0); i < ce.WhensCount; i++ {
			when := doc.CaseWhens[ce.WhensStart+i]
			if nodeHasJSON(src, doc, when.Condition) || nodeHasJSON(src, doc, when.Value) {
				return true
			}
		}
		return ce.HasElse && nodeHasJSON(src, doc, ce.Else)
	case parser.NodeKindFunctionExpr:
		return functionNodeHasJSON(src, doc, ref)
	default:
		return false
	}
}

func jsonPathOperand(src []byte, doc *parser.QueryDoc, ref parser.NodeRef) bool {
	if doc == nil || ref.Kind != parser.NodeKindString || ref.ID < 0 || int(ref.ID) >= len(doc.Strings) {
		return false
	}
	lit := doc.Strings[ref.ID]
	if lit.Start >= uint32(len(src)) || lit.End > uint32(len(src)) || lit.End <= lit.Start+1 {
		return false
	}
	path := strings.TrimSpace(sourceSpan(src, lit.Start+1, lit.End-1))
	if strings.HasPrefix(strings.ToLower(path), "strict ") || strings.HasPrefix(strings.ToLower(path), "lax ") {
		path = strings.TrimSpace(path[4:])
	}
	return strings.HasPrefix(path, "$")
}

func functionNodeHasJSON(src []byte, doc *parser.QueryDoc, ref parser.NodeRef) bool {
	if doc == nil || ref.Kind != parser.NodeKindFunctionExpr || ref.ID < 0 || int(ref.ID) >= len(doc.FunctionExprs) {
		return false
	}
	fn := doc.FunctionExprs[ref.ID]
	if fn.NameEnd > uint32(len(src)) {
		return false
	}
	name := src[fn.NameStart:fn.NameEnd]
	for _, candidate := range []string{
		"jsonb_set", "json_set", "jsonb_insert", "json_insert",
		"jsonb_build_array", "json_build_array", "jsonb_build_object", "json_build_object",
		"jsonb_populate_record", "json_populate_record", "to_jsonb", "to_json",
		"jsonb_array_length", "jsonb_typeof", "json_typeof",
	} {
		if strings.EqualFold(string(name), candidate) {
			return true
		}
	}
	return false
}

// virtualSelectHasOrderedSetAggregate identifies ordered-set aggregates that
// require the query-local evaluator. The physical aggregate planner only
// understands the ordinary COUNT/SUM/AVG/MIN/MAX family.
func virtualSelectHasOrderedSetAggregate(doc *parser.QueryDoc, stmt *parser.SelectStmt) bool {
	if doc == nil || stmt == nil {
		return false
	}
	for j := int32(0); j < stmt.ProjectionsCount; j++ {
		index := stmt.ProjectionsStart + j
		if index < 0 || int(index) >= len(doc.Projections) {
			continue
		}
		projection := doc.Projections[index]
		if projection.Expr.Kind != parser.NodeKindAggregateExpr || projection.Expr.ID < 0 || int(projection.Expr.ID) >= len(doc.AggregateExprs) {
			continue
		}
		if doc.AggregateExprs[projection.Expr.ID].OrderedSet {
			return true
		}
	}
	return false
}

// projectVirtualAggregateRows evaluates the aggregate-only SELECT shape used
// by scalar subqueries.  It deliberately follows SQL's NULL rules: COUNT
// returns zero for an empty input, while SUM/AVG/MIN/MAX return NULL when no
// non-NULL argument exists.  Correlated predicates have already been applied
// to rows by evaluateVirtualSelectRows, so this works for both uncorrelated
// and correlated aggregate subqueries without rewriting SQL text.
func (db *Database) projectVirtualAggregateRows(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) ([]virtualSQLRow, []string, error) {
	columns := make([]string, 0, stmt.ProjectionsCount)
	values := make(map[string]interface{}, stmt.ProjectionsCount)
	aggregateRow := virtualSQLRow{Values: values}
	if err := db.materializeVirtualAggregates(ctx, src, doc, stmt, rows, &aggregateRow, params, legacy); err != nil {
		return nil, nil, err
	}
	for j := int32(0); j < stmt.ProjectionsCount; j++ {
		projection := &doc.Projections[stmt.ProjectionsStart+j]
		if (projection.Expr.Kind == parser.NodeKindFunctionExpr && projection.Expr.ID >= 0 && int(projection.Expr.ID) < len(doc.FunctionExprs) && doc.FunctionExprs[projection.Expr.ID].HasWindow) || (projection.Expr.Kind == parser.NodeKindAggregateExpr && projection.Expr.ID >= 0 && int(projection.Expr.ID) < len(doc.AggregateExprs) && doc.AggregateExprs[projection.Expr.ID].HasWindow) {
			name, err := virtualProjectionName(src, doc, projection)
			if err != nil {
				return nil, nil, err
			}
			columns = append(columns, name)
			continue
		}
		switch projection.Expr.Kind {
		case parser.NodeKindAggregateExpr:
			if projection.Expr.ID < 0 || int(projection.Expr.ID) >= len(doc.AggregateExprs) {
				return nil, nil, fmt.Errorf("invalid aggregate expression reference")
			}
			ae := &doc.AggregateExprs[projection.Expr.ID]
			name := aggregateColumnName(uint8(ae.Func))
			if projection.AliasEnd > projection.Alias {
				name = sourceSpan(src, projection.Alias, projection.AliasEnd)
			}
			columns = append(columns, name)
			value, _, err := db.virtualExprValue(ctx, src, doc, projection.Expr, aggregateRow, params, legacy)
			if err != nil {
				return nil, nil, err
			}
			values[name] = value
		case parser.NodeKindBinaryExpr, parser.NodeKindCaseExpr, parser.NodeKindCastExpr:
			name, err := virtualProjectionName(src, doc, projection)
			if err != nil {
				return nil, nil, err
			}
			columns = append(columns, name)
			value, ok, err := db.virtualExprValue(ctx, src, doc, projection.Expr, aggregateRow, params, legacy)
			if err != nil {
				return nil, nil, err
			}
			if ok {
				values[name] = value
			} else {
				values[name] = nil
			}
		case parser.NodeKindFunctionExpr:
			if !virtualFunctionNameIsCollectionAggregate(src, doc, projection.Expr) {
				return nil, nil, fmt.Errorf("aggregate virtual SELECT supports aggregate and window projections only")
			}
			name, err := virtualProjectionName(src, doc, projection)
			if err != nil {
				return nil, nil, err
			}
			columns = append(columns, name)
			value, err := db.evaluateVirtualCollectionAggregate(ctx, src, doc, projection.Expr, rows, params, legacy)
			if err != nil {
				return nil, nil, err
			}
			values[name] = value
		default:
			return nil, nil, fmt.Errorf("aggregate virtual SELECT supports aggregate and window projections only")
		}
	}
	return []virtualSQLRow{{Values: values}}, columns, nil
}

// projectVirtualGroupedAggregateRows evaluates the grouped SELECT shape used
// by aggregate queries whose HAVING clause contains a scalar subquery.  The
// physical planner cannot retain the correlated virtual scope for that shape,
// so grouping and HAVING are performed over the already epoch-visible rows.
func (db *Database) projectVirtualGroupedAggregateRows(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) ([]virtualSQLRow, []string, error) {
	type virtualGroup struct {
		rows []virtualSQLRow
	}
	groups := make(map[string]*virtualGroup)
	order := make([]string, 0)
	for _, row := range rows {
		key := ""
		for _, ref := range stmt.GroupBy {
			value, ok, err := db.virtualExprValue(ctx, src, doc, ref, row, params, legacy)
			if err != nil {
				return nil, nil, err
			}
			if !ok {
				value = nil
			}
			key += aggregateValueKey(value) + "\x00"
		}
		if _, exists := groups[key]; !exists {
			groups[key] = &virtualGroup{}
			order = append(order, key)
		}
		groups[key].rows = append(groups[key].rows, row)
	}

	columns := make([]string, 0, stmt.ProjectionsCount)
	for j := int32(0); j < stmt.ProjectionsCount; j++ {
		projection := &doc.Projections[stmt.ProjectionsStart+j]
		name, err := virtualProjectionName(src, doc, projection)
		if err != nil {
			return nil, nil, err
		}
		columns = append(columns, name)
	}

	out := make([]virtualSQLRow, 0, len(order))
	for _, key := range order {
		group := groups[key]
		if len(group.rows) == 0 {
			continue
		}
		representative := group.rows[0]
		havingRow := virtualSQLRow{ID: representative.ID, Values: cloneMetadata(representative.Values)}
		if havingRow.Values == nil {
			havingRow.Values = make(map[string]interface{})
		}
		if err := db.materializeVirtualAggregates(ctx, src, doc, stmt, group.rows, &havingRow, params, legacy); err != nil {
			return nil, nil, err
		}
		for j := int32(0); j < stmt.ProjectionsCount; j++ {
			projection := &doc.Projections[stmt.ProjectionsStart+j]
			switch projection.Expr.Kind {
			case parser.NodeKindAggregateExpr:
				if projection.Expr.ID < 0 || int(projection.Expr.ID) >= len(doc.AggregateExprs) {
					return nil, nil, fmt.Errorf("invalid aggregate expression reference")
				}
				ae := &doc.AggregateExprs[projection.Expr.ID]
				if ae.HasWindow {
					continue
				}
				value, ok, err := db.virtualExprValue(ctx, src, doc, projection.Expr, havingRow, params, legacy)
				if err != nil {
					return nil, nil, err
				}
				if !ok {
					continue
				}
				if projection.AliasEnd > projection.Alias {
					havingRow.Values[sourceSpan(src, projection.Alias, projection.AliasEnd)] = value
				}
			case parser.NodeKindFunctionExpr:
				if !virtualFunctionNameIsCollectionAggregate(src, doc, projection.Expr) {
					continue
				}
				value, err := db.evaluateVirtualCollectionAggregate(ctx, src, doc, projection.Expr, group.rows, params, legacy)
				if err != nil {
					return nil, nil, err
				}
				fn := doc.FunctionExprs[projection.Expr.ID]
				fnName := sourceSpan(src, fn.NameStart, fn.NameEnd)
				havingRow.Values[fnName] = value
				if projection.AliasEnd > projection.Alias {
					havingRow.Values[sourceSpan(src, projection.Alias, projection.AliasEnd)] = value
				}
			}
			if projection.Expr.Kind != parser.NodeKindFunctionExpr && projection.AliasEnd > projection.Alias {
				value, ok, err := db.virtualExprValue(ctx, src, doc, projection.Expr, havingRow, params, legacy)
				if err != nil {
					return nil, nil, err
				}
				if ok {
					havingRow.Values[sourceSpan(src, projection.Alias, projection.AliasEnd)] = value
				}
			}
		}
		if stmt.HavingExpr.Kind != parser.NodeKindUnknown {
			ok, err := db.evalVirtualExpr(ctx, src, doc, stmt.HavingExpr, havingRow, params, legacy)
			if err != nil {
				return nil, nil, err
			}
			if !ok {
				continue
			}
		}
		values := make(map[string]interface{}, stmt.ProjectionsCount)
		for j := int32(0); j < stmt.ProjectionsCount; j++ {
			projection := &doc.Projections[stmt.ProjectionsStart+j]
			name := columns[j]
			var value interface{}
			var ok bool
			var err error
			switch projection.Expr.Kind {
			case parser.NodeKindIdentifier:
				value, ok = virtualIdentifierValue(src, &doc.Identifiers[projection.Expr.ID], representative)
			case parser.NodeKindAggregateExpr:
				ae := &doc.AggregateExprs[projection.Expr.ID]
				if ae.HasWindow {
					// Window aggregates are evaluated in the subsequent window pass.
					continue
				}
				value, ok, err = db.virtualExprValue(ctx, src, doc, projection.Expr, havingRow, params, legacy)
			case parser.NodeKindBinaryExpr:
				value, ok, err = db.virtualExprValue(ctx, src, doc, projection.Expr, havingRow, params, legacy)
			case parser.NodeKindFunctionExpr:
				if virtualFunctionNameIsCollectionAggregate(src, doc, projection.Expr) {
					value, err = db.evaluateVirtualCollectionAggregate(ctx, src, doc, projection.Expr, group.rows, params, legacy)
					ok = true
				} else if projection.Expr.ID >= 0 && int(projection.Expr.ID) < len(doc.FunctionExprs) && doc.FunctionExprs[projection.Expr.ID].HasWindow {
					// Window values are evaluated in the subsequent window pass.
					continue
				} else {
					return nil, nil, fmt.Errorf("unsupported grouped function projection")
				}
			case parser.NodeKindSubqueryExpr:
				value, ok, err = db.virtualSubqueryValue(ctx, src, doc, projection.Expr, representative, params, legacy)
			case parser.NodeKindCaseExpr, parser.NodeKindCastExpr:
				value, ok, err = db.virtualExprValue(ctx, src, doc, projection.Expr, representative, params, legacy)
			default:
				return nil, nil, fmt.Errorf("unsupported grouped virtual projection")
			}
			if err != nil {
				return nil, nil, err
			}
			if ok {
				values[name] = value
			} else {
				values[name] = nil
			}
		}
		out = append(out, virtualSQLRow{ID: representative.ID, Values: values})
	}
	return out, columns, nil
}

func virtualProjectionName(src []byte, doc *parser.QueryDoc, projection *parser.Projection) (string, error) {
	if projection == nil {
		return "", fmt.Errorf("nil virtual projection")
	}
	name := ""
	switch projection.Expr.Kind {
	case parser.NodeKindIdentifier:
		id := &doc.Identifiers[projection.Expr.ID]
		name = sourceSpan(src, id.Start, id.End)
	case parser.NodeKindAggregateExpr:
		if projection.Expr.ID < 0 || int(projection.Expr.ID) >= len(doc.AggregateExprs) {
			return "", fmt.Errorf("invalid aggregate projection reference")
		}
		name = aggregateColumnName(uint8(doc.AggregateExprs[projection.Expr.ID].Func))
	case parser.NodeKindSubqueryExpr:
		name = "scalar_subquery"
	case parser.NodeKindFunctionExpr:
		if projection.Expr.ID < 0 || int(projection.Expr.ID) >= len(doc.FunctionExprs) {
			return "", fmt.Errorf("invalid function projection reference")
		}
		fn := &doc.FunctionExprs[projection.Expr.ID]
		name = sourceSpan(src, fn.NameStart, fn.NameEnd)
	case parser.NodeKindCaseExpr:
		name = "case"
	case parser.NodeKindCastExpr:
		name = "cast"
	case parser.NodeKindBinaryExpr:
		name = "expression"
	default:
		return "", fmt.Errorf("unsupported virtual projection %d", projection.Expr.Kind)
	}
	if projection.AliasEnd > projection.Alias {
		name = sourceSpan(src, projection.Alias, projection.AliasEnd)
	}
	return name, nil
}

// virtualAggregateExprKey distinguishes aggregate expressions with the same
// function name. SUM(alpha) and SUM(alpha + beta) must remain separate values
// when a later expression combines them.
func virtualAggregateExprKey(id int32) string {
	return "__virtual_aggregate_" + strconv.FormatInt(int64(id), 10)
}

func (db *Database) materializeVirtualAggregates(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt, rows []virtualSQLRow, aggregateRow *virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) error {
	if aggregateRow == nil {
		return fmt.Errorf("nil virtual aggregate row")
	}
	if aggregateRow.Values == nil {
		aggregateRow.Values = make(map[string]interface{})
	}
	refs := make(map[int32]struct{})
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		collectVirtualAggregateRefs(doc, doc.Projections[stmt.ProjectionsStart+i].Expr, refs)
	}
	collectVirtualAggregateRefs(doc, stmt.HavingExpr, refs)
	for id := range refs {
		if id < 0 || int(id) >= len(doc.AggregateExprs) {
			return fmt.Errorf("invalid aggregate expression reference")
		}
		ae := &doc.AggregateExprs[id]
		if ae.HasWindow {
			continue
		}
		value, err := db.evaluateVirtualAggregate(ctx, src, doc, ae, rows, params, legacy)
		if err != nil {
			return err
		}
		aggregateRow.Values[virtualAggregateExprKey(id)] = value
		// Keep the legacy function-name slot for existing direct aggregate and
		// HAVING paths. Expression-identity lookup remains authoritative.
		name := aggregateColumnName(uint8(ae.Func))
		if _, exists := aggregateRow.Values[name]; !exists {
			aggregateRow.Values[name] = value
		}
	}
	return nil
}

func collectVirtualAggregateRefs(doc *parser.QueryDoc, ref parser.NodeRef, seen map[int32]struct{}) {
	if doc == nil || ref.Kind == parser.NodeKindUnknown {
		return
	}
	switch ref.Kind {
	case parser.NodeKindAggregateExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.AggregateExprs) {
			seen[ref.ID] = struct{}{}
		}
	case parser.NodeKindBinaryExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.BinaryExprs) {
			be := doc.BinaryExprs[ref.ID]
			collectVirtualAggregateRefs(doc, be.Left, seen)
			collectVirtualAggregateRefs(doc, be.Right, seen)
		}
	case parser.NodeKindUnaryExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.UnaryExprs) {
			collectVirtualAggregateRefs(doc, doc.UnaryExprs[ref.ID].Expr, seen)
		}
	case parser.NodeKindCaseExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.CaseExprs) {
			ce := doc.CaseExprs[ref.ID]
			for i := int32(0); i < ce.WhensCount; i++ {
				when := doc.CaseWhens[ce.WhensStart+i]
				collectVirtualAggregateRefs(doc, when.Condition, seen)
				collectVirtualAggregateRefs(doc, when.Value, seen)
			}
			if ce.HasElse {
				collectVirtualAggregateRefs(doc, ce.Else, seen)
			}
		}
	case parser.NodeKindFunctionExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.FunctionExprs) {
			fn := doc.FunctionExprs[ref.ID]
			for i := int32(0); i < fn.ArgsCount; i++ {
				collectVirtualAggregateRefs(doc, doc.FunctionArgs[fn.ArgsStart+i], seen)
			}
		}
	case parser.NodeKindBetweenExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.BetweenExprs) {
			between := doc.BetweenExprs[ref.ID]
			collectVirtualAggregateRefs(doc, between.Expr, seen)
			collectVirtualAggregateRefs(doc, between.Lower, seen)
			collectVirtualAggregateRefs(doc, between.Upper, seen)
		}
	case parser.NodeKindInExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.InExprs) {
			in := doc.InExprs[ref.ID]
			collectVirtualAggregateRefs(doc, in.Expr, seen)
			for i := int32(0); i < in.ListCount; i++ {
				collectVirtualAggregateRefs(doc, doc.Nodes[in.ListStart+i], seen)
			}
		}
	}
}

func (db *Database) evaluateVirtualAggregate(ctx context.Context, src []byte, doc *parser.QueryDoc, ae *parser.AggregateExpr, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) (interface{}, error) {
	if ae == nil {
		return nil, fmt.Errorf("nil aggregate expression")
	}
	if ae.OrderedSet {
		return db.evaluateOrderedSetAggregate(ctx, src, doc, ae, rows, params, legacy)
	}
	count := int64(0)
	var sum float64
	var minValue, maxValue interface{}
	seen := make(map[string]struct{})
	hasValue := false
	for _, row := range rows {
		var value interface{}
		valueOK := true
		if ae.Expr.Kind != parser.NodeKindUnknown {
			var err error
			value, valueOK, err = db.virtualExprValue(ctx, src, doc, ae.Expr, row, params, legacy)
			if err != nil {
				return nil, err
			}
		}
		if ae.Expr.Kind == parser.NodeKindUnknown {
			value = row.ID
			valueOK = true
		}
		if !valueOK || value == nil {
			continue
		}
		if ae.Distinct {
			key := aggregateValueKey(value)
			if _, exists := seen[key]; exists {
				continue
			}
			seen[key] = struct{}{}
		}
		if ae.Func == parser.AggCount {
			count++
			continue
		}
		if !hasValue {
			minValue, maxValue = value, value
			hasValue = true
		} else {
			if aggregateValueLess(value, minValue) {
				minValue = value
			}
			if aggregateValueLess(maxValue, value) {
				maxValue = value
			}
		}
		if ae.Func == parser.AggSum || ae.Func == parser.AggAvg {
			n, ok := toFloat(value)
			if !ok {
				return nil, fmt.Errorf("aggregate %s requires numeric values", aggregateColumnName(uint8(ae.Func)))
			}
			sum += n
		}
		count++
	}
	if ae.Func == parser.AggCount {
		return count, nil
	}
	if !hasValue {
		return nil, nil
	}
	switch ae.Func {
	case parser.AggSum:
		return sum, nil
	case parser.AggAvg:
		return sum / float64(count), nil
	case parser.AggMin:
		return minValue, nil
	case parser.AggMax:
		return maxValue, nil
	default:
		return nil, fmt.Errorf("unsupported aggregate function %d", ae.Func)
	}
}

func (db *Database) evaluateVirtualCollectionAggregate(ctx context.Context, src []byte, doc *parser.QueryDoc, ref parser.NodeRef, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) (interface{}, error) {
	if !virtualFunctionNameIsCollectionAggregate(src, doc, ref) {
		return nil, fmt.Errorf("unsupported collection aggregate")
	}
	fn := doc.FunctionExprs[ref.ID]
	name := sourceSpan(src, fn.NameStart, fn.NameEnd)
	if fn.ArgsCount < 1 || (strings.EqualFold(name, "string_agg") && fn.ArgsCount != 2) || (strings.EqualFold(name, "array_agg") && fn.ArgsCount != 1) {
		return nil, fmt.Errorf("%s expects %d argument(s)", strings.ToUpper(name), map[bool]int{true: 2, false: 1}[strings.EqualFold(name, "string_agg")])
	}

	values := make([]interface{}, 0, len(rows))
	var delimiter string
	if strings.EqualFold(name, "string_agg") && len(rows) > 0 {
		delimiterValue, ok, err := db.virtualExprValue(ctx, src, doc, doc.FunctionArgs[fn.ArgsStart+1], rows[0], params, legacy)
		if err != nil {
			return nil, err
		}
		if !ok || delimiterValue == nil {
			return nil, nil
		}
		delimiter = virtualAggregateText(delimiterValue)
	}

	for _, row := range rows {
		value, ok, err := db.virtualExprValue(ctx, src, doc, doc.FunctionArgs[fn.ArgsStart], row, params, legacy)
		if err != nil {
			return nil, err
		}
		if strings.EqualFold(name, "array_agg") {
			if !ok {
				value = nil
			}
			values = append(values, value)
			continue
		}
		if !ok || value == nil {
			continue
		}
		values = append(values, virtualAggregateText(value))
	}
	if len(values) == 0 {
		return nil, nil
	}
	if strings.EqualFold(name, "string_agg") {
		var out strings.Builder
		for i, value := range values {
			if i > 0 {
				out.WriteString(delimiter)
			}
			out.WriteString(value.(string))
		}
		return out.String(), nil
	}
	return values, nil
}

func virtualAggregateText(value interface{}) string {
	switch v := value.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	case nil:
		return ""
	default:
		return fmt.Sprint(v)
	}
}

type orderedSetValue struct {
	value interface{}
	valid bool
}

// evaluateOrderedSetAggregate implements the high-value PostgreSQL ordered
// set aggregates for scalar and grouped queries. Values are sorted according
// to the WITHIN GROUP order expression; NULL order values are ignored, as
// they cannot contribute to a percentile or mode result.
func (db *Database) evaluateOrderedSetAggregate(ctx context.Context, src []byte, doc *parser.QueryDoc, ae *parser.AggregateExpr, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) (interface{}, error) {
	if ae.OrderExpr.Kind == parser.NodeKindUnknown {
		return nil, fmt.Errorf("ordered-set aggregate is missing WITHIN GROUP ORDER BY expression")
	}
	values := make([]orderedSetValue, 0, len(rows))
	for _, row := range rows {
		value, ok, err := db.virtualExprValue(ctx, src, doc, ae.OrderExpr, row, params, legacy)
		if err != nil {
			return nil, err
		}
		if ok && value != nil {
			values = append(values, orderedSetValue{value: value, valid: true})
		}
	}
	if len(values) == 0 {
		return nil, nil
	}
	sort.SliceStable(values, func(i, j int) bool {
		cmp := compareVirtualValues(values[i].value, values[j].value)
		if ae.OrderDesc {
			return cmp > 0
		}
		return cmp < 0
	})

	switch ae.Func {
	case parser.AggMode:
		best := values[0].value
		bestCount := 0
		for i := 0; i < len(values); {
			j := i + 1
			for j < len(values) && compareVirtualValues(values[i].value, values[j].value) == 0 {
				j++
			}
			if j-i > bestCount {
				best, bestCount = values[i].value, j-i
			}
			i = j
		}
		return best, nil
	case parser.AggPercentileCont, parser.AggPercentileDisc:
		if ae.Expr.Kind == parser.NodeKindUnknown {
			return nil, fmt.Errorf("ordered-set percentile is missing fraction")
		}
		fractionValue, ok, err := db.virtualExprValue(ctx, src, doc, ae.Expr, virtualSQLRow{}, params, legacy)
		if err != nil {
			return nil, err
		}
		if !ok || fractionValue == nil {
			return nil, nil
		}
		fraction, ok := toFloat(fractionValue)
		if !ok || fraction < 0 || fraction > 1 {
			return nil, fmt.Errorf("ordered-set percentile fraction must be between 0 and 1")
		}
		if ae.Func == parser.AggPercentileDisc {
			index := int(math.Ceil(fraction*float64(len(values)))) - 1
			if index < 0 {
				index = 0
			}
			if index >= len(values) {
				index = len(values) - 1
			}
			return values[index].value, nil
		}
		if len(values) == 1 {
			value, ok := toFloat(values[0].value)
			if !ok {
				return nil, fmt.Errorf("PERCENTILE_CONT requires numeric ORDER BY values")
			}
			return value, nil
		}
		position := fraction * float64(len(values)-1)
		lower := int(math.Floor(position))
		upper := int(math.Ceil(position))
		lowerValue, lowerOK := toFloat(values[lower].value)
		upperValue, upperOK := toFloat(values[upper].value)
		if !lowerOK || !upperOK {
			return nil, fmt.Errorf("PERCENTILE_CONT requires numeric ORDER BY values")
		}
		if lower == upper {
			return lowerValue, nil
		}
		weight := position - float64(lower)
		return lowerValue + (upperValue-lowerValue)*weight, nil
	default:
		return nil, fmt.Errorf("unsupported ordered-set aggregate %d", ae.Func)
	}
}

func aggregateValueKey(value interface{}) string {
	return fmt.Sprintf("%T:%v", value, value)
}

func aggregateValueLess(left, right interface{}) bool {
	return compareVirtualValues(left, right) < 0
}

func compareVirtualValues(left, right interface{}) int {
	lf, lok := toFloat(left)
	rf, rok := toFloat(right)
	if lok && rok {
		switch {
		case lf < rf:
			return -1
		case lf > rf:
			return 1
		default:
			return 0
		}
	}
	ls, rs := recordMetaToString(left), recordMetaToString(right)
	if ls < rs {
		return -1
	}
	if ls > rs {
		return 1
	}
	return 0
}

func (db *Database) virtualSubqueryValue(ctx context.Context, src []byte, doc *parser.QueryDoc, ref parser.NodeRef, outer virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) (interface{}, bool, error) {
	if ref.Kind != parser.NodeKindSubqueryExpr || ref.ID < 0 || int(ref.ID) >= len(doc.SubqueryExprs) {
		return nil, false, fmt.Errorf("invalid scalar subquery reference")
	}
	sq := &doc.SubqueryExprs[ref.ID]
	if sq.Stmt.Kind != parser.NodeKindSelectStmt || sq.Stmt.ID < 0 || int(sq.Stmt.ID) >= len(doc.SelectStmts) {
		return nil, false, fmt.Errorf("invalid scalar subquery SELECT")
	}
	rows, columns, err := db.evaluateVirtualSelectRows(ctx, src, doc, &doc.SelectStmts[sq.Stmt.ID], &outer, params, legacy)
	if err != nil {
		return nil, false, err
	}
	if sq.Exists {
		return len(rows) > 0, true, nil
	}
	if len(rows) == 0 {
		// A scalar subquery with no rows is SQL NULL.
		return nil, true, nil
	}
	value, ok := firstVirtualValue(rows[0], columns)
	if !ok {
		return nil, true, nil
	}
	return value, true, nil
}

func (db *Database) evalVirtualExpr(ctx context.Context, src []byte, doc *parser.QueryDoc, ref parser.NodeRef, row virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) (bool, error) {
	switch ref.Kind {
	case parser.NodeKindCaseExpr, parser.NodeKindCastExpr:
		value, ok, err := db.virtualExprValue(ctx, src, doc, ref, row, params, legacy)
		if err != nil || !ok || value == nil {
			return false, err
		}
		if boolean, isBool := value.(bool); isBool {
			return boolean, nil
		}
		return true, nil
	case parser.NodeKindBinaryExpr:
		be := &doc.BinaryExprs[ref.ID]
		if be.Operator == uint8(lexer.KindAnd) {
			left, err := db.evalVirtualExpr(ctx, src, doc, be.Left, row, params, legacy)
			if err != nil || !left {
				return left, err
			}
			return db.evalVirtualExpr(ctx, src, doc, be.Right, row, params, legacy)
		}
		if be.Operator == uint8(lexer.KindOr) {
			left, err := db.evalVirtualExpr(ctx, src, doc, be.Left, row, params, legacy)
			if err != nil || left {
				return left, err
			}
			return db.evalVirtualExpr(ctx, src, doc, be.Right, row, params, legacy)
		}
		left, lok, err := db.virtualExprValue(ctx, src, doc, be.Left, row, params, legacy)
		if err != nil {
			return false, err
		}
		if be.NullTest != parser.NullTestNone {
			isNull := !lok || left == nil
			if be.NullTest == parser.NullTestIsNull {
				return isNull, nil
			}
			return !isNull, nil
		}
		right, rok, err := db.virtualExprValue(ctx, src, doc, be.Right, row, params, legacy)
		if err != nil {
			return false, err
		}
		if !lok || !rok || left == nil || right == nil {
			return false, nil
		}
		switch be.Operator {
		case uint8(lexer.KindEquals):
			return sqlValueEqual(left, right), nil
		case uint8(lexer.KindGreaterThan):
			return compareVirtualValues(left, right) > 0, nil
		case uint8(lexer.KindLessThan):
			return compareVirtualValues(left, right) < 0, nil
		case uint8(lexer.KindGreaterEqual):
			return compareVirtualValues(left, right) >= 0, nil
		case uint8(lexer.KindLessEqual):
			return compareVirtualValues(left, right) <= 0, nil
		case uint8(lexer.KindNotEqual):
			return !sqlValueEqual(left, right), nil
		case uint8(lexer.KindLike), uint8(lexer.KindILike):
			return virtualLikeMatch(recordMetaToString(left), recordMetaToString(right), be.Operator == uint8(lexer.KindILike)), nil
		case uint8(lexer.KindJSONContains), uint8(lexer.KindJSONContainedBy), uint8(lexer.KindJSONExists),
			uint8(lexer.KindJSONAny), uint8(lexer.KindJSONAll), uint8(lexer.KindJSONPathExists), uint8(lexer.KindFTSMatch):
			matched, _, evalErr := evaluateJSONBinary(be.Operator, left, right)
			if evalErr != nil {
				return false, evalErr
			}
			boolean, _ := matched.(bool)
			return boolean, nil
		default:
			return false, nil
		}
	case parser.NodeKindBetweenExpr:
		bw := &doc.BetweenExprs[ref.ID]
		value, vok, err := db.virtualExprValue(ctx, src, doc, bw.Expr, row, params, legacy)
		if err != nil || !vok || value == nil {
			return false, err
		}
		lower, lok, err := db.virtualExprValue(ctx, src, doc, bw.Lower, row, params, legacy)
		if err != nil || !lok || lower == nil {
			return false, err
		}
		upper, uok, err := db.virtualExprValue(ctx, src, doc, bw.Upper, row, params, legacy)
		if err != nil || !uok || upper == nil {
			return false, err
		}
		matched := compareVirtualValues(value, lower) >= 0 && compareVirtualValues(value, upper) <= 0
		if bw.Not {
			return !matched, nil
		}
		return matched, nil
	case parser.NodeKindInExpr:
		in := &doc.InExprs[ref.ID]
		value, ok, err := db.virtualExprValue(ctx, src, doc, in.Expr, row, params, legacy)
		if err != nil || !ok || value == nil {
			return false, err
		}
		matched := false
		if in.HasSubquery {
			if semijoin, optimized := virtualGraphSemijoinFromContext(ctx); optimized && semijoin.subqueryID == in.Subquery.ID {
				_, matched = semijoin.candidateSet[recordMetaToString(value)]
				if in.Not && !matched && semijoin.subqueryHasNull {
					return false, nil
				}
			} else {
				candidate, exists, err := db.virtualSubqueryRows(ctx, src, doc, in.Subquery, row, params, legacy)
				if err != nil {
					return false, err
				}
				if exists {
					for _, subrow := range candidate {
						candidateValue, candidateOK := firstVirtualValue(subrow, nil)
						if candidateOK && candidateValue != nil && sqlValueEqual(value, candidateValue) {
							matched = true
							break
						}
					}
				}
			}
		} else {
			for i := int32(0); i < in.ListCount; i++ {
				candidate, candidateOK, err := db.virtualExprValue(ctx, src, doc, doc.Nodes[in.ListStart+i], row, params, legacy)
				if err != nil {
					return false, err
				}
				if candidateOK && candidate != nil && sqlValueEqual(value, candidate) {
					matched = true
					break
				}
			}
		}
		if in.Not {
			return !matched, nil
		}
		return matched, nil
	case parser.NodeKindSubqueryExpr:
		value, ok, err := db.virtualSubqueryValue(ctx, src, doc, ref, row, params, legacy)
		if err != nil || !ok || value == nil {
			return false, err
		}
		if boolean, ok := value.(bool); ok {
			return boolean, nil
		}
		return true, nil
	case parser.NodeKindUnaryExpr:
		un := &doc.UnaryExprs[ref.ID]
		value, err := db.evalVirtualExpr(ctx, src, doc, un.Expr, row, params, legacy)
		if err != nil {
			return false, err
		}
		if un.Operator == uint8(lexer.KindNot) {
			return !value, nil
		}
		return value, nil
	case parser.NodeKindFunctionExpr:
		value, ok, err := db.virtualExprValue(ctx, src, doc, ref, row, params, legacy)
		if err != nil || !ok || value == nil {
			return false, err
		}
		if boolean, isBool := value.(bool); isBool {
			return boolean, nil
		}
		return true, nil
	}
	return false, nil
}

func (db *Database) virtualExprValue(ctx context.Context, src []byte, doc *parser.QueryDoc, ref parser.NodeRef, row virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) (interface{}, bool, error) {
	switch ref.Kind {
	case parser.NodeKindIdentifier:
		id := &doc.Identifiers[ref.ID]
		identifierText := sourceSpan(src, id.Start, id.End)
		// TRUE/FALSE are ordinary identifier tokens in the shared lexer. The
		// virtual JSON/subquery path intentionally runs before catalog binding,
		// so recognize these literals from their source span as well as from a
		// binder-provided ResolvedKindLiteral.
		switch {
		case strings.EqualFold(identifierText, "NULL"):
			return nil, true, nil
		case strings.EqualFold(identifierText, "TRUE"):
			return true, true, nil
		case strings.EqualFold(identifierText, "FALSE"):
			return false, true, nil
		}
		if len(identifierText) >= 6 && strings.EqualFold(identifierText[:6], "array[") {
			array, ok := parseJSONArrayConstructor(identifierText)
			return array, ok, nil
		}
		if id.Start < uint32(len(src)) && (src[id.Start] == '$' || src[id.Start] == '@') && params != nil {
			if value, found := params.Lookup(src, id.Start, id.End); found {
				return virtualScalarInterface(value), true, nil
			}
		}
		value, ok := virtualIdentifierValue(src, id, row)
		return value, ok, nil
	case parser.NodeKindString:
		literal := &doc.Strings[ref.ID]
		if literal.End <= literal.Start+1 {
			return "", true, nil
		}
		return sourceSpan(src, literal.Start+1, literal.End-1), true, nil
	case parser.NodeKindNumber:
		literal := &doc.Numbers[ref.ID]
		text := sourceSpan(src, literal.Start, literal.End)
		if strings.ContainsAny(text, ".eE") {
			value, err := strconv.ParseFloat(text, 64)
			if err != nil {
				return nil, false, fmt.Errorf("invalid numeric literal %q", text)
			}
			return value, true, nil
		}
		value, err := strconv.ParseInt(text, 10, 64)
		if err == nil {
			return value, true, nil
		}
		unsigned, unsignedErr := strconv.ParseUint(text, 10, 64)
		if unsignedErr == nil {
			return unsigned, true, nil
		}
		return nil, false, fmt.Errorf("invalid numeric literal %q", text)
	case parser.NodeKindCastExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.CastExprs) {
			return nil, false, fmt.Errorf("invalid cast expression reference")
		}
		cast := doc.CastExprs[ref.ID]
		value, ok, err := db.virtualExprValue(ctx, src, doc, cast.Expr, row, params, legacy)
		if err != nil || !ok {
			return value, ok, err
		}
		if value == nil {
			return nil, true, nil
		}
		typeName := sourceSpan(src, cast.TypeStart, cast.TypeEnd)
		typeName = strings.ToLower(strings.TrimSpace(typeName))
		switch typeName {
		case "json", "jsonb":
			decoded, valid := decodeJSONValue(value)
			if !valid {
				return nil, false, fmt.Errorf("cannot cast value to %s: invalid JSON", typeName)
			}
			return decoded, true, nil
		case "text", "varchar", "character varying", "char", "string":
			if object, isObject := value.(map[string]interface{}); isObject {
				encoded, err := json.Marshal(object)
				if err != nil {
					return nil, false, err
				}
				return string(encoded), true, nil
			}
			if array, isArray := value.([]interface{}); isArray {
				encoded, err := json.Marshal(array)
				if err != nil {
					return nil, false, err
				}
				return string(encoded), true, nil
			}
			return recordMetaToString(value), true, nil
		case "uuid":
			text := recordMetaToString(value)
			if !validVirtualUUID(text) {
				return nil, false, fmt.Errorf("cannot cast %q to uuid", text)
			}
			return text, true, nil
		case "int", "int2", "int4", "integer", "smallint", "bigint":
			parsed, err := strconv.ParseInt(strings.TrimSpace(recordMetaToString(value)), 10, 64)
			if err != nil {
				return nil, false, fmt.Errorf("cannot cast %q to %s: %w", recordMetaToString(value), typeName, err)
			}
			return parsed, true, nil
		case "float", "float4", "float8", "real", "double", "double precision", "numeric", "decimal":
			parsed, err := strconv.ParseFloat(strings.TrimSpace(recordMetaToString(value)), 64)
			if err != nil {
				return nil, false, fmt.Errorf("cannot cast %q to %s: %w", recordMetaToString(value), typeName, err)
			}
			return parsed, true, nil
		case "bool", "boolean":
			parsed, err := strconv.ParseBool(strings.TrimSpace(recordMetaToString(value)))
			if err != nil {
				return nil, false, fmt.Errorf("cannot cast %q to %s: %w", recordMetaToString(value), typeName, err)
			}
			return parsed, true, nil
		case "vector":
			vector := parseVectorLiteral(strings.TrimSpace(recordMetaToString(value)))
			if len(vector) == 0 {
				return nil, false, fmt.Errorf("cannot cast %q to vector", recordMetaToString(value))
			}
			return vector, true, nil
		case "timestamp", "timestamptz", "date":
			instant, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(recordMetaToString(value)))
			if err != nil {
				return nil, false, fmt.Errorf("cannot cast %q to %s: %w", recordMetaToString(value), typeName, err)
			}
			return instant, true, nil
		default:
			// Never silently treat an unknown cast target as the source type.
			// That would make a typo such as ::jston appear to succeed while
			// returning a value with the wrong SQL type and semantics.
			return nil, false, fmt.Errorf("unsupported cast target type %q", typeName)
		}
	case parser.NodeKindCaseExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.CaseExprs) {
			return nil, false, fmt.Errorf("invalid CASE expression reference")
		}
		ce := doc.CaseExprs[ref.ID]
		if ce.WhensStart < 0 || ce.WhensCount < 0 || ce.WhensStart+ce.WhensCount > int32(len(doc.CaseWhens)) {
			return nil, false, fmt.Errorf("invalid CASE branch range")
		}
		for i := int32(0); i < ce.WhensCount; i++ {
			when := doc.CaseWhens[ce.WhensStart+i]
			matched, err := db.evalVirtualExpr(ctx, src, doc, when.Condition, row, params, legacy)
			if err != nil {
				return nil, false, err
			}
			if matched {
				return db.virtualExprValue(ctx, src, doc, when.Value, row, params, legacy)
			}
		}
		if ce.HasElse {
			return db.virtualExprValue(ctx, src, doc, ce.Else, row, params, legacy)
		}
		return nil, true, nil
	case parser.NodeKindBinaryExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
			return nil, false, fmt.Errorf("invalid binary expression reference")
		}
		be := &doc.BinaryExprs[ref.ID]
		if !isJSONOperator(be.Operator) && !isJSONPathPredicateOperator(be.Operator) && lexer.Kind(be.Operator) != lexer.KindConcat {
			return db.evaluateVirtualScalarBinary(ctx, src, doc, be, row, params, legacy)
		}
		left, leftOK, err := db.virtualExprValue(ctx, src, doc, be.Left, row, params, legacy)
		if err != nil || !leftOK {
			return nil, false, err
		}
		right, rightOK, err := db.virtualExprValue(ctx, src, doc, be.Right, row, params, legacy)
		if err != nil || !rightOK {
			return nil, false, err
		}
		return evaluateJSONBinary(be.Operator, left, right)
	case parser.NodeKindFunctionExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.FunctionExprs) {
			return nil, false, fmt.Errorf("invalid function expression reference")
		}
		fn := doc.FunctionExprs[ref.ID]
		if fn.NameEnd > uint32(len(src)) {
			return nil, false, fmt.Errorf("invalid function name span")
		}
		args := make([]interface{}, 0, fn.ArgsCount)
		for i := int32(0); i < fn.ArgsCount; i++ {
			if fn.ArgsStart+i < 0 || int(fn.ArgsStart+i) >= len(doc.FunctionArgs) {
				return nil, false, fmt.Errorf("invalid function argument reference")
			}
			value, ok, err := db.virtualExprValue(ctx, src, doc, doc.FunctionArgs[fn.ArgsStart+i], row, params, legacy)
			if err != nil {
				return nil, false, err
			}
			if !ok {
				return nil, false, nil
			}
			args = append(args, value)
		}
		name := sourceSpan(src, fn.NameStart, fn.NameEnd)
		switch {
		case strings.EqualFold(name, "now"):
			if len(args) != 0 {
				return nil, false, fmt.Errorf("NOW() does not accept arguments")
			}
			return time.Now().UTC(), true, nil
		case strings.EqualFold(name, "nullif"):
			if len(args) != 2 {
				return nil, false, fmt.Errorf("NULLIF requires exactly two arguments")
			}
			if args[0] == nil || args[1] == nil {
				return args[0], true, nil
			}
			if sqlValueEqual(args[0], args[1]) {
				return nil, true, nil
			}
			return args[0], true, nil
		default:
			return evaluateJSONFunction(name, args)
		}
	case parser.NodeKindSubqueryExpr:
		return db.virtualSubqueryValue(ctx, src, doc, ref, row, params, legacy)
	case parser.NodeKindAggregateExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.AggregateExprs) {
			return nil, false, fmt.Errorf("invalid aggregate expression reference")
		}
		if value, ok := row.Values[virtualAggregateExprKey(ref.ID)]; ok {
			return value, true, nil
		}
		name := aggregateColumnName(uint8(doc.AggregateExprs[ref.ID].Func))
		value, ok := row.Values[name]
		return value, ok, nil
	}
	return nil, false, nil
}

func (db *Database) virtualSubqueryRows(ctx context.Context, src []byte, doc *parser.QueryDoc, ref parser.NodeRef, outer virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) ([]virtualSQLRow, bool, error) {
	if ref.Kind != parser.NodeKindSubqueryExpr || ref.ID < 0 || int(ref.ID) >= len(doc.SubqueryExprs) {
		return nil, false, fmt.Errorf("invalid IN subquery reference")
	}
	sq := &doc.SubqueryExprs[ref.ID]
	if sq.Stmt.Kind != parser.NodeKindSelectStmt || sq.Stmt.ID < 0 || int(sq.Stmt.ID) >= len(doc.SelectStmts) {
		return nil, false, fmt.Errorf("invalid IN subquery SELECT")
	}
	rows, _, err := db.evaluateVirtualSelectRows(ctx, src, doc, &doc.SelectStmts[sq.Stmt.ID], &outer, params, legacy)
	return rows, true, err
}

func virtualScalarInterface(value optimizer.ScalarValue) interface{} {
	switch value.Kind {
	case optimizer.ScalarNull, optimizer.ScalarInvalid:
		return nil
	case optimizer.ScalarString, optimizer.ScalarBytes:
		return string(value.BytesData)
	case optimizer.ScalarJSON:
		if decoded, ok := decodeJSONValue(value.BytesData); ok {
			return decoded
		}
		return string(value.BytesData)
	case optimizer.ScalarInt:
		return value.Int
	case optimizer.ScalarFloat:
		return value.Float
	case optimizer.ScalarBool:
		return value.Bool
	case optimizer.ScalarVector:
		return value.Vector
	case optimizer.ScalarTimestamp:
		return value.Time
	default:
		return string(value.Bytes())
	}
}

func validVirtualUUID(value string) bool {
	if len(value) != 36 {
		return false
	}
	for i := 0; i < len(value); i++ {
		if i == 8 || i == 13 || i == 18 || i == 23 {
			if value[i] != '-' {
				return false
			}
			continue
		}
		c := value[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

func (db *Database) evaluateVirtualScalarBinary(ctx context.Context, src []byte, doc *parser.QueryDoc, be *parser.BinaryExpr, row virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) (interface{}, bool, error) {
	if be == nil {
		return nil, false, fmt.Errorf("nil binary expression")
	}
	if lexer.Kind(be.Operator) == lexer.KindAnd || lexer.Kind(be.Operator) == lexer.KindOr {
		left, lok, err := db.virtualExprValue(ctx, src, doc, be.Left, row, params, legacy)
		if err != nil {
			return nil, false, err
		}
		leftBool := lok && left != nil && isVirtualTrue(left)
		if lexer.Kind(be.Operator) == lexer.KindAnd && !leftBool {
			return false, true, nil
		}
		if lexer.Kind(be.Operator) == lexer.KindOr && leftBool {
			return true, true, nil
		}
		right, rok, err := db.virtualExprValue(ctx, src, doc, be.Right, row, params, legacy)
		if err != nil {
			return nil, false, err
		}
		return isVirtualTrue(right) && rok && right != nil, true, nil
	}
	left, lok, err := db.virtualExprValue(ctx, src, doc, be.Left, row, params, legacy)
	if err != nil {
		return nil, false, err
	}
	right, rok, err := db.virtualExprValue(ctx, src, doc, be.Right, row, params, legacy)
	if err != nil {
		return nil, false, err
	}
	if !lok || !rok || left == nil || right == nil {
		return nil, true, nil
	}
	op := lexer.Kind(be.Operator)
	switch op {
	case lexer.KindEquals:
		return sqlValueEqual(left, right), true, nil
	case lexer.KindNotEqual:
		return !sqlValueEqual(left, right), true, nil
	case lexer.KindGreaterThan:
		return compareVirtualValues(left, right) > 0, true, nil
	case lexer.KindLessThan:
		return compareVirtualValues(left, right) < 0, true, nil
	case lexer.KindGreaterEqual:
		return compareVirtualValues(left, right) >= 0, true, nil
	case lexer.KindLessEqual:
		return compareVirtualValues(left, right) <= 0, true, nil
	case lexer.KindConcat:
		return recordMetaToString(left) + recordMetaToString(right), true, nil
	case lexer.KindShiftLeft, lexer.KindShiftRight:
		li, lok := toInt64(left)
		ri, rok := toInt64(right)
		if !lok || !rok || ri < 0 || ri >= 64 {
			return nil, false, fmt.Errorf("shift requires integer operands and a shift count in [0,63]")
		}
		if op == lexer.KindShiftLeft {
			return li << uint(ri), true, nil
		}
		return li >> uint(ri), true, nil
	case lexer.KindPlus, lexer.KindDash, lexer.KindAsterisk, lexer.KindSlash, lexer.KindPercent:
		return virtualArithmetic(left, right, op)
	default:
		return nil, false, fmt.Errorf("unsupported scalar operator %d", be.Operator)
	}
}

func isVirtualTrue(value interface{}) bool {
	if b, ok := value.(bool); ok {
		return b
	}
	return strings.EqualFold(strings.TrimSpace(recordMetaToString(value)), "true")
}

func toInt64(value interface{}) (int64, bool) {
	switch n := value.(type) {
	case int:
		return int64(n), true
	case int64:
		return n, true
	case int32:
		return int64(n), true
	case uint64:
		if n > math.MaxInt64 {
			return 0, false
		}
		return int64(n), true
	case float64:
		return int64(n), n == math.Trunc(n)
	case string:
		parsed, err := strconv.ParseInt(strings.TrimSpace(n), 10, 64)
		return parsed, err == nil
	default:
		return 0, false
	}
}

func virtualArithmetic(left, right interface{}, op lexer.Kind) (interface{}, bool, error) {
	// Preserve fractional SQL division when either operand is a floating-point
	// value. Aggregate SUM over FLOAT metadata returns float64 even when the
	// numeric value happens to be integral (for example 4.0 / 6.0).
	if op == lexer.KindSlash {
		_, leftFloat := left.(float32)
		if _, ok := left.(float64); ok {
			leftFloat = true
		}
		_, rightFloat := right.(float32)
		if _, ok := right.(float64); ok {
			rightFloat = true
		}
		if leftFloat || rightFloat {
			lf, lok := toFloat(left)
			rf, rok := toFloat(right)
			if !lok || !rok {
				return nil, false, fmt.Errorf("operator %d requires numeric operands", op)
			}
			if rf == 0 {
				return nil, false, fmt.Errorf("division by zero")
			}
			return lf / rf, true, nil
		}
	}
	if li, lok := toInt64(left); lok {
		if ri, rok := toInt64(right); rok {
			switch op {
			case lexer.KindPlus:
				return li + ri, true, nil
			case lexer.KindDash:
				return li - ri, true, nil
			case lexer.KindAsterisk:
				return li * ri, true, nil
			case lexer.KindSlash:
				if ri == 0 {
					return nil, false, fmt.Errorf("division by zero")
				}
				return li / ri, true, nil
			case lexer.KindPercent:
				if ri == 0 {
					return nil, false, fmt.Errorf("modulo by zero")
				}
				return li % ri, true, nil
			}
		}
	}
	lf, lok := toFloat(left)
	rf, rok := toFloat(right)
	if !lok || !rok {
		return nil, false, fmt.Errorf("operator %d requires numeric operands", op)
	}
	switch op {
	case lexer.KindPlus:
		return lf + rf, true, nil
	case lexer.KindDash:
		return lf - rf, true, nil
	case lexer.KindAsterisk:
		return lf * rf, true, nil
	case lexer.KindSlash:
		if rf == 0 {
			return nil, false, fmt.Errorf("division by zero")
		}
		return lf / rf, true, nil
	default:
		return nil, false, fmt.Errorf("operator %d requires integer operands", op)
	}
}

func sourceSpan(src []byte, start, end uint32) string {
	if start > end || end > uint32(len(src)) {
		return ""
	}
	return string(src[start:end])
}

func searchRowsToVirtual(results *SearchResults) []virtualSQLRow {
	if results == nil {
		return nil
	}
	rows := make([]virtualSQLRow, 0, len(results.Results))
	for _, result := range results.Results {
		values := cloneMetadata(result.Metadata)
		if values == nil {
			values = make(map[string]interface{})
		}
		if _, ok := values["id"]; !ok {
			values["id"] = result.ID
		}
		rows = append(rows, virtualSQLRow{ID: result.ID, Values: values})
	}
	return rows
}

func projectVirtualRows(doc *parser.QueryDoc, src []byte, stmt *parser.SelectStmt, rows []virtualSQLRow) (*SearchResults, error) {
	projected, columns, err := projectRows(doc, src, stmt, rows)
	if err != nil {
		return nil, err
	}
	return finishVirtualRows(nil, doc, src, stmt, projected, columns, nil), nil
}

func executeVirtualCTEJoin(ctx context.Context, db *Database, src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt, from *parser.TableExpr, join *parser.JoinClause, cteRows []virtualSQLRow, cteName string) (*SearchResults, error) {
	collection := sourceSpan(src, from.Start, from.End)
	col, err := db.GetCollection(collection)
	if err != nil {
		return nil, err
	}
	leftRecords, err := recordsVisibleInContext(ctx, col)
	if err != nil {
		return nil, err
	}
	leftAlias := sourceSpan(src, from.Alias, from.AliasEnd)
	if leftAlias == "" {
		leftAlias = collection
	}
	rightAlias := sourceSpan(src, join.Alias, join.AliasEnd)
	if rightAlias == "" {
		rightAlias = cteName
	}
	leftColumn, rightColumn, err := virtualJoinColumns(doc, src, join.OnExpr, leftAlias, rightAlias)
	if err != nil {
		return nil, err
	}

	rows := make([]virtualSQLRow, 0)
	for _, record := range leftRecords {
		leftValue, leftOK := recordColumn(record, leftColumn)
		if !leftOK {
			continue
		}
		for _, right := range cteRows {
			rightValue, rightOK := right.Values[rightColumn]
			if !rightOK || !sqlValueEqual(leftValue, rightValue) {
				continue
			}
			values := cloneMetadata(record.Metadata)
			if values == nil {
				values = make(map[string]interface{})
			}
			values["id"] = record.ID
			for key, value := range right.Values {
				// Keep unqualified names usable for the common `SELECT c.id`
				// shape. Qualified lookup remains driven by the AST offsets.
				if _, exists := values[key]; !exists {
					values[key] = value
				}
			}
			row := virtualSQLRow{ID: record.ID, Values: values}
			if stmt.WhereExpr.Kind != parser.NodeKindUnknown && !evalVirtualExpr(doc, src, stmt.WhereExpr, row, leftAlias, rightAlias, nil) {
				continue
			}
			rows = append(rows, row)
		}
	}
	projected, columns, err := projectRows(doc, src, stmt, rows)
	if err != nil {
		return nil, err
	}
	return finishVirtualRows(db, doc, src, stmt, projected, columns, nil), nil
}

func virtualJoinColumns(doc *parser.QueryDoc, src []byte, ref parser.NodeRef, leftAlias, rightAlias string) (string, string, error) {
	if ref.Kind != parser.NodeKindBinaryExpr {
		return "", "", fmt.Errorf("CTE JOIN requires an equality ON expression")
	}
	be := &doc.BinaryExprs[ref.ID]
	if be.Operator != uint8(lexer.KindEquals) || be.Left.Kind != parser.NodeKindIdentifier || be.Right.Kind != parser.NodeKindIdentifier {
		return "", "", fmt.Errorf("CTE JOIN requires alias.column = alias.column")
	}
	left := &doc.Identifiers[be.Left.ID]
	right := &doc.Identifiers[be.Right.ID]
	leftQ := sourceSpan(src, left.QualStart, left.QualEnd)
	rightQ := sourceSpan(src, right.QualStart, right.QualEnd)
	leftCol := sourceSpan(src, left.Start, left.End)
	rightCol := sourceSpan(src, right.Start, right.End)
	if strings.EqualFold(leftQ, leftAlias) && strings.EqualFold(rightQ, rightAlias) {
		return leftCol, rightCol, nil
	}
	if strings.EqualFold(rightQ, leftAlias) && strings.EqualFold(leftQ, rightAlias) {
		return rightCol, leftCol, nil
	}
	return "", "", fmt.Errorf("CTE JOIN aliases do not match FROM/JOIN aliases")
}

func recordColumn(record Record, column string) (interface{}, bool) {
	if strings.EqualFold(column, "id") {
		return record.ID, record.ID != ""
	}
	for name, value := range record.Metadata {
		if strings.EqualFold(name, column) {
			return value, value != nil
		}
	}
	return nil, false
}

func sqlValueEqual(left, right interface{}) bool {
	if left == nil || right == nil {
		return false
	}
	return recordMetaToString(left) == recordMetaToString(right)
}

func projectRows(doc *parser.QueryDoc, src []byte, stmt *parser.SelectStmt, rows []virtualSQLRow) ([]virtualSQLRow, []string, error) {
	if stmt.ProjectionsCount == 0 {
		return rows, nil, nil
	}
	columns := make([]string, 0, stmt.ProjectionsCount)
	projectionNames := make([]string, stmt.ProjectionsCount)
	nonStarProjections := 0
	hasStar := false
	// Derive the RowDescription independently of the row count.  Empty CTEs
	// must still expose their selected column names through pgwire.
	for j := int32(0); j < stmt.ProjectionsCount; j++ {
		projection := &doc.Projections[stmt.ProjectionsStart+j]
		if projection.Star {
			hasStar = true
			if len(rows) > 0 {
				for key := range rows[0].Values {
					columns = append(columns, key)
				}
				sort.Strings(columns)
			}
			continue
		}
		if projection.Expr.Kind != parser.NodeKindIdentifier {
			return nil, nil, fmt.Errorf("generic CTE projection supports identifiers and * only")
		}
		id := &doc.Identifiers[projection.Expr.ID]
		name := sourceSpan(src, id.Start, id.End)
		if projection.AliasEnd > projection.Alias {
			name = sourceSpan(src, projection.Alias, projection.AliasEnd)
		}
		projectionNames[j] = name
		nonStarProjections++
		columns = append(columns, name)
	}
	// A lone star already describes the complete row map.  The source rows
	// are query-local and are not reused after projection, so returning them
	// directly avoids one map allocation and one key copy per row.
	if hasStar && stmt.ProjectionsCount == 1 {
		for i := range rows {
			if rows[i].Values == nil {
				rows[i].Values = make(map[string]interface{})
			}
		}
		return rows, columns, nil
	}
	out := make([]virtualSQLRow, 0, len(rows))
	for i := range rows {
		valueCapacity := nonStarProjections
		if hasStar {
			valueCapacity += len(rows[i].Values)
		}
		values := make(map[string]interface{}, valueCapacity)
		for j := int32(0); j < stmt.ProjectionsCount; j++ {
			projection := &doc.Projections[stmt.ProjectionsStart+j]
			if projection.Star {
				for key, value := range rows[i].Values {
					values[key] = value
				}
				continue
			}
			id := &doc.Identifiers[projection.Expr.ID]
			if value, ok := virtualIdentifierValue(src, id, rows[i]); ok {
				values[projectionNames[j]] = value
			}
		}
		out = append(out, virtualSQLRow{ID: rows[i].ID, Values: values})
	}
	return out, columns, nil
}

func virtualIdentifierValue(src []byte, id *parser.Identifier, row virtualSQLRow) (interface{}, bool) {
	if id.QualEnd > id.QualStart {
		qualified := sourceSpan(src, id.QualStart, id.QualEnd) + "." + sourceSpan(src, id.Start, id.End)
		if value, ok := row.Values[qualified]; ok {
			return value, true
		}
		for key, value := range row.Values {
			if strings.EqualFold(key, qualified) {
				return value, true
			}
		}
	}
	name := sourceSpan(src, id.Start, id.End)
	value, ok := row.Values[name]
	if ok {
		return value, true
	}
	for key, value := range row.Values {
		if strings.EqualFold(key, name) {
			return value, true
		}
	}
	return nil, false
}

func virtualLikeMatch(value, pattern string, insensitive bool) bool {
	if insensitive {
		value = strings.ToLower(value)
		pattern = strings.ToLower(pattern)
	}
	if pattern == "%" {
		return true
	}
	// Small allocation-free semantics are unnecessary here: this evaluator is
	// for query-local virtual rows, not the hot physical scan path.
	if !strings.Contains(pattern, "%") && !strings.Contains(pattern, "_") {
		return value == pattern
	}
	parts := strings.Split(pattern, "%")
	position := 0
	for i, part := range parts {
		if part == "" {
			continue
		}
		idx := strings.Index(value[position:], part)
		if idx < 0 || (i == 0 && !strings.HasPrefix(value, part)) {
			return false
		}
		position += idx + len(part)
	}
	if !strings.HasSuffix(pattern, "%") && position != len(value) {
		return false
	}
	return true
}

func (db *Database) applyVirtualSelectClauses(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt, rows []virtualSQLRow, columns []string, params *optimizer.ParameterSet, legacy QueryParams) ([]virtualSQLRow, error) {
	_ = columns
	if len(stmt.OrderTerms) == 0 && stmt.OrderBy.Kind != parser.NodeKindUnknown {
		stmt.OrderTerms = []parser.OrderTerm{{Expr: stmt.OrderBy, IsDesc: stmt.IsDesc}}
	}
	if len(stmt.OrderTerms) > 0 {
		term := stmt.OrderTerms[0]
		type orderedRow struct {
			row   virtualSQLRow
			value interface{}
			valid bool
		}
		ordered := make([]orderedRow, 0, len(rows))
		for _, row := range rows {
			value, ok, err := db.virtualExprValue(ctx, src, doc, term.Expr, row, params, legacy)
			if err != nil {
				return nil, err
			}
			ordered = append(ordered, orderedRow{row: row, value: value, valid: ok && value != nil})
		}
		sort.SliceStable(ordered, func(i, j int) bool {
			left, right := ordered[i], ordered[j]
			if left.valid != right.valid {
				return !left.valid
			}
			if left.valid {
				cmp := compareVirtualValues(left.value, right.value)
				if cmp != 0 {
					if term.IsDesc {
						return cmp > 0
					}
					return cmp < 0
				}
			}
			return left.row.ID < right.row.ID
		})
		rows = make([]virtualSQLRow, len(ordered))
		for i := range ordered {
			rows[i] = ordered[i].row
		}
	}
	offset := virtualClauseInt(doc, src, stmt.Offset, stmt.OffsetExpr, params)
	if offset > 0 {
		if offset >= len(rows) {
			return nil, nil
		}
		rows = rows[offset:]
	}
	limit := virtualClauseInt(doc, src, stmt.Limit, stmt.LimitExpr, params)
	if limit >= 0 && limit < len(rows) {
		rows = rows[:limit]
	}
	return rows, nil
}

func finishVirtualRows(db *Database, doc *parser.QueryDoc, src []byte, stmt *parser.SelectStmt, rows []virtualSQLRow, columns []string, params *optimizer.ParameterSet) *SearchResults {
	out := &SearchResults{Columns: columns, ColumnTypes: virtualProjectionTypes(db, doc, src, stmt, columns), Results: make([]*SearchResult, 0, len(rows))}
	for _, row := range rows {
		out.Results = append(out.Results, &SearchResult{ID: row.ID, Score: 1, Metadata: row.Values})
	}
	if stmt.Distinct {
		plan := &optimizer.PhysicalPlan{Distinct: true, Projections: columns}
		out.Results = distinctSearchResults(out.Results, columns)
		_ = plan
	}
	if stmt.OrderBy.Kind != parser.NodeKindUnknown {
		if stmt.OrderBy.Kind == parser.NodeKindIdentifier {
			id := &doc.Identifiers[stmt.OrderBy.ID]
			plan := &optimizer.PhysicalPlan{OrderBy: sourceSpan(src, id.Start, id.End), IsDesc: stmt.IsDesc}
			newExecutor(nil).applyOrderBy(out, plan)
		}
	}
	offset := virtualClauseInt(doc, src, stmt.Offset, stmt.OffsetExpr, params)
	if offset > 0 {
		if offset >= len(out.Results) {
			out.Results = nil
		} else {
			out.Results = out.Results[offset:]
		}
	}
	limit := virtualClauseInt(doc, src, stmt.Limit, stmt.LimitExpr, params)
	if limit >= 0 && limit < len(out.Results) {
		out.Results = out.Results[:limit]
	}
	out.Total = len(out.Results)
	return out
}

// virtualProjectionTypes carries the type of expression projections across
// the SQL executor/protocol boundary. Row values are intentionally decoded
// JSON values, so inferring a type from the first row would make empty result
// sets and NULL-only columns describe incorrectly.
func virtualProjectionTypes(db *Database, doc *parser.QueryDoc, src []byte, stmt *parser.SelectStmt, columns []string) []uint16 {
	if doc == nil || stmt == nil || len(columns) == 0 || stmt.ProjectionsCount != int32(len(columns)) {
		return nil
	}
	types := make([]uint16, len(columns))
	for i := range columns {
		projection := &doc.Projections[stmt.ProjectionsStart+int32(i)]
		types[i] = virtualExprCatalogType(doc, src, projection.Expr)
		if projection.Expr.Kind == parser.NodeKindFunctionExpr && projection.Expr.ID >= 0 && int(projection.Expr.ID) < len(doc.FunctionExprs) {
			fn := &doc.FunctionExprs[projection.Expr.ID]
			name := strings.ToLower(sourceSpan(src, fn.NameStart, fn.NameEnd))
			switch name {
			case "json_set", "jsonb_set", "json_insert", "jsonb_insert",
				"json_build_array", "jsonb_build_array", "json_build_object", "jsonb_build_object",
				"json_populate_record", "jsonb_populate_record", "to_json", "to_jsonb":
				types[i] = catalog.TypeJSONB
			}
		}
		if projection.Expr.Kind != parser.NodeKindBinaryExpr || projection.Expr.ID < 0 || int(projection.Expr.ID) >= len(doc.BinaryExprs) {
			continue
		}
		be := &doc.BinaryExprs[projection.Expr.ID]
		switch lexer.Kind(be.Operator) {
		case lexer.KindJSONContains, lexer.KindJSONContainedBy, lexer.KindJSONExists, lexer.KindJSONAny, lexer.KindJSONAll, lexer.KindJSONPathExists:
			types[i] = catalog.TypeBool
		case lexer.KindJSONExtractText, lexer.KindJSONPathText:
			types[i] = catalog.TypeString
		case lexer.KindArrowRight, lexer.KindJSONExtract, lexer.KindJSONPath:
			types[i] = virtualJSONValueType(db, doc, src, stmt, be.Left)
		}
	}
	return types
}

func virtualExprCatalogType(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) uint16 {
	if doc == nil {
		return 0
	}
	switch ref.Kind {
	case parser.NodeKindString:
		return catalog.TypeString
	case parser.NodeKindNumber:
		if ref.ID >= 0 && int(ref.ID) < len(doc.Numbers) {
			text := sourceSpan(src, doc.Numbers[ref.ID].Start, doc.Numbers[ref.ID].End)
			if strings.ContainsAny(text, ".eE") {
				return catalog.TypeFloat
			}
		}
		return catalog.TypeBigInt
	case parser.NodeKindCastExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.CastExprs) {
			return 0
		}
		typeName := strings.ToLower(strings.TrimSpace(sourceSpan(src, doc.CastExprs[ref.ID].TypeStart, doc.CastExprs[ref.ID].TypeEnd)))
		switch typeName {
		case "json":
			return catalog.TypeJSON
		case "jsonb":
			return catalog.TypeJSONB
		case "vector":
			return catalog.TypeVector
		case "text", "varchar", "character varying", "char", "string":
			return catalog.TypeString
		case "uuid":
			return catalog.TypeUUID
		case "bigint":
			return catalog.TypeBigInt
		case "int", "int2", "int4", "integer", "smallint":
			return catalog.TypeInt
		case "float", "float4", "float8", "real", "double", "double precision", "numeric", "decimal":
			return catalog.TypeFloat
		case "bool", "boolean":
			return catalog.TypeBool
		default:
			return virtualExprCatalogType(doc, src, doc.CastExprs[ref.ID].Expr)
		}
	case parser.NodeKindFunctionExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.FunctionExprs) {
			fn := doc.FunctionExprs[ref.ID]
			name := sourceSpan(src, fn.NameStart, fn.NameEnd)
			if strings.EqualFold(name, "now") {
				return catalog.TypeTimestamp
			}
			if strings.EqualFold(name, "nullif") && fn.ArgsCount > 0 && fn.ArgsStart >= 0 && fn.ArgsStart < int32(len(doc.FunctionArgs)) {
				return virtualExprCatalogType(doc, src, doc.FunctionArgs[fn.ArgsStart])
			}
		}
	case parser.NodeKindAggregateExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.AggregateExprs) {
			return 0
		}
		ae := doc.AggregateExprs[ref.ID]
		switch ae.Func {
		case parser.AggCount:
			return catalog.TypeBigInt
		case parser.AggSum, parser.AggAvg:
			return catalog.TypeFloat
		case parser.AggVectorAvg:
			return catalog.TypeVector
		default:
			return virtualExprCatalogType(doc, src, ae.Expr)
		}
	case parser.NodeKindCaseExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.CaseExprs) {
			return 0
		}
		ce := doc.CaseExprs[ref.ID]
		for i := int32(0); i < ce.WhensCount; i++ {
			if typ := virtualExprCatalogType(doc, src, doc.CaseWhens[ce.WhensStart+i].Value); typ != 0 {
				return typ
			}
		}
		if ce.HasElse {
			return virtualExprCatalogType(doc, src, ce.Else)
		}
	case parser.NodeKindBinaryExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.BinaryExprs) {
			op := lexer.Kind(doc.BinaryExprs[ref.ID].Operator)
			switch op {
			case lexer.KindEquals, lexer.KindNotEqual, lexer.KindGreaterThan, lexer.KindLessThan, lexer.KindGreaterEqual, lexer.KindLessEqual, lexer.KindAnd, lexer.KindOr:
				return catalog.TypeBool
			case lexer.KindConcat:
				return catalog.TypeString
			case lexer.KindSlash:
				return catalog.TypeFloat
			}
			return virtualExprCatalogType(doc, src, doc.BinaryExprs[ref.ID].Left)
		}
	case parser.NodeKindIdentifier:
		if ref.ID >= 0 && int(ref.ID) < len(doc.Identifiers) {
			id := doc.Identifiers[ref.ID]
			name := strings.ToLower(sourceSpan(src, id.Start, id.End))
			if name == "true" || name == "false" {
				return catalog.TypeBool
			}
		}
	}
	return 0
}

func virtualJSONValueType(db *Database, doc *parser.QueryDoc, src []byte, stmt *parser.SelectStmt, ref parser.NodeRef) uint16 {
	if db == nil || doc == nil || stmt == nil || ref.Kind != parser.NodeKindIdentifier || ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
		return catalog.TypeJSONB
	}
	if stmt.FromTable.Kind != parser.NodeKindTableExpr || stmt.FromTable.ID < 0 || int(stmt.FromTable.ID) >= len(doc.TableExprs) {
		return catalog.TypeJSONB
	}
	table := &doc.TableExprs[stmt.FromTable.ID]
	if table.Start >= uint32(len(src)) || table.End > uint32(len(src)) || table.Start >= table.End {
		return catalog.TypeJSONB
	}
	collection, err := db.GetCollection(sourceSpan(src, table.Start, table.End))
	if err != nil || collection == nil {
		return catalog.TypeJSONB
	}
	name := sourceSpan(src, doc.Identifiers[ref.ID].Start, doc.Identifiers[ref.ID].End)
	for field, fieldType := range collection.Config().MetadataSchema {
		if !strings.EqualFold(field, name) {
			continue
		}
		if fieldType == JSONField {
			return catalog.TypeJSON
		}
		if fieldType == JSONBField {
			return catalog.TypeJSONB
		}
	}
	return catalog.TypeJSONB
}

func virtualClauseInt(doc *parser.QueryDoc, src []byte, numberID int32, expr parser.NodeRef, params *optimizer.ParameterSet) int {
	if numberID >= 0 && int(numberID) < len(doc.Numbers) {
		n := &doc.Numbers[numberID]
		v, _ := strconv.Atoi(sourceSpan(src, n.Start, n.End))
		return v
	}
	if expr.Kind == parser.NodeKindIdentifier && expr.ID >= 0 && int(expr.ID) < len(doc.Identifiers) && params != nil {
		id := doc.Identifiers[expr.ID]
		if value, found := params.Lookup(src, id.Start, id.End); found {
			if integer, ok := toInt64(virtualScalarInterface(value)); ok && integer >= 0 && integer <= int64(^uint(0)>>1) {
				return int(integer)
			}
		}
	}
	return -1
}

func evalVirtualExpr(doc *parser.QueryDoc, src []byte, ref parser.NodeRef, row virtualSQLRow, leftAlias, rightAlias string, membership map[string]struct{}) bool {
	switch ref.Kind {
	case parser.NodeKindBinaryExpr:
		be := &doc.BinaryExprs[ref.ID]
		if be.Operator == uint8(lexer.KindAnd) {
			return evalVirtualExpr(doc, src, be.Left, row, leftAlias, rightAlias, membership) && evalVirtualExpr(doc, src, be.Right, row, leftAlias, rightAlias, membership)
		}
		if be.Operator == uint8(lexer.KindOr) {
			return evalVirtualExpr(doc, src, be.Left, row, leftAlias, rightAlias, membership) || evalVirtualExpr(doc, src, be.Right, row, leftAlias, rightAlias, membership)
		}
		left, lok := virtualExprValue(doc, src, be.Left, row)
		right, rok := virtualExprValue(doc, src, be.Right, row)
		if !lok || !rok {
			return false
		}
		switch be.Operator {
		case uint8(lexer.KindEquals):
			return sqlValueEqual(left, right)
		case uint8(lexer.KindGreaterThan):
			return recordMetaToString(left) > recordMetaToString(right)
		case uint8(lexer.KindLessThan):
			return recordMetaToString(left) < recordMetaToString(right)
		}
	case parser.NodeKindInExpr:
		in := &doc.InExprs[ref.ID]
		value, ok := virtualExprValue(doc, src, in.Expr, row)
		if !ok {
			return false
		}
		matched := false
		for i := int32(0); i < in.ListCount; i++ {
			if candidate, ok := virtualExprValue(doc, src, doc.Nodes[in.ListStart+i], row); ok && sqlValueEqual(value, candidate) {
				matched = true
				break
			}
		}
		if in.HasSubquery && membership != nil {
			_, matched = membership[recordMetaToString(value)]
		}
		if in.Not {
			return !matched
		}
		return matched
	case parser.NodeKindSubqueryExpr:
		return len(membership) > 0
	}
	return false
}

func virtualExprValue(doc *parser.QueryDoc, src []byte, ref parser.NodeRef, row virtualSQLRow) (interface{}, bool) {
	switch ref.Kind {
	case parser.NodeKindIdentifier:
		return virtualIdentifierValue(src, &doc.Identifiers[ref.ID], row)
	case parser.NodeKindString:
		s := &doc.Strings[ref.ID]
		return sourceSpan(src, s.Start+1, s.End-1), true
	case parser.NodeKindNumber:
		n := &doc.Numbers[ref.ID]
		return sourceSpan(src, n.Start, n.End), true
	}
	return nil, false
}
