package libravdb

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/graph"
	"github.com/xDarkicex/libravdb/internal/optimizer"
)

func (db *Database) executeCypherPipe(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt, params *optimizer.ParameterSet, legacy QueryParams) (*SearchResults, error) {
	if stmt == nil || stmt.FromTable.Kind != parser.NodeKindGraphTable || stmt.FromTable.ID < 0 || int(stmt.FromTable.ID) >= len(doc.GraphTables) {
		return nil, fmt.Errorf("Cypher WITH requires a native MATCH source")
	}
	path := doc.GraphTables[stmt.FromTable.ID].MatchPath
	if path.Kind != parser.NodeKindMatchPath || path.ID < 0 || int(path.ID) >= len(doc.MatchPaths) {
		return nil, fmt.Errorf("Cypher WITH requires a valid MATCH path")
	}
	collection, err := db.cypherMatchGraphCollection(doc, src, &doc.MatchPaths[path.ID])
	if err != nil {
		return nil, err
	}
	if collection == nil {
		return nil, fmt.Errorf("Cypher WITH requires a graph-backed collection")
	}
	epoch := epochFromContext(ctx)
	ownedEpoch := false
	if epoch == nil {
		epoch, err = db.BeginEpochTx(ctx)
		if err != nil {
			return nil, err
		}
		ctx = epoch.Context(ctx)
		ownedEpoch = true
	}
	if ownedEpoch {
		defer func() {
			_ = epoch.Rollback(ctx)
		}()
	}
	bindings, err := db.collectCypherMatchBindings(ctx, src, doc, &doc.MatchPaths[path.ID], collection, stmt.WhereExpr, epoch, params, legacy)
	if err != nil {
		return nil, err
	}
	rows := make([]virtualSQLRow, 0, len(bindings))
	for _, binding := range bindings {
		rows = append(rows, cypherBindingRow(binding))
	}
	for i := int32(0); i < stmt.PipeWithCount; i++ {
		clause := &doc.WithClauses[stmt.PipeWithStart+i]
		rows, err = db.applyCypherWith(ctx, src, doc, clause, rows, params, legacy)
		if err != nil {
			return nil, err
		}
		if clause.MatchPath.Kind == parser.NodeKindMatchPath && clause.MatchPath.ID >= 0 && int(clause.MatchPath.ID) < len(doc.MatchPaths) {
			nextPath := &doc.MatchPaths[clause.MatchPath.ID]
			nextCollection, collectionErr := db.cypherMatchGraphCollection(doc, src, nextPath)
			if collectionErr != nil {
				return nil, collectionErr
			}
			if nextCollection == nil || nextCollection.GetGraph() == nil {
				return nil, fmt.Errorf("MATCH after WITH requires a graph-backed collection")
			}
			rowsBindings, matchErr := db.collectCypherMatchBindingsFromRows(ctx, src, doc, nextPath, nextCollection, clause.MatchWhere, epoch, params, legacy, rows)
			if matchErr != nil {
				return nil, matchErr
			}
			rows = make([]virtualSQLRow, 0, len(rowsBindings))
			for _, binding := range rowsBindings {
				rows = append(rows, cypherBindingRow(binding))
			}
		}
	}
	rows, columns, err := db.projectCypherReturn(ctx, src, doc, stmt, rows, params, legacy)
	if err != nil {
		return nil, err
	}
	return finishVirtualRows(db, doc, src, stmt, rows, columns, params), nil
}

func cypherBindingRow(binding cypherMatchBinding) virtualSQLRow {
	aliases := make([]string, 0, len(binding.vertices)+len(binding.edges))
	for alias := range binding.vertices {
		aliases = append(aliases, alias)
	}
	for alias := range binding.edges {
		aliases = append(aliases, alias)
	}
	sort.Strings(aliases)
	row := binding.base
	row.Scopes = append([]virtualSQLScope(nil), virtualRowScopes(binding.base)...)
	if row.Values != nil {
		row.Values = cloneMetadata(row.Values)
	}
	for _, alias := range aliases {
		if record, ok := binding.vertices[alias]; ok {
			values := cloneMetadata(record.Metadata)
			if values == nil {
				values = make(map[string]interface{})
			}
			values["id"] = record.ID
			row.Scopes = append(row.Scopes, virtualSQLScope{Alias: alias, Values: values, Vector: record.Vector})
			if row.ID == "" {
				row.ID = record.ID
			}
			continue
		}
		if edge, ok := binding.edges[alias]; ok {
			values := cloneMetadata(edge.properties)
			if values == nil {
				values = make(map[string]interface{})
			}
			values["source_id"] = edge.from
			values["target_id"] = edge.target
			values["edge_type"] = graphEdgeKindName(edge.kind)
			values["edge_weight"] = edge.weight
			row.Scopes = append(row.Scopes, virtualSQLScope{Alias: alias, Values: values})
		}
	}
	return row
}

func graphEdgeKindName(kind uint8) string {
	return graph.EdgeKindName(kind)
}

func (db *Database) applyCypherWith(ctx context.Context, src []byte, doc *parser.QueryDoc, clause *parser.WithClause, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) ([]virtualSQLRow, error) {
	if clause == nil || len(clause.Projections) == 0 {
		return nil, fmt.Errorf("WITH requires projections")
	}
	if cypherWithHasAggregate(doc, clause) {
		rows = cypherWithAggregateRows(ctx, src, doc, clause, rows, params, legacy, db)
	} else {
		projected := make([]virtualSQLRow, 0, len(rows))
		for _, row := range rows {
			value, err := db.projectCypherRow(ctx, src, doc, clause.Projections, row, params, legacy)
			if err != nil {
				return nil, err
			}
			projected = append(projected, value)
		}
		rows = projected
	}
	if clause.Where.Kind != parser.NodeKindUnknown {
		filtered := rows[:0]
		for _, row := range rows {
			value, ok, err := db.virtualExprValue(ctx, src, doc, clause.Where, row, params, legacy)
			if err != nil {
				return nil, err
			}
			if ok && isVirtualTrue(value) {
				filtered = append(filtered, row)
			}
		}
		rows = filtered
	}
	if clause.Distinct {
		rows = distinctVirtualRows(rows)
	}
	rows, err := db.applyCypherWithOrder(ctx, src, doc, clause, rows, params, legacy, db)
	if err != nil {
		return nil, err
	}
	return applyCypherWithWindow(rows, clause, src, doc, params), nil
}

func (db *Database) projectCypherRow(ctx context.Context, src []byte, doc *parser.QueryDoc, projections []parser.Projection, row virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) (virtualSQLRow, error) {
	out := virtualSQLRow{ID: row.ID, Values: make(map[string]interface{})}
	for _, projection := range projections {
		handled := false
		name := cypherProjectionName(src, doc, projection)
		if projection.Star {
			out.Scopes = append(out.Scopes, virtualRowScopes(row)...)
			continue
		}
		if projection.Expr.Kind == parser.NodeKindIdentifier && projection.Expr.ID >= 0 && int(projection.Expr.ID) < len(doc.Identifiers) {
			id := &doc.Identifiers[projection.Expr.ID]
			if id.QualStart == id.QualEnd {
				alias := strings.ToLower(sourceSpan(src, id.Start, id.End))
				for _, scope := range virtualRowScopes(row) {
					if strings.EqualFold(scope.Alias, alias) {
						bound := projectionAlias(src, projection, alias)
						copyScope := virtualSQLScope{Alias: bound, Values: scope.Values}
						out.Scopes = append(out.Scopes, copyScope)
						out.Values[bound] = scope.Values
						handled = true
						break
					}
				}
			}
		}
		if handled {
			continue
		}
		value, ok, err := db.virtualExprValue(ctx, src, doc, projection.Expr, row, params, legacy)
		if err != nil {
			return virtualSQLRow{}, err
		}
		if ok {
			out.Values[name] = value
		} else {
			out.Values[name] = nil
		}
	}
	return out, nil
}

func (db *Database) projectCypherReturn(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) ([]virtualSQLRow, []string, error) {
	projections := make([]parser.Projection, 0, stmt.ProjectionsCount)
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		projections = append(projections, doc.Projections[stmt.ProjectionsStart+i])
	}
	columns := make([]string, 0, len(projections))
	for _, projection := range projections {
		columns = append(columns, cypherProjectionName(src, doc, projection))
	}
	out := make([]virtualSQLRow, 0, len(rows))
	for _, row := range rows {
		projected, err := db.projectCypherRow(ctx, src, doc, projections, row, params, legacy)
		if err != nil {
			return nil, nil, err
		}
		for i, projection := range projections {
			if scope, ok := cypherBareScope(src, doc, projection, row); ok {
				name := columns[i]
				projected.Values[name] = scope.Values
				projected.Scopes = nil
			}
		}
		out = append(out, projected)
	}
	if len(stmt.OrderTerms) > 0 {
		var err error
		out, err = applyCypherOrderTerms(ctx, src, doc, stmt.OrderTerms, out, params, legacy, db)
		if err != nil {
			return nil, nil, err
		}
	}
	offset := virtualClauseInt(doc, src, stmt.Offset, stmt.OffsetExpr, params)
	if offset > 0 {
		if offset >= len(out) {
			out = nil
		} else {
			out = out[offset:]
		}
	}
	limit := virtualClauseInt(doc, src, stmt.Limit, stmt.LimitExpr, params)
	if limit >= 0 && limit < len(out) {
		out = out[:limit]
	}
	return out, columns, nil
}

func cypherProjectionName(src []byte, doc *parser.QueryDoc, projection parser.Projection) string {
	if projection.AliasEnd > projection.Alias {
		return sourceSpan(src, projection.Alias, projection.AliasEnd)
	}
	if projection.Star {
		return "*"
	}
	if projection.Expr.Kind == parser.NodeKindIdentifier && projection.Expr.ID >= 0 && int(projection.Expr.ID) < len(doc.Identifiers) {
		return sourceSpan(src, doc.Identifiers[projection.Expr.ID].Start, doc.Identifiers[projection.Expr.ID].End)
	}
	return "expr"
}

func projectionAlias(src []byte, projection parser.Projection, fallback string) string {
	if projection.AliasEnd > projection.Alias {
		return sourceSpan(src, projection.Alias, projection.AliasEnd)
	}
	return fallback
}

func cypherWithHasAggregate(doc *parser.QueryDoc, clause *parser.WithClause) bool {
	for _, projection := range clause.Projections {
		if projection.Expr.Kind == parser.NodeKindAggregateExpr {
			return true
		}
	}
	return false
}

func cypherWithAggregateRows(ctx context.Context, src []byte, doc *parser.QueryDoc, clause *parser.WithClause, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams, db *Database) []virtualSQLRow {
	type group struct{ rows []virtualSQLRow }
	groups := make(map[string]*group)
	order := make([]string, 0)
	for _, row := range rows {
		key := ""
		for _, projection := range clause.Projections {
			if projection.Expr.Kind == parser.NodeKindAggregateExpr {
				continue
			}
			value, ok := cypherExprValue(ctx, src, doc, projection.Expr, row, params, legacy, db)
			if !ok {
				value = nil
			}
			key += aggregateValueKey(value) + "\x00"
		}
		if _, ok := groups[key]; !ok {
			groups[key] = &group{}
			order = append(order, key)
		}
		groups[key].rows = append(groups[key].rows, row)
	}
	out := make([]virtualSQLRow, 0, len(order))
	for _, key := range order {
		groupRows := groups[key].rows
		if len(groupRows) == 0 {
			continue
		}
		row := virtualSQLRow{ID: groupRows[0].ID, Values: make(map[string]interface{})}
		for _, projection := range clause.Projections {
			name := cypherProjectionName(src, doc, projection)
			if projection.Expr.Kind == parser.NodeKindAggregateExpr {
				value := cypherAggregateValue(ctx, src, doc, projection.Expr, groupRows, params, legacy, db)
				row.Values[name] = value
				continue
			}
			if scope, ok := cypherBareScope(src, doc, projection, groupRows[0]); ok {
				bound := projectionAlias(src, projection, scope.Alias)
				row.Scopes = append(row.Scopes, virtualSQLScope{Alias: bound, Values: scope.Values})
				row.Values[bound] = scope.Values
				continue
			}
			value, ok, _ := db.virtualExprValue(ctx, src, doc, projection.Expr, groupRows[0], params, legacy)
			if ok {
				row.Values[name] = value
			}
		}
		out = append(out, row)
	}
	return out
}

func cypherBareScope(src []byte, doc *parser.QueryDoc, projection parser.Projection, row virtualSQLRow) (virtualSQLScope, bool) {
	if projection.Expr.Kind != parser.NodeKindIdentifier || projection.Expr.ID < 0 || int(projection.Expr.ID) >= len(doc.Identifiers) {
		return virtualSQLScope{}, false
	}
	id := &doc.Identifiers[projection.Expr.ID]
	if id.QualStart != id.QualEnd {
		return virtualSQLScope{}, false
	}
	name := sourceSpan(src, id.Start, id.End)
	for _, scope := range virtualRowScopes(row) {
		if strings.EqualFold(scope.Alias, name) {
			return scope, true
		}
	}
	return virtualSQLScope{}, false
}

func cypherAggregateValue(ctx context.Context, src []byte, doc *parser.QueryDoc, ref parser.NodeRef, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams, db *Database) interface{} {
	if ref.ID < 0 || int(ref.ID) >= len(doc.AggregateExprs) {
		return nil
	}
	ae := &doc.AggregateExprs[ref.ID]
	values := make([]interface{}, 0, len(rows))
	for _, row := range rows {
		if ae.Expr.Kind == parser.NodeKindUnknown {
			values = append(values, int64(1))
			continue
		}
		value, ok := cypherExprValue(ctx, src, doc, ae.Expr, row, params, legacy, db)
		if ok && value != nil {
			values = append(values, value)
		}
	}
	switch ae.Func {
	case parser.AggCount:
		return int64(len(values))
	case parser.AggSum, parser.AggAvg:
		var total float64
		for _, value := range values {
			if n, ok := toFloat(value); ok {
				total += n
			}
		}
		if ae.Func == parser.AggAvg && len(values) > 0 {
			return total / float64(len(values))
		}
		if len(values) == 0 {
			return nil
		}
		return total
	case parser.AggMin, parser.AggMax:
		if len(values) == 0 {
			return nil
		}
		best := values[0]
		for _, value := range values[1:] {
			cmp := compareVirtualValues(value, best)
			if (ae.Func == parser.AggMin && cmp < 0) || (ae.Func == parser.AggMax && cmp > 0) {
				best = value
			}
		}
		return best
	default:
		return nil
	}
}

func cypherExprValue(ctx context.Context, src []byte, doc *parser.QueryDoc, ref parser.NodeRef, row virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams, db *Database) (interface{}, bool) {
	if ref.Kind == parser.NodeKindIdentifier && ref.ID >= 0 && int(ref.ID) < len(doc.Identifiers) {
		id := &doc.Identifiers[ref.ID]
		if id.QualStart == id.QualEnd {
			name := sourceSpan(src, id.Start, id.End)
			for _, scope := range virtualRowScopes(row) {
				if strings.EqualFold(scope.Alias, name) {
					return scope.Values, true
				}
			}
		}
	}
	value, ok, _ := db.virtualExprValue(ctx, src, doc, ref, row, params, legacy)
	return value, ok
}

func distinctVirtualRows(rows []virtualSQLRow) []virtualSQLRow {
	out := make([]virtualSQLRow, 0, len(rows))
	for _, row := range rows {
		duplicate := false
		for _, existing := range out {
			if aggregateValueKey(cloneVisibleVirtualValues(row)) == aggregateValueKey(cloneVisibleVirtualValues(existing)) {
				duplicate = true
				break
			}
		}
		if !duplicate {
			out = append(out, row)
		}
	}
	return out
}

func (db *Database) applyCypherWithOrder(ctx context.Context, src []byte, doc *parser.QueryDoc, clause *parser.WithClause, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams, _ *Database) ([]virtualSQLRow, error) {
	return applyCypherOrderTerms(ctx, src, doc, clause.OrderTerms, rows, params, legacy, db)
}

func applyCypherOrderTerms(ctx context.Context, src []byte, doc *parser.QueryDoc, terms []parser.OrderTerm, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams, db *Database) ([]virtualSQLRow, error) {
	if len(terms) == 0 {
		return rows, nil
	}
	type ordered struct {
		row   virtualSQLRow
		value interface{}
		valid bool
	}
	orderedRows := make([]ordered, 0, len(rows))
	for _, row := range rows {
		value, ok, err := db.virtualExprValue(ctx, src, doc, terms[0].Expr, row, params, legacy)
		if err != nil {
			return nil, err
		}
		orderedRows = append(orderedRows, ordered{row: row, value: value, valid: ok && value != nil})
	}
	term := terms[0]
	sort.SliceStable(orderedRows, func(i, j int) bool {
		left, right := orderedRows[i], orderedRows[j]
		if left.valid != right.valid {
			return !left.valid
		}
		cmp := compareVirtualValues(left.value, right.value)
		if cmp != 0 {
			if term.IsDesc {
				return cmp > 0
			}
			return cmp < 0
		}
		return left.row.ID < right.row.ID
	})
	out := make([]virtualSQLRow, len(orderedRows))
	for i := range orderedRows {
		out[i] = orderedRows[i].row
	}
	return out, nil
}

func applyCypherWithWindow(rows []virtualSQLRow, clause *parser.WithClause, src []byte, doc *parser.QueryDoc, params *optimizer.ParameterSet) []virtualSQLRow {
	start := 0
	if clause.Skip.Kind != parser.NodeKindUnknown {
		if value, ok := virtualScalarExprValue(doc, src, clause.Skip, rows, params); ok {
			if n, valid := toInt64(value); valid && n > 0 {
				start = int(n)
			}
		}
	}
	if start >= len(rows) {
		return nil
	}
	rows = rows[start:]
	if clause.Limit.Kind != parser.NodeKindUnknown {
		if value, ok := virtualScalarExprValue(doc, src, clause.Limit, rows, params); ok {
			if n, valid := toInt64(value); valid && n >= 0 && n < int64(len(rows)) {
				rows = rows[:n]
			}
		}
	}
	return rows
}

func virtualScalarExprValue(doc *parser.QueryDoc, src []byte, ref parser.NodeRef, rows []virtualSQLRow, params *optimizer.ParameterSet) (interface{}, bool) {
	if len(rows) == 0 {
		return nil, false
	}
	return virtualExprValue(doc, src, ref, rows[0])
}
