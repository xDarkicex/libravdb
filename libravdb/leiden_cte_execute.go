package libravdb

import (
	"bytes"
	"context"
	"fmt"
	"sort"

	"github.com/xDarkicex/lexer"
	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// =============================================================================
// ExecuteLeidenCTE
// =============================================================================

// ExecuteLeidenCTE materializes the local_clusters virtual relation from the
// bound CTE plan and joins it against the outer SELECT's FROM table using the
// ON condition from the parsed JOIN clause.
//
// Pipeline:
//
//	ExecuteLeidenMatchPlan → build node→community index →
//	iterate outer FROM rows → resolve node IDs → JOIN → SearchResults
func (e *EpochTx) ExecuteLeidenCTE(
	ctx context.Context,
	src []byte,
	doc *parser.QueryDoc,
	bound *BoundLeidenCTE,
	selectIndex int,
	params QueryParams,
) (*SearchResults, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if bound == nil {
		return nil, fmt.Errorf("BoundLeidenCTE must not be nil")
	}

	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil, ErrEpochClosed
	}
	e.mu.Unlock()

	if selectIndex < 0 || selectIndex >= len(doc.SelectStmts) {
		return nil, fmt.Errorf("selectIndex %d out of range", selectIndex)
	}
	stmt := doc.SelectStmts[selectIndex]
	col, err := e.db.GetCollection(bound.Collection)
	if err != nil {
		return nil, fmt.Errorf("get CTE collection %q: %w", bound.Collection, err)
	}

	// ── Phase 1: Execute the Leiden plan ──
	leidenResult, err := e.ExecuteBoundLeidenMatchPlan(ctx, bound.Plan)
	if err != nil {
		return nil, fmt.Errorf("execute CTE Leiden plan: %w", err)
	}

	// ── Phase 2: Build node→community lookup from Leiden relation ──
	// local_clusters columns: node_id, community_id, collection, record_id, ...
	nodeToCommunity := make(map[uint64]*LeidenRelationRow, len(leidenResult.Relation.Rows))
	for i := range leidenResult.Relation.Rows {
		row := &leidenResult.Relation.Rows[i]
		nodeToCommunity[row.NodeID] = row
	}

	// ── Phase 3: Get outer FROM table rows ──
	outerRecords, err := e.ListRecords(ctx, bound.Collection)
	if err != nil {
		return nil, fmt.Errorf("list outer FROM records: %w", err)
	}

	// ── Phase 4: Evaluate ON condition and build joined rows ──
	// The ON expression is `left_alias.column = right_alias.column`.
	join := findCTEJoin(src, stmt, bound.Name)
	if join == nil {
		return nil, fmt.Errorf("CTE %q is not referenced by a JOIN", bound.Name)
	}

	// Parse ON: extract left column, right column, and aliases.
	leftAlias, leftCol, rightAlias, rightCol, err := parseJoinOn(src, doc, join)
	if err != nil {
		return nil, fmt.Errorf("parse JOIN ON: %w", err)
	}

	outerAlias := bound.Collection
	if stmt.FromTable.Kind == parser.NodeKindTableExpr {
		from := doc.TableExprs[stmt.FromTable.ID]
		if from.AliasEnd > from.Alias {
			outerAlias = string(src[from.Alias:from.AliasEnd])
		}
	}

	// Build joined results.
	var results []*SearchResult
	columns := buildProjectedColumns(src, doc, stmt)

	for _, rec := range outerRecords {
		// Resolve the record's graph node ID for JOIN on d.node_id.
		nodeID, err := e.LookupNodeID(ctx, bound.Collection, rec.ID)
		if err != nil {
			continue
		}

		communityRow, ok := nodeToCommunity[nodeID]
		if !ok {
			continue // inner join: skip unmatched
		}

		joined := &cteJoinedRow{
			record:     rec,
			nodeID:     nodeID,
			community:  communityRow,
			outerAlias: outerAlias,
			cteAlias:   bound.JoinAlias,
		}
		joinMatch, err := evaluateJoin(join, src, doc, joined)
		if err != nil {
			return nil, fmt.Errorf("evaluate JOIN ON: %w", err)
		}
		if !joinMatch {
			continue
		}
		if stmt.WhereExpr.Kind != parser.NodeKindUnknown {
			match, err := evaluateCTEPredicate(src, doc, stmt.WhereExpr, joined)
			if err != nil {
				return nil, fmt.Errorf("evaluate WHERE: %w", err)
			}
			if !match {
				continue
			}
		}

		// Build projected row with metadata from both sides.
		metadata := make(map[string]interface{}, len(columns))
		for i := int32(0); i < stmt.ProjectionsCount; i++ {
			proj := doc.Projections[stmt.ProjectionsStart+i]
			name := projectionName(src, doc, proj)
			if name == "" {
				continue
			}
			value, ok, err := evaluateCTEProjection(ctx, src, doc, proj, joined, col, params)
			if err != nil {
				return nil, fmt.Errorf("evaluate projection %q: %w", name, err)
			}
			if ok {
				metadata[name] = value
			}
		}

		results = append(results, &SearchResult{
			ID:       rec.ID,
			Score:    1.0,
			Metadata: metadata,
		})
	}

	_ = leftAlias
	_ = leftCol
	_ = rightAlias
	_ = rightCol

	// Apply ORDER BY and LIMIT from outer SELECT.
	out := &SearchResults{
		Results: results,
		Total:   len(results),
		Columns: columns,
	}

	if stmt.OrderBy.Kind != parser.NodeKindUnknown {
		applySelectOrderBy(out, src, doc, stmt)
	}
	if stmt.Limit >= 0 && len(out.Results) > int(stmt.Limit) {
		out.Results = out.Results[:stmt.Limit]
		out.Total = len(out.Results)
	}

	return out, nil
}

// ── Helpers ──

// findCTEJoin returns the JOIN clause whose table name matches the CTE name.
func findCTEJoin(src []byte, stmt parser.SelectStmt, cteName string) *parser.JoinClause {
	for i := range stmt.Joins {
		join := &stmt.Joins[i]
		if join.TableStart >= join.TableEnd {
			continue
		}
		if bytes.EqualFold(src[join.TableStart:join.TableEnd], []byte(cteName)) {
			return join
		}
	}
	return nil
}

// buildProjectedColumns extracts the SELECT column list from the parser AST.
func buildProjectedColumns(src []byte, doc *parser.QueryDoc, stmt parser.SelectStmt) []string {
	if stmt.ProjectionsCount <= 0 {
		return nil
	}
	cols := make([]string, 0, stmt.ProjectionsCount)
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		proj := doc.Projections[stmt.ProjectionsStart+i]
		if proj.Star {
			return nil // SELECT * — all columns
		}
		// Extract alias if present, otherwise the expression text.
		if proj.AliasEnd > proj.Alias {
			cols = append(cols, string(src[proj.Alias:proj.AliasEnd]))
		} else {
			// Use the expression source text as column name.
			expr := proj.Expr
			colName := extractExprName(src, doc, expr)
			if colName != "" {
				cols = append(cols, colName)
			}
		}
	}
	return cols
}

// extractExprName gets the column name from an identifier or qualified identifier.
func extractExprName(src []byte, doc *parser.QueryDoc, ref parser.NodeRef) string {
	switch ref.Kind {
	case parser.NodeKindIdentifier:
		id := doc.Identifiers[ref.ID]
		return string(src[id.Start:id.End])
	default:
		return ""
	}
}

// parseJoinOn extracts the left and right alias.column from the ON expression.
func parseJoinOn(src []byte, doc *parser.QueryDoc, join *parser.JoinClause) (leftAlias, leftCol, rightAlias, rightCol string, err error) {
	if join.OnExpr.Kind != parser.NodeKindBinaryExpr {
		return "", "", "", "", fmt.Errorf("ON expression must be a binary comparison")
	}
	bin := doc.BinaryExprs[join.OnExpr.ID]
	if bin.Operator != uint8(lexer.KindEquals) {
		return "", "", "", "", fmt.Errorf("ON expression must use '=', got op %d", bin.Operator)
	}

	if bin.Left.Kind != parser.NodeKindIdentifier || bin.Right.Kind != parser.NodeKindIdentifier {
		return "", "", "", "", fmt.Errorf("ON expression must compare identifiers")
	}

	left := doc.Identifiers[bin.Left.ID]
	right := doc.Identifiers[bin.Right.ID]

	if left.QualStart != 0 {
		leftAlias = string(src[left.QualStart:left.QualEnd])
	}
	leftCol = string(src[left.Start:left.End])

	if right.QualStart != 0 {
		rightAlias = string(src[right.QualStart:right.QualEnd])
	}
	rightCol = string(src[right.Start:right.End])

	return leftAlias, leftCol, rightAlias, rightCol, nil
}

type cteJoinedRow struct {
	record     Record
	nodeID     uint64
	community  *LeidenRelationRow
	outerAlias string
	cteAlias   string
}

func projectionName(src []byte, doc *parser.QueryDoc, proj parser.Projection) string {
	if proj.AliasEnd > proj.Alias {
		return string(src[proj.Alias:proj.AliasEnd])
	}
	return extractExprName(src, doc, proj.Expr)
}

func aliasEqual(a, b string) bool {
	return bytes.EqualFold([]byte(a), []byte(b))
}

func resolveJoinedColumn(row *cteJoinedRow, alias, column string) (interface{}, bool, error) {
	if row == nil {
		return nil, false, fmt.Errorf("nil joined row")
	}
	if alias == "" || aliasEqual(alias, row.outerAlias) {
		return resolveRecordColumnWithNode(row.record, row.nodeID, column)
	}
	if aliasEqual(alias, row.cteAlias) {
		value := resolveCTEColumn(row.community, column)
		return value, value != nil, nil
	}
	return nil, false, fmt.Errorf("unknown relation alias %q", alias)
}

func resolveRecordColumnWithNode(rec Record, nodeID uint64, column string) (interface{}, bool, error) {
	if column == "node_id" {
		return nodeID, true, nil
	}
	value := resolveRecordColumn(rec, column)
	return value, value != nil, nil
}

func evaluateJoin(join *parser.JoinClause, src []byte, doc *parser.QueryDoc, row *cteJoinedRow) (bool, error) {
	if join == nil || join.OnExpr.Kind == parser.NodeKindUnknown {
		return false, fmt.Errorf("JOIN is missing an ON expression")
	}
	if join.OnExpr.Kind != parser.NodeKindBinaryExpr {
		return false, fmt.Errorf("ON expression must be a binary comparison")
	}
	bin := doc.BinaryExprs[join.OnExpr.ID]
	if bin.Operator != uint8(lexer.KindEquals) || bin.Left.Kind != parser.NodeKindIdentifier || bin.Right.Kind != parser.NodeKindIdentifier {
		return false, fmt.Errorf("ON expression must be alias-qualified equality")
	}
	left := doc.Identifiers[bin.Left.ID]
	right := doc.Identifiers[bin.Right.ID]
	lv, lok, err := resolveJoinedColumn(row, identifierQualifier(src, left), identifierName(src, left))
	if err != nil {
		return false, err
	}
	rv, rok, err := resolveJoinedColumn(row, identifierQualifier(src, right), identifierName(src, right))
	if err != nil {
		return false, err
	}
	if !lok || !rok || !scalarEqual(lv, rv) {
		return false, nil
	}
	return true, nil
}

func evaluateCTEPredicate(src []byte, doc *parser.QueryDoc, ref parser.NodeRef, row *cteJoinedRow) (bool, error) {
	switch ref.Kind {
	case parser.NodeKindBinaryExpr:
		bin := doc.BinaryExprs[ref.ID]
		if bin.Operator == uint8(lexer.KindAnd) || bin.Operator == uint8(lexer.KindOr) {
			left, err := evaluateCTEPredicate(src, doc, bin.Left, row)
			if err != nil {
				return false, err
			}
			right, err := evaluateCTEPredicate(src, doc, bin.Right, row)
			if err != nil {
				return false, err
			}
			if bin.Operator == uint8(lexer.KindAnd) {
				return left && right, nil
			}
			return left || right, nil
		}
		lv, lok, err := evaluateCTEScalar(src, doc, bin.Left, row)
		if err != nil {
			return false, err
		}
		rv, rok, err := evaluateCTEScalar(src, doc, bin.Right, row)
		if err != nil {
			return false, err
		}
		if !lok || !rok {
			return false, nil
		}
		switch lexer.Kind(bin.Operator) {
		case lexer.KindEquals:
			return scalarEqual(lv, rv), nil
		case lexer.KindGreaterThan:
			return scalarCompare(lv, rv) > 0, nil
		case lexer.KindLessThan:
			return scalarCompare(lv, rv) < 0, nil
		default:
			return false, fmt.Errorf("unsupported WHERE operator %d", bin.Operator)
		}
	default:
		return false, fmt.Errorf("unsupported WHERE expression kind %d", ref.Kind)
	}
}

func evaluateCTEScalar(src []byte, doc *parser.QueryDoc, ref parser.NodeRef, row *cteJoinedRow) (interface{}, bool, error) {
	switch ref.Kind {
	case parser.NodeKindIdentifier:
		id := doc.Identifiers[ref.ID]
		return resolveJoinedColumn(row, identifierQualifier(src, id), identifierName(src, id))
	case parser.NodeKindNumber:
		n := doc.Numbers[ref.ID]
		var value float64
		if _, err := fmt.Sscanf(string(src[n.Start:n.End]), "%f", &value); err != nil {
			return nil, false, err
		}
		return value, true, nil
	case parser.NodeKindString:
		s := doc.Strings[ref.ID]
		value := string(src[s.Start:s.End])
		if len(value) >= 2 && value[0] == '\'' && value[len(value)-1] == '\'' {
			value = value[1 : len(value)-1]
		}
		return value, true, nil
	default:
		return nil, false, fmt.Errorf("unsupported scalar expression kind %d", ref.Kind)
	}
}

func identifierQualifier(src []byte, id parser.Identifier) string {
	if id.QualEnd > id.QualStart {
		return string(src[id.QualStart:id.QualEnd])
	}
	return ""
}

func identifierName(src []byte, id parser.Identifier) string {
	return string(src[id.Start:id.End])
}

func scalarNumber(v interface{}) (float64, bool) {
	switch n := v.(type) {
	case uint64:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case int32:
		return float64(n), true
	case float64:
		return n, true
	case float32:
		return float64(n), true
	default:
		return 0, false
	}
}

func scalarEqual(a, b interface{}) bool {
	if af, ok := scalarNumber(a); ok {
		if bf, ok := scalarNumber(b); ok {
			return af == bf
		}
	}
	return fmt.Sprint(a) == fmt.Sprint(b)
}

func scalarCompare(a, b interface{}) int {
	if af, ok := scalarNumber(a); ok {
		if bf, ok := scalarNumber(b); ok {
			if af < bf {
				return -1
			}
			if af > bf {
				return 1
			}
			return 0
		}
	}
	as, bs := fmt.Sprint(a), fmt.Sprint(b)
	if as < bs {
		return -1
	}
	if as > bs {
		return 1
	}
	return 0
}

// resolveCTEColumn returns the value for a column from the Leiden relation row.
func resolveCTEColumn(row *LeidenRelationRow, col string) interface{} {
	if row == nil {
		return nil
	}
	switch col {
	case "node_id":
		return row.NodeID
	case "community_id":
		return row.CommunityID
	case "collection":
		return row.Collection
	case "record_id":
		return row.RecordID
	default:
		return nil
	}
}

func evaluateCTEProjection(
	ctx context.Context,
	src []byte,
	doc *parser.QueryDoc,
	proj parser.Projection,
	row *cteJoinedRow,
	col *Collection,
	params QueryParams,
) (interface{}, bool, error) {
	_ = ctx
	if proj.Star {
		return nil, false, nil
	}
	switch proj.Expr.Kind {
	case parser.NodeKindIdentifier:
		id := doc.Identifiers[proj.Expr.ID]
		return resolveJoinedColumn(row, identifierQualifier(src, id), identifierName(src, id))
	case parser.NodeKindVectorFunc:
		vf := doc.VectorFuncs[proj.Expr.ID]
		if vf.VectorA.Kind != parser.NodeKindIdentifier {
			return nil, false, fmt.Errorf("VECTOR_DISTANCE first operand must be an identifier")
		}
		vectorID := doc.Identifiers[vf.VectorA.ID]
		vectorColumn := identifierName(src, vectorID)
		if vectorColumn != "embedding" && vectorColumn != "vector" && vectorColumn != "vec" {
			return nil, false, fmt.Errorf("unsupported vector column %q", vectorColumn)
		}
		queryVector, err := resolveCTEVectorOperand(src, doc, vf.VectorB, params)
		if err != nil {
			return nil, false, err
		}
		if len(queryVector) == 0 || len(queryVector) != len(row.record.Vector) {
			return nil, false, fmt.Errorf("vector dimension mismatch")
		}
		vfp := optimizer.VectorFuncProjection{
			IsDistance:  !vf.IsMaxSim,
			QueryVector: queryVector,
		}
		return float64(computeVectorScore(col, vfp, row.record.Vector)), true, nil
	default:
		return nil, false, fmt.Errorf("unsupported projection expression kind %d", proj.Expr.Kind)
	}
}

func resolveCTEVectorOperand(src []byte, doc *parser.QueryDoc, ref parser.NodeRef, params QueryParams) ([]float32, error) {
	switch ref.Kind {
	case parser.NodeKindString:
		s := doc.Strings[ref.ID]
		literal := string(src[s.Start:s.End])
		if len(literal) >= 2 && literal[0] == '\'' && literal[len(literal)-1] == '\'' {
			literal = literal[1 : len(literal)-1]
		}
		return parseVectorLiteral(literal), nil
	case parser.NodeKindIdentifier:
		id := doc.Identifiers[ref.ID]
		if id.Start >= id.End || int(id.End) > len(src) {
			return nil, fmt.Errorf("invalid vector parameter identifier")
		}
		name := string(src[id.Start:id.End])
		if len(name) < 2 || (name[0] != '$' && name[0] != '@') {
			return nil, fmt.Errorf("vector operand must be a named parameter or literal")
		}
		value, ok := params[name[1:]]
		if !ok {
			return nil, fmt.Errorf("vector parameter %q is missing", name)
		}
		vector, ok := value.([]float32)
		if !ok {
			return nil, fmt.Errorf("vector parameter %q must be []float32", name)
		}
		return append([]float32(nil), vector...), nil
	default:
		return nil, fmt.Errorf("vector operand must be a vector literal or named parameter")
	}
}

// resolveRecordColumn returns the value for a column from the outer table record.
func resolveRecordColumn(rec Record, col string) interface{} {
	if col == "id" || col == "ID" {
		return rec.ID
	}
	if rec.Metadata != nil {
		if v, ok := rec.Metadata[col]; ok {
			return v
		}
	}
	// Try common column mappings.
	switch col {
	case "title":
		if rec.Metadata != nil {
			if v, ok := rec.Metadata["title"]; ok {
				return v
			}
		}
		return rec.ID
	}
	return nil
}

// applySelectOrderBy sorts results by the ORDER BY column from the outer SELECT.
func applySelectOrderBy(out *SearchResults, src []byte, doc *parser.QueryDoc, stmt parser.SelectStmt) {
	if stmt.OrderBy.Kind != parser.NodeKindIdentifier {
		return
	}
	orderID := doc.Identifiers[stmt.OrderBy.ID]
	orderCol := string(src[orderID.Start:orderID.End])

	sort.SliceStable(out.Results, func(i, j int) bool {
		vi := out.Results[i].Metadata[orderCol]
		vj := out.Results[j].Metadata[orderCol]
		if vi == nil && vj == nil {
			return false
		}
		if vi == nil {
			return true
		}
		if vj == nil {
			return false
		}
		cmp := scalarCompare(vi, vj)
		if stmt.IsDesc {
			return cmp > 0
		}
		return cmp < 0
	})
}
