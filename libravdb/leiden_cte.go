package libravdb

import (
	"context"
	"fmt"
	"strings"

	"github.com/xDarkicex/lexer/parser"
)

// =============================================================================
// Bound CTE type
// =============================================================================

// BoundLeidenCTE is a fully-bound COMPUTE LEIDEN CTE ready for execution.
// The outer SELECT's FROM table provides the collection; the CTE body's
// Leiden plan is lowered, collection-resolved, and bound against the epoch.
type BoundLeidenCTE struct {
	Name       string
	Collection string
	JoinAlias  string
	Plan       *BoundLeidenMatchPlan
}

// =============================================================================
// BindLeidenCTE
// =============================================================================

// BindLeidenCTE binds a parsed WITH ... COMPUTE LEIDEN CTE to the outer
// SELECT's FROM collection. The CTE body is lowered into a logical plan,
// collection-resolved from the outer FROM table, and bound against the
// active epoch. The JOIN reference is validated against the CTE name.
//
// No Leiden execution occurs. No rows are materialized.
func (e *EpochTx) BindLeidenCTE(
	ctx context.Context,
	src []byte,
	doc *parser.QueryDoc,
	selectIndex int,
) (*BoundLeidenCTE, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if doc == nil {
		return nil, fmt.Errorf("QueryDoc must not be nil")
	}

	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil, ErrEpochClosed
	}
	e.mu.Unlock()

	if selectIndex < 0 || selectIndex >= len(doc.SelectStmts) {
		return nil, fmt.Errorf("selectIndex %d out of range [0, %d)", selectIndex, len(doc.SelectStmts))
	}

	stmt := doc.SelectStmts[selectIndex]

	// ── Validate exactly one CTE ──
	if stmt.CTEsCount != 1 {
		return nil, fmt.Errorf("expected exactly 1 CTE, got %d", stmt.CTEsCount)
	}
	if int(stmt.CTEsStart) >= len(doc.CTEs) {
		return nil, fmt.Errorf("CTEsStart %d out of range", stmt.CTEsStart)
	}

	cte := doc.CTEs[stmt.CTEsStart]
	if cte.NameStart >= cte.NameEnd || int(cte.NameEnd) > len(src) {
		return nil, fmt.Errorf("CTE name offsets out of range")
	}
	cteName := string(src[cte.NameStart:cte.NameEnd])
	if cteName == "" {
		return nil, fmt.Errorf("CTE name must not be empty")
	}

	if cte.Body.Kind != parser.NodeKindComputeLeidenStmt {
		return nil, fmt.Errorf("CTE body must be COMPUTE LEIDEN, got %v", cte.Body.Kind)
	}
	bodyIndex := int(cte.Body.ID)
	if bodyIndex < 0 || bodyIndex >= len(doc.ComputeLeidenStmts) {
		return nil, fmt.Errorf("CTE body index %d out of range", bodyIndex)
	}

	// ── Resolve outer FROM table as the collection ──
	if stmt.FromTable.Kind != parser.NodeKindTableExpr {
		return nil, fmt.Errorf("outer FROM must be a table expression, got %v", stmt.FromTable.Kind)
	}
	fromID := int(stmt.FromTable.ID)
	if fromID < 0 || fromID >= len(doc.TableExprs) {
		return nil, fmt.Errorf("FROM table index %d out of range", fromID)
	}
	tbl := doc.TableExprs[fromID]
	if tbl.Start >= tbl.End || int(tbl.End) > len(src) {
		return nil, fmt.Errorf("FROM table name offsets out of range")
	}
	collection := string(src[tbl.Start:tbl.End])
	if collection == "" {
		return nil, fmt.Errorf("outer FROM table name must not be empty")
	}

	// Verify collection exists and has a graph.
	col, err := e.db.GetCollection(collection)
	if err != nil {
		return nil, fmt.Errorf("outer collection %q: %w", collection, err)
	}
	if col.GetGraph() == nil {
		return nil, fmt.Errorf("outer collection %q has no graph", collection)
	}

	// ── Lower CTE body into logical plan ──
	logicalPlan, err := LowerComputeLeidenPlan(src, doc, bodyIndex)
	if err != nil {
		return nil, fmt.Errorf("lower CTE body: %w", err)
	}

	// Collection: outer FROM is authoritative. Reject conflicts.
	if logicalPlan.Collection != "" && logicalPlan.Collection != collection {
		return nil, fmt.Errorf("CTE body collection %q conflicts with outer FROM %q",
			logicalPlan.Collection, collection)
	}
	logicalPlan.Collection = collection

	// ── Bind against the epoch using the outer collection ──
	bound, err := e.BindLeidenMatchPlan(ctx, logicalPlan, collection)
	if err != nil {
		return nil, fmt.Errorf("bind CTE plan: %w", err)
	}

	// ── Validate JOIN reference ──
	joinAlias, err := resolveJoinAlias(src, stmt, cteName)
	if err != nil {
		return nil, err
	}

	return &BoundLeidenCTE{
		Name:       cteName,
		Collection: collection,
		JoinAlias:  joinAlias,
		Plan:       bound,
	}, nil
}

// resolveJoinAlias finds the JOIN clause whose table name matches the CTE
// name and extracts its alias. Returns an error if no matching JOIN exists.
func resolveJoinAlias(src []byte, stmt parser.SelectStmt, cteName string) (string, error) {
	cteLower := strings.ToLower(cteName)

	for _, join := range stmt.Joins {
		if join.TableStart >= join.TableEnd || int(join.TableEnd) > len(src) {
			continue
		}
		joinTable := string(src[join.TableStart:join.TableEnd])
		if strings.ToLower(joinTable) != cteLower {
			continue
		}

		if join.Alias == 0 || join.AliasEnd <= join.Alias {
			return "", fmt.Errorf("JOIN on CTE %q must have an alias", cteName)
		}
		alias := string(src[join.Alias:join.AliasEnd])
		if alias == "" {
			return "", fmt.Errorf("JOIN alias for CTE %q must not be empty", cteName)
		}
		return alias, nil
	}

	return "", fmt.Errorf("CTE %q is not referenced by a JOIN", cteName)
}
