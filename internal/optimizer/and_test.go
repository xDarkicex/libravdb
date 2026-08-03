package optimizer

import (
	"testing"

	"github.com/xDarkicex/lexer/parser"
)

func TestExtractAndPredicates(t *testing.T) {
	sql := "SELECT id, name FROM users WHERE id > 5 AND name = 'alice'"
	doc := &parser.QueryDoc{}
	if err := parser.Parse([]byte(sql), doc); err != nil {
		t.Fatalf("Parse: %v", err)
	}

	stmt := &doc.SelectStmts[0]
	t.Logf("Projections: %d", stmt.ProjectionsCount)
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		proj := &doc.Projections[stmt.ProjectionsStart+i]
		if proj.Expr.Kind == parser.NodeKindIdentifier {
			id := &doc.Identifiers[proj.Expr.ID]
			t.Logf("  proj[%d]: %q", i, string([]byte(sql)[id.Start:id.End]))
		}
	}

	// Walk WHERE tree
	whereNode := stmt.WhereExpr
	t.Logf("WHERE kind: %v", whereNode.Kind)
	if whereNode.Kind == parser.NodeKindBinaryExpr {
		be := &doc.BinaryExprs[whereNode.ID]
		t.Logf("Top-level op: %d (KindAnd=%d)", be.Operator, 25)
	}
}
