package optimizer

import (
	"context"
	"testing"

	"github.com/xDarkicex/lexer/parser"
	btree "github.com/xDarkicex/libravdb/internal/index/btree"
)

func TestRelational_E2E(t *testing.T) {
	// 1. Create a B-tree and insert keys
	tree, _ := btree.New(btree.DefaultConfig())
	defer tree.Close()
	ctx := context.Background()

	for i := 0; i < 10; i++ {
		key := []byte{byte('a' + i)}
		val := btree.EncodeValue(uint32(i), 1, 0)
		tree.Insert(ctx, key, val)
	}

	// 2. Parse SQL with WHERE id = 'c'
	sql := "SELECT id FROM users WHERE id = 'c'"
	doc := &parser.QueryDoc{}
	if err := parser.Parse([]byte(sql), doc); err != nil {
		t.Fatalf("Parse: %v", err)
	}

	// 3. Verify AST has the predicate
	stmt := &doc.SelectStmts[0]
	be := &doc.BinaryExprs[stmt.WhereExpr.ID]
	if be.Left.Kind != parser.NodeKindIdentifier {
		t.Fatal("LEFT is not Identifier")
	}
	sl := &doc.Strings[be.Right.ID]
	src := []byte(sql)
	value := src[sl.Start+1 : sl.End-1] // strip quotes
	t.Logf("Predicate: id = %q (operator=%d)", string(value), be.Operator)

	// 4. Search B-tree directly (simulating executeRelational)
	val, err := tree.Search(ctx, value)
	if err != nil {
		t.Fatalf("B-tree Search(%q): %v", value, err)
	}
	ord, ver, _ := btree.DecodeValue(val)
	t.Logf("Found: ordinal=%d version=%d key=%q", ord, ver, string(value))
	if ord != 2 {
		t.Errorf("Expected ordinal=2 for key 'c', got %d", ord)
	}

	// 5. Test range scan: WHERE id > 'e'
	sql2 := "SELECT id FROM users WHERE id > 'e'"
	doc2 := &parser.QueryDoc{}
	parser.Parse([]byte(sql2), doc2)
	stmt2 := &doc2.SelectStmts[0]
	be2 := &doc2.BinaryExprs[stmt2.WhereExpr.ID]
	sl2 := &doc2.Strings[be2.Right.ID]
	value2 := []byte(sql2)[sl2.Start+1 : sl2.End-1]

	c := tree.Seek(value2)
	count := 0
	for c.Valid() {
		key := string(c.Key())
		if key == string(value2) {
			c.Next()
			continue // skip exact match for >
		}
		count++
		t.Logf("  Range[%d]: key=%q", count-1, key)
		c.Next()
	}
	t.Logf("Range > %q: %d results", value2, count)
	if count != 5 { // f, g, h, i, j (strictly greater than 'e')
		t.Errorf("Expected 5 results for > 'e', got %d", count)
	}
}
