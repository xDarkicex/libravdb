package libravdb

import (
	"context"
	"testing"
	"time"

	"github.com/xDarkicex/lexer/parser"
)

func TestSQLSessionSettings(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:session-settings"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	s, err := db.NewSQLSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Exec("SET statement_timeout = '5s'"); err != nil {
		t.Fatal(err)
	}
	if got := s.SessionConfig().StatementTimeout; got != 5*time.Second {
		t.Fatalf("statement_timeout=%v", got)
	}
	if err := s.Exec("SET max_recursion_depth TO 42"); err != nil {
		t.Fatal(err)
	}
	if got := s.SessionConfig().MaxRecursionDepth; got != 42 {
		t.Fatalf("max_recursion_depth=%d", got)
	}
	if err := s.Exec("RESET statement_timeout"); err != nil {
		t.Fatal(err)
	}
	if got := s.SessionConfig().StatementTimeout; got != 0 {
		t.Fatalf("RESET statement_timeout=%v", got)
	}
	if err := s.Exec("SET enable_seqscan = off"); err == nil {
		t.Fatal("enable_seqscan must not be silently accepted")
	}
	if err := s.Exec("SET unknown_setting = 1"); err == nil {
		t.Fatal("unknown setting must be rejected")
	}
}

func TestSessionSettingParserPreservesTypedValues(t *testing.T) {
	var doc parser.QueryDoc
	if err := parser.Parse([]byte("SET max_recursion_depth = 100"), &doc); err != nil {
		t.Fatal(err)
	}
	if len(doc.SessionSettingStmts) != 1 || doc.SessionSettingStmts[0].Value.Kind != parser.NodeKindNumber {
		t.Fatalf("unexpected AST: %#v", doc.SessionSettingStmts)
	}
	if err := parser.Parse([]byte("RESET statement_timeout"), &doc); err != nil {
		t.Fatal(err)
	}
	if len(doc.SessionSettingStmts) != 1 || !doc.SessionSettingStmts[0].Reset {
		t.Fatalf("unexpected RESET AST: %#v", doc.SessionSettingStmts)
	}
}

func TestSessionMaxRecursionDepth(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:session-recursion"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.CreateCollection(ctx, "tree_nodes", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{"parent_id": StringField})); err != nil {
		t.Fatal(err)
	}
	collection, _ := db.GetCollection("tree_nodes")
	for _, row := range []struct{ id, parent string }{{"root", ""}, {"child", "root"}, {"grandchild", "child"}} {
		if err := collection.Insert(ctx, row.id, nil, map[string]interface{}{"parent_id": row.parent}); err != nil {
			t.Fatal(err)
		}
	}
	session, err := db.NewSQLSession(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Exec("SET max_recursion_depth = 1"); err != nil {
		t.Fatal(err)
	}
	_, err = session.Query(`WITH RECURSIVE tree AS (
		SELECT id, parent_id FROM tree_nodes WHERE id = 'root'
		UNION ALL
		SELECT c.id, c.parent_id FROM tree_nodes c JOIN tree t ON c.parent_id = t.id
	) SELECT id FROM tree`)
	if err == nil {
		t.Fatal("recursive query exceeded configured depth without an error")
	}
}
