package libravdb

import (
	"context"
	"testing"
)

func TestSQLOrPredicatesOnMetadataCollection(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/or-metadata.libravdb"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	col, err := db.CreateCollection(ctx, "users", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{
		"email":         StringField,
		"password_hash": StringField,
		"name":          StringField,
		"created_at":    StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	rows := []struct {
		id   string
		name string
	}{
		{id: "foo-user", name: "Alice"},
		{id: "u-2", name: "foo bar"},
		{id: "u-3", name: "Carol"},
	}
	for _, row := range rows {
		if err := col.Insert(ctx, row.id, nil, map[string]interface{}{
			"email":         row.id + "@example.com",
			"password_hash": "hash",
			"name":          row.name,
			"created_at":    "2026-08-12",
		}); err != nil {
			t.Fatalf("insert %s: %v", row.id, err)
		}
	}

	result, err := db.QueryWithParams(ctx,
		"SELECT id, email, password_hash, name, created_at FROM users WHERE name LIKE $1 OR id LIKE $2 ORDER BY name",
		QueryParams{"1": "%foo%", "2": "%foo%"})
	if err != nil {
		t.Fatalf("OR metadata query: %v", err)
	}
	if result.Total != 2 {
		t.Fatalf("OR metadata rows=%d, want 2: %#v", result.Total, result.Results)
	}
	seen := map[string]bool{}
	for _, row := range result.Results {
		seen[row.ID] = true
	}
	if !seen["foo-user"] || !seen["u-2"] || seen["u-3"] {
		t.Fatalf("OR metadata IDs=%v, want foo-user and u-2", seen)
	}
}

func TestSQLOrPredicatesAcrossGraphJoinAliases(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:or-graph-join"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	graph, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	if !RegisterEdgeKind("OR_FOLLOWS", 246) {
		t.Fatal("register OR_FOLLOWS")
	}

	col, err := db.CreateCollection(ctx, "users", WithMetadataOnly(), WithGraph(graph), WithMetadataSchema(MetadataSchema{
		"name": StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for id, name := range map[string]string{
		"alice": "gentry source",
		"bob":   "ordinary",
		"carol": "gentry target",
		"dave":  "ordinary",
	} {
		if err := col.Insert(ctx, id, nil, map[string]interface{}{"name": name}); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}
	node := func(id string) uint64 {
		nodeID, nodeErr := db.GetNodeID(ctx, "users", id)
		if nodeErr != nil {
			t.Fatalf("node %s: %v", id, nodeErr)
		}
		return nodeID
	}
	txn := graph.BeginTxn()
	for _, edge := range [][2]string{{"alice", "bob"}, {"bob", "carol"}, {"dave", "bob"}, {"bob", "dave"}} {
		if err := txn.AddEdge(node(edge[0]), node(edge[1]), 1, 246); err != nil {
			t.Fatal(err)
		}
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	result, err := db.QueryWithParams(ctx, `SELECT src.id, tgt.id
        FROM users src JOIN MATCH (src)-[r:OR_FOLLOWS]->(tgt)
        WHERE src.name LIKE $1 OR tgt.name LIKE $2 OR src.id LIKE $3 OR tgt.id LIKE $4`, QueryParams{
		"1": "%gentry%", "2": "%gentry%", "3": "%gentry%", "4": "%gentry%",
	})
	if err != nil {
		t.Fatalf("OR graph query: %v", err)
	}
	seen := map[string]bool{}
	for _, row := range result.Results {
		seen[row.ID] = true
	}
	if len(seen) != 2 || !seen["alice|bob"] || !seen["bob|carol"] {
		t.Fatalf("OR graph IDs=%v, want alice|bob and bob|carol", seen)
	}
}
