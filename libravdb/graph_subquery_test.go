package libravdb

import (
	"context"
	"testing"
)

func TestSQLJoinMatchSubqueryIn(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:join-match-subquery"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer gr.Close()
	if !RegisterEdgeKind("SUBQUERY_FOLLOWS", 245) {
		t.Fatal("register SUBQUERY_FOLLOWS")
	}

	users, err := db.CreateCollection(ctx, "users", WithMetadataOnly(), WithGraph(gr), WithMetadataSchema(MetadataSchema{
		"email": StringField,
		"name":  StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for id, metadata := range map[string]map[string]interface{}{
		"alice": {"email": "alice@example.com", "name": "Alice"},
		"bob":   {"email": "bob@example.com", "name": "Bob"},
		"carol": {"email": "carol@example.com", "name": "catherine"},
	} {
		if err := users.Insert(ctx, id, nil, metadata); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}

	alice, err := db.GetNodeID(ctx, "users", "alice")
	if err != nil {
		t.Fatal(err)
	}
	bob, err := db.GetNodeID(ctx, "users", "bob")
	if err != nil {
		t.Fatal(err)
	}
	carol, err := db.GetNodeID(ctx, "users", "carol")
	if err != nil {
		t.Fatal(err)
	}
	txn := gr.BeginTxn()
	if err := txn.AddEdge(alice, carol, 1, 245); err != nil {
		t.Fatal(err)
	}
	if err := txn.AddEdge(bob, alice, 1, 245); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	query := `
		SELECT id, email, name FROM users
		WHERE id IN (
			SELECT src.id
			FROM users src
			JOIN MATCH (src)-[r:SUBQUERY_FOLLOWS]->(tgt)
			WHERE tgt.name LIKE '%ca%'
		)
		ORDER BY name`
	rows, err := db.Query(ctx, query)
	if err != nil {
		t.Fatalf("JOIN MATCH subquery: %v", err)
	}
	if len(rows.Results) != 1 {
		t.Fatalf("JOIN MATCH subquery rows=%d, want 1: %#v", len(rows.Results), rows)
	}
	if got := rows.Results[0].Metadata["id"]; got != "alice" {
		t.Fatalf("JOIN MATCH subquery id=%#v, want alice; metadata=%#v", got, rows.Results[0].Metadata)
	}

	parameterized, err := db.QueryWithParams(ctx, `
		SELECT id FROM users
		WHERE id IN (
			SELECT src.id
			FROM users src
			JOIN MATCH (src)-[r:SUBQUERY_FOLLOWS]->(tgt)
			WHERE tgt.name LIKE $1
		)`, QueryParams{"1": "%ca%"})
	if err != nil {
		t.Fatalf("parameterized JOIN MATCH subquery: %v", err)
	}
	if len(parameterized.Results) != 1 || parameterized.Results[0].Metadata["id"] != "alice" {
		t.Fatalf("parameterized JOIN MATCH subquery = %#v, want alice", parameterized)
	}
}
