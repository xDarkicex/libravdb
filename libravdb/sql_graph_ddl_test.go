package libravdb

import (
	"context"
	"strings"
	"testing"
)

func TestSQLGraphDDLBootstrapsAndReopensUnifiedGraph(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/graph-ddl.libravdb"

	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "CREATE GRAPH TABLE users (name TEXT)"); err != nil {
		t.Fatalf("CREATE GRAPH TABLE: %v", err)
	}
	if _, err := db.Query(ctx, "CREATE EDGE TYPE SQL_DDL_FOLLOWS"); err != nil {
		t.Fatalf("CREATE EDGE TYPE: %v", err)
	}
	for _, query := range []string{
		"INSERT INTO users (id, name) VALUES ('alice', 'Alice')",
		"INSERT INTO users (id, name) VALUES ('bob', 'Bob')",
		"INSERT INTO GRAPH_EDGES VALUES ('alice', 'SQL_DDL_FOLLOWS', 'bob')",
	} {
		if _, err := db.Query(ctx, query); err != nil {
			t.Fatalf("%s: %v", query, err)
		}
	}
	rows, err := db.Query(ctx, "SELECT tgt.id FROM users src JOIN MATCH (src)-[:SQL_DDL_FOLLOWS]->(tgt) WHERE src.id = 'alice'")
	if err != nil {
		t.Fatalf("JOIN MATCH before reopen: %v", err)
	}
	if len(rows.Results) != 1 || rows.Results[0].Metadata["id"] != "bob" {
		t.Fatalf("JOIN MATCH before reopen = %#v, want bob", rows.Results)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer db.Close()
	users, err := db.GetCollection("users")
	if err != nil {
		t.Fatalf("reopened users: %v", err)
	}
	if users.GetGraph() == nil {
		t.Fatal("reopened graph table has no graph attachment")
	}
	rows, err = db.Query(ctx, "SELECT tgt.id FROM users src JOIN MATCH (src)-[:SQL_DDL_FOLLOWS]->(tgt) WHERE src.id = 'alice'")
	if err != nil {
		t.Fatalf("JOIN MATCH after reopen: %v", err)
	}
	if len(rows.Results) != 1 || rows.Results[0].Metadata["id"] != "bob" {
		t.Fatalf("JOIN MATCH after reopen = %#v, want bob", rows.Results)
	}
}

func TestSQLUndirectedEdgeTypeTraversesBothEndpointsAndSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/undirected-graph-ddl.libravdb"

	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE GRAPH TABLE undirected_users (name TEXT)",
		"CREATE EDGE TYPE SQL_DDL_UNDIRECTED_KNOWS UNDIRECTED",
		"INSERT INTO undirected_users (id, name) VALUES ('alice', 'Alice')",
		"INSERT INTO undirected_users (id, name) VALUES ('bob', 'Bob')",
		"INSERT INTO GRAPH_EDGES VALUES ('alice', 'SQL_DDL_UNDIRECTED_KNOWS', 'bob')",
	} {
		if _, err := db.Query(ctx, query); err != nil {
			t.Fatalf("%s: %v", query, err)
		}
	}
	rows, err := db.Query(ctx, "SELECT tgt.id FROM undirected_users src JOIN MATCH (src)-[:SQL_DDL_UNDIRECTED_KNOWS]->(tgt) WHERE src.id = 'bob'")
	if err != nil {
		t.Fatalf("reverse undirected MATCH before reopen: %v", err)
	}
	if len(rows.Results) != 1 || !strings.HasSuffix(rows.Results[0].ID, "alice") {
		t.Fatalf("reverse undirected MATCH before reopen = %#v, want alice", rows.Results)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer db.Close()
	users, err := db.GetCollection("undirected_users")
	if err != nil {
		t.Fatalf("reopened undirected_users: %v", err)
	}
	rows, err = db.Query(ctx, "SELECT tgt.id FROM undirected_users src JOIN MATCH (src)-[:SQL_DDL_UNDIRECTED_KNOWS]->(tgt) WHERE src.id = 'bob'")
	if err != nil {
		t.Fatalf("reverse undirected MATCH after reopen: %v", err)
	}
	if len(rows.Results) != 1 || !strings.HasSuffix(rows.Results[0].ID, "alice") {
		t.Fatalf("reverse undirected MATCH after reopen = %#v, want alice", rows.Results)
	}
	if _, err := db.Query(ctx, "DELETE FROM GRAPH_EDGES WHERE source = 'bob' AND type = 'SQL_DDL_UNDIRECTED_KNOWS' AND target = 'alice'"); err != nil {
		t.Fatalf("reverse undirected GRAPH_EDGES delete: %v", err)
	}
	rows, err = db.Query(ctx, "SELECT tgt.id FROM undirected_users src JOIN MATCH (src)-[:SQL_DDL_UNDIRECTED_KNOWS]->(tgt) WHERE src.id = 'bob'")
	if err != nil {
		t.Fatalf("undirected MATCH after delete: %v", err)
	}
	aliceNode, err := db.GetNodeID(ctx, "undirected_users", "alice")
	if err != nil {
		t.Fatalf("resolve alice after delete: %v", err)
	}
	bobNode, err := db.GetNodeID(ctx, "undirected_users", "bob")
	if err != nil {
		t.Fatalf("resolve bob after delete: %v", err)
	}
	aliceEdges, err := users.GetGraph().Neighbors(aliceNode)
	if err != nil {
		t.Fatalf("read alice neighbors after delete: %v", err)
	}
	bobEdges, err := users.GetGraph().Neighbors(bobNode)
	if err != nil {
		t.Fatalf("read bob neighbors after delete: %v", err)
	}
	if len(aliceEdges) != 0 || len(bobEdges) != 0 {
		t.Fatalf("undirected graph state after delete: alice=%#v bob=%#v, want no edges", aliceEdges, bobEdges)
	}
	if len(rows.Results) != 0 {
		t.Fatalf("undirected MATCH after delete = %#v, want no rows", rows.Results)
	}
}
