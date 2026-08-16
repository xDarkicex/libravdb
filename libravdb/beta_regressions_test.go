package libravdb

import (
	"context"
	"testing"
)

func TestBetaGraphEdgesDeleteSQL(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/graph-delete.libravdb"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	graph, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	if !RegisterEdgeKind("FOLLOWS", 241) {
		t.Fatal("register FOLLOWS")
	}
	col, err := db.CreateCollection(ctx, "users", WithMetadataOnly(), WithGraph(graph))
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"alice", "bob", "carol"} {
		if err := col.Insert(ctx, id, nil, nil); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}
	if _, err := db.Query(ctx, "INSERT INTO GRAPH_EDGES VALUES ('alice', 'FOLLOWS', 'bob')"); err != nil {
		t.Fatalf("insert edge: %v", err)
	}
	if _, err := db.QueryWithParams(ctx, "INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)", QueryParams{
		"1": "alice", "2": "FOLLOWS", "3": "carol",
	}); err != nil {
		t.Fatalf("parameterized insert edge: %v", err)
	}
	if _, err := db.QueryWithParams(ctx, "DELETE FROM GRAPH_EDGES WHERE source = $1 AND type = $2 AND target = $3", QueryParams{
		"1": "alice", "2": "FOLLOWS", "3": "bob",
	}); err != nil {
		t.Fatalf("parameterized delete edge: %v", err)
	}
	if _, err := db.Query(ctx, "DELETE FROM GRAPH_EDGES WHERE source = 'alice' AND type = 'FOLLOWS' AND target = 'carol'"); err != nil {
		t.Fatalf("delete edge: %v", err)
	}
	var remaining int
	graph.ForEachEdge(func(src, tgt uint64, edge Edge) bool {
		if edge.GetKind() == 241 {
			remaining++
		}
		return true
	})
	if remaining != 0 {
		t.Fatalf("remaining FOLLOWS edges = %d, want 0", remaining)
	}
}

func TestBetaGraphEdgesDeleteSQLEpoch(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/graph-delete-epoch.libravdb"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	graph, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	if !RegisterEdgeKind("EPOCH_FOLLOWS", 243) {
		t.Fatal("register EPOCH_FOLLOWS")
	}
	col, err := db.CreateCollection(ctx, "users", WithMetadataOnly(), WithGraph(graph))
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"alice", "bob"} {
		if err := col.Insert(ctx, id, nil, nil); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}
	if _, err := db.Query(ctx, "INSERT INTO GRAPH_EDGES VALUES ('alice', 'EPOCH_FOLLOWS', 'bob')"); err != nil {
		t.Fatalf("insert edge: %v", err)
	}

	rollbackEpoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rollbackEpoch.Query(ctx, "DELETE FROM GRAPH_EDGES WHERE source = 'alice' AND type = 'EPOCH_FOLLOWS' AND target = 'bob'", nil); err != nil {
		t.Fatalf("epoch delete: %v", err)
	}
	if edges, err := graph.Neighbors(mustNodeID(t, db, ctx, "users", "alice")); err != nil || len(edges) != 1 {
		t.Fatalf("live graph changed before rollback: edges=%v err=%v", edges, err)
	}
	if err := rollbackEpoch.Rollback(ctx); err != nil {
		t.Fatalf("rollback epoch delete: %v", err)
	}

	commitEpoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := commitEpoch.Query(ctx, "DELETE FROM GRAPH_EDGES WHERE source = $1 AND type = $2 AND target = $3", QueryParams{
		"1": "alice", "2": "EPOCH_FOLLOWS", "3": "bob",
	}); err != nil {
		t.Fatalf("parameterized epoch delete: %v", err)
	}
	if err := commitEpoch.Commit(ctx); err != nil {
		t.Fatalf("commit epoch delete: %v", err)
	}
	if edges, err := graph.Neighbors(mustNodeID(t, db, ctx, "users", "alice")); err != nil || len(edges) != 0 {
		t.Fatalf("edge survived committed epoch delete: edges=%v err=%v", edges, err)
	}
}

func mustNodeID(t *testing.T, db *Database, ctx context.Context, collection, id string) uint64 {
	t.Helper()
	nodeID, err := db.GetNodeID(ctx, collection, id)
	if err != nil {
		t.Fatalf("GetNodeID(%s,%s): %v", collection, id, err)
	}
	return nodeID
}

func TestBetaCatalogAndGraphReopen(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/reopen.libravdb"
	graph, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if !RegisterEdgeKind("FOLLOWS", 241) {
		t.Fatal("register FOLLOWS")
	}
	db, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatal(err)
	}
	col, err := db.CreateCollection(ctx, "users", WithMetadataOnly(), WithGraph(graph), WithMetadataSchema(MetadataSchema{
		"email": StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "alice", nil, map[string]interface{}{"email": "a@example.com"}); err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "bob", nil, map[string]interface{}{"email": "b@example.com"}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "CREATE TABLE follows (source TEXT, target TEXT, created_at TEXT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO GRAPH_EDGES VALUES ('alice', 'FOLLOWS', 'bob')"); err != nil {
		t.Fatal(err)
	}
	if rows, err := db.Query(ctx, "SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt)"); err != nil || len(rows.Results) != 1 {
		t.Fatalf("initial JOIN MATCH rows=%v err=%v, want 1 row", rows, err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	graph.Close()

	db, err = Open(WithStoragePath(path))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	defer func() {
		if col, err := db.GetCollection("users"); err == nil {
			col.Close()
		}
	}()
	if _, err := db.Query(ctx, "SELECT source, target FROM follows"); err != nil {
		t.Fatalf("reopened follows columns: %v", err)
	}
	if _, err := db.Query(ctx, "SELECT email FROM users"); err != nil {
		t.Fatalf("reopened users metadata column: %v", err)
	}
	users, err := db.GetCollection("users")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := users.Get(ctx, "alice"); err != nil {
		t.Fatalf("reopened users record: %v", err)
	}
	reopenedGraph, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer reopenedGraph.Close()
	users.SetGraph(reopenedGraph)
	src, err := db.GetNodeID(ctx, "users", "alice")
	if err != nil {
		t.Fatal(err)
	}
	edges, err := reopenedGraph.Neighbors(src)
	if err != nil {
		t.Fatal(err)
	}
	bobNode, err := db.GetNodeID(ctx, "users", "bob")
	if err != nil {
		t.Fatal(err)
	}
	if len(edges) != 1 || edges[0].Target != bobNode || edges[0].GetKind() != 241 {
		t.Fatalf("reopened graph edges = %#v, want alice→bob FOLLOWS edge", edges)
	}
	if rows, err := db.Query(ctx, "SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt)"); err != nil || len(rows.Results) != 1 {
		t.Fatalf("reopened JOIN MATCH rows=%v err=%v, want 1 row", rows, err)
	}
}

func TestBetaJoinMatchProjectionAndSourceFilter(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:join-match-beta"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	graph, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	if !RegisterEdgeKind("BETA_FOLLOWS", 242) {
		t.Fatal("register BETA_FOLLOWS")
	}
	col, err := db.CreateCollection(ctx, "users", WithMetadataOnly(), WithGraph(graph), WithMetadataSchema(MetadataSchema{
		"name": StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for id, name := range map[string]string{"alice": "Alice", "bob": "Bob", "carol": "Carol"} {
		if err := col.Insert(ctx, id, nil, map[string]interface{}{"name": name}); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}
	alice, _ := db.GetNodeID(ctx, "users", "alice")
	bob, _ := db.GetNodeID(ctx, "users", "bob")
	carol, _ := db.GetNodeID(ctx, "users", "carol")
	txn := graph.BeginTxn()
	if err := txn.AddEdge(alice, bob, 1, 242); err != nil {
		t.Fatal(err)
	}
	if err := txn.AddEdge(carol, bob, 1, 242); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	rows, err := db.Query(ctx, "SELECT tgt.id, tgt.name FROM users src JOIN MATCH (src)-[:BETA_FOLLOWS]->(tgt) WHERE src.id = 'alice'")
	if err != nil {
		t.Fatal(err)
	}
	if len(rows.Results) != 1 {
		t.Fatalf("JOIN MATCH rows=%d, want 1", len(rows.Results))
	}
	if got := rows.Results[0].Metadata["id"]; got != "bob" {
		t.Fatalf("JOIN MATCH projected id=%#v, want bob; metadata=%#v", got, rows.Results[0].Metadata)
	}
	if got := rows.Results[0].Metadata["name"]; got != "Bob" {
		t.Fatalf("JOIN MATCH projected name=%#v, want Bob; metadata=%#v", got, rows.Results[0].Metadata)
	}
}

func TestBetaJoinMatchFiltersDistinctEdgeKinds(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/join-match-edge-kinds"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	graph, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()

	if !RegisterEdgeKind("BETA_FAMILY_OF", 247) {
		t.Fatal("register BETA_FAMILY_OF")
	}
	if !RegisterEdgeKind("BETA_FRIEND_OF", 248) {
		t.Fatal("register BETA_FRIEND_OF")
	}
	col, err := db.CreateCollection(ctx, "people", WithMetadataOnly(), WithGraph(graph))
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"alice", "family", "friend"} {
		if err := col.Insert(ctx, id, nil, nil); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}
	alice, _ := db.GetNodeID(ctx, "people", "alice")
	family, _ := db.GetNodeID(ctx, "people", "family")
	friend, _ := db.GetNodeID(ctx, "people", "friend")
	txn := graph.BeginTxn()
	if err := txn.AddEdge(alice, family, 1, 247); err != nil {
		t.Fatal(err)
	}
	if err := txn.AddEdge(alice, friend, 1, 248); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	rows, err := db.Query(ctx, "SELECT tgt.id FROM people src JOIN MATCH (src)-[:BETA_FAMILY_OF]->(tgt) WHERE src.id = 'alice'")
	if err != nil {
		t.Fatal(err)
	}
	if len(rows.Results) != 1 || rows.Results[0].Metadata["id"] != "family" {
		t.Fatalf("family JOIN MATCH rows=%#v, want only family", rows)
	}
	unknown, err := db.Query(ctx, "SELECT tgt.id FROM people src JOIN MATCH (src)-[:BETA_NOT_REGISTERED]->(tgt) WHERE src.id = 'alice'")
	if err == nil {
		t.Fatalf("unknown edge label unexpectedly succeeded with rows=%#v", unknown)
	}
}

func TestBetaChainedJoinMatchAppliesTerminalFilter(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:chained-join-match-beta"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	graph, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	if !RegisterEdgeKind("CHAIN_FOLLOWS", 244) {
		t.Fatal("register CHAIN_FOLLOWS")
	}
	col, err := db.CreateCollection(ctx, "users", WithMetadataOnly(), WithGraph(graph))
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"alice", "bob", "dave"} {
		if err := col.Insert(ctx, id, nil, nil); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}
	node := func(id string) uint64 { return mustNodeID(t, db, ctx, "users", id) }
	txn := graph.BeginTxn()
	for _, edge := range [][2]string{{"alice", "bob"}, {"bob", "alice"}, {"bob", "dave"}} {
		if err := txn.AddEdge(node(edge[0]), node(edge[1]), 1, 244); err != nil {
			t.Fatal(err)
		}
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	rows, err := db.QueryWithParams(ctx, "SELECT tgt.id FROM users me JOIN MATCH (me)-[f1:CHAIN_FOLLOWS]->(mid) JOIN MATCH (mid)-[f2:CHAIN_FOLLOWS]->(tgt) WHERE me.id = $1 AND tgt.id <> $1", QueryParams{"1": "alice"})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows.Results) != 1 {
		for _, row := range rows.Results {
			t.Logf("chained result id=%q metadata=%#v", row.ID, row.Metadata)
		}
		t.Fatalf("chained JOIN MATCH rows=%d, want 1 (dave only): %#v", len(rows.Results), rows)
	}
	if got := rows.Results[0].Metadata["id"]; got != "dave" {
		t.Fatalf("chained JOIN MATCH target=%#v, want dave; metadata=%#v id=%q", got, rows.Results[0].Metadata, rows.Results[0].ID)
	}

	grouped, err := db.QueryWithParams(ctx, "SELECT tgt.id, COUNT(*) AS mutual FROM users me JOIN MATCH (me)-[f1:CHAIN_FOLLOWS]->(mid) JOIN MATCH (mid)-[f2:CHAIN_FOLLOWS]->(tgt) WHERE me.id = $1 AND tgt.id <> $1 GROUP BY tgt.id", QueryParams{"1": "alice"})
	if err != nil {
		t.Fatal(err)
	}
	if len(grouped.Results) != 1 || grouped.Results[0].Metadata["id"] != "dave" || grouped.Results[0].Metadata["mutual"] != int64(1) {
		t.Fatalf("grouped chained JOIN MATCH = %#v, want dave/mutual=1", grouped)
	}

	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	epochRows, err := epoch.Query(ctx, "SELECT tgt.id FROM users me JOIN MATCH (me)-[f1:CHAIN_FOLLOWS]->(mid) JOIN MATCH (mid)-[f2:CHAIN_FOLLOWS]->(tgt) WHERE me.id = $1 AND tgt.id <> $1", QueryParams{"1": "alice"})
	if err != nil {
		t.Fatalf("epoch chained JOIN MATCH: %v", err)
	}
	if len(epochRows.Results) != 1 || epochRows.Results[0].Metadata["id"] != "dave" {
		t.Fatalf("epoch chained JOIN MATCH = %#v, want dave only", epochRows)
	}
	if err := epoch.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
}
