package libravdb

import (
	"context"
	"testing"

	apexjson "github.com/xDarkicex/apexJSON/v2"
	"github.com/xDarkicex/libravdb/internal/graph"
)

func newCypherPipelineDB(t *testing.T) (*Database, context.Context) {
	t.Helper()
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:cypher_pipeline"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewGraph(GraphConfig{})
	if err != nil {
		db.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		g.Close()
		db.Close()
	})
	if _, err := db.CreateCollection(ctx, "people", WithMetadataOnly(), WithGraph(g), WithMetadataSchema(MetadataSchema{
		"uuid": StringField,
		"name": StringField,
	})); err != nil {
		t.Fatal(err)
	}
	for _, person := range []string{"alice", "bob", "carol"} {
		col, getErr := db.GetCollection("people")
		if getErr != nil {
			t.Fatal(getErr)
		}
		if err := col.Insert(ctx, person, nil, map[string]interface{}{"uuid": person, "name": person}); err != nil {
			t.Fatal(err)
		}
	}
	if !graph.RegisterEdgeKind("CYPHER_PIPE_REL", 231) {
		t.Fatal("edge kind registration failed")
	}
	return db, ctx
}

func cypherPipelineCollection(t *testing.T, db *Database) *Collection {
	t.Helper()
	col, err := db.GetCollection("people")
	if err != nil {
		t.Fatal(err)
	}
	return col
}

func TestCypherUniversalSetAndDetachDelete(t *testing.T) {
	db, ctx := newCypherPipelineDB(t)
	returned, err := db.QueryWithParams(ctx, `MERGE (a:Person {uuid: $id}) SET a.name = $name RETURN a.uuid AS uuid`, QueryParams{"id": "alice", "name": "Alice"})
	if err != nil {
		t.Fatalf("universal SET: %v", err)
	}
	if returned.Total != 1 || returned.Results[0].Metadata["uuid"] != "alice" {
		t.Fatalf("MERGE RETURN result=%#v", returned.Results)
	}
	col := cypherPipelineCollection(t, db)
	row, err := col.Get(ctx, "alice")
	if err != nil || row.Metadata["name"] != "Alice" {
		t.Fatalf("universal SET result=%#v err=%v", row.Metadata, err)
	}

	alice, _ := db.GetNodeID(ctx, "people", "alice")
	bob, _ := db.GetNodeID(ctx, "people", "bob")
	txn := col.GetGraph().BeginTxn()
	if err := col.GetGraph().AddEdge(txn, alice, bob, 1, 231); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `MATCH (a)-[r:CYPHER_PIPE_REL]->(b) DETACH DELETE a`); err != nil {
		t.Fatalf("DETACH DELETE: %v", err)
	}
	if _, err := col.Get(ctx, "alice"); err == nil {
		t.Fatal("DETACH DELETE left alice")
	}
	if _, err := col.Get(ctx, "bob"); err != nil {
		t.Fatalf("DETACH DELETE removed bob: %v", err)
	}
}

func TestCypherPipeWithAndAggregate(t *testing.T) {
	db, ctx := newCypherPipelineDB(t)
	rows, err := db.Query(ctx, `MATCH (n) WITH n.uuid AS u WHERE u <> 'bob' RETURN u ORDER BY u`)
	if err != nil {
		t.Fatalf("pipe WITH: %v", err)
	}
	if rows.Total != 2 || rows.Results[0].Metadata["u"] != "alice" || rows.Results[1].Metadata["u"] != "carol" {
		t.Fatalf("pipe rows=%#v", rows.Results)
	}
	count, err := db.Query(ctx, `MATCH (n) WITH count(n) AS c RETURN c`)
	if err != nil {
		t.Fatalf("aggregate WITH: %v", err)
	}
	if count.Total != 1 || count.Results[0].Metadata["c"] != int64(3) {
		t.Fatalf("aggregate rows=%#v metadata=%#v", count.Results, count.Results[0].Metadata)
	}
	chained, err := db.Query(ctx, `MATCH (n) WITH n.uuid AS u MATCH (m {uuid: u}) RETURN m.uuid AS uuid ORDER BY uuid`)
	if err != nil {
		t.Fatalf("MATCH after WITH: %v", err)
	}
	if chained.Total != 3 || chained.Results[0].Metadata["uuid"] != "alice" || chained.Results[2].Metadata["uuid"] != "carol" {
		t.Fatalf("chained rows=%#v", chained.Results)
	}
}

func TestCypherRelationshipSetAndDelete(t *testing.T) {
	db, ctx := newCypherPipelineDB(t)
	col := cypherPipelineCollection(t, db)
	alice, _ := db.GetNodeID(ctx, "people", "alice")
	bob, _ := db.GetNodeID(ctx, "people", "bob")
	txn := col.GetGraph().BeginTxn()
	if err := col.GetGraph().AddEdge(txn, alice, bob, 1, 231); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := db.QueryWithParams(ctx, `MERGE (a {uuid: $a})-[r:CYPHER_PIPE_REL]->(b {uuid: $b}) SET r.fact = $fact`, QueryParams{"a": "alice", "b": "bob", "fact": "family"}); err != nil {
		t.Fatalf("relationship SET: %v", err)
	}
	views, err := col.GetGraph().NeighborsWithProperties(alice)
	if err != nil || len(views) != 1 {
		t.Fatalf("relationship lookup views=%#v err=%v", views, err)
	}
	raw, err := graph.EdgePropertyJSON(views[0].Properties)
	if err != nil {
		t.Fatal(err)
	}
	var properties map[string]interface{}
	if err := apexjson.Unmarshal(raw, &properties); err != nil || properties["fact"] != "family" {
		t.Fatalf("relationship properties=%#v err=%v", properties, err)
	}
	if _, err := db.Query(ctx, `MATCH (a)-[r:CYPHER_PIPE_REL]->(b) DELETE r`); err != nil {
		t.Fatalf("relationship DELETE: %v", err)
	}
	views, _ = col.GetGraph().NeighborsWithProperties(alice)
	if len(views) != 0 {
		t.Fatalf("relationship DELETE left edges=%#v", views)
	}
	// Re-add the edge and verify non-detach vertex deletion fails atomically.
	txn = col.GetGraph().BeginTxn()
	if err := col.GetGraph().AddEdge(txn, alice, bob, 1, 231); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `MATCH (a) DELETE a`); err == nil {
		t.Fatal("non-detach vertex DELETE unexpectedly succeeded")
	}
	if views, _ := col.GetGraph().Neighbors(alice); len(views) != 1 {
		t.Fatalf("failed DELETE mutated graph: %#v", views)
	}
}
