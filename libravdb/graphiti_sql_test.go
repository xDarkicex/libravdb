package libravdb

import (
	"context"
	"reflect"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

func TestGraphitiOptionalMatchAndVertexProperties(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:graphiti_optional"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	col, err := db.CreateCollection(ctx, "people", WithMetadataOnly(), WithGraph(g), WithMetadataSchema(MetadataSchema{
		"uuid": StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for _, person := range []string{"alice", "bob", "carol"} {
		if err := col.Insert(ctx, person, nil, map[string]interface{}{"uuid": person}); err != nil {
			t.Fatal(err)
		}
	}
	if !graph.RegisterEdgeKind("GRAPHITI_KNOWS", 221) {
		t.Fatal("edge kind registration failed")
	}
	alice, err := db.GetNodeID(ctx, "people", "alice")
	if err != nil {
		t.Fatal(err)
	}
	bob, err := db.GetNodeID(ctx, "people", "bob")
	if err != nil {
		t.Fatal(err)
	}
	txn := g.BeginTxn()
	if err := g.AddEdge(txn, alice, bob, 1, 221); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	rows, err := db.Query(ctx, `
		SELECT source_id, target_id
		FROM people AS p
		OPTIONAL MATCH (p)-[r:GRAPHITI_KNOWS]->(target:Person {uuid: 'bob'})
		ORDER BY source_id`)
	if err != nil {
		t.Fatalf("OPTIONAL MATCH: %v", err)
	}
	if rows.Total != 3 {
		t.Fatalf("OPTIONAL MATCH rows=%d, want 3: %#v", rows.Total, rows.Results)
	}
	if rows.Results[0].Metadata["source_id"] != "alice" || rows.Results[0].Metadata["target_id"] != "bob" {
		t.Fatalf("alice optional row=%#v", rows.Results[0].Metadata)
	}
	for _, row := range rows.Results[1:] {
		if row.Metadata["target_id"] != nil {
			t.Fatalf("unmatched optional row has target: %#v", row.Metadata)
		}
	}
}

func TestGraphitiMergeAtomicNodesEdgesAndLabels(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:graphiti_merge"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	col, err := db.CreateCollection(ctx, "people", WithMetadataOnly(), WithGraph(g), WithMetadataSchema(MetadataSchema{
		"uuid": StringField,
		"name": StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	if !graph.RegisterEdgeKind("GRAPHITI_MERGE_KNOWS", 222) {
		t.Fatal("edge kind registration failed")
	}
	merge := `MERGE (a:Person {uuid: $a})-[r:GRAPHITI_MERGE_KNOWS {weight: $w}]->(b:Person {uuid: $b})
ON CREATE SET a.name = $an, b.name = $bn
ON MATCH SET a.name = $an
`
	if _, err := db.QueryWithParams(ctx, merge, QueryParams{
		"a": "alice", "b": "bob", "w": 0.75, "an": "Alice", "bn": "Bob",
	}); err != nil {
		t.Fatalf("MERGE create: %v", err)
	}
	if count, err := col.Count(ctx); err != nil || count != 2 {
		t.Fatalf("MERGE created count=%d err=%v, want 2", count, err)
	}
	rows, err := db.Query(ctx, "SELECT source_id, target_id, edge_type, edge_weight FROM people p JOIN MATCH (p)-[r:GRAPHITI_MERGE_KNOWS]->(target:Person {uuid: 'bob'})")
	if err != nil {
		t.Fatalf("MERGE graph read: %v", err)
	}
	if rows.Total != 1 {
		t.Fatalf("MERGE graph rows=%d: %#v", rows.Total, rows.Results)
	}
	if rows.Results[0].Metadata["source_id"] != "alice" || rows.Results[0].Metadata["target_id"] != "bob" {
		t.Fatalf("MERGE endpoints=%#v", rows.Results[0].Metadata)
	}
	if rows.Results[0].Metadata["edge_type"] != "GRAPHITI_MERGE_KNOWS" || rows.Results[0].Metadata["edge_weight"] != float32(0.75) {
		t.Fatalf("MERGE edge projection=%#v", rows.Results[0].Metadata)
	}
	if _, err := db.QueryWithParams(ctx, merge, QueryParams{
		"a": "alice", "b": "bob", "w": 0.75, "an": "Alice 2", "bn": "Bob 2",
	}); err != nil {
		t.Fatalf("MERGE match: %v", err)
	}
	if count, err := col.Count(ctx); err != nil || count != 2 {
		t.Fatalf("MERGE match count=%d err=%v, want 2", count, err)
	}
	rows, err = db.Query(ctx, "SELECT source_id, target_id FROM people p JOIN MATCH (p)-[:GRAPHITI_MERGE_KNOWS]->(target:Person {uuid: 'bob'})")
	if err != nil || rows.Total != 1 {
		t.Fatalf("MERGE match edge rows=%d err=%v, want one relationship", rows.Total, err)
	}
	alice, err := col.Get(ctx, "alice")
	if err != nil || alice.Metadata["name"] != "Alice 2" {
		t.Fatalf("MERGE ON MATCH metadata=%#v err=%v", alice.Metadata, err)
	}
	if _, err := db.QueryWithParams(ctx, `MERGE (c:Person {uuid: $c}) ON CREATE SET c.name = $cn`, QueryParams{
		"c": "carol", "cn": "Carol",
	}); err != nil {
		t.Fatalf("node-only MERGE: %v", err)
	}
	if count, err := col.Count(ctx); err != nil || count != 3 {
		t.Fatalf("node-only MERGE count=%d err=%v, want 3", count, err)
	}
}

func TestGraphitiPathVariableProjection(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:graphiti_path"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	col, err := db.CreateCollection(ctx, "people", WithMetadataOnly(), WithGraph(g), WithMetadataSchema(MetadataSchema{
		"uuid": StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for _, person := range []string{"alice", "bob", "carol"} {
		if err := col.Insert(ctx, person, nil, map[string]interface{}{"uuid": person}); err != nil {
			t.Fatal(err)
		}
	}
	if !graph.RegisterEdgeKind("GRAPHITI_PATH", 223) {
		t.Fatal("edge kind registration failed")
	}
	alice, err := db.GetNodeID(ctx, "people", "alice")
	if err != nil {
		t.Fatal(err)
	}
	bob, err := db.GetNodeID(ctx, "people", "bob")
	if err != nil {
		t.Fatal(err)
	}
	carol, err := db.GetNodeID(ctx, "people", "carol")
	if err != nil {
		t.Fatal(err)
	}
	txn := g.BeginTxn()
	if err := g.AddEdge(txn, alice, bob, 0.5, 223); err != nil {
		t.Fatal(err)
	}
	if err := g.AddEdge(txn, bob, carol, 0.75, 223); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	rows, err := db.Query(ctx, `
		SELECT p
		FROM people AS src
		JOIN MATCH p = (src)-[:GRAPHITI_PATH]->(mid)-[:GRAPHITI_PATH]->(target)
		WHERE src.id = 'alice'`)
	if err != nil {
		t.Fatalf("path projection: %v", err)
	}
	if rows.Total != 1 || len(rows.Results) != 1 {
		t.Fatalf("path rows=%d results=%d: %#v", rows.Total, len(rows.Results), rows.Results)
	}
	path, ok := rows.Results[0].Metadata["p"].(GraphPath)
	if !ok {
		t.Fatalf("path type=%T value=%#v", rows.Results[0].Metadata["p"], rows.Results[0].Metadata["p"])
	}
	if got, want := path.Nodes, []string{"alice", "bob", "carol"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("path nodes=%v, want %v", got, want)
	}
	if got, want := path.EdgeTypes, []string{"GRAPHITI_PATH", "GRAPHITI_PATH"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("path edge types=%v, want %v", got, want)
	}
	if got, want := path.EdgeWeights, []float32{0.5, 0.75}; !reflect.DeepEqual(got, want) {
		t.Fatalf("path edge weights=%v, want %v", got, want)
	}
}

func TestGraphitiNativeCypherShortestLabelsAndAnonymousProperties(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:graphiti_native_cypher"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	col, err := db.CreateCollection(ctx, "people", WithMetadataOnly(), WithGraph(g), WithMetadataSchema(MetadataSchema{
		"uuid": StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for _, person := range []string{"alice", "bob", "carol"} {
		if err := col.Insert(ctx, person, nil, map[string]interface{}{"uuid": person}); err != nil {
			t.Fatal(err)
		}
	}
	for _, item := range []struct {
		id    string
		label string
	}{
		{"alice", "Person"}, {"alice", "Active"},
		{"bob", "Person"}, {"carol", "Person"},
	} {
		node, err := db.GetNodeID(ctx, "people", item.id)
		if err != nil {
			t.Fatal(err)
		}
		g.RegisterVertexLabel(node, item.label)
	}
	alice, _ := db.GetNodeID(ctx, "people", "alice")
	bob, _ := db.GetNodeID(ctx, "people", "bob")
	carol, _ := db.GetNodeID(ctx, "people", "carol")
	txn := g.BeginTxn()
	if err := g.AddEdge(txn, alice, bob, 0.75, 0); err != nil {
		t.Fatal(err)
	}
	if err := g.AddEdge(txn, alice, carol, 0.25, 0); err != nil {
		t.Fatal(err)
	}
	if err := g.AddEdge(txn, bob, carol, 0.75, 0); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	cypherSQL := `MATCH (a:Person:Active)-{weight > 0.5}->(b:Person)
RETURN a.id AS source_id, b.id AS target_id`
	rows, err := db.Query(ctx, cypherSQL)
	if err != nil {
		t.Fatalf("native MATCH RETURN: %v", err)
	}
	if rows.Total != 1 || rows.Results[0].Metadata["source_id"] != "alice" || rows.Results[0].Metadata["target_id"] != "bob" {
		t.Fatalf("native MATCH rows=%d, want alice -> bob", rows.Total)
	}

	rows, err = db.Query(ctx, `MATCH shortestPath((a)-[*1..3]->(b))
RETURN a.id AS source_id, b.id AS target_id`)
	if err != nil {
		t.Fatalf("shortestPath: %v", err)
	}
	if rows.Total != 3 {
		t.Fatalf("shortestPath rows=%d, want 3 unique source/terminal pairs: %#v", rows.Total, rows.Results)
	}
	rows, err = db.Query(ctx, `MATCH p = shortestPath((a)-[*1..3]->(b))
RETURN p`)
	if err != nil {
		t.Fatalf("shortestPath path alias: %v", err)
	}
	if rows.Total != 3 {
		t.Fatalf("shortestPath path alias rows=%d, want 3: %#v", rows.Total, rows.Results)
	}
	if _, ok := rows.Results[0].Metadata["p"].(GraphPath); !ok {
		t.Fatalf("shortestPath path alias value=%#v", rows.Results[0].Metadata["p"])
	}
	rows, err = db.Query(ctx, `MATCH (a)
RETURN shortestPath((a)-[*1..3]->(b)) AS path`)
	if err != nil {
		t.Fatalf("shortestPath expression: %v", err)
	}
	if rows.Total != 3 {
		t.Fatalf("shortestPath expression rows=%d, want 3: %#v", rows.Total, rows.Results)
	}
	if paths, ok := rows.Results[0].Metadata["path"].([]interface{}); !ok || len(paths) == 0 {
		t.Fatalf("shortestPath expression value=%#v", rows.Results[0].Metadata["path"])
	}

	if !graph.RegisterEdgeKind("GRAPHITI_COMP", 224) {
		t.Fatal("pattern-comprehension edge kind registration failed")
	}
	txn = g.BeginTxn()
	if err := g.AddEdge(txn, alice, bob, 1, 224); err != nil {
		t.Fatal(err)
	}
	if err := g.AddEdge(txn, bob, carol, 1, 224); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	rows, err = db.Query(ctx, `MATCH (a) RETURN [(a)-[:GRAPHITI_COMP]->(b) | b.id] AS friends`)
	if err != nil {
		t.Fatalf("pattern comprehension: %v", err)
	}
	if rows.Total != 3 {
		t.Fatalf("pattern comprehension rows=%d, want 3", rows.Total)
	}
	if friends, ok := rows.Results[0].Metadata["friends"].([]interface{}); !ok || len(friends) != 1 || friends[0] != "bob" {
		t.Fatalf("alice friends=%#v", rows.Results[0].Metadata["friends"])
	}
}
