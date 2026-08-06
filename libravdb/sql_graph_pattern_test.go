package libravdb

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// TestSQL_GraphLabelSeedAndKindFilter verifies the M1 graph pattern
// completeness end-to-end through db.Query:
//  1. Label-scan seeding — a labeled start vertex seeds traversal without
//     a WHERE id predicate or vector anchor.
//  2. Edge-kind filtering — typed edges ([e:TYPE]) constrain traversal;
//     a kind-1 edge does not traverse kind-2 edges.
func TestSQL_GraphLabelSeedAndKindFilter(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:m1_smoke"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	// Create a graph and attach it to the collection.
	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()

	col, err := db.CreateCollection(ctx, "g", WithDimension(4), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Insert records -> these become graph nodes.
	recs := map[string][]float32{
		"svc1": {0.1, 0.2, 0.3, 0.4},
		"api1": {0.5, 0.6, 0.7, 0.8},
		"doc1": {0.9, 0.8, 0.7, 0.6},
	}
	for id, vec := range recs {
		if err := col.Insert(ctx, id, vec, nil); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}

	// Resolve node IDs for each record.
	nid := func(id string) uint64 {
		n, err := db.GetNodeID(ctx, "g", id)
		if err != nil {
			t.Fatalf("GetNodeID(%s): %v", id, err)
		}
		return n
	}
	svc, api, doc := nid("svc1"), nid("api1"), nid("doc1")
	t.Logf("nodeIDs: svc1=%d api1=%d doc1=%d", svc, api, doc)

	// Register edge kinds. Use high kind numbers to avoid collisions
	// with the global per-process registry shared across test suites.
	// Use unique kind numbers to avoid global registry collisions with
	// other test suites that also register edge kinds.
	if !graph.RegisterEdgeKind("DEPENDS_ON", 71) {
		t.Fatalf("RegisterEdgeKind DEPENDS_ON=71 failed: kind already claimed")
	}
	if !graph.RegisterEdgeKind("DOCUMENTED_BY", 72) {
		t.Fatalf("RegisterEdgeKind DOCUMENTED_BY=72 failed: kind already claimed")
	}

	// Add typed edges: svc1 -DEPENDS_ON-> api1, api1 -DOCUMENTED_BY-> doc1.
	g := col.GetGraph()
	if g == nil {
		t.Fatalf("no graph on collection")
	}
	txn := g.BeginTxn()
	if err := g.AddEdge(txn, svc, api, 1.0, 71); err != nil {
		t.Fatalf("edge svc->api: %v", err)
	}
	if err := g.AddEdge(txn, api, doc, 1.0, 72); err != nil {
		t.Fatalf("edge api->doc: %v", err)
	}

	// Label the start node.
	g.RegisterVertexLabel(svc, "Service")
	t.Logf("label nodes for Service: %v", g.GetLabelNodes("Service"))

	// Query 1: label-scan seeding, no WHERE, no vector anchor.
	res, err := db.Query(ctx, "SELECT id FROM GRAPH_TABLE(g MATCH (a:Service)-[e:DEPENDS_ON]->(b))")
	if err != nil {
		t.Fatalf("label-seed query failed: %v", err)
	}
	t.Logf("label-seed query rows: %d", len(res.Results))
	for _, r := range res.Results {
		t.Logf("  id=%s", r.ID)
	}
	if len(res.Results) == 0 {
		t.Fatalf("label-seed query returned 0 rows (expected api1 via svc1 seed)")
	}

	// Query 2: kind-filtered traversal — DEPENDS_ON*1..3 must reach api1
	// but NOT doc1 (DOCUMENTED_BY edge excluded by kind filter).
	res2, err := db.Query(ctx, "SELECT id FROM GRAPH_TABLE(g MATCH (a:Service)-[e:DEPENDS_ON*1..3]->(x))")
	if err != nil {
		t.Fatalf("kind-filter query failed: %v", err)
	}
	ids := map[string]bool{}
	for _, r := range res2.Results {
		ids[r.ID] = true
	}
	t.Logf("kind-filter rows: %v", ids)
	if !ids["api1"] {
		t.Errorf("kind-filter: expected api1 in results, got %v", ids)
	}
	if ids["doc1"] {
		t.Errorf("kind-filter: doc1 should NOT be reachable via DEPENDS_ON only, got %v", ids)
	}
}

// A LIMIT on a graph query is a result-projection limit, never a traversal
// limit. Otherwise a later graph match could not safely be vector-ranked.
func TestGraphLimitCompletesBoundedTraversalBeforeProjection(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:graph_complete_limit"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer gr.Close()
	col, err := db.CreateCollection(ctx, "graph_limit", WithDimension(2), WithGraph(gr))
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"seed", "first", "late"} {
		if err := col.Insert(ctx, id, []float32{0, 0}, nil); err != nil {
			t.Fatal(err)
		}
	}
	seed, err := db.GetNodeID(ctx, "graph_limit", "seed")
	if err != nil {
		t.Fatal(err)
	}
	first, _ := db.GetNodeID(ctx, "graph_limit", "first")
	late, _ := db.GetNodeID(ctx, "graph_limit", "late")
	txn := gr.BeginTxn()
	if err := gr.AddEdge(txn, seed, first, 1, 0); err != nil {
		t.Fatal(err)
	}
	if err := gr.AddEdge(txn, seed, late, 1, 0); err != nil {
		t.Fatal(err)
	}
	before := gr.Stats().BFSNodesVisited
	plan := &optimizer.PhysicalPlan{
		CollectionName:    "graph_limit",
		Kind:              optimizer.QueryKindGraph,
		HasGraphTraversal: true,
		HasExplicitSeed:   true,
		ExplicitSeedID:    seed,
		GraphEdges:        []optimizer.GraphEdgePlan{{Direction: 1, QuantMin: 1, QuantMax: 1}},
		MaxHops:           1,
		Limit:             1,
	}
	results, err := newExecutor(db).Execute(ctx, plan)
	if err != nil {
		t.Fatal(err)
	}
	if len(results.Results) != 1 {
		t.Fatalf("projected rows = %d, want 1", len(results.Results))
	}
	if visited := gr.Stats().BFSNodesVisited - before; visited < 3 {
		t.Fatalf("BFS visited %d nodes with LIMIT 1, want seed plus both bounded matches", visited)
	}
}
