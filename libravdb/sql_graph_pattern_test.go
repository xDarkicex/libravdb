package libravdb

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
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

	// Register edge kinds (idempotent).
	graph.RegisterEdgeKind("DEPENDS_ON", 1)
	graph.RegisterEdgeKind("DOCUMENTED_BY", 2)

	// Add typed edges: svc1 -DEPENDS_ON-> api1, api1 -DOCUMENTED_BY-> doc1.
	g := col.GetGraph()
	if g == nil {
		t.Fatalf("no graph on collection")
	}
	txn := g.BeginTxn()
	if err := g.AddEdge(txn, svc, api, 1.0, 1); err != nil {
		t.Fatalf("edge svc->api: %v", err)
	}
	if err := g.AddEdge(txn, api, doc, 1.0, 2); err != nil {
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
