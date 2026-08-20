package libravdb

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// TestSQL_JoinMatchGraphJoin verifies the M3a graph join end-to-end through
// db.Query: FROM services s JOIN MATCH (s)-[:DEPENDS_ON*1..3]->(api:Endpoint)
// -[:DOCUMENTED_BY]->(doc:Manual).
//
// The left (services) table is scanned relationally; every row's key resolves
// to a graph node which seeds a BFS over the match-path edges. Each reached
// vertex emits a joined row (leftKey|vertexRecID).
func TestSQL_JoinMatchGraphJoin(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:m3a_join_match"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	// Create a graph and attach it to the services collection.
	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()

	col, err := db.CreateCollection(ctx, "services", WithMetadataOnly(), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Insert records -> these become graph nodes.
	recs := map[string][]float32{
		"svc1": nil,
		"svc2": nil,
		"api1": nil,
		"api2": nil,
		"doc1": nil,
		"doc2": nil,
	}
	for id, vec := range recs {
		if err := col.Insert(ctx, id, vec, nil); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}

	// Resolve node IDs for each record.
	nid := func(id string) uint64 {
		n, err := db.GetNodeID(ctx, "services", id)
		if err != nil {
			t.Fatalf("GetNodeID(%s): %v", id, err)
		}
		return n
	}
	svc1, svc2, api1, api2, doc1, doc2 := nid("svc1"), nid("svc2"), nid("api1"), nid("api2"), nid("doc1"), nid("doc2")

	// Register edge kinds. Use high kind numbers to avoid collisions
	// with the global per-process registry shared across test suites.
	if !graph.RegisterEdgeKind("DEPENDS_ON", 71) {
		t.Fatalf("RegisterEdgeKind DEPENDS_ON=71 failed: kind already claimed")
	}
	if !graph.RegisterEdgeKind("DOCUMENTED_BY", 72) {
		t.Fatalf("RegisterEdgeKind DOCUMENTED_BY=72 failed: kind already claimed")
	}

	g := col.GetGraph()
	if g == nil {
		t.Fatalf("no graph on collection")
	}
	// The MATCH pattern declares endpoint labels. Register them on the test
	// vertices so this fixture exercises label filtering explicitly.
	g.RegisterVertexLabel(api1, "Endpoint")
	g.RegisterVertexLabel(api2, "Endpoint")
	g.RegisterVertexLabel(doc1, "Manual")
	g.RegisterVertexLabel(doc2, "Manual")
	txn := g.BeginTxn()
	// svc1 -DEPENDS_ON-> api1 -DOCUMENTED_BY-> doc1
	if err := g.AddEdge(txn, svc1, api1, 1.0, 71); err != nil {
		t.Fatalf("edge svc1->api1: %v", err)
	}
	if err := g.AddEdge(txn, api1, doc1, 1.0, 72); err != nil {
		t.Fatalf("edge api1->doc1: %v", err)
	}
	// svc2 -DEPENDS_ON-> api2 (separate chain) -DOCUMENTED_BY-> doc2
	if err := g.AddEdge(txn, svc2, api2, 1.0, 71); err != nil {
		t.Fatalf("edge svc2->api2: %v", err)
	}
	if err := g.AddEdge(txn, api2, doc2, 1.0, 72); err != nil {
		t.Fatalf("edge api2->doc2: %v", err)
	}

	// Query: full two-edge graph join. Both services should reach their docs
	// through their respective api nodes (separate chains prevent cross-contamination).
	res, err := db.Query(ctx, "SELECT id FROM services s JOIN MATCH (s)-[:DEPENDS_ON*1..3]->(api:Endpoint)-[:DOCUMENTED_BY]->(doc:Manual)")
	if err != nil {
		t.Fatalf("JOIN MATCH query failed: %v", err)
	}
	ids := map[string]bool{}
	for _, r := range res.Results {
		ids[r.ID] = true
	}
	t.Logf("JOIN MATCH rows: %v", ids)

	// svc1 -> api1 -> doc1
	if !ids["svc1|doc1"] {
		t.Errorf("expected svc1|doc1 in results, got %v", ids)
	}
	// svc2 -> api1 -> doc2
	if !ids["svc2|doc2"] {
		t.Errorf("expected svc2|doc2 in results, got %v", ids)
	}
	// The intermediate vertex api1 must also appear (JOIN MATCH emits every
	// reached vertex, including the endpoint between the two edges).
	if !ids["svc1|api1"] {
		t.Errorf("expected svc1|api1 (intermediate vertex) in results, got %v", ids)
	}
	if !ids["svc2|api2"] {
		t.Errorf("expected svc2|api2 (intermediate vertex) in results, got %v", ids)
	}
	// Cross-doc leakage must NOT happen: svc1 cannot reach doc2.
	if ids["svc1|doc2"] {
		t.Errorf("svc1|doc2 should NOT be reachable (no path svc1->doc2), got %v", ids)
	}
	if ids["svc2|doc1"] {
		t.Errorf("svc2|doc1 should NOT be reachable (no path svc2->doc1), got %v", ids)
	}

	// Query 2: LEFT JOIN MATCH — a service with no edges must still appear.
	if err := col.Insert(ctx, "svc3", nil, nil); err != nil {
		t.Fatalf("insert svc3: %v", err)
	}
	res2, err := db.Query(ctx, "SELECT id FROM services s LEFT JOIN MATCH (s)-[:DEPENDS_ON*1..3]->(api:Endpoint)-[:DOCUMENTED_BY]->(doc:Manual)")
	if err != nil {
		t.Fatalf("LEFT JOIN MATCH query failed: %v", err)
	}
	ids2 := map[string]bool{}
	for _, r := range res2.Results {
		ids2[r.ID] = true
	}
	t.Logf("LEFT JOIN MATCH rows: %v", ids2)
	if !ids2["svc3|"] {
		t.Errorf("LEFT JOIN MATCH: expected svc3| (unmatched left row), got %v", ids2)
	}
	// LEFT JOIN also produces empty-right rows for non-service nodes
	// (api1, api2, doc1, doc2 have no graph edges). This is expected
	// because the relational scan processes every collection record.
	if !ids2["api1|"] {
		t.Errorf("LEFT JOIN MATCH: expected api1| (no graph matches), got %v", ids2)
	}
}
