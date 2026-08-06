package libravdb

import (
	"context"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/internal/util"
)

// TestGraphScored_FullAcceptanceQuery verifies the full graph-algorithmic
// vector search pipeline using the internal executor API:
//
//	(1.0 - VECTOR_DISTANCE(doc.embedding, $v)) * GRAPH_CENTRALITY(doc)
//
// Inbound MATCH: (doc)<-[:CITES]-(ref:Document)
func TestGraphScored_FullAcceptanceQuery(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/scored_acceptance.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "documents",
		WithDimension(3), WithGraph(gr),
		WithMetadataSchema(MetadataSchema{"content": StringField}),
	)

	// Insert documents with vectors.
	col.Insert(context.Background(), "D1", []float32{1, 0, 0},
		map[string]interface{}{"content": "auth module docs"})
	col.Insert(context.Background(), "D2", []float32{0, 1, 0},
		map[string]interface{}{"content": "network config"})
	col.Insert(context.Background(), "D3", []float32{0, 0, 1},
		map[string]interface{}{"content": "backup guide"})

	d1, _ := db.GetNodeID(context.Background(), "documents", "D1")
	d2, _ := db.GetNodeID(context.Background(), "documents", "D2")
	d3, _ := db.GetNodeID(context.Background(), "documents", "D3")

	// D2 and D3 cite D1 (inbound edges to D1). Kind 1 = CITES.
	txn := gr.BeginTxn()
	txn.AddEdge(d2, d1, 1.0, 1) // D2 -[:CITES]-> D1
	txn.AddEdge(d3, d1, 1.0, 1) // D3 -[:CITES]-> D1
	txn.Commit(context.Background())
	// Wait for graph edge batch flush and verify edges are live.
	time.Sleep(100 * time.Millisecond)
	edges, _ := gr.Neighbors(d2)
	if len(edges) == 0 {
		t.Fatal("edges not committed after 100ms — batch flush may have failed")
	}

	snap, err := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	if err != nil {
		t.Fatalf("SnapshotAt: %v", err)
	}
	defer snap.Close()

	// Build the scored multimodal plan.
	queryVec := []float32{1, 0, 0}
	plan := &optimizer.PhysicalPlan{
		Kind:               optimizer.QueryKindMultiModal,
		CollectionName:     "documents",
		QueryVector:        queryVec,
		HasVectorSearch:    true,
		HasGraphTraversal:  true,
		HasScoreExpr:       true,
		ScoreArithOp:       11, // multiply
		HasGraphCentrality: true,
		ScoreLiteralValue:  1.0,
		GraphEdges: []optimizer.GraphEdgePlan{
			{Direction: -1, QuantMin: 0, QuantMax: 1, EdgeKind: 1}, // inbound CITES
		},
		GraphJoins: []optimizer.GraphJoinPlan{{
			LeftCollection: "documents",
			GraphEdges: []optimizer.GraphEdgePlan{
				{Direction: -1, QuantMin: 0, QuantMax: 1, EdgeKind: 1},
			},
			MaxHops: 1,
		}},
		MaxHops:     1,
		Limit:       5,
		Projections: []string{"id", "content"},
		SnapshotLSN: snap.LSN,
	}

	exec := newExecutor(db)

	// Step 1: Generate graph candidates (inbound MATCH) using the plan.
	col2, _ := db.GetCollection("documents")
	candidates, err := exec.multiModalGraphCandidatesAtLSN(context.Background(), col2,
		plan, []string{"D1", "D2", "D3"}, snap.LSN)
	if err != nil {
		t.Fatalf("graph candidates: %v", err)
	}
	t.Logf("candidates: %v", candidates)

	// Verify centrality via direct API.
	t.Logf("D1 centrality=%.4f", gr.GraphCentrality(d1))
	t.Logf("D2 centrality=%.4f", gr.GraphCentrality(d2))

	// Step 2: Build scoring expression.
	distFn, _ := util.GetDistanceFunc(util.DistanceMetric(col.Config().Metric))
	expr := buildScoreExpr(plan, distFn, queryVec)

	// Pre-compute centrality for each candidate.
	centrality := map[string]float64{
		"D1": gr.GraphCentrality(d1),
		"D2": gr.GraphCentrality(d2),
		"D3": gr.GraphCentrality(d3),
	}
	t.Logf("centrality: D1=%.4f D2=%.4f D3=%.4f", centrality["D1"], centrality["D2"], centrality["D3"])

	// Step 3: Score candidates with pre-computed centrality.
	results, err := exec.executeScoredMultiModalWithCentrality(context.Background(), col, plan, candidates, expr, centrality, 0, true)
	if err != nil {
		t.Fatalf("executeScoredMultiModal: %v", err)
	}

	t.Logf("results: %d", results.Total)
	for _, r := range results.Results {
		t.Logf("  %s: score=%.4f", r.ID, r.Score)
	}

	if results.Total == 0 {
		t.Error("expected at least 1 result")
	}
	// D1 should have the highest centrality (2 inbound CITES edges).
	if results.Total > 0 && results.Results[0].ID != "D1" {
		t.Errorf("expected D1 as top result (highest centrality), got %s", results.Results[0].ID)
	}
}

// TestGraphScored_VectorDistanceOnly verifies vector-only scoring.
func TestGraphScored_VectorDistanceOnly(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/scored_vec.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3))
	col.Insert(context.Background(), "A", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "B", []float32{0, 0, 1}, nil)
	time.Sleep(20 * time.Millisecond)

	snap, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	defer snap.Close()

	queryVec := []float32{1, 0, 0}
	distFn, _ := util.GetDistanceFunc(util.DistanceMetric(col.Config().Metric))
	expr := buildScoreExpr(&optimizer.PhysicalPlan{
		QueryVector: queryVec,
	}, distFn, queryVec)

	// A is closer to queryVec than B.
	candidates := map[string]struct{}{"A": {}, "B": {}}
	results, _ := newExecutor(db).executeScoredMultiModal(context.Background(), col,
		&optimizer.PhysicalPlan{Limit: 2, QueryVector: queryVec},
		candidates, expr, snap.LSN, false) // asc: lower distance = better

	if results.Results[0].ID != "A" {
		t.Errorf("A should be top (closer to query vector), got %s", results.Results[0].ID)
	}
}

// TestGraphScored_TopK verifies the bounded heap correctly returns top-k.
func TestGraphScored_TopK(t *testing.T) {
	heap := newTopKHeap(3, false) // ascending, keep 3 smallest
	heap.push(&Record{ID: "a"}, 5.0)
	heap.push(&Record{ID: "b"}, 3.0)
	heap.push(&Record{ID: "c"}, 8.0)
	heap.push(&Record{ID: "d"}, 1.0)
	heap.push(&Record{ID: "e"}, 4.0)

	sorted := heap.sorted()
	if len(sorted) != 3 {
		t.Fatalf("got %d, want 3", len(sorted))
	}
	if sorted[0].record.ID != "d" || sorted[0].score != 1.0 {
		t.Errorf("position 0: %s %.1f, want d 1.0", sorted[0].record.ID, sorted[0].score)
	}
	if sorted[1].record.ID != "b" || sorted[1].score != 3.0 {
		t.Errorf("position 1: %s %.1f, want b 3.0", sorted[1].record.ID, sorted[1].score)
	}
	if sorted[2].record.ID != "e" || sorted[2].score != 4.0 {
		t.Errorf("position 2: %s %.1f, want e 4.0", sorted[2].record.ID, sorted[2].score)
	}
}

// TestGraphScored_InboundMatchDirection verifies inbound edge direction works.
func TestGraphScored_InboundMatchDirection(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/scored_inbound.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "A", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "B", []float32{0, 0, 1}, nil)
	a, _ := db.GetNodeID(context.Background(), "docs", "A")
	b, _ := db.GetNodeID(context.Background(), "docs", "B")

	// A -[:CITES]-> B  (A cites B, so B's inbound neighbor is A).
	txn := gr.BeginTxn()
	txn.AddEdge(a, b, 1.0, 1) // outbound from A to B, inbound to B
	txn.Commit(context.Background())
	time.Sleep(20 * time.Millisecond)

	// Inbound from B's perspective: who cites B?
	inbound, _ := gr.InboundNeighbors(b)
	if len(inbound) != 1 || inbound[0].Target != a {
		t.Errorf("inbound to B: %v, want [A]", inbound)
	}
}
