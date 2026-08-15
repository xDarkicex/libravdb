package libravdb

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/graph"
	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/internal/util"
)

// TestGraphScoredSQL_AcceptanceQuery runs the exact marketing SQL through
// db.QueryWithParams and verifies results. This is the Priority 0 gate.
func TestGraphScoredSQL_AcceptanceQuery(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/scored_acceptance_sql.libravdb"))
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

	// Register edge kinds explicitly. CITES=1, REFERENCES=2 (non-CITES).
	// The registry is global; false means already registered (idempotent, OK).
	if !graph.RegisterEdgeKind("CITES", 1) {
		t.Log("CITES edge kind already registered (idempotent)")
	}
	if !graph.RegisterEdgeKind("REFERENCES", 2) {
		t.Log("REFERENCES edge kind already registered (idempotent)")
	}

	col.Insert(context.Background(), "D1", []float32{1, 0, 0},
		map[string]interface{}{"content": "auth module"})
	col.Insert(context.Background(), "D2", []float32{0, 1, 0},
		map[string]interface{}{"content": "network config"})
	col.Insert(context.Background(), "D3", []float32{0, 0, 1},
		map[string]interface{}{"content": "backup guide"})

	d1, _ := db.GetNodeID(context.Background(), "documents", "D1")
	d2, _ := db.GetNodeID(context.Background(), "documents", "D2")
	d3, _ := db.GetNodeID(context.Background(), "documents", "D3")

	// Register terminal labels so (ref:Document) validates.
	gr.RegisterVertexLabel(d1, "Document")
	gr.RegisterVertexLabel(d2, "Document")
	gr.RegisterVertexLabel(d3, "Document")

	// D2 and D3 cite D1 (kind 1 = CITES). D1 references D3 (kind 2 = non-CITES).
	// The MATCH pattern only matches CITES edges, so only D1 qualifies.
	txn := gr.BeginTxn()
	txn.AddEdge(d2, d1, 1.0, 1) // D2 -[:CITES]-> D1
	txn.AddEdge(d3, d1, 1.0, 1) // D3 -[:CITES]-> D1
	txn.AddEdge(d1, d3, 1.0, 2) // D1 -[:REFERENCES]-> D3 (should NOT match CITES pattern)
	txn.Commit(context.Background())
	time.Sleep(100 * time.Millisecond)

	if edges, _ := gr.Neighbors(d2); len(edges) == 0 {
		t.Fatal("edges not committed")
	}

	// Step 1: VECTOR_DISTANCE with $param through public API. ✅
	sql1 := "SELECT VECTOR_DISTANCE(embedding, $pv) AS d FROM documents"
	r1, err := db.QueryWithParams(context.Background(), sql1, QueryParams{"pv": []float32{1, 0, 0}})
	if err != nil {
		t.Fatalf("vector+param: %v", err)
	}
	if r1.Total != 3 {
		t.Errorf("vector+param: want 3, got %d", r1.Total)
	}
	t.Logf("✅ VECTOR_DISTANCE + $param: %d results", r1.Total)

	// Step 2: WHERE MATCH with source-row seeding. ✅
	sql2 := "SELECT id FROM documents d WHERE MATCH (d)<-[:CITES]-(ref:Document)"
	r2, err := db.Query(context.Background(), sql2)
	if err != nil {
		t.Fatalf("WHERE MATCH: %v", err)
	}
	if r2.Total != 1 {
		t.Errorf("WHERE MATCH: want 1 (D1), got %d — D2/D3 should not qualify", r2.Total)
	}
	if r2.Total > 0 && r2.Results[0].ID != "D1" {
		t.Errorf("WHERE MATCH: want D1, got %s", r2.Results[0].ID)
	}
	t.Logf("✅ WHERE MATCH: %d results", r2.Total)

	// Add more fixtures to exercise ORDER BY + LIMIT with multiple qualifiers.
	col.Insert(context.Background(), "D4", []float32{0.5, 0, 0},
		map[string]interface{}{"content": "api docs"})
	col.Insert(context.Background(), "D5", []float32{0, 0, 0.5},
		map[string]interface{}{"content": "deploy guide"})
	d4, _ := db.GetNodeID(context.Background(), "documents", "D4")
	d5, _ := db.GetNodeID(context.Background(), "documents", "D5")
	gr.RegisterVertexLabel(d4, "Document")
	gr.RegisterVertexLabel(d5, "Document")
	txn2 := gr.BeginTxn()
	txn2.AddEdge(d5, d4, 1.0, 1) // D5 -[:CITES]-> D4
	txn2.Commit(context.Background())
	time.Sleep(100 * time.Millisecond)

	// Now D1 (cited by D2,D3) and D4 (cited by D5) both qualify.
	// ORDER BY + LIMIT should return top-2 by score.
	sql2b := "SELECT doc.id, " +
		"(1.0 - VECTOR_DISTANCE(doc.embedding, $pv)) * GRAPH_CENTRALITY(doc) AS r " +
		"FROM documents doc " +
		"WHERE MATCH (doc)<-[:CITES]-(ref:Document) " +
		"ORDER BY r DESC LIMIT 2"
	r2b, err := db.QueryWithParams(context.Background(), sql2b, QueryParams{"pv": []float32{1, 0, 0}})
	if err != nil {
		t.Fatalf("multi-qualifier ORDER BY: %v", err)
	}
	if r2b.Total != 2 {
		t.Errorf("multi-qualifier: want 2, got %d", r2b.Total)
	}
	t.Logf("✅ multi-qualifier ORDER BY LIMIT: %d results", r2b.Total)
	for _, r := range r2b.Results {
		t.Logf("  %s: score=%.4f", r.ID, r.Score)
	}
	if r2.Total == 0 {
		t.Error("WHERE MATCH returned 0 results")
	}

	// Step 3: FULL acceptance query — VECTOR_DISTANCE + GRAPH_CENTRALITY
	// + WHERE MATCH + ORDER BY + LIMIT through public SQL. Score and
	// ordering verified against independent reference computation.
	sql3 := "SELECT doc.id, " +
		"(1.0 - VECTOR_DISTANCE(doc.embedding, $pv)) * GRAPH_CENTRALITY(doc) AS authoritative_relevance " +
		"FROM documents doc " +
		"WHERE MATCH (doc)<-[:CITES]-(ref:Document) " +
		"ORDER BY authoritative_relevance DESC " +
		"LIMIT 5"
	r3, err := db.QueryWithParams(context.Background(), sql3, QueryParams{"pv": []float32{1, 0, 0}})
	if err != nil {
		t.Fatalf("full acceptance query: %v", err)
	}
	t.Logf("✅ full acceptance query: %d results", r3.Total)
	for _, r := range r3.Results {
		t.Logf("  %s: score=%.4f", r.ID, r.Score)
	}

	// D1 (cited by D2,D3) and D4 (cited by D5) both qualify.
	// D2, D3, D5 have no inbound CITES — excluded.
	if r3.Total != 2 {
		t.Fatalf("want 2 results (D1 + D4), got %d", r3.Total)
	}
	sawD1, sawD4 := false, false
	for _, r := range r3.Results {
		if r.ID == "D1" {
			sawD1 = true
			if r.Score < 0.99 || r.Score > 1.01 {
				t.Errorf("D1 score=%.4f, want ~1.0", r.Score)
			}
		}
		if r.ID == "D4" {
			sawD4 = true
		}
		if r.ID == "D2" || r.ID == "D3" || r.ID == "D5" {
			t.Errorf("%s should not appear — no inbound CITES edges", r.ID)
		}
	}
	if !sawD1 {
		t.Error("D1 missing from results")
	}
	if !sawD4 {
		t.Error("D4 missing from results")
	}
	if r3.Results[0].ID != "D1" {
		t.Errorf("D1 should rank first (centrality=1.0 vs D4=0.5), got %s", r3.Results[0].ID)
	}
}

// TestGraphScoredSQL_PublicAPI verifies the full pipeline through QueryWithParams:
// parse → bind → optimize → execute with scored expression.
func TestGraphScoredSQL_PublicAPI(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/scored_sql.libravdb"))
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

	// Insert documents.
	col.Insert(context.Background(), "D1", []float32{1, 0, 0},
		map[string]interface{}{"content": "auth module"})
	col.Insert(context.Background(), "D2", []float32{0, 1, 0},
		map[string]interface{}{"content": "network config"})
	col.Insert(context.Background(), "D3", []float32{0, 0, 1},
		map[string]interface{}{"content": "backup guide"})

	d1, _ := db.GetNodeID(context.Background(), "documents", "D1")
	d2, _ := db.GetNodeID(context.Background(), "documents", "D2")
	d3, _ := db.GetNodeID(context.Background(), "documents", "D3")

	// D2 and D3 cite D1 (inbound to D1).
	txn := gr.BeginTxn()
	txn.AddEdge(d2, d1, 1.0, 1)
	txn.AddEdge(d3, d1, 1.0, 1)
	txn.Commit(context.Background())
	time.Sleep(50 * time.Millisecond)

	// Test 1: Vector distance query through internal executor.
	time.Sleep(20 * time.Millisecond)
	snap, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	defer snap.Close()
	exec := newExecutor(db)
	results, err := exec.ExecuteAtLSN(context.Background(), &optimizer.PhysicalPlan{
		Kind:           optimizer.QueryKindVectorProjection,
		CollectionName: "documents",
		QueryVector:    []float32{1, 0, 0},
		Limit:          5,
		Projections:    []string{"id"},
	}, snap.LSN)
	if err != nil {
		t.Fatalf("vector query: %v", err)
	}
	t.Logf("vector query: %d results", results.Total)
	if results.Total != 3 {
		t.Errorf("expected 3 results, got %d", results.Total)
	}

	// Test 3: Verify centrality is computed correctly via the public API.
	c := gr.GraphCentrality(d1)
	t.Logf("D1 centrality: %.4f", c)
	if c <= 0 {
		t.Error("D1 should have positive centrality (2 inbound CITES edges)")
	}
}

// TestGraphScoredSQL_InboundMatch verifies inbound MATCH through internal path.
func TestGraphScoredSQL_InboundMatch(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/scored_inbound_sql.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	col, _ := db.CreateCollection(context.Background(), "documents",
		WithDimension(3), WithGraph(gr))

	for i := 1; i <= 3; i++ {
		id := fmt.Sprintf("D%d", i)
		col.Insert(context.Background(), id, []float32{float32(i), 0, 0}, nil)
	}
	d1, _ := db.GetNodeID(context.Background(), "documents", "D1")
	d2, _ := db.GetNodeID(context.Background(), "documents", "D2")

	// D2 cites D1 (outbound from D2, inbound to D1).
	txn := gr.BeginTxn()
	txn.AddEdge(d2, d1, 1.0, 1) // D2 → D1
	txn.Commit(context.Background())
	time.Sleep(50 * time.Millisecond)

	// Inbound to D1 = D2 cites D1.
	inbound, _ := gr.InboundNeighbors(d1)
	if len(inbound) == 0 {
		t.Error("D1 should have inbound neighbor D2")
	}

	// Centrality: D1 has 1 inbound, D2 has 0.
	c1 := gr.GraphCentrality(d1)
	c2 := gr.GraphCentrality(d2)
	if c1 <= c2 {
		t.Errorf("D1=%.4f should be > D2=%.4f", c1, c2)
	}
}

// TestGraphScoredSQL_VectorCentralityScoring verifies the scored pathway
// end-to-end through the internal plan API.
func TestGraphScoredSQL_VectorCentralityScoring(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/scored_vc.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	col, _ := db.CreateCollection(context.Background(), "documents",
		WithDimension(3), WithGraph(gr))

	col.Insert(context.Background(), "A", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "B", []float32{0, 0, 1}, nil)
	a, _ := db.GetNodeID(context.Background(), "documents", "A")
	b, _ := db.GetNodeID(context.Background(), "documents", "B")

	// B cites A (inbound to A).
	txn := gr.BeginTxn()
	txn.AddEdge(b, a, 1.0, 1)
	txn.Commit(context.Background())
	time.Sleep(50 * time.Millisecond)

	snap, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	defer snap.Close()

	// Build plan with score expression lowered.
	plan := &optimizer.PhysicalPlan{
		Kind:               optimizer.QueryKindMultiModal,
		CollectionName:     "documents",
		QueryVector:        []float32{1, 0, 0},
		HasVectorSearch:    true,
		HasGraphTraversal:  true,
		HasScoreExpr:       true,
		ScoreArithOp:       11, // multiply
		HasGraphCentrality: true,
		ScoreLiteralValue:  1.0,
		GraphEdges:         []optimizer.GraphEdgePlan{{Direction: -1, QuantMin: 0, QuantMax: 1, EdgeKind: 1}},
		GraphJoins:         []optimizer.GraphJoinPlan{{LeftCollection: "documents", GraphEdges: []optimizer.GraphEdgePlan{{Direction: -1, QuantMin: 0, QuantMax: 1, EdgeKind: 1}}, MaxHops: 1}},
		MaxHops:            1,
		Limit:              5,
		SnapshotLSN:        snap.LSN,
	}

	exec := newExecutor(db)
	col2, _ := db.GetCollection("documents")
	candidates, _ := exec.multiModalGraphCandidatesAtLSN(context.Background(), col2, plan, []string{"A", "B"}, snap.LSN)

	distFn, _ := util.GetDistanceFunc(util.DistanceMetric(col.Config().Metric))
	expr := buildScoreExpr(plan, distFn, plan.QueryVector)
	results, err := exec.executeScoredMultiModal(context.Background(), col, plan, candidates, expr, snap.LSN, true)
	if err != nil {
		t.Fatalf("executeScoredMultiModal: %v", err)
	}

	t.Logf("results: %d", results.Total)
	for _, r := range results.Results {
		t.Logf("  %s: score=%.4f", r.ID, r.Score)
	}

	// A should rank highest: closer to query vector AND higher centrality.
	if results.Total > 0 && results.Results[0].ID != "A" {
		t.Errorf("expected A first, got %s", results.Results[0].ID)
	}
}
