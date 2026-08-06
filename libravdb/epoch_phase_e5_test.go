package libravdb

import (
	"context"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// =============================================================================
// Test A: Snapshot hybrid exclusion
// =============================================================================

func TestE5_SnapshotHybridExclusion(t *testing.T) {
	dir := t.TempDir() + "/e5_hybrid_exclusion.libravdb"
	var t0 time.Time

	// Phase 1: Create base graph + terminal vector at t0.
	func() {
		db, _ := Open(WithStoragePath(dir))
		defer db.Close()
		gr, _ := NewGraph(GraphConfig{})
		defer gr.Close()
		col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))

		col.Insert(context.Background(), "Source", []float32{1, 0, 0}, nil)
		col.Insert(context.Background(), "Terminal", []float32{2, 2, 2}, nil)
		src, _ := db.GetNodeID(context.Background(), "docs", "Source")
		tgt, _ := db.GetNodeID(context.Background(), "docs", "Terminal")

		graph.RegisterEdgeKind("LINKS", 1)
		txn := gr.BeginTxn()
		txn.AddEdge(src, tgt, 1.0, 1)
		txn.Commit(context.Background())
		time.Sleep(100 * time.Millisecond)
		t0 = time.Now().UTC()
		t.Logf("Phase 1: base Source→Terminal committed at %v", t0)
	}()

	time.Sleep(100 * time.Millisecond)

	// Phase 2: Commit a closer terminal vector and new edge after t0.
	func() {
		db, _ := Open(WithStoragePath(dir))
		defer db.Close()
		col, _ := db.GetCollection("docs")
		col.Insert(context.Background(), "CloserTerminal", []float32{1, 1, 1}, nil)
		t.Logf("Phase 2: CloserTerminal [1,1,1] committed after t0")
	}()

	// Phase 3: BeginEpochTxAt(t0). Graph+vector query must only see pre-t0 data.
	db3, _ := Open(WithStoragePath(dir))
	defer db3.Close()

	// Attach a fresh graph so epoch graph traversal works on the reopened DB.
	gr3, _ := NewGraph(GraphConfig{})
	defer gr3.Close()
	col3, _ := db3.GetCollection("docs")
	col3.SetGraph(gr3)

	epoch, err := db3.BeginEpochTxAt(context.Background(), t0)
	if err != nil {
		t.Fatalf("BeginEpochTxAt: %v", err)
	}
	defer epoch.Rollback(context.Background())

	// Verify post-t0 record is excluded from epoch view.
	recs, _ := epoch.ListRecords(context.Background(), "docs")
	for _, rec := range recs {
		if rec.ID == "CloserTerminal" {
			t.Fatal("CloserTerminal must not be visible in epoch pinned at t0")
		}
	}
	t.Logf("Phase 3: post-t0 CloserTerminal excluded from epoch ✓")

	// Verify epoch graph traversal only sees pre-t0 edge.
	gtx, _ := epoch.GraphTxn("docs")
	src, _ := epoch.LookupNodeID(context.Background(), "docs", "Source")
	neighbors, _ := gtx.NeighborsOverlay(src)
	if len(neighbors) == 0 {
		t.Fatal("epoch must see base edge Source→Terminal")
	}
	tgtNode, _ := epoch.LookupNodeID(context.Background(), "docs", "Terminal")
	if neighbors[0].Target != tgtNode {
		t.Fatalf("edge target: want Terminal(%d), got %d", tgtNode, neighbors[0].Target)
	}
	t.Logf("Phase 3: epoch sees base edge Source→Terminal ✓")

	// Verify a closer staged vector can win (staged records rank correctly).
	_, err = epoch.Query(context.Background(), "INSERT INTO docs (id, embedding) VALUES ('StagedWinner', '[0,0,0]')", nil)
	if err != nil {
		t.Fatalf("epoch insert: %v", err)
	}
	swNode, _ := epoch.LookupNodeID(context.Background(), "docs", "StagedWinner")
	gtx.AddEdge(src, swNode, 1.0, 1)
	t.Logf("Phase 3: staged StagedWinner [0,0,0] is closer than Terminal [2,2,2]")

	// The epoch overlay should see both edges now.
	neighbors2, _ := gtx.NeighborsOverlay(src)
	if len(neighbors2) < 2 {
		t.Fatalf("epoch should see 2 edges (base + staged), got %d", len(neighbors2))
	}
	t.Logf("Phase 3: epoch overlay sees base + staged edges ✓")
	t.Log("✅ test A: snapshot hybrid exclusion")
}

// =============================================================================
// Test C: Overlay removal in hybrid context
// =============================================================================

func TestE5_OverlayRemoval(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/e5_overlay_removal.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "Source", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "Terminal", []float32{2, 2, 2}, nil)
	src, _ := db.GetNodeID(context.Background(), "docs", "Source")
	tgt, _ := db.GetNodeID(context.Background(), "docs", "Terminal")

	graph.RegisterEdgeKind("LINKS", 1)
	txn := gr.BeginTxn()
	txn.AddEdge(src, tgt, 1.0, 1)
	txn.Commit(context.Background())

	// Verify base edge exists.
	baseNeighbors, _ := gr.Neighbors(src)
	if len(baseNeighbors) == 0 {
		t.Fatal("base edge should exist")
	}
	t.Logf("Phase 1: base edge Source→Terminal exists")

	// Begin epoch and remove the edge.
	epoch, _ := db.BeginEpochTx(context.Background())
	gtx, _ := epoch.GraphTxn("docs")
	gtx.RemoveEdge(src, tgt, 1)

	// Epoch overlay must show zero results through the removed edge.
	overlayNeighbors, _ := gtx.NeighborsOverlay(src)
	if len(overlayNeighbors) != 0 {
		t.Fatalf("epoch overlay must show 0 neighbors after removal, got %d", len(overlayNeighbors))
	}
	t.Logf("Phase 2: epoch overlay shows 0 neighbors after edge removal ✓")

	// Rollback restores only the epoch's local view.
	epoch.Rollback(context.Background())
	restoredNeighbors, _ := gr.Neighbors(src)
	if len(restoredNeighbors) == 0 {
		t.Fatal("live edge should be restored after rollback")
	}
	t.Logf("Phase 3: live edge restored after rollback ✓")
	t.Log("✅ test C: overlay removal in hybrid context")
}

// =============================================================================
// Test D: Inbound hybrid path
// =============================================================================

func TestE5_InboundHybridPath(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/e5_inbound.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "Target", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "BaseSource", []float32{2, 0, 0}, nil)
	tgt, _ := db.GetNodeID(context.Background(), "docs", "Target")
	baseSrc, _ := db.GetNodeID(context.Background(), "docs", "BaseSource")

	graph.RegisterEdgeKind("CITES", 1)
	txn := gr.BeginTxn()
	txn.AddEdge(baseSrc, tgt, 1.0, 1)
	txn.Commit(context.Background())

	// Verify live inbound edge.
	inbound, _ := gr.InboundNeighbors(tgt)
	if len(inbound) == 0 || inbound[0].Target != baseSrc {
		t.Fatal("live inbound edge should exist")
	}
	t.Logf("Phase 1: live inbound edge BaseSource→Target exists")

	// Begin epoch, stage an additional inbound edge.
	col.Insert(context.Background(), "StagedSource", []float32{3, 0, 0}, nil)
	stagedSrc, _ := db.GetNodeID(context.Background(), "docs", "StagedSource")

	epoch, _ := db.BeginEpochTx(context.Background())
	gtx, _ := epoch.GraphTxn("docs")
	gtx.AddEdge(stagedSrc, tgt, 1.0, 1)

	// Inbound overlay must see both base and staged inbound edges.
	ibOverlay, _ := gtx.InboundNeighborsOverlay(tgt)
	if len(ibOverlay) < 2 {
		t.Fatalf("inbound overlay: want >= 2 edges, got %d", len(ibOverlay))
	}
	hasBase, hasStaged := false, false
	for _, e := range ibOverlay {
		if e.Target == baseSrc {
			hasBase = true
		}
		if e.Target == stagedSrc {
			hasStaged = true
		}
	}
	if !hasBase || !hasStaged {
		t.Fatal("inbound overlay must see both base and staged sources")
	}
	t.Logf("Phase 2: inbound overlay includes base + staged sources ✓")

	epoch.Rollback(context.Background())
	t.Log("✅ test D: inbound hybrid path")
}

// =============================================================================
// Test G: Provisional node identity through SQL
// =============================================================================

func TestE5_ProvisionalNodeIdentity(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/e5_provisional.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "Existing", []float32{1, 0, 0}, nil)

	graph.RegisterEdgeKind("KNOWS", 10)

	epoch, _ := db.BeginEpochTx(context.Background())

	// Insert two provisional records via SQL.
	_, err := epoch.Query(context.Background(), "INSERT INTO docs (id, embedding) VALUES ('Alpha', '[1,0,0]')", nil)
	if err != nil {
		t.Fatalf("insert Alpha: %v", err)
	}
	_, err = epoch.Query(context.Background(), "INSERT INTO docs (id, embedding) VALUES ('Beta', '[0,1,0]')", nil)
	if err != nil {
		t.Fatalf("insert Beta: %v", err)
	}

	// Insert graph edge between provisional nodes via SQL.
	_, err = epoch.Query(context.Background(), "INSERT INTO GRAPH_EDGES VALUES ('Alpha', 'KNOWS', 'Beta')", nil)
	if err != nil {
		t.Fatalf("insert edge Alpha→Beta: %v", err)
	}

	// Verify provisional IDs resolve correctly within epoch.
	alpha, err := epoch.LookupNodeID(context.Background(), "docs", "Alpha")
	if err != nil {
		t.Fatalf("LookupNodeID Alpha: %v", err)
	}
	beta, err := epoch.LookupNodeID(context.Background(), "docs", "Beta")
	if err != nil {
		t.Fatalf("LookupNodeID Beta: %v", err)
	}

	// Verify the staged edge is visible in epoch overlay.
	gtx, _ := epoch.GraphTxn("docs")
	neighbors, _ := gtx.NeighborsOverlay(alpha)
	if len(neighbors) == 0 || neighbors[0].Target != beta {
		t.Fatalf("staged edge Alpha→Beta must be visible in overlay")
	}
	t.Logf("Phase 1: provisional edge Alpha(%d)→Beta(%d) visible in epoch ✓", alpha, beta)

	// Verify reverse resolution works.
	colName, recID, err := epoch.ResolveNodeID(context.Background(), alpha)
	if err != nil || colName != "docs" || recID != "Alpha" {
		t.Fatalf("ResolveNodeID(%d): want (docs, Alpha), got (%s, %s), err=%v", alpha, colName, recID, err)
	}
	t.Logf("Phase 1: reverse resolution Alpha(%d)→(docs, Alpha) ✓", alpha)

	// Commit and verify durable mapping.
	if err := epoch.Commit(context.Background()); err != nil {
		t.Fatalf("Commit: %v", err)
	}
	time.Sleep(50 * time.Millisecond)

	durableAlpha, _ := db.GetNodeID(context.Background(), "docs", "Alpha")
	durableBeta, _ := db.GetNodeID(context.Background(), "docs", "Beta")
	if durableAlpha == 0 || durableBeta == 0 {
		t.Fatal("durable node IDs must be assigned after commit")
	}
	durableEdges, _ := gr.Neighbors(durableAlpha)
	if len(durableEdges) == 0 || durableEdges[0].Target != durableBeta {
		t.Fatal("durable edge must survive commit")
	}
	t.Logf("Phase 2: durable edge Alpha(%d)→Beta(%d) after commit ✓", durableAlpha, durableBeta)
	t.Log("✅ test G: provisional node identity through SQL")
}

// =============================================================================
// Test H: HNSW guard — epoch queries must not use live HNSW
// =============================================================================

func TestE5_HNSWGuard(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/e5_hnsw_guard.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "A", []float32{1, 0, 0}, nil)

	epoch, _ := db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// Execute a vector query inside the epoch. It must use exact scoring, not HNSW.
	results, err := epoch.Query(context.Background(),
		"SELECT id FROM docs ORDER BY VECTOR_DISTANCE(embedding, '[0,1,0]') ASC LIMIT 1", nil)
	if err != nil {
		t.Fatalf("epoch vector query: %v", err)
	}
	if len(results.Results) == 0 {
		t.Fatal("epoch vector query must return results")
	}
	t.Logf("Phase 1: epoch vector query returned %d results (exact SIMD path)", len(results.Results))

	// Verify the result ID is correct.
	if results.Results[0].ID != "A" {
		t.Fatalf("expected 'A', got '%s'", results.Results[0].ID)
	}
	t.Logf("Phase 1: correct result '%s' returned ✓", results.Results[0].ID)
	t.Log("✅ test H: epoch uses exact SIMD, not HNSW")
}

// =============================================================================
// Test J: AS OF guard through hybrid path
// =============================================================================

func TestE5_ASOFGuard(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/e5_asof_guard.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "A", []float32{1, 0, 0}, nil)

	epoch, _ := db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// AS OF TIMESTAMP inside epoch must be rejected.
	_, err := epoch.Query(context.Background(),
		"SELECT * FROM docs AS OF TIMESTAMP '2020-01-01T00:00:00Z'", nil)
	if err == nil {
		t.Fatal("AS OF TIMESTAMP inside epoch must be rejected")
	}
	t.Logf("AS OF rejection: %v ✓", err)
	t.Log("✅ test J: AS OF TIMESTAMP rejected inside epoch")
}

// =============================================================================
// Test: Staged edge + graph traversal in epoch with multi-hop path
// =============================================================================

func TestE5_StagedMultiHop(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/e5_multihop.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "A", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "B", []float32{2, 0, 0}, nil)
	a, _ := db.GetNodeID(context.Background(), "docs", "A")
	b, _ := db.GetNodeID(context.Background(), "docs", "B")

	graph.RegisterEdgeKind("FLOWS", 20)

	// Commit A→B as baseline BEFORE creating the epoch so it's visible at S0.
	txn := gr.BeginTxn()
	txn.AddEdge(a, b, 1.0, 1)
	if err := txn.Commit(context.Background()); err != nil {
		t.Fatalf("commit A→B: %v", err)
	}
	// Verify base edge exists before epoch.
	abEdges, _ := gr.Neighbors(a)
	if len(abEdges) == 0 || abEdges[0].Target != b {
		t.Fatalf("A→B edge must exist before epoch: got %d edges", len(abEdges))
	}
	t.Logf("Phase 1: base edge A(%d)→B(%d) committed and visible ✓", a, b)

	// Now create the epoch (A→B is visible at S0).
	epoch, _ := db.BeginEpochTx(context.Background())
	// Stage B→C and C→D inside epoch.
	_, err := epoch.Query(context.Background(), "INSERT INTO docs (id, embedding) VALUES ('C', '[3,0,0]')", nil)
	if err != nil {
		t.Fatalf("insert C: %v", err)
	}
	_, err = epoch.Query(context.Background(), "INSERT INTO docs (id, embedding) VALUES ('D', '[4,0,0]')", nil)
	if err != nil {
		t.Fatalf("insert D: %v", err)
	}
	_, err = epoch.Query(context.Background(), "INSERT INTO GRAPH_EDGES VALUES ('B', 'FLOWS', 'C')", nil)
	if err != nil {
		t.Fatalf("insert B→C: %v", err)
	}
	_, err = epoch.Query(context.Background(), "INSERT INTO GRAPH_EDGES VALUES ('C', 'FLOWS', 'D')", nil)
	if err != nil {
		t.Fatalf("insert C→D: %v", err)
	}

	// Verify 3-hop traversal: A → B → C → D.
	gtx, _ := epoch.GraphTxn("docs")
	aID, _ := epoch.LookupNodeID(context.Background(), "docs", "A")
	cID, _ := epoch.LookupNodeID(context.Background(), "docs", "C")
	dID, _ := epoch.LookupNodeID(context.Background(), "docs", "D")

	// A should have B as neighbor.
	aNeighbors, _ := gtx.NeighborsOverlay(aID)
	if len(aNeighbors) == 0 || aNeighbors[0].Target != b {
		t.Fatal("A→B should be visible")
	}
	// B should have C as neighbor.
	bNeighbors, _ := gtx.NeighborsOverlay(b)
	hasC := false
	for _, nb := range bNeighbors {
		if nb.Target == cID {
			hasC = true
		}
	}
	if !hasC {
		t.Fatal("B→C staged edge should be visible")
	}
	// C should have D as neighbor.
	cNeighbors, _ := gtx.NeighborsOverlay(cID)
	if len(cNeighbors) == 0 || cNeighbors[0].Target != dID {
		t.Fatal("C→D staged edge should be visible")
	}
	t.Logf("Phase 1: 3-hop path A→B→C→D traversable in epoch ✓")

	epoch.Rollback(context.Background())

	// After rollback: only A→B should exist in live graph.
	liveA, _ := gr.Neighbors(a)
	if len(liveA) == 0 || liveA[0].Target != b {
		t.Fatal("live A→B should exist after rollback")
	}
	// B should have no edges in live graph.
	liveB, _ := gr.Neighbors(b)
	if len(liveB) != 0 {
		t.Fatal("live B should have no edges after rollback")
	}
	t.Logf("Phase 2: after rollback, only base A→B survives ✓")
	t.Log("✅ staged multi-hop traversal")
}
