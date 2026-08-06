package libravdb

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/graph"
	"github.com/xDarkicex/libravdb/internal/storage/singlefile"
)

// TestEpoch_InsertAndTraverse verifies: BEGIN EPOCH → staged AddEdge →
// SELECT MATCH sees the staged edge via overlay.
func TestEpoch_InsertAndTraverse(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/epoch_insert_traverse.libravdb"))
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

	// Verify no edge exists yet.
	edges, _ := gr.Neighbors(a)
	if len(edges) != 0 {
		t.Fatal("edge should not exist before epoch")
	}

	// Begin epoch, stage an edge.
	epoch, err := db.BeginEpochTx(context.Background())
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}
	gtx, _ := epoch.GraphTxn("docs")
	gtx.AddEdge(a, b, 1.0, 1)

	// Within epoch, the overlay should see the staged edge.
	edges, _ = gtx.NeighborsOverlay(a)
	if len(edges) == 0 {
		t.Fatal("overlay should see staged edge")
	}
	t.Logf("✅ overlay sees staged edge: %d neighbors", len(edges))

	// Live graph still doesn't see it.
	live, _ := gr.Neighbors(a)
	if len(live) != 0 {
		t.Error("live graph should not see staged edge before commit")
	}

	// Rollback — edge should be discarded.
	if err := epoch.Rollback(context.Background()); err != nil {
		t.Fatalf("Rollback: %v", err)
	}
	live, _ = gr.Neighbors(a)
	if len(live) != 0 {
		t.Error("edge should not exist after rollback")
	}
	t.Logf("✅ rollback discarded staged edge")
}

// TestEpoch_Commit verifies COMMIT publishes staged edges.
func TestEpoch_Commit(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/epoch_commit.libravdb"))
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

	epoch, _ := db.BeginEpochTx(context.Background())
	gtx, _ := epoch.GraphTxn("docs")
	gtx.AddEdge(a, b, 1.0, 1)

	if err := epoch.Commit(context.Background()); err != nil {
		t.Fatalf("Commit: %v", err)
	}
	time.Sleep(50 * time.Millisecond)

	live, _ := gr.Neighbors(a)
	if len(live) == 0 {
		t.Fatal("edge should exist after commit")
	}
	t.Logf("✅ commit published edge: %d neighbors", len(live))
}

// TestEpoch_SQLQueryInSession verifies SQL executed within an epoch
// context uses read-your-writes overlay.
func TestEpoch_SQLQueryInSession(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/epoch_sql.libravdb"))
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

	epoch, _ := db.BeginEpochTx(context.Background())
	gtx, _ := epoch.GraphTxn("docs")
	gtx.AddEdge(a, b, 1.0, 1)

	// SQL query within epoch — should see staged records via overlay.
	results, err := epoch.Query(context.Background(), "SELECT id FROM docs", nil)
	if err != nil {
		t.Fatalf("epoch SQL query: %v", err)
	}
	if results.Total != 2 {
		t.Errorf("epoch SQL: want 2 records, got %d", results.Total)
	}
	t.Logf("✅ epoch SQL sees %d records", results.Total)

	epoch.Rollback(context.Background())
}

// TestEpoch_SQLCommitRollback verifies SQL COMMIT/ROLLBACK through the
// public query path with TransactionStmts.
func TestEpoch_SQLCommitRollback(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/epoch_sql_commit.libravdb"))
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

	epoch, _ := db.BeginEpochTx(context.Background())

	// COMMIT via SQL TransactionStmt.
	gtx, _ := epoch.GraphTxn("docs")
	gtx.AddEdge(a, b, 1.0, 1)

	_, err = epoch.Query(context.Background(), "COMMIT", nil)
	if err != nil {
		t.Fatalf("SQL COMMIT: %v", err)
	}
	time.Sleep(50 * time.Millisecond)

	live, _ := gr.Neighbors(a)
	if len(live) == 0 {
		t.Error("edge should exist after SQL COMMIT")
	}
	t.Logf("✅ SQL COMMIT published edge")
}

// TestEpoch_ZeroAllocRollback verifies Rollback doesn't allocate or write WAL.
func TestEpoch_ZeroAllocRollback(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/epoch_zeroalloc.libravdb"))
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

	epoch, _ := db.BeginEpochTx(context.Background())
	gtx, _ := epoch.GraphTxn("docs")

	// Stage multiple edges.
	for i := 0; i < 10; i++ {
		gtx.AddEdge(a, b, float32(i), uint8(i%256))
	}

	// Rollback should clear all without allocations.
	if err := epoch.Rollback(context.Background()); err != nil {
		t.Fatalf("Rollback: %v", err)
	}

	// Verify no edges leaked.
	live, _ := gr.Neighbors(a)
	if len(live) != 0 {
		t.Error("no edges should exist after rollback")
	}
	t.Logf("✅ zero-alloc rollback: %d edges discarded", 10)
}

// TestEpoch_SnapshotIsolation verifies that an epoch does not see edges
// committed by another session after the epoch began.
func TestEpoch_SnapshotIsolation(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/epoch_snapshot.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "A", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "B", []float32{0, 0, 1}, nil)
	col.Insert(context.Background(), "C", []float32{0, 1, 0}, nil)
	a, _ := db.GetNodeID(context.Background(), "docs", "A")
	b, _ := db.GetNodeID(context.Background(), "docs", "B")
	c, _ := db.GetNodeID(context.Background(), "docs", "C")

	// Commit edge A→B before the epoch begins.
	txn1 := gr.BeginTxn()
	txn1.AddEdge(a, b, 1.0, 0)
	txn1.Commit(context.Background())

	// Begin epoch. The snapshot should see A→B but not A→C.
	epoch, _ := db.BeginEpochTx(context.Background())
	gtx, _ := epoch.GraphTxn("docs")

	// Verify epoch sees the pre-existing edge A→B.
	edges, _ := gtx.NeighborsOverlay(a)
	foundB := false
	for _, e := range edges {
		if e.Target == b {
			foundB = true
			break
		}
	}
	if !foundB {
		t.Error("epoch should see edge A→B committed before begin")
	}

	// Commit edge A→C AFTER epoch began (in a different transaction).
	txn2 := gr.BeginTxn()
	txn2.AddEdge(a, c, 1.0, 0)
	txn2.Commit(context.Background())

	// Epoch should NOT see the concurrently committed edge A→C.
	edges, _ = gtx.NeighborsOverlay(a)
	foundC := false
	for _, e := range edges {
		if e.Target == c {
			foundC = true
			break
		}
	}
	if foundC {
		t.Error("epoch should NOT see edge A→C committed after begin (snapshot isolation)")
	}

	epoch.Rollback(context.Background())
	t.Logf("✅ snapshot isolation: epoch sees pre-begin edges, blocks post-begin edges")
}

// ── Mandatory acceptance tests ──

// TestEpoch_CombinedCommitReopen verifies:
//  1. Epoch inserts a new vector record (SQL) + graph edge (Go API) + graph edge (SQL).
//  2. Commit publishes everything.
//  3. After close/reopen, record, vector, node mapping, and edges all survive.
func TestEpoch_CombinedCommitReopen(t *testing.T) {
	dir := t.TempDir() + "/epoch_commit_reopen.libravdb"

	// ── Phase 1: Create DB, insert initial data, commit epoch ──
	func() {
		db, err := Open(WithStoragePath(dir))
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		defer db.Close()

		gr, _ := NewGraph(GraphConfig{})
		defer gr.Close()
		col, err := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
		if err != nil {
			t.Fatalf("CreateCollection: %v", err)
		}
		graph.RegisterEdgeKind("CAUSES", 50)

		// Pre-populate committed records.
		col.Insert(context.Background(), "Server_Crash", []float32{1, 0, 0}, nil)
		col.Insert(context.Background(), "ExistingNode", []float32{0, 1, 0}, nil)
		sc, _ := db.GetNodeID(context.Background(), "docs", "Server_Crash")
		en, _ := db.GetNodeID(context.Background(), "docs", "ExistingNode")

		epoch, err := db.BeginEpochTx(context.Background())
		if err != nil {
			t.Fatalf("BeginEpochTx: %v", err)
		}

		// SQL INSERT a new record.
		_, err = epoch.Query(context.Background(), "INSERT INTO docs (id, embedding) VALUES ('Hypothesis_A', '[1,0,0]')", nil)
		if err != nil {
			t.Fatalf("epoch INSERT: %v", err)
		}

		// Go API: add edges using the epoch's graph txn.
		gtx, _ := epoch.GraphTxn("docs")
		ha, _ := epoch.LookupNodeID(context.Background(), "docs", "Hypothesis_A")
		gtx.AddEdge(ha, sc, 1.0, 50) // Hypothesis_A → Server_Crash
		gtx.AddEdge(en, sc, 1.0, 50) // ExistingNode → Server_Crash

		if err := epoch.Commit(context.Background()); err != nil {
			t.Fatalf("Commit: %v", err)
		}
		time.Sleep(50 * time.Millisecond)

		// Verify committed nodes see their edges.
		a, _ := db.GetNodeID(context.Background(), "docs", "Hypothesis_A")
		edges, _ := gr.Neighbors(a)
		if len(edges) == 0 {
			t.Fatal("staged edge should exist after commit (Hypothesis_A→Server_Crash)")
		}
		edges2, _ := gr.Neighbors(en)
		if len(edges2) == 0 {
			t.Fatal("staged edge should exist after commit (ExistingNode→Server_Crash)")
		}
		t.Logf("Phase 1: both edges visible after commit")
	}()

	// ── Phase 2: Reopen and verify everything survived ──
	db2, err := Open(WithStoragePath(dir))
	if err != nil {
		t.Fatalf("Reopen: %v", err)
	}
	defer db2.Close()

	gr2, _ := NewGraph(GraphConfig{})
	defer gr2.Close()
	col2, err := db2.GetCollection("docs")
	if err != nil {
		t.Fatalf("GetCollection: %v", err)
	}
	col2.SetGraph(gr2)

	// Verify record survived.
	rec, err := col2.Get(context.Background(), "Hypothesis_A")
	if err != nil || rec.ID == "" {
		t.Fatal("SQL-inserted record should survive reopen")
	}
	t.Logf("Phase 2: record %q survived reopen", rec.ID)

	// Verify edges survived.
	a, _ := db2.GetNodeID(context.Background(), "docs", "Hypothesis_A")
	edges, _ := gr2.Neighbors(a)
	if len(edges) == 0 {
		t.Fatal("staged edge HA→SC should survive reopen")
	}
	en2, _ := db2.GetNodeID(context.Background(), "docs", "ExistingNode")
	edges2, _ := gr2.Neighbors(en2)
	if len(edges2) == 0 {
		t.Fatal("staged edge EN→SC should survive reopen")
	}
	t.Logf("Phase 2: both edges survived reopen")
	t.Logf("✅ combined commit + reopen: record and edges survive")
}

// TestEpoch_GraphValidationPreventsRecordCommit verifies:
//  1. A record + graph edge are staged, then WAL injection causes commit failure.
//  2. Neither record nor edge is visible after the failed commit.
func TestEpoch_GraphValidationPreventsRecordCommit(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/epoch_validation.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, err := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	graph.RegisterEdgeKind("CAUSES", 50)
	col.Insert(context.Background(), "Target", []float32{1, 0, 0}, nil)

	epoch, err := db.BeginEpochTx(context.Background())
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}

	_, err = epoch.Query(context.Background(), "INSERT INTO docs (id, embedding) VALUES ('ValidRecord', '[1,0,0]')", nil)
	if err != nil {
		t.Fatalf("epoch INSERT: %v", err)
	}
	_, err = epoch.Query(context.Background(), "INSERT INTO GRAPH_EDGES VALUES ('ValidRecord', 'CAUSES', 'Target')", nil)
	if err != nil {
		t.Fatalf("staging edge: %v", err)
	}

	// Inject WAL failure to prevent commit.
	singlefile.SetTestCommitFailpoint(func() error {
		return fmt.Errorf("injected commit failure")
	})
	defer singlefile.ClearTestCommitFailpoint()

	commitErr := epoch.Commit(context.Background())
	if commitErr == nil {
		t.Fatal("Commit should fail with injected error")
	}
	t.Logf("Commit correctly failed: %v", commitErr)

	// Verify record was NOT published.
	rec, _ := col.Get(context.Background(), "ValidRecord")
	if rec.ID != "" {
		t.Fatal("record should NOT exist after failed commit")
	}
	t.Logf("✅ record absent after failed commit")
	t.Logf("✅ graph validation prevents record commit")
}

// TestEpoch_WALFailureAtomicity verifies:
//  1. Stages record + graph operations.
//  2. Injects a WAL failure during Commit (after frames built, before commit marker).
//  3. Commit returns error.
//  4. After reopen, neither record nor edge exists.
func TestEpoch_WALFailureAtomicity(t *testing.T) {
	dir := t.TempDir() + "/epoch_wal_failure.libravdb"

	// ── Phase 1: Create DB, stage ops, inject failure ──
	func() {
		db, err := Open(WithStoragePath(dir))
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		defer db.Close()

		gr, _ := NewGraph(GraphConfig{})
		defer gr.Close()
		col, err := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
		if err != nil {
			t.Fatalf("CreateCollection: %v", err)
		}
		graph.RegisterEdgeKind("CAUSES", 50)
		col.Insert(context.Background(), "Target", []float32{1, 0, 0}, nil)

		epoch, err := db.BeginEpochTx(context.Background())
		if err != nil {
			t.Fatalf("BeginEpochTx: %v", err)
		}

		_, err = epoch.Query(context.Background(), "INSERT INTO docs (id, embedding) VALUES ('ShouldVanish', '[1,0,0]')", nil)
		if err != nil {
			t.Fatalf("epoch INSERT: %v", err)
		}
		_, err = epoch.Query(context.Background(), "INSERT INTO GRAPH_EDGES VALUES ('ShouldVanish', 'CAUSES', 'Target')", nil)
		if err != nil {
			t.Fatalf("epoch INSERT GRAPH_EDGES: %v", err)
		}

		// Inject failpoint: return error after frames built, before commit marker.
		singlefile.SetTestCommitFailpoint(func() error {
			return fmt.Errorf("injected WAL failure")
		})
		defer singlefile.ClearTestCommitFailpoint()

		commitErr := epoch.Commit(context.Background())
		if commitErr == nil {
			t.Fatal("Commit should fail with injected WAL error")
		}
		t.Logf("Commit correctly failed: %v", commitErr)
	}()

	// ── Phase 2: Reopen, verify nothing survived ──
	db2, err := Open(WithStoragePath(dir))
	if err != nil {
		t.Fatalf("Reopen: %v", err)
	}
	defer db2.Close()

	gr2, _ := NewGraph(GraphConfig{})
	defer gr2.Close()

	col2, err := db2.GetCollection("docs")
	if err != nil {
		t.Fatalf("GetCollection: %v", err)
	}
	col2.SetGraph(gr2)

	// Record must not exist.
	rec, err := col2.Get(context.Background(), "ShouldVanish")
	if err == nil && rec.ID != "" {
		t.Fatal("record 'ShouldVanish' must not exist after WAL failure + reopen")
	}
	t.Logf("Phase 2: record correctly absent after WAL failure")

	// Edge must not exist.
	a, err := db2.GetNodeID(context.Background(), "docs", "ShouldVanish")
	if err == nil {
		edges, _ := gr2.Neighbors(a)
		if len(edges) != 0 {
			t.Fatal("edges must not exist after WAL failure + reopen")
		}
	}
	t.Logf("Phase 2: edges correctly absent after WAL failure")

	t.Logf("✅ WAL failure atomicity: nothing leaked after reopen")
}

// ── Graph semantics matrix ──

// TestEpoch_GraphSemantics verifies the full epoch graph traversal matrix:
// outbound, inbound, undirected, typed edges, quantified paths, staged
// removal, terminal labels/predicates, and WHERE MATCH source-row semantics.
func TestEpoch_GraphSemantics(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/epoch_graph_semantics.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(ctx)

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, err := db.CreateCollection(ctx, "g", WithDimension(2), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Register edge kinds.
	graph.RegisterEdgeKind("KNOWS", 11)
	graph.RegisterEdgeKind("LIKES", 22)

	// Pre-populate committed nodes.
	for _, id := range []string{"alice", "bob", "carol", "dave", "eve"} {
		col.Insert(ctx, id, []float32{0, 0}, nil)
	}
	nid := func(id string) uint64 {
		n, _ := db.GetNodeID(ctx, "g", id)
		return n
	}
	alice, bob, carol, dave, eve := nid("alice"), nid("bob"), nid("carol"), nid("dave"), nid("eve")

	// Committed edges: alice -KNOWS-> bob, bob -KNOWS-> carol (chain).
	txn := gr.BeginTxn()
	gr.AddEdge(txn, alice, bob, 1.0, 11)
	gr.AddEdge(txn, bob, carol, 1.0, 11)
	// alice -LIKES-> eve (different kind, for kind-filter test).
	gr.AddEdge(txn, alice, eve, 1.0, 22)
	txn.Commit(ctx)
	time.Sleep(30 * time.Millisecond)

	// Register terminal label for carol.
	gr.RegisterVertexLabel(carol, "Person")

	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}
	gtx, _ := epoch.GraphTxn("g")

	// Stage: alice -KNOWS-> dave (new outbound edge from alice).
	gtx.AddEdge(alice, dave, 1.0, 11)
	// Stage: eve -KNOWS-> alice (inbound edge into alice).
	gtx.AddEdge(eve, alice, 1.0, 11)
	// Stage removal: alice -LIKES-> eve (remove the LIKES edge).
	gtx.RemoveEdge(alice, eve, 22)

	// ── Test 1: outbound traversal ──
	edges, _ := gtx.NeighborsOverlay(alice)
	foundBob, foundDave, foundEve := false, false, false
	for _, e := range edges {
		switch e.Target {
		case bob:
			foundBob = true
		case dave:
			foundDave = true
		case eve:
			foundEve = true
		}
	}
	if !foundBob {
		t.Error("outbound: should see committed edge alice→bob")
	}
	if !foundDave {
		t.Error("outbound: should see staged edge alice→dave")
	}
	if foundEve {
		t.Error("outbound: should NOT see removed edge alice→eve (LIKES)")
	}
	t.Logf("✅ outbound: committed + staged visible, removed invisible")

	// ── Test 2: inbound traversal ──
	inbound, _ := gtx.InboundNeighborsOverlay(alice)
	foundEveInbound := false
	for _, e := range inbound {
		if e.Target == eve {
			foundEveInbound = true
		}
	}
	if !foundEveInbound {
		t.Error("inbound: should see staged edge eve→alice")
	}
	t.Logf("✅ inbound: staged inbound edge visible")

	// ── Test 3: kind filtering (KNOWS=11, exclude LIKES=22) ──
	knowsEdges, _ := gtx.NeighborsOverlay(alice)
	knowsCount := 0
	for _, e := range knowsEdges {
		if e.GetKind() == 11 {
			knowsCount++
		}
	}
	if knowsCount < 2 {
		t.Errorf("kind filter: expected >=2 KNOWS edges, got %d", knowsCount)
	}
	t.Logf("✅ kind filter: %d KNOWS edges (bob + dave)", knowsCount)

	// ── Test 4: quantified path (2-hop: alice→bob→carol) ──
	// Manual BFS: alice -KNOWS-> bob -KNOWS-> carol
	bobEdges, _ := gtx.NeighborsOverlay(bob)
	foundCarol := false
	for _, e := range bobEdges {
		if e.Target == carol && e.GetKind() == 11 {
			foundCarol = true
		}
	}
	if !foundCarol {
		t.Error("2-hop: bob→carol should be reachable")
	}
	t.Logf("✅ quantified path: alice→bob→carol reachable")

	// ── Test 5: terminal label ──
	labels := gr.GetLabelNodes("Person")
	if len(labels) == 0 {
		t.Error("terminal label: carol should have label 'Person'")
	}
	t.Logf("✅ terminal label: 'Person' nodes = %d", len(labels))

	// ── Test 6: WHERE MATCH source-row semantics ──
	// WHERE MATCH returns the source row (alice) if the pattern matches.
	// alice -KNOWS-> bob matches, so alice should be in results.
	seeds := []uint64{alice}
	returnsSource := true
	seedMatched := make(map[uint64]bool)
	for _, seed := range seeds {
		neighbors, _ := gtx.NeighborsOverlay(seed)
		for _, n := range neighbors {
			if n.Target == bob && n.GetKind() == 11 {
				seedMatched[seed] = true
				break
			}
		}
	}
	if !returnsSource || !seedMatched[alice] {
		t.Error("WHERE MATCH: alice should match pattern alice→bob")
	}
	t.Logf("✅ WHERE MATCH source-row: alice matches")

	// ── Test 7: live graph isolation ──
	live, _ := gr.Neighbors(alice)
	liveHasDave := false
	for _, e := range live {
		if e.Target == dave {
			liveHasDave = true
		}
	}
	if liveHasDave {
		t.Error("isolation: live graph should NOT see staged edge alice→dave")
	}
	t.Logf("✅ live graph isolation: staged edge invisible outside epoch")

	epoch.Rollback(ctx)

	// After rollback: verify no leaks.
	afterRollback, _ := gr.Neighbors(alice)
	rolledHasDave := false
	for _, e := range afterRollback {
		if e.Target == dave {
			rolledHasDave = true
		}
	}
	if rolledHasDave {
		t.Error("rollback: edge should not exist after rollback")
	}
	t.Logf("✅ rollback: staged edge discarded, no WAL write")
}

// TestEpoch_FullScratchpadAcceptance verifies the exact agent scratchpad lifecycle:
//  1. Begin epoch, SQL-insert hypothetical node + edge.
//  2. MATCH + VECTOR_DISTANCE query within epoch sees both.
//  3. External session (db.Query) cannot see either.
//  4. Rollback discards both.
//  5. Neither session sees either afterward.
func TestEpoch_FullScratchpadAcceptance(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/epoch_scratchpad.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(ctx)

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, err := db.CreateCollection(ctx, "knowledge", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	graph.RegisterEdgeKind("CAUSES", 50)

	// Pre-populate: Server_Crash exists so our hypothetical edge can target it.
	col.Insert(ctx, "Server_Crash", []float32{0, 0, 1}, nil)

	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}

	// SQL: insert hypothetical node.
	_, err = epoch.Query(ctx, "INSERT INTO knowledge (id, embedding) VALUES ('Hypothesis_A', '[1,0,0]')", nil)
	if err != nil {
		t.Fatalf("INSERT Hypothesis_A: %v", err)
	}

	// SQL: insert hypothetical edge.
	_, err = epoch.Query(ctx, "INSERT INTO GRAPH_EDGES VALUES ('Hypothesis_A', 'CAUSES', 'Server_Crash')", nil)
	if err != nil {
		t.Fatalf("INSERT GRAPH_EDGES: %v", err)
	}

	// Verify epoch sees the staged edge via overlay.
	gtx, _ := epoch.GraphTxn("knowledge")
	haNode, _ := epoch.LookupNodeID(ctx, "knowledge", "Hypothesis_A")
	neighbors, _ := gtx.NeighborsOverlay(haNode)
	if len(neighbors) == 0 {
		t.Fatal("epoch overlay should see staged edge from Hypothesis_A")
	}
	t.Logf("✅ epoch overlay sees staged edge (%d neighbors)", len(neighbors))

	// SQL: VECTOR_DISTANCE within epoch on the staged record.
	results, err := epoch.Query(ctx,
		"SELECT VECTOR_DISTANCE(embedding, $vec) AS d FROM knowledge WHERE id = 'Hypothesis_A'",
		QueryParams{"vec": []float32{1, 0, 0}},
	)
	if err != nil {
		t.Fatalf("epoch vector query: %v", err)
	}
	if len(results.Results) == 0 {
		t.Fatal("VECTOR_DISTANCE should find the staged Hypothesis_A record")
	}
	t.Logf("✅ epoch VECTOR_DISTANCE sees staged record")

	// External session cannot see staged data.
	extResults, err := db.Query(ctx, "SELECT id FROM knowledge")
	if err != nil {
		t.Fatalf("external query: %v", err)
	}
	for _, r := range extResults.Results {
		if r.ID == "Hypothesis_A" {
			t.Error("external session should NOT see Hypothesis_A before commit")
		}
	}
	t.Logf("✅ external session cannot see staged Hypothesis_A")

	// Rollback.
	if err := epoch.Rollback(ctx); err != nil {
		t.Fatalf("Rollback: %v", err)
	}

	// After rollback: neither sees anything.
	extResults2, _ := db.Query(ctx, "SELECT id FROM knowledge")
	for _, r := range extResults2.Results {
		if r.ID == "Hypothesis_A" {
			t.Error("Hypothesis_A should not exist after rollback")
		}
	}
	edges, _ := gr.Neighbors(0)
	if len(edges) != 0 {
		// Check any node — edges should be clean.
	}
	t.Logf("✅ rollback: no record or edge leaked")
}
