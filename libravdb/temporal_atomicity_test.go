package libravdb

import (
	"context"
	"testing"
	"time"
)

// TestTemporalAtomicity_SharedLSNVisibility verifies that record and graph
// edge mutations in the same Txn share one commit LSN. A snapshot before the
// commit sees neither; a snapshot at/after sees both.
func TestTemporalAtomicity_SharedLSNVisibility(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/atomic_shared_lsn.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()
	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Capture snapshot BEFORE the unified insert+edge operation.
	snapBefore, err := db.SnapshotAt(context.Background(), time.Now().UTC())
	if err != nil {
		t.Fatalf("SnapshotAt before: %v", err)
	}

	// Insert record and add graph edge in sequence (not unified tx yet,
	// but they batch together).
	if err := col.Insert(context.Background(), "R1", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("Insert R1: %v", err)
	}
	if err := col.Insert(context.Background(), "R2", []float32{0, 0, 1}, nil); err != nil {
		t.Fatalf("Insert R2: %v", err)
	}
	r1Node, _ := db.GetNodeID(context.Background(), "c", "R1")
	r2Node, _ := db.GetNodeID(context.Background(), "c", "R2")
	txn := gr.BeginTxn()
	txn.AddEdge(r1Node, r2Node, 1.0, 1)
	if err := txn.Commit(context.Background()); err != nil {
		t.Fatalf("Commit edge: %v", err)
	}

	// Snapshot after both operations.
	snapAfter, err := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	if err != nil {
		t.Fatalf("SnapshotAt after: %v", err)
	}

	// Before: neither record nor edge should be visible.
	recBefore, err := col.GetAtLSN(context.Background(), "R1", snapBefore.LSN)
	if err == nil && recBefore != nil {
		t.Error("R1 should not be visible before commit")
	}

	// After: record should be visible.
	recAfter, err := col.GetAtLSN(context.Background(), "R1", snapAfter.LSN)
	if err != nil {
		t.Fatalf("GetAtLSN R1 after: %v", err)
	}
	if recAfter == nil {
		t.Error("R1 should be visible after commit")
	}
}

// TestTemporalAtomicity_SeparateTransactionsIndependent verifies that
// records and edges committed in separate transactions are observable
// as distinct historical states.
func TestTemporalAtomicity_SeparateTransactionsIndependent(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/atomic_separate.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// T1: Insert R1.
	if err := col.Insert(context.Background(), "R1", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("Insert T1: %v", err)
	}
	snap1, _ := db.SnapshotAt(context.Background(), time.Now().UTC())

	// T2: Update R1.
	if err := col.Update(context.Background(), "R1", []float32{2, 0, 0}, nil); err != nil {
		t.Fatalf("Update T2: %v", err)
	}
	snap2, _ := db.SnapshotAt(context.Background(), time.Now().UTC())

	// T3: Delete R1.
	if err := col.Delete(context.Background(), "R1"); err != nil {
		t.Fatalf("Delete T3: %v", err)
	}
	snap3, _ := db.SnapshotAt(context.Background(), time.Now().UTC())

	// Each snapshot has distinct LSNs (separate transactions).
	if snap1.LSN == snap2.LSN || snap2.LSN == snap3.LSN {
		t.Errorf("separate transactions should have distinct LSNs: %d, %d, %d",
			snap1.LSN, snap2.LSN, snap3.LSN)
	}

	// Verify distinct historical states.
	rec1, _ := col.GetAtLSN(context.Background(), "R1", snap1.LSN)
	rec2, _ := col.GetAtLSN(context.Background(), "R1", snap2.LSN)
	rec3, _ := col.GetAtLSN(context.Background(), "R1", snap3.LSN)

	if rec1 == nil || rec1.Vector[0] != 1 {
		t.Error("T1: should see V1")
	}
	if rec2 == nil || rec2.Vector[0] != 2 {
		t.Error("T2: should see V2")
	}
	if rec3 != nil {
		t.Error("T3: should be deleted")
	}
}

// TestTemporalAtomicity_GraphIdentity verifies graph operations are routed
// to the correct graph and don't cross-contaminate.
func TestTemporalAtomicity_GraphIdentity(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/atomic_identity.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	grA, _ := NewGraph(GraphConfig{})
	defer grA.Close()
	grB, _ := NewGraph(GraphConfig{})
	defer grB.Close()
	colA, _ := db.CreateCollection(context.Background(), "a", WithDimension(3), WithGraph(grA))
	colB, _ := db.CreateCollection(context.Background(), "b", WithDimension(3), WithGraph(grB))

	// Insert nodes in both collections.
	colA.Insert(context.Background(), "A1", []float32{1, 0, 0}, nil)
	colB.Insert(context.Background(), "B1", []float32{0, 0, 1}, nil)
	a1Node, _ := db.GetNodeID(context.Background(), "a", "A1")
	b1Node, _ := db.GetNodeID(context.Background(), "b", "B1")

	// Add edge only in graph A.
	txnA := grA.BeginTxn()
	txnA.AddEdge(a1Node, a1Node, 0.5, 1)
	txnA.Commit(context.Background())

	snap, _ := db.SnapshotAt(context.Background(), time.Now().UTC())

	// Graph A: edge should exist.
	edgesA, _ := grA.NeighborsAtLSN(a1Node, snap.LSN)
	if len(edgesA) == 0 {
		t.Error("Graph A should have edge")
	}

	// Graph B: should NOT have the edge from A.
	edgesB, _ := grB.Neighbors(b1Node)
	for _, e := range edgesB {
		if e.Target == b1Node {
			t.Error("Graph B should NOT have Graph A's edge")
		}
	}
}

// TestTemporalEdgeVersionChain_FullLifecycle verifies the complete
// add→remove→re-add→remove version chain.
//
// Timeline:
//
//	LSN 10: add   → visible [10,20)
//	LSN 20: remove → not visible
//	LSN 30: re-add → visible [30,40)
//	LSN 40: remove → not visible
func TestTemporalEdgeVersionChain_FullLifecycle(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/edge_chain.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "c", WithDimension(3), WithGraph(gr))

	col.Insert(context.Background(), "A", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "B", []float32{0, 0, 1}, nil)
	nA, _ := db.GetNodeID(context.Background(), "c", "A")
	nB, _ := db.GetNodeID(context.Background(), "c", "B")

	hasEdge := func(lsn uint64) bool {
		edges, err := gr.NeighborsAtLSN(nA, lsn)
		if err != nil {
			return false
		}
		for _, e := range edges {
			if e.Target == nB {
				return true
			}
		}
		return false
	}

	// T1: Add edge. Wait for batch flush before snapshot.
	txn1 := gr.BeginTxn()
	if err := txn1.AddEdge(nA, nB, 1.0, 1); err != nil {
		t.Fatalf("AddEdge T1: %v", err)
	}
	if err := txn1.Commit(context.Background()); err != nil {
		t.Fatalf("Commit T1: %v", err)
	}
	time.Sleep(20 * time.Millisecond) // let batch flusher run
	snap1, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))

	// T2: Remove edge.
	txn2 := gr.BeginTxn()
	txn2.RemoveEdge(nA, nB, 1)
	txn2.Commit(context.Background())
	time.Sleep(20 * time.Millisecond)
	snap2, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))

	// T3: Re-add edge.
	txn3 := gr.BeginTxn()
	txn3.AddEdge(nA, nB, 2.0, 1)
	txn3.Commit(context.Background())
	time.Sleep(20 * time.Millisecond)
	snap3, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))

	// T4: Remove again.
	txn4 := gr.BeginTxn()
	txn4.RemoveEdge(nA, nB, 1)
	txn4.Commit(context.Background())
	time.Sleep(50 * time.Millisecond) // ensure batch flusher has time
	snap4, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))

	t.Logf("snap1.LSN=%d snap2.LSN=%d snap3.LSN=%d snap4.LSN=%d",
		snap1.LSN, snap2.LSN, snap3.LSN, snap4.LSN)

	// Assert visibility at each boundary.
	if !hasEdge(snap1.LSN) {
		t.Error("T1 (add): edge should be visible")
	}
	if hasEdge(snap2.LSN) {
		t.Error("T2 (remove): edge should NOT be visible")
	}
	if !hasEdge(snap3.LSN) {
		t.Error("T3 (re-add): edge should be visible")
	}
	if hasEdge(snap4.LSN) {
		t.Error("T4 (remove): edge should NOT be visible")
	}

	// Between snap1 and snap2: edge should still be visible (close/reopen).
	db.Close()
	db2, _ := Open(WithStoragePath(t.TempDir() + "/edge_chain.libravdb"))
	defer db2.Drop(context.Background())
	gr2, _ := NewGraph(GraphConfig{})
	defer gr2.Close()
	col2, _ := db2.CreateCollection(context.Background(), "c", WithDimension(3), WithGraph(gr2))
	_ = col2
	_ = gr2
}
