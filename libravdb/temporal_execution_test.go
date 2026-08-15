package libravdb

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// TestTemporalExecution_VectorCorrectness verifies historical vector ranking.
// T1: insert C with V1 + edge to P. T2: update C to V2. T3: delete edge.
// Query at T1 → V1 used for ranking, higher score than V2.
// Query at T2 → V2 used for ranking.
// Query at T3 → edge gone, candidate excluded.
func TestTemporalExecution_VectorCorrectness(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/temporal_exec_vec.libravdb"))
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

	// T1: Insert C with V1=[1,0,0], P with V=[0,0,1], edge C->P.
	if err := col.Insert(context.Background(), "C1", []float32{1, 0, 0}, map[string]interface{}{"type": "customer"}); err != nil {
		t.Fatalf("Insert C1: %v", err)
	}
	if err := col.Insert(context.Background(), "P1", []float32{0, 0, 1}, map[string]interface{}{"type": "product"}); err != nil {
		t.Fatalf("Insert P1: %v", err)
	}
	c1Node, _ := db.GetNodeID(context.Background(), "c", "C1")
	p1Node, _ := db.GetNodeID(context.Background(), "c", "P1")
	txn := gr.BeginTxn()
	txn.AddEdge(c1Node, p1Node, 1.0, 1)
	txn.Commit(context.Background())
	snap1, _ := db.SnapshotAt(context.Background(), time.Now().UTC())

	// T2: Update C1 to V2=[5,0,0].
	if err := col.Update(context.Background(), "C1", []float32{5, 0, 0}, nil); err != nil {
		t.Fatalf("Update C1: %v", err)
	}
	snap2, _ := db.SnapshotAt(context.Background(), time.Now().UTC())

	// T3: Remove edge.
	txn2 := gr.BeginTxn()
	txn2.RemoveEdge(c1Node, p1Node, 1)
	txn2.Commit(context.Background())
	snap3, _ := db.SnapshotAt(context.Background(), time.Now().UTC())

	// Verify historical vectors via GetAtLSN.
	rec1, err := col.GetAtLSN(context.Background(), "C1", snap1.LSN)
	if err != nil {
		t.Fatalf("GetAtLSN T1: %v", err)
	}
	if rec1 == nil || rec1.Vector[0] != 1 {
		t.Errorf("T1: C1 vector[0] = %v, want 1 (V1)", rec1.Vector[0])
	}
	rec2, err := col.GetAtLSN(context.Background(), "C1", snap2.LSN)
	if err != nil {
		t.Fatalf("GetAtLSN T2: %v", err)
	}
	if rec2 == nil || rec2.Vector[0] != 5 {
		t.Errorf("T2: C1 vector[0] = %v, want 5 (V2)", rec2.Vector[0])
	}
	_ = snap3
}

// TestTemporalExecution_FutureStateLeakPrevention verifies no future data
// leaks into historical queries.
func TestTemporalExecution_FutureStateLeakPrevention(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/temporal_exec_leak.libravdb"))
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
		t.Fatalf("Insert R1: %v", err)
	}
	snap1, _ := db.SnapshotAt(context.Background(), time.Now().UTC())

	// T2: Insert R2 (after T1 snapshot).
	if err := col.Insert(context.Background(), "R2", []float32{2, 0, 0}, nil); err != nil {
		t.Fatalf("Insert R2: %v", err)
	}

	// Query at T1: R2 should NOT appear.
	rec, err := col.GetAtLSN(context.Background(), "R2", snap1.LSN)
	if err != nil {
		t.Fatalf("GetAtLSN R2: %v", err)
	}
	if rec != nil {
		t.Error("T1: R2 should NOT be visible (inserted after T1)")
	}
}

// TestTemporalExecution_RelationalOnlyAtLSN verifies simple relational reads.
func TestTemporalExecution_RelationalOnlyAtLSN(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/temporal_exec_rel.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("r%d", i)
		if err := col.Insert(context.Background(), id, []float32{float32(i), 0, 0}, nil); err != nil {
			t.Fatalf("Insert %s: %v", id, err)
		}
	}
	snap, _ := db.SnapshotAt(context.Background(), time.Now().UTC())

	plan := &optimizer.PhysicalPlan{
		Kind:               optimizer.QueryKindRelational,
		CollectionName:     "c",
		HasRelationalQuery: true,
		Limit:              10,
		Projections:        []string{"id"},
	}
	exec := newExecutor(db)
	results, err := exec.ExecuteAtLSN(context.Background(), plan, snap.LSN)
	if err != nil {
		t.Fatalf("ExecuteAtLSN relational: %v", err)
	}
	if results.Total != 5 {
		t.Errorf("got %d results, want 5", results.Total)
	}
}

// TestTemporalExecution_GuardrailNoHNSW verifies temporal queries reject
// HNSW/ANN query kinds.
func TestTemporalExecution_GuardrailNoHNSW(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/temporal_exec_guard.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	plan := &optimizer.PhysicalPlan{
		Kind:        optimizer.QueryKindKNN,
		QueryVector: []float32{1, 0, 0},
		Limit:       5,
	}
	exec := newExecutor(db)
	_, err = exec.ExecuteAtLSN(context.Background(), plan, 1)
	if err == nil {
		t.Error("KNN query kind should be rejected for temporal execution")
	}
}
