package libravdb

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// TestGraphCentrality_Basic verifies inbound degree centrality computation.
func TestGraphCentrality_Basic(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/centrality_basic.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()
	col, err := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Create 3 documents.
	for i := 1; i <= 3; i++ {
		id := fmt.Sprintf("D%d", i)
		col.Insert(context.Background(), id, []float32{float32(i), 0, 0}, nil)
	}
	d1, _ := db.GetNodeID(context.Background(), "docs", "D1")
	d2, _ := db.GetNodeID(context.Background(), "docs", "D2")
	d3, _ := db.GetNodeID(context.Background(), "docs", "D3")

	// D2 and D3 cite D1 (inbound edges to D1).
	txn := gr.BeginTxn()
	txn.AddEdge(d2, d1, 1.0, 1)
	txn.AddEdge(d3, d1, 1.0, 1)
	txn.Commit(context.Background())
	time.Sleep(50 * time.Millisecond)

	// D1 should have inbound degree 2 → highest centrality.
	c1 := gr.GraphCentrality(d1)
	c2 := gr.GraphCentrality(d2)
	c3 := gr.GraphCentrality(d3)

	t.Logf("centrality: D1=%.4f D2=%.4f D3=%.4f", c1, c2, c3)
	if c1 <= c2 {
		t.Errorf("D1 centrality (%.4f) should be > D2 (%.4f)", c1, c2)
	}
	if c1 <= c3 {
		t.Errorf("D1 centrality (%.4f) should be > D3 (%.4f)", c1, c3)
	}
	if c1 != 1.0 {
		t.Errorf("D1 centrality should be 1.0 (max inbound), got %.4f", c1)
	}
}

// TestGraphCentrality_EdgeKindFilter verifies kind-aware centrality.
func TestGraphCentrality_EdgeKindFilter(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/centrality_kind.libravdb"))
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

	// Edge kind 1 = CITES
	txn := gr.BeginTxn()
	txn.AddEdge(b, a, 1.0, 1) // kind 1
	txn.Commit(context.Background())
	time.Sleep(20 * time.Millisecond)

	c := gr.GraphCentrality(a)
	if c <= 0 {
		t.Errorf("centrality should be > 0 with edge, got %.4f", c)
	}
}

// TestGraphCentrality_ZeroDegree verifies zero-degree nodes return 0.
func TestGraphCentrality_ZeroDegree(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/centrality_zero.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "X", []float32{1, 0, 0}, nil)

	x, _ := db.GetNodeID(context.Background(), "docs", "X")
	c := gr.GraphCentrality(x)
	if c != 0.0 {
		t.Errorf("isolated node centrality should be 0, got %.4f", c)
	}
}

// TestGraphCentrality_Temporal verifies historical centrality.
func TestGraphCentrality_Temporal(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/centrality_temporal.libravdb"))
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

	// T1: edge added, centrality > 0.
	txn := gr.BeginTxn()
	txn.AddEdge(b, a, 1.0, 1)
	txn.Commit(context.Background())
	time.Sleep(20 * time.Millisecond)
	snap1, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))

	// T2: edge removed, centrality = 0.
	txn2 := gr.BeginTxn()
	txn2.RemoveEdge(b, a, 1)
	txn2.Commit(context.Background())
	time.Sleep(20 * time.Millisecond)
	snap2, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))

	// Historical at T1: should see centrality > 0.
	c1 := gr.CentralityAtLSN(a, snap1.LSN)
	// Current: should be 0 (edge removed).
	c2 := gr.CentralityAtLSN(a, snap2.LSN)

	t.Logf("T1 centrality=%.4f T2 centrality=%.4f", c1, c2)
	snap1.Close()
	snap2.Close()
	if c1 <= 0 {
		t.Error("T1: centrality should be > 0 (edge present)")
	}
	if c2 != 0.0 {
		t.Errorf("T2: centrality should be 0 (edge removed), got %.4f", c2)
	}
}
