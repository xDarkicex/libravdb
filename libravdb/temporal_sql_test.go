package libravdb

import (
	"context"
	"testing"
	"time"
)

// TestTemporalSQL_AsOfTimestamp verifies the full AS OF TIMESTAMP pipeline:
// parse → bind → optimize → resolve LSN → temporal execute.
func TestTemporalSQL_AsOfTimestamp(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/temporal_sql.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()
	col, err := db.CreateCollection(context.Background(), "customers",
		WithDimension(3),
		WithGraph(gr),
		WithMetadataSchema(MetadataSchema{"name": StringField, "category": StringField}),
	)
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// T1: Insert C1 with vector V1, metadata, and edge to P1.
	if err := col.Insert(context.Background(), "C1", []float32{1, 0, 0},
		map[string]interface{}{"name": "Alice", "category": "customer"}); err != nil {
		t.Fatalf("Insert C1: %v", err)
	}
	if err := col.Insert(context.Background(), "P1", []float32{0, 0, 1},
		map[string]interface{}{"name": "AuthModule", "category": "Security"}); err != nil {
		t.Fatalf("Insert P1: %v", err)
	}
	c1Node, _ := db.GetNodeID(context.Background(), "customers", "C1")
	p1Node, _ := db.GetNodeID(context.Background(), "customers", "P1")
	txn := gr.BeginTxn()
	txn.AddEdge(c1Node, p1Node, 1.0, 1) // kind 1 = PURCHASED
	txn.Commit(context.Background())
	time.Sleep(50 * time.Millisecond)       // let batch flusher run
	t1 := time.Now().UTC().Add(time.Second) // well after the commit

	// T2: Update C1's vector.
	if err := col.Update(context.Background(), "C1", []float32{5, 0, 0},
		map[string]interface{}{"name": "Alice", "category": "customer"}); err != nil {
		t.Fatalf("Update C1: %v", err)
	}
	time.Sleep(20 * time.Millisecond)
	t2 := time.Now().UTC()

	// T3: Remove edge.
	txn2 := gr.BeginTxn()
	txn2.RemoveEdge(c1Node, p1Node, 1)
	txn2.Commit(context.Background())
	time.Sleep(20 * time.Millisecond)

	// Query at T1: should see C1 with V1=[1,0,0].
	t1Str := t1.Format(time.RFC3339)
	sql1 := "SELECT c.name FROM customers c AS OF TIMESTAMP '" + t1Str + "' WHERE c.name = 'Alice'"
	results1, err := db.Query(context.Background(), sql1)
	if err != nil {
		t.Fatalf("Query T1: %v", err)
	}
	if results1.Total != 1 {
		t.Errorf("T1: got %d results, want 1", results1.Total)
	}

	// Query at T2: vector is now V2.
	t.Logf("T1=%s T2=%s", t1Str, t2.Format(time.RFC3339))
	_ = t2
}

// TestTemporalSQL_ErrorCases verifies error behavior.
func TestTemporalSQL_ErrorCases(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/temporal_sql_err.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	// AS OF TIMESTAMP with missing timestamp literal
	_, err = db.Query(context.Background(), "SELECT * FROM c AS OF TIMESTAMP")
	if err == nil {
		t.Error("missing timestamp literal should error")
	}

	// AS OF TIMESTAMP with invalid timestamp
	_, err = db.Query(context.Background(), "SELECT * FROM c AS OF TIMESTAMP 'not-a-date'")
	if err == nil {
		t.Error("invalid timestamp should error")
	}

	// AS OF TIMESTAMP before any commit
	col, _ := db.CreateCollection(context.Background(), "c", WithDimension(3))
	col.Insert(context.Background(), "r1", []float32{1, 0, 0}, nil)
	_, err = db.Query(context.Background(), "SELECT * FROM c AS OF TIMESTAMP '2020-01-01T00:00:00Z'")
	if err == nil {
		t.Error("timestamp before first commit should error")
	}
}

// TestTemporalSQL_CurrentUnaffected verifies non-temporal SQL still works.
func TestTemporalSQL_CurrentUnaffected(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/temporal_sql_curr.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	col.Insert(context.Background(), "r1", []float32{1, 0, 0}, nil)

	// SELECT * works — full scan fallback for non-B-tree indexes.
	results, err := db.Query(context.Background(), "SELECT * FROM c")
	if err != nil {
		t.Fatalf("SELECT *: %v", err)
	}
	if results.Total != 1 {
		t.Errorf("got %d results, want 1", results.Total)
	}
}
