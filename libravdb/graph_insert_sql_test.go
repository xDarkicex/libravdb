package libravdb

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// TestSQLInsertGraphEdge_EpochStageAndRollback verifies that
// INSERT INTO GRAPH_EDGES VALUES (...) within an epoch transaction
// stages the edge, makes it visible via the overlay, and discards it on ROLLBACK.
func TestSQLInsertGraphEdge_EpochStageAndRollback(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/graph_insert_epoch.libravdb"))
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

	// Register an edge kind.
	graph.RegisterEdgeKind("CAUSES", 50)

	// Insert records so we have node IDs to reference.
	col.Insert(context.Background(), "Hypothesis_A", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "Server_Crash", []float32{0, 0, 1}, nil)
	a, _ := db.GetNodeID(context.Background(), "docs", "Hypothesis_A")

	// Verify no edge exists yet.
	edges, _ := gr.Neighbors(a)
	if len(edges) != 0 {
		t.Fatal("edge should not exist before epoch")
	}

	// Begin epoch transaction programmatically.
	epoch, err := db.BeginEpochTx(context.Background())
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}

	// Insert edge via SQL within the epoch session.
	results, err := epoch.Query(context.Background(), "INSERT INTO GRAPH_EDGES VALUES ('Hypothesis_A', 'CAUSES', 'Server_Crash')", nil)
	if err != nil {
		t.Fatalf("INSERT INTO GRAPH_EDGES: %v", err)
	}
	if results.Total != 1 {
		t.Errorf("expected 1 row affected, got %d", results.Total)
	}

	// Within the epoch, the overlay should see the staged edge.
	gtx, _ := epoch.GraphTxn("docs")
	edges, _ = gtx.NeighborsOverlay(a)
	if len(edges) == 0 {
		t.Fatal("overlay should see staged edge after SQL insert")
	}
	t.Logf("✅ SQL INSERT INTO GRAPH_EDGES staged: %d neighbors", len(edges))

	// Live graph still doesn't see it.
	live, _ := gr.Neighbors(a)
	if len(live) != 0 {
		t.Error("live graph should not see staged edge before commit")
	}

	// Rollback via SQL.
	_, err = epoch.Query(context.Background(), "ROLLBACK", nil)
	if err != nil {
		t.Fatalf("ROLLBACK: %v", err)
	}

	// Verify edge is gone from live graph.
	live, _ = gr.Neighbors(a)
	if len(live) != 0 {
		t.Error("edge should not exist after rollback")
	}
	t.Logf("✅ SQL ROLLBACK discarded staged edge")
}

// TestSQLInsertGraphEdge_DirectCommit verifies that INSERT INTO GRAPH_EDGES
// outside an epoch transaction commits immediately.
func TestSQLInsertGraphEdge_DirectCommit(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/graph_insert_direct.libravdb"))
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

	col.Insert(context.Background(), "Hypothesis_A", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "Server_Crash", []float32{0, 0, 1}, nil)
	a, _ := db.GetNodeID(context.Background(), "docs", "Hypothesis_A")

	// Insert edge via SQL outside any epoch.
	results, err := db.Query(context.Background(), "INSERT INTO GRAPH_EDGES VALUES ('Hypothesis_A', 'CAUSES', 'Server_Crash')")
	if err != nil {
		t.Fatalf("INSERT INTO GRAPH_EDGES: %v", err)
	}
	if results.Total != 1 {
		t.Errorf("expected 1 row affected, got %d", results.Total)
	}

	// Verify edge is in the live graph.
	edges, _ := gr.Neighbors(a)
	if len(edges) == 0 {
		t.Fatal("edge should exist in live graph after direct insert")
	}
	t.Logf("✅ direct INSERT INTO GRAPH_EDGES published immediately: %d neighbors", len(edges))
}

// TestSQLInsertGraphEdge_BadKindName verifies error handling for unknown edge kinds.
func TestSQLInsertGraphEdge_BadKindName(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/graph_insert_badkind.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	_, err = db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	_, err = db.Query(context.Background(), "INSERT INTO GRAPH_EDGES VALUES ('A', 'NONEXISTENT_KIND', 'B')")
	if err == nil {
		t.Fatal("expected error for unknown edge kind")
	}
	t.Logf("✅ unknown edge kind correctly rejected: %v", err)
}

// TestSQLInsertGraphEdge_NonexistentNode verifies error handling for
// record IDs that don't exist.
func TestSQLInsertGraphEdge_NonexistentNode(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/graph_insert_badnode.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	_, err = db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	graph.RegisterEdgeKind("CAUSES", 50)

	_, err = db.Query(context.Background(), "INSERT INTO GRAPH_EDGES VALUES ('Ghost_Node', 'CAUSES', 'Server_Crash')")
	if err == nil {
		t.Fatal("expected error for nonexistent source node")
	}
	t.Logf("✅ nonexistent node correctly rejected: %v", err)
}
