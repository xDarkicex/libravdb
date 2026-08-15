package libravdb

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// =============================================================================
// CTE execution integration tests
// =============================================================================

func TestLeidenCTE_Execute_BasicJoin(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/cte_exec.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()

	col, err := db.CreateCollection(context.Background(), "documents", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	graph.RegisterEdgeKind("LINK", 10)

	// Insert seed records and graph edges.
	col.Insert(context.Background(), "seed1", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "seed2", []float32{1, 1, 0}, nil)
	col.Insert(context.Background(), "docA", []float32{0, 1, 0}, nil)
	col.Insert(context.Background(), "docB", []float32{0, 0, 1}, nil)

	s1, _ := db.GetNodeID(context.Background(), "documents", "seed1")
	s2, _ := db.GetNodeID(context.Background(), "documents", "seed2")
	dA, _ := db.GetNodeID(context.Background(), "documents", "docA")
	dB, _ := db.GetNodeID(context.Background(), "documents", "docB")

	gr.RegisterVertexLabel(s1, "seeds")
	gr.RegisterVertexLabel(s2, "seeds")

	txn := gr.BeginTxn()
	txn.AddEdge(s1, dA, 1.0, 10)
	txn.AddEdge(s2, dB, 1.0, 10)
	txn.Commit(context.Background())

	s, err := db.NewSQLSession(context.Background())
	if err != nil {
		t.Fatalf("NewSQLSession: %v", err)
	}
	defer s.Close()

	// BEGIN EPOCH
	if err := s.Exec("BEGIN EPOCH TRANSACTION"); err != nil {
		t.Fatalf("BEGIN EPOCH: %v", err)
	}

	// Execute CTE SELECT.
	sql := `WITH local_clusters AS (
    COMPUTE LEIDEN FROM MATCH
    (s:seeds)-[:LINK*1..1]->(target)
)
SELECT d.title, c.community_id
FROM documents d
JOIN local_clusters c ON d.node_id = c.node_id`

	results, err := s.Query(sql)
	if err != nil {
		t.Fatalf("CTE query: %v", err)
	}

	if results == nil {
		t.Fatal("results must not be nil")
	}
	t.Logf("CTE JOIN returned %d rows", len(results.Results))

	// Each matched terminal (docA, docB) should appear joined with its community.
	if len(results.Results) == 0 {
		t.Fatal("expected at least one joined row")
	}

	for i, r := range results.Results {
		t.Logf("  [%d] id=%q metadata=%v", i, r.ID, r.Metadata)
		if r.Metadata == nil {
			t.Errorf("row[%d]: Metadata must not be nil", i)
		}
		if _, ok := r.Metadata["community_id"]; !ok {
			t.Errorf("row[%d]: missing community_id", i)
		}
	}

	s.Exec("ROLLBACK")
	t.Log("✅ CTE basic JOIN execution")
}

func TestLeidenCTE_Execute_SnapshotIsolation(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/cte_snap.libravdb"))
	defer db.Drop(context.Background())
	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "documents", WithDimension(3), WithGraph(gr))
	graph.RegisterEdgeKind("LINK", 10)

	col.Insert(context.Background(), "s1", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "pre", []float32{0, 1, 0}, nil)
	s1, _ := db.GetNodeID(context.Background(), "documents", "s1")
	pre, _ := db.GetNodeID(context.Background(), "documents", "pre")
	gr.RegisterVertexLabel(s1, "seeds")

	txn := gr.BeginTxn()
	txn.AddEdge(s1, pre, 1.0, 10)
	txn.Commit(context.Background())

	s, _ := db.NewSQLSession(context.Background())
	defer s.Close()
	s.Exec("BEGIN EPOCH TRANSACTION")

	// Stage a post-epoch record.
	epoch := s.epoch
	epoch.Insert(context.Background(), "documents", "post", []float32{1, 1, 1}, nil)
	postID, _ := epoch.LookupNodeID(context.Background(), "documents", "post")
	epoch.AddGraphEdge("documents", s1, postID, 1.0, 10)

	sql := `WITH local_clusters AS (
    COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*1..1]->(target)
)
SELECT d.title, c.community_id
FROM documents d
JOIN local_clusters c ON d.node_id = c.node_id`

	results, err := s.Query(sql)
	if err != nil {
		t.Fatalf("CTE query: %v", err)
	}

	// Both pre-epoch "pre" and staged "post" should appear.
	foundPre, foundPost := false, false
	for _, r := range results.Results {
		switch r.ID {
		case "pre":
			foundPre = true
		case "post":
			foundPost = true
		}
	}
	if !foundPre {
		t.Error("pre-epoch record missing from CTE JOIN")
	}
	if !foundPost {
		t.Error("staged record missing from CTE JOIN")
	}

	s.Exec("ROLLBACK")
	t.Log("✅ CTE snapshot/staged overlay")
}

func TestLeidenCTE_Execute_NoEpochError(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/cte_noepoch.libravdb"))
	defer db.Drop(context.Background())
	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	db.CreateCollection(context.Background(), "documents", WithDimension(3), WithGraph(gr))
	graph.RegisterEdgeKind("LINK", 10)

	s, _ := db.NewSQLSession(context.Background())
	defer s.Close()

	sql := `WITH c AS (COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target))
SELECT 1 FROM documents d JOIN c x ON d.node_id = x.node_id`
	_, err := s.Query(sql)
	if err == nil {
		t.Fatal("expected error for CTE without epoch")
	}
	t.Logf("no-epoch CTE rejected: %v", err)

	t.Log("✅ CTE without epoch rejected")
}
