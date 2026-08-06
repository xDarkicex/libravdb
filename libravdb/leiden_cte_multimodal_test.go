package libravdb

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// This acceptance test covers the CTE path's non-Leiden SQL semantics:
// parser-backed WHERE filtering, VECTOR_DISTANCE projection with @params, and
// numeric ORDER BY before LIMIT.
func TestLeidenCTE_Execute_MultimodalProjectionFilterOrder(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/cte_multimodal.libravdb"))
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
	const edgeName = "CTE_MM_CONNECTED_7"
	if !graph.RegisterEdgeKind(edgeName, 61) && graph.ResolveEdgeKind(edgeName) != 61 {
		t.Fatalf("%s registration conflict", edgeName)
	}

	rows := []struct {
		id    string
		vec   []float32
		title string
	}{
		{"seed", []float32{1, 0, 0}, "seed"},
		{"near", []float32{1, 0, 0}, "near"},
		{"mid", []float32{0.5, 0.5, 0}, "mid"},
		{"far", []float32{0, 1, 0}, "far"},
	}
	for _, row := range rows {
		if err := col.Insert(context.Background(), row.id, row.vec, map[string]interface{}{"title": row.title}); err != nil {
			t.Fatalf("Insert %s: %v", row.id, err)
		}
	}
	seed, _ := db.GetNodeID(context.Background(), "documents", "seed")
	near, _ := db.GetNodeID(context.Background(), "documents", "near")
	mid, _ := db.GetNodeID(context.Background(), "documents", "mid")
	far, _ := db.GetNodeID(context.Background(), "documents", "far")
	gr.RegisterVertexLabel(seed, "seeds")
	txn := gr.BeginTxn()
	txn.AddEdge(seed, near, 1, 61)
	txn.AddEdge(seed, mid, 1, 61)
	txn.AddEdge(seed, far, 1, 61)
	if err := txn.Commit(context.Background()); err != nil {
		t.Fatalf("graph commit: %v", err)
	}

	s, err := db.NewSQLSession(context.Background())
	if err != nil {
		t.Fatalf("NewSQLSession: %v", err)
	}
	defer s.Close()
	if err := s.Exec("BEGIN EPOCH TRANSACTION"); err != nil {
		t.Fatalf("BEGIN EPOCH: %v", err)
	}

	query := `WITH local_clusters AS (
    COMPUTE LEIDEN FROM MATCH (s:seeds)-[:CTE_MM_CONNECTED_7*1..1]->(target)
)
SELECT d.title,
       c.community_id,
       VECTOR_DISTANCE(d.embedding, @query_vec) AS semantic_score
FROM documents d
JOIN local_clusters c ON d.node_id = c.node_id
WHERE c.community_id = 1
ORDER BY semantic_score ASC
LIMIT 2`
	result, err := s.QueryWithParams(query, QueryParams{"query_vec": []float32{1, 0, 0}})
	if err != nil {
		t.Fatalf("multimodal CTE query: %v", err)
	}
	if result.Total == 0 {
		t.Fatal("expected filtered CTE rows")
	}
	if result.Total > 2 {
		t.Fatalf("LIMIT 2 ignored: got %d rows", result.Total)
	}
	for i, row := range result.Results {
		if _, ok := row.Metadata["semantic_score"]; !ok {
			t.Fatalf("row %d missing semantic_score: %#v", i, row.Metadata)
		}
		if row.Metadata["community_id"] != uint64(1) && row.Metadata["community_id"] != float64(1) {
			t.Fatalf("row %d escaped WHERE filter: %#v", i, row.Metadata)
		}
	}
	if len(result.Results) > 1 {
		prev := result.Results[0].Metadata["semantic_score"].(float64)
		for i := 1; i < len(result.Results); i++ {
			current := result.Results[i].Metadata["semantic_score"].(float64)
			if current < prev {
				t.Fatalf("ORDER BY semantic_score ASC violated: %v then %v", prev, current)
			}
			prev = current
		}
	}
	if err := s.Exec("ROLLBACK"); err != nil {
		t.Fatalf("ROLLBACK: %v", err)
	}
}
