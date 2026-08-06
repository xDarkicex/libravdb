package libravdb

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// =============================================================================
// Full SQL integration: COMPUTE LEIDEN → SearchResults
// =============================================================================

func TestComputeLeiden_SQL_BasicExecution(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/leiden_sql.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()

	col, err := db.CreateCollection(context.Background(), "nodes", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	graph.RegisterEdgeKind("LINK", 10)

	// Insert records and label seeds.
	if err := col.Insert(context.Background(), "alpha", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("Insert alpha: %v", err)
	}
	if err := col.Insert(context.Background(), "beta", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("Insert beta: %v", err)
	}
	if err := col.Insert(context.Background(), "gamma", []float32{0, 0, 1}, nil); err != nil {
		t.Fatalf("Insert gamma: %v", err)
	}

	alphaID, _ := db.GetNodeID(context.Background(), "nodes", "alpha")
	betaID, _ := db.GetNodeID(context.Background(), "nodes", "beta")
	gammaID, _ := db.GetNodeID(context.Background(), "nodes", "gamma")

	gr.RegisterVertexLabel(alphaID, "roots")
	gr.RegisterVertexLabel(gammaID, "roots")

	// alpha → beta, alpha → gamma
	baseTxn := gr.BeginTxn()
	baseTxn.AddEdge(alphaID, betaID, 1.0, 10)
	baseTxn.AddEdge(alphaID, gammaID, 1.0, 10)
	baseTxn.Commit(context.Background())

	s, err := db.NewSQLSession(context.Background())
	if err != nil {
		t.Fatalf("NewSQLSession: %v", err)
	}
	defer s.Close()

	// BEGIN EPOCH
	if err := s.Exec("BEGIN EPOCH TRANSACTION"); err != nil {
		t.Fatalf("BEGIN EPOCH: %v", err)
	}

	// Execute COMPUTE LEIDEN via SQL.
	sql := "COMPUTE LEIDEN FROM MATCH (r:roots)-[:LINK*1..2]->(target)"
	results, err := s.Query(sql)
	if err != nil {
		t.Fatalf("COMPUTE LEIDEN query: %v", err)
	}

	if results == nil {
		t.Fatal("results must not be nil")
	}
	if len(results.Results) == 0 {
		t.Fatal("expected at least one result row")
	}

	// Verify metadata fields.
	for _, r := range results.Results {
		if r.Metadata == nil {
			t.Fatal("Metadata must not be nil")
		}
		if _, ok := r.Metadata["node_id"]; !ok {
			t.Error("missing node_id in metadata")
		}
		if _, ok := r.Metadata["community_id"]; !ok {
			t.Error("missing community_id in metadata")
		}
		if _, ok := r.Metadata["record_id"]; !ok {
			t.Error("missing record_id in metadata")
		}
	}

	t.Logf("SQL COMPUTE LEIDEN returned %d rows", len(results.Results))
	for i, r := range results.Results {
		t.Logf("  [%d] node_id=%v community_id=%v record_id=%v",
			i, r.Metadata["node_id"], r.Metadata["community_id"], r.Metadata["record_id"])
	}

	s.Exec("ROLLBACK")
	t.Log("✅ SQL COMPUTE LEIDEN basic execution")
}

func TestComputeLeiden_SQL_NoEpochError(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/leiden_noepoch.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	db.CreateCollection(context.Background(), "nodes", WithDimension(3), WithGraph(gr))
	graph.RegisterEdgeKind("LINK", 10)

	s, err := db.NewSQLSession(context.Background())
	if err != nil {
		t.Fatalf("NewSQLSession: %v", err)
	}
	defer s.Close()

	// COMPUTE LEIDEN without epoch must error.
	_, err = s.Query("COMPUTE LEIDEN FROM MATCH (r:roots)-[:LINK]->(target)")
	if err == nil {
		t.Fatal("expected error for COMPUTE LEIDEN without epoch")
	}
	t.Logf("no-epoch error: %v", err)

	t.Log("✅ COMPUTE LEIDEN without epoch returns error")
}

func TestComputeLeiden_SQL_WithOptions(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/leiden_opts.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()

	col, err := db.CreateCollection(context.Background(), "nodes", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	graph.RegisterEdgeKind("CONNECT", 50)

	col.Insert(context.Background(), "n1", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "n2", []float32{0, 1, 0}, nil)
	col.Insert(context.Background(), "n3", []float32{0, 0, 1}, nil)

	n1, _ := db.GetNodeID(context.Background(), "nodes", "n1")
	n2, _ := db.GetNodeID(context.Background(), "nodes", "n2")
	n3, _ := db.GetNodeID(context.Background(), "nodes", "n3")

	gr.RegisterVertexLabel(n1, "seeds")

	baseTxn := gr.BeginTxn()
	baseTxn.AddEdge(n1, n2, 1.0, 50)
	baseTxn.AddEdge(n2, n3, 1.0, 50)
	baseTxn.Commit(context.Background())

	s, _ := db.NewSQLSession(context.Background())
	defer s.Close()

	s.Exec("BEGIN EPOCH TRANSACTION")

	// With OPTIONS.
	sql := `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:CONNECT*1..2]->(target) OPTIONS (resolution = 0.5, max_levels = 4)`
	results, err := s.Query(sql)
	if err != nil {
		t.Fatalf("COMPUTE LEIDEN with OPTIONS: %v", err)
	}

	if len(results.Results) == 0 {
		t.Fatal("expected results with options")
	}

	// Modularity should be propagated.
	if mod, ok := results.Results[0].Metadata["modularity"]; ok {
		t.Logf("modularity: %v", mod)
	}

	s.Exec("ROLLBACK")
	t.Log("✅ SQL COMPUTE LEIDEN with OPTIONS")
}

func TestComputeLeiden_SQL_SavepointBranch(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/leiden_branch.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()

	col, err := db.CreateCollection(context.Background(), "nodes", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	graph.RegisterEdgeKind("LINK", 10)

	col.Insert(context.Background(), "seed", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "base_target", []float32{0, 1, 0}, nil)

	seedID, _ := db.GetNodeID(context.Background(), "nodes", "seed")
	baseID, _ := db.GetNodeID(context.Background(), "nodes", "base_target")

	gr.RegisterVertexLabel(seedID, "seeds")

	baseTxn := gr.BeginTxn()
	baseTxn.AddEdge(seedID, baseID, 1.0, 10)
	baseTxn.Commit(context.Background())

	s, _ := db.NewSQLSession(context.Background())
	defer s.Close()
	s.Exec("BEGIN EPOCH TRANSACTION")

	// Baseline execution.
	baseline, err := s.Query("COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*1..1]->(target)")
	if err != nil {
		t.Fatalf("baseline: %v", err)
	}
	baselineRows := len(baseline.Results)

	// Savepoint + branch.
	s.Exec("SAVEPOINT branch")
	s.ExecWithParams("INSERT INTO nodes (id, embedding) VALUES ('bridge', '[1,1,1]')", nil)

	// Add an edge from seed to bridge.
	s.Exec("INSERT INTO GRAPH_EDGES VALUES ('seed', 'LINK', 'bridge')")

	// Branch execution — should see bridge.
	branchResult, err := s.Query("COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*1..1]->(target)")
	if err != nil {
		t.Fatalf("branch: %v", err)
	}
	if len(branchResult.Results) <= baselineRows {
		t.Fatalf("branch must have more rows than baseline (%d vs %d)", len(branchResult.Results), baselineRows)
	}

	// Rollback.
	s.Exec("ROLLBACK TO SAVEPOINT branch")

	// Restored execution must match baseline.
	restored, err := s.Query("COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*1..1]->(target)")
	if err != nil {
		t.Fatalf("restored: %v", err)
	}
	if len(restored.Results) != baselineRows {
		t.Fatalf("restored rows (%d) must match baseline (%d)", len(restored.Results), baselineRows)
	}

	s.Exec("ROLLBACK")
	t.Log("✅ SQL COMPUTE LEIDEN savepoint branch workflow")
}

func TestComputeLeiden_SQL_Determinism(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/leiden_determ.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "nodes", WithDimension(3), WithGraph(gr))
	graph.RegisterEdgeKind("LINK", 10)

	col.Insert(context.Background(), "s", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "t1", []float32{0, 1, 0}, nil)
	col.Insert(context.Background(), "t2", []float32{0, 0, 1}, nil)

	sID, _ := db.GetNodeID(context.Background(), "nodes", "s")
	t1ID, _ := db.GetNodeID(context.Background(), "nodes", "t1")
	t2ID, _ := db.GetNodeID(context.Background(), "nodes", "t2")

	gr.RegisterVertexLabel(sID, "seeds")
	txn := gr.BeginTxn()
	txn.AddEdge(sID, t1ID, 1.0, 10)
	txn.AddEdge(sID, t2ID, 1.0, 10)
	txn.Commit(context.Background())

	s, _ := db.NewSQLSession(context.Background())
	defer s.Close()
	s.Exec("BEGIN EPOCH TRANSACTION")

	first, err := s.Query("COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*1..1]->(target)")
	if err != nil {
		t.Fatalf("first query: %v", err)
	}

	for i := 0; i < 5; i++ {
		result, err := s.Query("COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*1..1]->(target)")
		if err != nil {
			t.Fatalf("query %d: %v", i, err)
		}
		if len(result.Results) != len(first.Results) {
			t.Fatalf("query %d: row count differs (%d vs %d)", i, len(result.Results), len(first.Results))
		}
		for j := range result.Results {
			if result.Results[j].ID != first.Results[j].ID {
				t.Fatalf("query %d row %d: ID differs", i, j)
			}
		}
	}

	s.Exec("ROLLBACK")
	t.Log("✅ SQL COMPUTE LEIDEN determinism across 5 calls")
}

func TestComputeLeiden_SQL_InvalidSyntax(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/leiden_syntax.libravdb"))
	defer db.Drop(context.Background())
	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	db.CreateCollection(context.Background(), "nodes", WithDimension(3), WithGraph(gr))
	graph.RegisterEdgeKind("LINK", 10)

	col, _ := db.GetCollection("nodes")
	col.Insert(context.Background(), "s", []float32{1, 0, 0}, nil)
	sID, _ := db.GetNodeID(context.Background(), "nodes", "s")
	gr.RegisterVertexLabel(sID, "seeds")

	s, _ := db.NewSQLSession(context.Background())
	defer s.Close()
	s.Exec("BEGIN EPOCH TRANSACTION")

	rejections := []string{
		"COMPUTE LEIDEN FROM MATCH (a)-[:LINK]->+(b)",              // unbounded +
		"COMPUTE LEIDEN MATCH (a)-[:LINK]->(b)",                    // missing FROM
		"COMPUTE LEIDEN FROM MATCH (a)-[:LINK]->(b) OPTIONS ()",    // empty options
		"COMPUTE LEIDEN FROM MATCH (s:seeds)-[:UNKNOWN]->(target)", // unknown edge kind
	}

	for _, sql := range rejections {
		_, err := s.Query(sql)
		if err == nil {
			t.Errorf("expected error for: %s", sql)
		} else {
			t.Logf("rejected %q: %v", sql, err)
		}
	}

	s.Exec("ROLLBACK")
	t.Log("✅ SQL COMPUTE LEIDEN invalid syntax rejected")
}
