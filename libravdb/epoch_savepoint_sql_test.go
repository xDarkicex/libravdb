package libravdb

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// =============================================================================
// Test A: Full agent branch workflow
// =============================================================================

func TestSession_Savepoint_AgentBranchWorkflow(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/savepoint_branch.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "Target", []float32{1, 0, 0}, nil)
	graph.RegisterEdgeKind("LINKS", 30)

	s, _ := db.NewSQLSession(context.Background())
	defer s.Close()

	// BEGIN EPOCH
	s.Exec("BEGIN EPOCH TRANSACTION")

	// Outer base insert.
	s.ExecWithParams("INSERT INTO docs (id, embedding) VALUES ('BaseHypothesis', '[1,0,0]')", nil)
	t.Log("Phase 1: inserted outer BaseHypothesis")

	// SAVEPOINT candidate_a
	if err := s.Exec("SAVEPOINT candidate_a"); err != nil {
		t.Fatalf("SAVEPOINT: %v", err)
	}

	// Candidate A mutations.
	s.ExecWithParams("INSERT INTO docs (id, embedding) VALUES ('CandidateA', '[0,1,0]')", nil)
	s.Exec("INSERT INTO GRAPH_EDGES VALUES ('CandidateA', 'LINKS', 'Target')")

	// Verify A is visible via VECTOR_DISTANCE query.
	recs, _ := s.QueryWithParams("SELECT id FROM docs ORDER BY VECTOR_DISTANCE(embedding, '[0,1,0]') ASC LIMIT 10", nil)
	foundA := false
	for _, r := range recs.Results {
		if r.ID == "CandidateA" {
			foundA = true
		}
	}
	if !foundA {
		t.Fatal("CandidateA must be visible in branch")
	}
	t.Log("Phase 2: CandidateA visible in branch A ✓")

	// ROLLBACK TO candidate_a
	if err := s.Exec("ROLLBACK TO SAVEPOINT candidate_a"); err != nil {
		t.Fatalf("ROLLBACK TO: %v", err)
	}

	// Verify A is gone but BaseHypothesis remains.
	recs2, _ := s.QueryWithParams("SELECT id FROM docs ORDER BY VECTOR_DISTANCE(embedding, '[1,0,0]') ASC LIMIT 10", nil)
	foundA2 := false
	foundBase := false
	for _, r := range recs2.Results {
		if r.ID == "CandidateA" {
			foundA2 = true
		}
		if r.ID == "BaseHypothesis" {
			foundBase = true
		}
	}
	if foundA2 {
		t.Fatal("CandidateA must be absent after rollback")
	}
	if !foundBase {
		t.Fatal("BaseHypothesis must survive rollback")
	}
	t.Log("Phase 3: CandidateA absent, BaseHypothesis present after rollback ✓")

	// SAVEPOINT candidate_b
	s.Exec("SAVEPOINT candidate_b")
	s.ExecWithParams("INSERT INTO docs (id, embedding) VALUES ('CandidateB', '[0,0,1]')", nil)
	s.Exec("INSERT INTO GRAPH_EDGES VALUES ('CandidateB', 'LINKS', 'Target')")

	recs3, _ := s.QueryWithParams("SELECT id FROM docs ORDER BY VECTOR_DISTANCE(embedding, '[1,0,0]') ASC LIMIT 10", nil)
	foundB := false
	for _, r := range recs3.Results {
		if r.ID == "CandidateB" {
			foundB = true
		}
	}
	if !foundB {
		t.Fatal("CandidateB must be visible in branch B")
	}
	t.Log("Phase 4: CandidateB visible in branch B ✓")

	// RELEASE candidate_b
	s.Exec("RELEASE SAVEPOINT candidate_b")

	// COMMIT
	if err := s.Exec("COMMIT"); err != nil {
		t.Fatalf("COMMIT: %v", err)
	}
	t.Log("Phase 5: COMMIT succeeded")

	// Verify durable state: BaseHypothesis + CandidateB, but NOT CandidateA.
	col2, _ := db.GetCollection("docs")
	_, errA := col2.Get(context.Background(), "CandidateA")
	if errA == nil {
		t.Fatal("CandidateA must not exist after commit")
	}
	_, errBase := col2.Get(context.Background(), "BaseHypothesis")
	if errBase != nil {
		t.Fatal("BaseHypothesis must exist after commit")
	}
	_, errB := col2.Get(context.Background(), "CandidateB")
	if errB != nil {
		t.Fatal("CandidateB must exist after commit")
	}
	t.Log("Phase 6: BaseHypothesis + CandidateB durable, CandidateA absent ✓")
	t.Log("✅ test A: full agent branch workflow")
}

// =============================================================================
// Test B: Nested savepoints
// =============================================================================

func TestSession_Savepoint_NestedSavepoints(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/savepoint_nested.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))

	s, _ := db.NewSQLSession(context.Background())
	defer s.Close()

	s.Exec("BEGIN EPOCH TRANSACTION")
	s.ExecWithParams("INSERT INTO docs (id, embedding) VALUES ('outer', '[1,0,0]')", nil)

	s.Exec("SAVEPOINT s1")
	s.ExecWithParams("INSERT INTO docs (id, embedding) VALUES ('one', '[0,1,0]')", nil)

	s.Exec("SAVEPOINT s2")
	s.ExecWithParams("INSERT INTO docs (id, embedding) VALUES ('two', '[0,0,1]')", nil)

	// All three visible.
	recs, _ := s.QueryWithParams("SELECT id FROM docs ORDER BY VECTOR_DISTANCE(embedding, '[1,0,0]') ASC LIMIT 10", nil)
	if countResults(recs, "outer", "one", "two") != 3 {
		t.Fatal("outer, one, two must all be visible")
	}

	// ROLLBACK TO s2
	s.Exec("ROLLBACK TO SAVEPOINT s2")
	recs2, _ := s.QueryWithParams("SELECT id FROM docs ORDER BY VECTOR_DISTANCE(embedding, '[1,0,0]') ASC LIMIT 10", nil)
	if countResults(recs2, "two") > 0 {
		t.Fatal("two must be absent after rollback to s2")
	}
	if countResults(recs2, "outer", "one") != 2 {
		t.Fatal("outer and one must be visible")
	}
	t.Log("Phase 1: ROLLBACK TO s2: two absent, outer+one present ✓")

	// ROLLBACK TO s1
	s.Exec("ROLLBACK TO SAVEPOINT s1")
	recs3, _ := s.QueryWithParams("SELECT id FROM docs ORDER BY VECTOR_DISTANCE(embedding, '[1,0,0]') ASC LIMIT 10", nil)
	if countResults(recs3, "one") > 0 || countResults(recs3, "two") > 0 {
		t.Fatal("one and two must be absent after rollback to s1")
	}
	if countResults(recs3, "outer") != 1 {
		t.Fatal("outer must be visible")
	}
	t.Log("Phase 2: ROLLBACK TO s1: only outer visible ✓")

	s.Exec("COMMIT")
	t.Log("✅ test B: nested savepoints")
}

func countResults(r *SearchResults, ids ...string) int {
	if r == nil {
		return 0
	}
	c := 0
	for _, want := range ids {
		for _, got := range r.Results {
			if got.ID == want {
				c++
				break
			}
		}
	}
	return c
}

// =============================================================================
// Test C: Last-write-wins undo
// =============================================================================

func TestSession_Savepoint_LastWriteWinsUndo(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/savepoint_lww.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	if _, err := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithMetric(L2Distance), WithGraph(gr)); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Insert initial records: rec=[1,0,0], competitor=[2.5,0,0].
	col, err := db.GetCollection("docs")
	if err != nil {
		t.Fatalf("GetCollection: %v", err)
	}
	if err := col.Insert(context.Background(), "rec", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("Insert rec: %v", err)
	}
	if err := col.Insert(context.Background(), "competitor", []float32{2.5, 0, 0}, nil); err != nil {
		t.Fatalf("Insert competitor: %v", err)
	}

	s, err := db.NewSQLSession(context.Background())
	if err != nil {
		t.Fatalf("NewSQLSession: %v", err)
	}
	defer s.Close()

	if err := s.Exec("BEGIN EPOCH TRANSACTION"); err != nil {
		t.Fatalf("BEGIN EPOCH: %v", err)
	}

	if err := s.Exec("SAVEPOINT p"); err != nil {
		t.Fatalf("SAVEPOINT: %v", err)
	}

	// Multiple vector writes to same record. SQL UPDATE currently updates only
	// metadata, not the stored vector; use EpochTx.Update for vector mutation.
	if err := s.epoch.Update(context.Background(), "docs", "rec", []float32{2, 0, 0}, nil); err != nil {
		t.Fatalf("epoch.Update to [2,0,0]: %v", err)
	}
	if err := s.epoch.Update(context.Background(), "docs", "rec", []float32{3, 0, 0}, nil); err != nil {
		t.Fatalf("epoch.Update to [3,0,0]: %v", err)
	}

	// Before rollback: ranking by L2 distance to [3,0,0] should put rec first
	// since rec was updated to [3,0,0] (distance 0) and competitor is at
	// [2.5,0,0] (distance 0.5).
	recs, err := s.QueryWithParams("SELECT id, VECTOR_DISTANCE(embedding, '[3,0,0]') AS d FROM docs ORDER BY d ASC LIMIT 1", nil)
	if err != nil {
		t.Fatalf("Query before rollback: %v", err)
	}
	if len(recs.Results) == 0 || recs.Results[0].ID != "rec" {
		if len(recs.Results) > 0 {
			t.Fatalf("before rollback: rec (updated to [3,0,0]) must be closest to [3,0,0], got top=%q", recs.Results[0].ID)
		}
		t.Fatal("before rollback: no results from ranking query")
	}
	t.Log("Phase 1: ranking confirms latest update [3,0,0] visible ✓")

	// ROLLBACK TO p
	if err := s.Exec("ROLLBACK TO SAVEPOINT p"); err != nil {
		t.Fatalf("ROLLBACK TO: %v", err)
	}

	// After rollback: rec=[1,0,0], competitor=[2.5,0,0].
	// Ranking by distance to [3,0,0] should put competitor closer (|2.5-3|=0.5 < |1-3|=2).
	recs2, err := s.QueryWithParams("SELECT id, VECTOR_DISTANCE(embedding, '[3,0,0]') AS d FROM docs ORDER BY d ASC LIMIT 1", nil)
	if err != nil {
		t.Fatalf("Query after rollback: %v", err)
	}
	if len(recs2.Results) == 0 {
		t.Fatal("after rollback: no results from ranking query")
	}
	// competitor [2.5,0,0] is closer to [3,0,0] than rec [1,0,0].
	if recs2.Results[0].ID != "competitor" {
		t.Fatalf("after rollback: expected competitor to rank closest to [3,0,0], got %q", recs2.Results[0].ID)
	}
	t.Log("Phase 2: ranking proves pre-savepoint vector restored ✓")

	// Commit and reopen: verify persisted rec remains at [1,0,0].
	if err := s.Exec("COMMIT"); err != nil {
		t.Fatalf("COMMIT: %v", err)
	}

	// Open fresh session.
	s2, err := db.NewSQLSession(context.Background())
	if err != nil {
		t.Fatalf("NewSQLSession 2: %v", err)
	}
	defer s2.Close()

	recs3, err := s2.QueryWithParams("SELECT id, VECTOR_DISTANCE(embedding, '[3,0,0]') AS d FROM docs ORDER BY d ASC LIMIT 1", nil)
	if err != nil {
		t.Fatalf("Query after commit: %v", err)
	}
	if len(recs3.Results) == 0 {
		t.Fatal("after commit: no results from ranking query")
	}
	if recs3.Results[0].ID != "competitor" {
		t.Fatalf("after commit: expected competitor to rank closest, got %q (rolled-back update persisted?)", recs3.Results[0].ID)
	}
	t.Log("Phase 3: commit/reopen proves rolled-back update absent ✓")
	t.Log("✅ test C: last-write-wins undo")
}

// =============================================================================
// Test D: Graph operation cancellation
// =============================================================================

func TestSession_Savepoint_GraphOpCancellation(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/savepoint_graph.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, err := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := col.Insert(context.Background(), "A", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("Insert A: %v", err)
	}
	if err := col.Insert(context.Background(), "B", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("Insert B: %v", err)
	}
	if err := col.Insert(context.Background(), "C", []float32{0, 0, 1}, nil); err != nil {
		t.Fatalf("Insert C: %v", err)
	}
	graph.RegisterEdgeKind("E", 1)

	// Base edge A→B via a direct graph Txn (no SQL syntax for pre-epoch setup).
	a, err := db.GetNodeID(context.Background(), "docs", "A")
	if err != nil {
		t.Fatalf("GetNodeID A: %v", err)
	}
	b, err := db.GetNodeID(context.Background(), "docs", "B")
	if err != nil {
		t.Fatalf("GetNodeID B: %v", err)
	}
	c, err := db.GetNodeID(context.Background(), "docs", "C")
	if err != nil {
		t.Fatalf("GetNodeID C: %v", err)
	}
	baseTxn := gr.BeginTxn()
	if err := baseTxn.AddEdge(a, b, 1.0, 1); err != nil {
		t.Fatalf("base AddEdge: %v", err)
	}
	if err := baseTxn.Commit(context.Background()); err != nil {
		t.Fatalf("base Commit: %v", err)
	}

	s, err := db.NewSQLSession(context.Background())
	if err != nil {
		t.Fatalf("NewSQLSession: %v", err)
	}
	defer s.Close()

	if err := s.Exec("BEGIN EPOCH TRANSACTION"); err != nil {
		t.Fatalf("BEGIN EPOCH: %v", err)
	}

	// Remove base edge A→B, add A→C via EpochTx wrappers.
	if err := s.epoch.RemoveGraphEdge("docs", a, b, 1); err != nil {
		t.Fatalf("RemoveGraphEdge A→B: %v", err)
	}
	if err := s.epoch.AddGraphEdge("docs", a, c, 1.0, 1); err != nil {
		t.Fatalf("AddGraphEdge A→C: %v", err)
	}

	// Verify via overlay: A has C but not B.
	gtx, err := s.epoch.GraphTxn("docs")
	if err != nil {
		t.Fatalf("GraphTxn: %v", err)
	}
	neighbors, err := gtx.NeighborsOverlay(a)
	if err != nil {
		t.Fatalf("NeighborsOverlay: %v", err)
	}
	hasB, hasC := false, false
	for _, nb := range neighbors {
		if nb.Target == b {
			hasB = true
		}
		if nb.Target == c {
			hasC = true
		}
	}
	if hasB || !hasC {
		t.Fatalf("Phase 1: A should have C but not B, got hasB=%v hasC=%v", hasB, hasC)
	}
	t.Log("Phase 1: A→B removed, A→C added in epoch ✓")

	// SAVEPOINT p
	if err := s.Exec("SAVEPOINT p"); err != nil {
		t.Fatalf("SAVEPOINT p: %v", err)
	}

	// Branch mutations: remove A→C, add A→D.
	// Public SQL for inserting D (record).
	if err := s.ExecWithParams("INSERT INTO docs (id, embedding) VALUES ('D', '[1,1,1]')", nil); err != nil {
		t.Fatalf("INSERT D: %v", err)
	}
	d, err := s.epoch.LookupNodeID(context.Background(), "docs", "D")
	if err != nil {
		t.Fatalf("LookupNodeID D: %v", err)
	}
	// Use EpochTx wrappers for graph mutations without SQL syntax.
	if err := s.epoch.RemoveGraphEdge("docs", a, c, 1); err != nil {
		t.Fatalf("RemoveGraphEdge A→C: %v", err)
	}
	if err := s.epoch.AddGraphEdge("docs", a, d, 1.0, 1); err != nil {
		t.Fatalf("AddGraphEdge A→D: %v", err)
	}

	// Verify branch: A has D but not C.
	gtx2, err := s.epoch.GraphTxn("docs")
	if err != nil {
		t.Fatalf("GraphTxn after branch: %v", err)
	}
	neighbors2, err := gtx2.NeighborsOverlay(a)
	if err != nil {
		t.Fatalf("NeighborsOverlay branch: %v", err)
	}
	hasC2, hasD2 := false, false
	for _, nb := range neighbors2 {
		if nb.Target == c {
			hasC2 = true
		}
		if nb.Target == d {
			hasD2 = true
		}
	}
	if hasC2 || !hasD2 {
		t.Fatalf("Phase 2: A should have D but not C in branch, got hasC=%v hasD=%v", hasC2, hasD2)
	}
	t.Log("Phase 2: A→C removed, A→D added in branch ✓")

	// ROLLBACK TO p
	if err := s.Exec("ROLLBACK TO SAVEPOINT p"); err != nil {
		t.Fatalf("ROLLBACK TO p: %v", err)
	}

	// After rollback: the graph Txn was rebuilt; fetch fresh.
	gtx3, err := s.epoch.GraphTxn("docs")
	if err != nil {
		t.Fatalf("GraphTxn after rollback: %v", err)
	}
	neighbors3, err := gtx3.NeighborsOverlay(a)
	if err != nil {
		t.Fatalf("NeighborsOverlay after rollback: %v", err)
	}
	hasB3, hasC3, hasD3 := false, false, false
	for _, nb := range neighbors3 {
		if nb.Target == b {
			hasB3 = true
		}
		if nb.Target == c {
			hasC3 = true
		}
		if nb.Target == d {
			hasD3 = true
		}
	}
	if hasB3 || !hasC3 || hasD3 {
		t.Fatalf("after rollback: hasB=%v hasC=%v hasD=%v (want hasB=false hasC=true hasD=false)", hasB3, hasC3, hasD3)
	}
	t.Log("Phase 3: A→C restored, A→B still removed, A→D absent after rollback ✓")

	if err := s.Exec("ROLLBACK"); err != nil {
		t.Fatalf("outer ROLLBACK: %v", err)
	}
	t.Log("✅ test D: graph operation cancellation")
}

// =============================================================================
// Test E: Error/state-machine tests
// =============================================================================

func TestSession_Savepoint_ErrorStates(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/savepoint_errors.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))

	s, _ := db.NewSQLSession(context.Background())
	defer s.Close()

	// SAVEPOINT outside epoch.
	err := s.Exec("SAVEPOINT x")
	if !errors.Is(err, ErrSavepointOutsideEpoch) {
		t.Fatalf("SAVEPOINT outside epoch: want ErrSavepointOutsideEpoch, got %v", err)
	}
	t.Log("SAVEPOINT outside epoch rejected ✓")

	// BEGIN EPOCH
	s.Exec("BEGIN EPOCH TRANSACTION")
	s.Exec("SAVEPOINT a")

	// Duplicate name.
	err = s.Exec("SAVEPOINT a")
	if !errors.Is(err, ErrSavepointExists) {
		t.Fatalf("duplicate SAVEPOINT: want ErrSavepointExists, got %v", err)
	}
	t.Log("duplicate SAVEPOINT rejected ✓")

	// Rollback to unknown name.
	err = s.Exec("ROLLBACK TO SAVEPOINT unknown")
	if !errors.Is(err, ErrSavepointNotFound) {
		t.Fatalf("ROLLBACK TO unknown: want ErrSavepointNotFound, got %v", err)
	}
	t.Log("ROLLBACK TO unknown rejected ✓")

	// Release non-top savepoint.
	s.Exec("SAVEPOINT b")
	err = s.Exec("RELEASE SAVEPOINT a")
	if !errors.Is(err, ErrSavepointNotTop) {
		t.Fatalf("RELEASE non-top: want ErrSavepointNotTop, got %v", err)
	}
	t.Log("RELEASE non-top savepoint rejected ✓")

	// Outer ROLLBACK clears savepoints.
	s.Exec("ROLLBACK")
	err = s.Exec("SAVEPOINT x")
	if !errors.Is(err, ErrSavepointOutsideEpoch) {
		t.Fatalf("SAVEPOINT after ROLLBACK: want ErrSavepointOutsideEpoch, got %v", err)
	}
	t.Log("SAVEPOINT after outer ROLLBACK rejected ✓")

	t.Log("✅ test E: error/state-machine tests")
}

// =============================================================================
// Test F: Concurrent savepoint safety
// =============================================================================

func TestSession_Savepoint_ConcurrentSafety(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/savepoint_race.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "Base", []float32{1, 0, 0}, nil)
	graph.RegisterEdgeKind("E", 1)

	var wg sync.WaitGroup

	// Session A: branch → rollback → commit outer.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			s, _ := db.NewSQLSession(context.Background())
			s.Exec("BEGIN EPOCH TRANSACTION")
			s.ExecWithParams("INSERT INTO docs (id, embedding) VALUES ('Outer', '[1,0,0]')", nil)
			s.Exec("SAVEPOINT branch")
			s.ExecWithParams("INSERT INTO docs (id, embedding) VALUES ('BranchOnly', '[0,1,0]')", nil)
			s.Exec("ROLLBACK TO SAVEPOINT branch")
			s.Exec("COMMIT")
			s.Close()
		}
	}()

	// Session B: live queries.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			s, _ := db.NewSQLSession(context.Background())
			results, _ := s.Query("SELECT id FROM docs WHERE id = 'BranchOnly'")
			if results != nil && len(results.Results) > 0 {
				t.Errorf("Session B observed rolled-back BranchOnly at iter %d", i)
				s.Close()
				return
			}
			s.Close()
		}
	}()

	wg.Wait()
	t.Log("✅ test F: concurrent savepoint safety")
}
