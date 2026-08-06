package libravdb

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// =============================================================================
// Test A: Full marketing flow, rollback
// =============================================================================

func TestSession_FullMarketingFlow_Rollback(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/session_flow_rollback.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "documents", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "Server_Crash", []float32{1, 0, 0}, nil)
	graph.RegisterEdgeKind("CAUSES", 50)

	s, _ := db.NewSQLSession(context.Background())
	defer s.Close()

	// BEGIN EPOCH
	if err := s.Exec("BEGIN EPOCH TRANSACTION"); err != nil {
		t.Fatalf("BEGIN EPOCH: %v", err)
	}

	// INSERT record
	if err := s.ExecWithParams("INSERT INTO documents (id, embedding) VALUES ('Hypothesis_A', '[1,0,0]')", nil); err != nil {
		t.Fatalf("INSERT: %v", err)
	}

	// INSERT graph edge
	if err := s.Exec("INSERT INTO GRAPH_EDGES VALUES ('Hypothesis_A', 'CAUSES', 'Server_Crash')"); err != nil {
		t.Fatalf("INSERT GRAPH_EDGES: %v", err)
	}

	// QUERY inside session: use VECTOR_DISTANCE (known-working path through executeVectorProjection).
	results, qerr := s.QueryWithParams(
		"SELECT id FROM documents ORDER BY VECTOR_DISTANCE(embedding, '[1,0,0]') ASC LIMIT 5", nil)
	if qerr != nil {
		t.Fatalf("SELECT: %v", qerr)
	}
	found := false
	for _, r := range results.Results {
		if r.ID == "Hypothesis_A" {
			found = true
		}
	}
	if !found {
		t.Fatal("session must see staged Hypothesis_A in vector query")
	}
	t.Logf("Phase 1: session vector query sees staged record ✓")

	// External live Database query must NOT see staged data.
	liveResults, _ := db.Query(context.Background(), "SELECT id FROM documents")
	if liveResults != nil {
		for _, r := range liveResults.Results {
			if r.ID == "Hypothesis_A" {
				t.Fatal("live DB must not see staged record")
			}
		}
	}
	t.Logf("Phase 1: live DB cannot see staged record ✓")

	// ROLLBACK
	if err := s.Exec("ROLLBACK"); err != nil {
		t.Fatalf("ROLLBACK: %v", err)
	}

	// After rollback, verify via Go API.
	col2, _ := db.GetCollection("documents")
	_, getErr := col2.Get(context.Background(), "Hypothesis_A")
	if getErr == nil {
		t.Fatal("after rollback, record must not exist")
	}
	t.Logf("Phase 2: after ROLLBACK, record not in storage ✓")

	// Live DB must still not see it.
	finalRecs, _ := col2.ListAll(context.Background())
	for _, r := range finalRecs {
		if r.ID == "Hypothesis_A" {
			t.Fatal("live DB should never see rolled-back record")
		}
	}
	t.Log("✅ test A: full marketing flow with ROLLBACK")
}

// =============================================================================
// Test B: Full marketing flow, commit and reopen
// =============================================================================

func TestSession_FullMarketingFlow_CommitReopen(t *testing.T) {
	dir := t.TempDir() + "/session_flow_commit.libravdb"

	// Phase 1: Commit through session.
	func() {
		db, _ := Open(WithStoragePath(dir))
		defer db.Close()

		gr, _ := NewGraph(GraphConfig{})
		defer gr.Close()
		col, _ := db.CreateCollection(context.Background(), "documents", WithDimension(3), WithGraph(gr))
		col.Insert(context.Background(), "Server_Crash", []float32{1, 0, 0}, nil)
		graph.RegisterEdgeKind("CAUSES", 50)

		s, _ := db.NewSQLSession(context.Background())
		defer s.Close()

		s.Exec("BEGIN EPOCH TRANSACTION")
		s.ExecWithParams("INSERT INTO documents (id, embedding) VALUES ('Hypothesis_A', '[1,0,0]')", nil)
		s.Exec("INSERT INTO GRAPH_EDGES VALUES ('Hypothesis_A', 'CAUSES', 'Server_Crash')")

		// Verify inside session via VECTOR_DISTANCE query.
		results, _ := s.QueryWithParams(
			"SELECT id FROM documents ORDER BY VECTOR_DISTANCE(embedding, '[1,0,0]') ASC LIMIT 5", nil)
		found := false
		for _, r := range results.Results {
			if r.ID == "Hypothesis_A" {
				found = true
			}
		}
		if !found {
			t.Fatal("session must see staged record before commit")
		}

		// External cannot see.
		liveResults, _ := db.Query(context.Background(), "SELECT id FROM documents")
		if liveResults != nil {
			for _, r := range liveResults.Results {
				if r.ID == "Hypothesis_A" {
					t.Fatal("live DB must not see staged record before commit")
				}
			}
		}

		// COMMIT.
		if err := s.Exec("COMMIT"); err != nil {
			t.Fatalf("COMMIT: %v", err)
		}
		t.Logf("Phase 1: COMMIT succeeded")
	}()

	// Phase 2: Reopen and verify persistence via Go API.
	db2, _ := Open(WithStoragePath(dir))
	defer db2.Close()

	col2, err := db2.GetCollection("documents")
	if err != nil {
		t.Fatalf("GetCollection after reopen: %v", err)
	}

	// Verify record survived.
	rec, err := col2.Get(context.Background(), "Hypothesis_A")
	if err != nil || rec.ID != "Hypothesis_A" {
		t.Fatalf("committed record must survive reopen: err=%v", err)
	}
	t.Logf("Phase 2: committed record visible after reopen ✓")

	// Verify GraphNodeID mapping survived.
	nodeID, err := db2.GetNodeID(context.Background(), "documents", "Hypothesis_A")
	if err != nil || nodeID == 0 {
		t.Fatal("GraphNodeID must survive reopen")
	}
	t.Logf("Phase 2: GraphNodeID %d survived reopen ✓", nodeID)

	// Verify edge survived.
	gr2, _ := NewGraph(GraphConfig{})
	defer gr2.Close()
	col2.SetGraph(gr2)
	scNode, _ := db2.GetNodeID(context.Background(), "documents", "Server_Crash")
	edges, _ := gr2.Neighbors(nodeID)
	if len(edges) == 0 || edges[0].Target != scNode {
		t.Fatal("edge must survive reopen")
	}
	t.Logf("Phase 2: edge %d→%d survived reopen ✓", nodeID, scNode)
	t.Log("✅ test B: full marketing flow with COMMIT + reopen")
}

// =============================================================================
// Test C: Two-session isolation
// =============================================================================

func TestSession_TwoSessionIsolation(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/session_isolation.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "Target", []float32{1, 0, 0}, nil)
	graph.RegisterEdgeKind("LINKS", 1)

	// Session A: begin epoch, stage data.
	sA, _ := db.NewSQLSession(context.Background())
	defer sA.Close()
	sA.Exec("BEGIN EPOCH TRANSACTION")
	sA.ExecWithParams("INSERT INTO docs (id, embedding) VALUES ('StagedA', '[1,0,0]')", nil)
	sA.Exec("INSERT INTO GRAPH_EDGES VALUES ('StagedA', 'LINKS', 'Target')")

	// Session A sees its staged data via VECTOR_DISTANCE query.
	resultsA, _ := sA.QueryWithParams(
		"SELECT id FROM docs ORDER BY VECTOR_DISTANCE(embedding, '[1,0,0]') ASC LIMIT 5", nil)
	found := false
	for _, r := range resultsA.Results {
		if r.ID == "StagedA" {
			found = true
		}
	}
	if !found {
		t.Fatal("Session A must see its own staged data")
	}
	t.Logf("Phase 1: Session A sees staged StagedA ✓")

	// Session B: cannot see A's staged data.
	sB, _ := db.NewSQLSession(context.Background())
	defer sB.Close()
	resultsB, _ := sB.QueryWithParams(
		"SELECT id FROM docs ORDER BY VECTOR_DISTANCE(embedding, '[1,0,0]') ASC LIMIT 5", nil)
	if resultsB != nil {
		for _, r := range resultsB.Results {
			if r.ID == "StagedA" {
				t.Fatal("Session B must NOT see A's staged data")
			}
		}
	}
	t.Logf("Phase 1: Session B cannot see StagedA ✓")

	// A commits.
	sA.Exec("COMMIT")

	// B can now see it in a fresh query.
	resultsB2, _ := sB.QueryWithParams(
		"SELECT id FROM docs ORDER BY VECTOR_DISTANCE(embedding, '[1,0,0]') ASC LIMIT 5", nil)
	found2 := false
	for _, r := range resultsB2.Results {
		if r.ID == "StagedA" {
			found2 = true
		}
	}
	if !found2 {
		t.Fatal("Session B must see StagedA after A commits")
	}
	t.Logf("Phase 2: Session B sees StagedA after commit ✓")
	t.Log("✅ test C: two-session isolation")
}

// =============================================================================
// Test E: Parameter propagation in epoch session
// =============================================================================

func TestSession_ParameterPropagation(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/session_params.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "A", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "B", []float32{0, 1, 0}, nil)

	s, _ := db.NewSQLSession(context.Background())
	defer s.Close()

	s.Exec("BEGIN EPOCH TRANSACTION")

	// Query with parameterized vector.
	results, err := s.QueryWithParams(
		"SELECT id FROM docs ORDER BY VECTOR_DISTANCE(embedding, $vec) ASC LIMIT 1",
		QueryParams{"vec": []float32{1, 0, 0}})
	if err != nil {
		t.Fatalf("QueryWithParams: %v", err)
	}
	if len(results.Results) == 0 || results.Results[0].ID != "A" {
		t.Fatalf("expected 'A' closest to [1,0,0], got %d results", len(results.Results))
	}
	t.Logf("Phase 1: parameterized vector query returns '%s' ✓", results.Results[0].ID)

	s.Exec("ROLLBACK")
	t.Log("✅ test E: parameter propagation in epoch session")
}

// =============================================================================
// Test F: State-machine errors
// =============================================================================

func TestSession_StateMachineErrors(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/session_errors.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))

	s, _ := db.NewSQLSession(context.Background())

	// COMMIT with no epoch.
	err := s.Exec("COMMIT")
	if !errors.Is(err, ErrNoActiveEpoch) && err.Error() != ErrNoActiveEpoch.Error() {
		t.Fatalf("COMMIT without epoch: want ErrNoActiveEpoch, got %v", err)
	}
	t.Logf("COMMIT without epoch rejected ✓")

	// ROLLBACK with no epoch.
	err = s.Exec("ROLLBACK")
	if !errors.Is(err, ErrNoActiveEpoch) && err.Error() != ErrNoActiveEpoch.Error() {
		t.Fatalf("ROLLBACK without epoch: want ErrNoActiveEpoch, got %v", err)
	}
	t.Logf("ROLLBACK without epoch rejected ✓")

	// BEGIN EPOCH works.
	s.Exec("BEGIN EPOCH TRANSACTION")

	// Nested BEGIN EPOCH must fail.
	err = s.Exec("BEGIN EPOCH")
	if !errors.Is(err, ErrEpochAlreadyActive) && err.Error() != ErrEpochAlreadyActive.Error() {
		t.Fatalf("nested BEGIN: want ErrEpochAlreadyActive, got %v", err)
	}
	t.Logf("nested BEGIN rejected ✓")

	s.Exec("ROLLBACK")

	// Close.
	s.Close()

	// SQL after Close must fail.
	_, err = s.Query("SELECT 1")
	if !errors.Is(err, ErrSessionClosed) && err.Error() != ErrSessionClosed.Error() {
		t.Fatalf("query after close: want ErrSessionClosed, got %v", err)
	}
	t.Logf("query after Close rejected ✓")

	// Close is idempotent.
	s.Close()
	t.Logf("double Close is idempotent ✓")

	t.Log("✅ test F: state-machine errors")
}

// =============================================================================
// Test: Close rolls back active epoch
// =============================================================================

func TestSession_CloseRollsBackEpoch(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/session_close_rollback.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "Existing", []float32{1, 0, 0}, nil)

	// Capture pre-session state.
	recCount := 0
	col.Iterate(context.Background(), func(rec Record) error { recCount++; return nil })

	s, _ := db.NewSQLSession(context.Background())
	s.Exec("BEGIN EPOCH TRANSACTION")
	s.ExecWithParams("INSERT INTO docs (id, embedding) VALUES ('Staged', '[0,1,0]')", nil)

	// Close without explicit ROLLBACK.
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Verify no record leaked.
	postCount := 0
	col.Iterate(context.Background(), func(rec Record) error { postCount++; return nil })
	if postCount != recCount {
		t.Fatalf("record leak: before=%d, after=%d", recCount, postCount)
	}
	t.Logf("Close rolled back staged record: count unchanged (%d) ✓", postCount)
	t.Log("✅ Close auto-rolls-back active epoch")
}

// =============================================================================
// Test: Race safety with concurrent sessions
// =============================================================================

func TestSession_ConcurrentRaceSafety(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/session_race.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	col.Insert(context.Background(), "Base", []float32{1, 0, 0}, nil)
	graph.RegisterEdgeKind("LINKS", 1)

	var wg sync.WaitGroup

	// Session A: repeatedly BEGIN → stage → ROLLBACK.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 20; i++ {
			s, _ := db.NewSQLSession(context.Background())
			s.Exec("BEGIN EPOCH TRANSACTION")
			s.ExecWithParams("INSERT INTO docs (id, embedding) VALUES ('Staged_A', '[1,1,1]')", nil)
			s.Exec("INSERT INTO GRAPH_EDGES VALUES ('Staged_A', 'LINKS', 'Base')")
			s.Exec("ROLLBACK")
			s.Close()
		}
	}()

	// Session B: repeatedly query live data.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 20; i++ {
			s, _ := db.NewSQLSession(context.Background())
			results, _ := s.Query("SELECT id FROM docs WHERE id = 'Staged_A'")
			if results != nil && len(results.Results) > 0 {
				t.Errorf("Session B observed Session A's staged ID at iteration %d", i)
				s.Close()
				return
			}
			s.Close()
		}
	}()

	wg.Wait()
	t.Logf("Concurrent sessions: no cross-session visibility ✓")
	t.Log("✅ concurrent session race safety")
}
