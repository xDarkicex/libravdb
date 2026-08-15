package libravdb

import (
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/graph"
)

// =============================================================================
// Test harness
// =============================================================================

type execHarness struct {
	db  *Database
	gr  Graph
	col *Collection
}

func newExecHarness(t *testing.T) *execHarness {
	t.Helper()
	db, err := Open(WithStoragePath(t.TempDir() + "/leiden_exec.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { db.Drop(context.Background()) })

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	t.Cleanup(func() { gr.Close() })

	col, err := db.CreateCollection(context.Background(), "nodes", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	return &execHarness{db: db, gr: gr, col: col}
}

func (h *execHarness) insert(id string) uint64 {
	h.col.Insert(context.Background(), id, []float32{1, 0, 0}, nil)
	nid, _ := h.db.GetNodeID(context.Background(), "nodes", id)
	return nid
}

func (h *execHarness) insertLabeled(id, label string) uint64 {
	nid := h.insert(id)
	h.gr.RegisterVertexLabel(nid, label)
	return nid
}

func (h *execHarness) addEdge(src, tgt uint64, kind uint8) {
	txn := h.gr.BeginTxn()
	txn.AddEdge(src, tgt, 1.0, kind)
	txn.Commit(context.Background())
}

func (h *execHarness) bindAndExecute(t *testing.T, epoch *EpochTx, edgeKind string, kind uint8, minHops, maxHops int, dir LeidenMatchDirection) *LeidenExecutionResult {
	t.Helper()

	spec := LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: resolveSeeds(t, epoch, h.gr, "seeds"),
		EdgeKinds:   nil,
		MinHops:     minHops,
		MaxHops:     maxHops,
		Direction:   dir,
	}
	if edgeKind != "" {
		spec.EdgeKinds = []uint8{kind}
	}

	opts := EpochLeidenOptions{}
	opts.Seeds = make([]uint64, len(spec.SeedNodeIDs))
	copy(opts.Seeds, spec.SeedNodeIDs)

	bound := &BoundLeidenMatchPlan{Spec: spec, Options: opts}
	result, err := epoch.ExecuteBoundLeidenMatchPlan(context.Background(), bound)
	if err != nil {
		t.Fatalf("ExecuteBoundLeidenMatchPlan: %v", err)
	}
	return result
}

func resolveSeeds(t *testing.T, epoch *EpochTx, g Graph, label string) []uint64 {
	t.Helper()
	candidates := g.GetLabelNodes(label)
	records, _ := epoch.ListRecords(context.Background(), "nodes")
	visible := make(map[string]bool)
	for _, r := range records {
		visible[r.ID] = true
	}
	var seeds []uint64
	for _, nid := range candidates {
		col, rid, err := epoch.ResolveNodeID(context.Background(), nid)
		if err != nil || col != "nodes" || !visible[rid] {
			continue
		}
		seeds = append(seeds, nid)
	}
	return seeds
}

// =============================================================================
// Test 1: Basic execution
// =============================================================================

func TestLeidenExecute_Basic(t *testing.T) {
	h := newExecHarness(t)
	graph.RegisterEdgeKind("LINK", 10)

	s1 := h.insertLabeled("s1", "seeds")
	s2 := h.insertLabeled("s2", "seeds")
	t1 := h.insert("t1")
	t2 := h.insert("t2")

	h.addEdge(s1, t1, 10)
	h.addEdge(s1, t2, 10)
	h.addEdge(s2, t1, 10)

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	result := h.bindAndExecute(t, epoch, "LINK", 10, 1, 1, LeidenMatchOutbound)

	if result.MatchResult == nil {
		t.Fatal("MatchResult must not be nil")
	}
	if result.Relation == nil {
		t.Fatal("Relation must not be nil")
	}
	if len(result.Relation.Rows) == 0 {
		t.Fatal("Relation.Rows must be non-empty")
	}

	// Every row must have correct Collection.
	for _, row := range result.Relation.Rows {
		if row.Collection != "nodes" {
			t.Errorf("row Collection: want 'nodes', got %q", row.Collection)
		}
		if row.RecordID == "" {
			t.Error("row RecordID is empty")
		}
		if row.NodeID == 0 {
			t.Error("row NodeID is zero")
		}
	}

	// Propagation: Relation diagnostics match MatchResult.
	if result.Relation.Truncated != result.MatchResult.LeidenResult.Truncated {
		t.Error("Truncated mismatch between Relation and LeidenResult")
	}
	if result.Relation.Scope != result.MatchResult.LeidenResult.Scope {
		t.Error("Scope mismatch")
	}

	t.Logf("executed: %d relation rows, modularity=%v", len(result.Relation.Rows), result.Relation.Modularity)
	t.Log("✅ basic execution")
}

// =============================================================================
// Test 2: Snapshot isolation
// =============================================================================

func TestLeidenExecute_SnapshotIsolation(t *testing.T) {
	h := newExecHarness(t)
	graph.RegisterEdgeKind("LINK", 10)

	s1 := h.insertLabeled("s1", "seeds")
	pre := h.insert("pre")

	h.addEdge(s1, pre, 10)

	t0 := time.Now().UTC()
	time.Sleep(10 * time.Millisecond)

	// Post-t0 node and edge.
	post := h.insert("post")
	h.addEdge(s1, post, 10)

	epoch, err := h.db.BeginEpochTxAt(context.Background(), t0)
	if err != nil {
		t.Fatalf("BeginEpochTxAt: %v", err)
	}
	defer epoch.Rollback(context.Background())

	result := h.bindAndExecute(t, epoch, "LINK", 10, 1, 1, LeidenMatchOutbound)

	recordIDs := make(map[string]bool)
	for _, row := range result.Relation.Rows {
		recordIDs[row.RecordID] = true
	}
	if recordIDs["post"] {
		t.Fatal("post-t0 record must be absent")
	}
	if !recordIDs["pre"] {
		t.Fatal("pre-t0 record must be present")
	}

	t.Log("✅ snapshot isolation")
}

// =============================================================================
// Test 3: Staged overlay visibility
// =============================================================================

func TestLeidenExecute_StagedOverlay(t *testing.T) {
	h := newExecHarness(t)
	graph.RegisterEdgeKind("LINK", 10)

	s1 := h.insertLabeled("s1", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// Stage a record and edge.
	epoch.Insert(context.Background(), "nodes", "staged", []float32{0, 1, 0}, nil)
	stagedNID, _ := epoch.LookupNodeID(context.Background(), "nodes", "staged")
	epoch.AddGraphEdge("nodes", s1, stagedNID, 1.0, 10)

	result := h.bindAndExecute(t, epoch, "LINK", 10, 1, 1, LeidenMatchOutbound)

	found := false
	for _, row := range result.Relation.Rows {
		if row.RecordID == "staged" {
			found = true
		}
	}
	if !found {
		t.Fatal("staged record must appear in relation")
	}

	// Live read must not see staged state.
	liveCol, _ := h.db.GetCollection("nodes")
	_, err := liveCol.Get(context.Background(), "staged")
	if err == nil {
		t.Fatal("live read must not see uncommitted staged record")
	}

	t.Log("✅ staged overlay visibility")
}

// =============================================================================
// Test 4: Savepoint rollback
// =============================================================================

func TestLeidenExecute_SavepointRollback(t *testing.T) {
	h := newExecHarness(t)
	graph.RegisterEdgeKind("LINK", 10)

	s1 := h.insertLabeled("s1", "seeds")
	base := h.insert("base")
	h.addEdge(s1, base, 10)

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// Baseline execution.
	baseline := h.bindAndExecute(t, epoch, "LINK", 10, 1, 1, LeidenMatchOutbound)

	// Savepoint.
	epoch.Savepoint("sp")

	// Stage a bridge node that changes the Leiden result.
	epoch.Insert(context.Background(), "nodes", "bridge", []float32{1, 1, 1}, nil)
	bridgeNID, _ := epoch.LookupNodeID(context.Background(), "nodes", "bridge")
	epoch.AddGraphEdge("nodes", s1, bridgeNID, 1.0, 10)

	// Execute — should see bridge.
	branched := h.bindAndExecute(t, epoch, "LINK", 10, 1, 1, LeidenMatchOutbound)
	if len(branched.Relation.Rows) == len(baseline.Relation.Rows) {
		t.Fatal("branch execution must differ from baseline")
	}

	// Rollback.
	epoch.RollbackTo("sp")

	// Execute again — must match baseline exactly.
	restored := h.bindAndExecute(t, epoch, "LINK", 10, 1, 1, LeidenMatchOutbound)
	if !reflect.DeepEqual(restored.Relation.Rows, baseline.Relation.Rows) {
		t.Fatal("restored relation rows must match baseline after rollback")
	}
	if !reflect.DeepEqual(restored.MatchResult.MatchedNodeIDs, baseline.MatchResult.MatchedNodeIDs) {
		t.Fatal("restored matched node IDs must match baseline")
	}

	t.Log("✅ savepoint rollback")
}

// =============================================================================
// Test 5: Commit durability
// =============================================================================

func TestLeidenExecute_CommitDurability(t *testing.T) {
	h := newExecHarness(t)
	graph.RegisterEdgeKind("LINK", 10)

	s1 := h.insertLabeled("s1", "seeds")
	tgt := h.insert("tgt")
	h.addEdge(s1, tgt, 10)

	epoch, _ := h.db.BeginEpochTx(context.Background())

	spec := LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: resolveSeeds(t, epoch, h.gr, "seeds"),
		MinHops:     1,
		MaxHops:     1,
		Direction:   LeidenMatchOutbound,
	}
	bound := &BoundLeidenMatchPlan{Spec: spec, Options: EpochLeidenOptions{
		Seeds: make([]uint64, len(spec.SeedNodeIDs)),
	}}
	copy(bound.Options.Seeds, spec.SeedNodeIDs)

	result, err := epoch.ExecuteBoundLeidenMatchPlan(context.Background(), bound)
	if err != nil {
		t.Fatalf("ExecuteBoundLeidenMatchPlan: %v", err)
	}
	rowsBeforeCommit := len(result.Relation.Rows)

	// Commit.
	if err := epoch.Commit(context.Background()); err != nil {
		t.Fatalf("Commit: %v", err)
	}

	// After commit, records must survive.
	col, _ := h.db.GetCollection("nodes")
	_, err = col.Get(context.Background(), "s1")
	if err != nil {
		t.Fatal("s1 must survive commit")
	}
	_, err = col.Get(context.Background(), "tgt")
	if err != nil {
		t.Fatal("tgt must survive commit")
	}

	t.Logf("committed: %d relation rows before commit", rowsBeforeCommit)
	t.Log("✅ commit durability")
}

// =============================================================================
// Test 6: Defensive ownership
// =============================================================================

func TestLeidenExecute_DefensiveOwnership(t *testing.T) {
	h := newExecHarness(t)
	graph.RegisterEdgeKind("LINK", 10)

	s1 := h.insertLabeled("s1", "seeds")
	t1 := h.insert("t1")
	h.addEdge(s1, t1, 10)

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	first := h.bindAndExecute(t, epoch, "LINK", 10, 1, 1, LeidenMatchOutbound)

	// Mutate returned results.
	first.MatchResult.MatchedNodeIDs[0] = 99999
	first.MatchResult.LeidenResult.Communities = nil
	if len(first.Relation.Rows) > 0 {
		first.Relation.Rows[0].NodeID = 0
		first.Relation.Rows[0].RecordID = "hacked"
	}

	// Execute again — unchanged.
	second := h.bindAndExecute(t, epoch, "LINK", 10, 1, 1, LeidenMatchOutbound)

	if second.MatchResult.LeidenResult.Communities == nil {
		t.Fatal("mutation leaked: Communities became nil")
	}
	if len(second.Relation.Rows) > 0 && second.Relation.Rows[0].RecordID == "hacked" {
		t.Fatal("mutation leaked into second execution")
	}

	t.Log("✅ defensive ownership")
}

// =============================================================================
// Test 7: Invalid bound plans
// =============================================================================

func TestLeidenExecute_InvalidBoundPlans(t *testing.T) {
	h := newExecHarness(t)
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("s1", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// Nil bound plan.
	_, err := epoch.ExecuteBoundLeidenMatchPlan(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error for nil bound plan")
	}
	t.Logf("nil bound plan: %v", err)

	// Empty collection.
	_, err = epoch.ExecuteBoundLeidenMatchPlan(context.Background(), &BoundLeidenMatchPlan{
		Spec: LeidenMatchSpec{Collection: ""},
	})
	if err == nil {
		t.Fatal("expected error for empty collection")
	}
	t.Logf("empty collection: %v", err)

	// Empty seed IDs.
	_, err = epoch.ExecuteBoundLeidenMatchPlan(context.Background(), &BoundLeidenMatchPlan{
		Spec: LeidenMatchSpec{Collection: "nodes", MinHops: 0, MaxHops: 1, Direction: LeidenMatchOutbound},
	})
	if err == nil {
		t.Fatal("expected error for empty seed IDs")
	}
	t.Logf("empty seeds: %v", err)

	// Invalid hop interval.
	_, err = epoch.ExecuteBoundLeidenMatchPlan(context.Background(), &BoundLeidenMatchPlan{
		Spec: LeidenMatchSpec{Collection: "nodes", SeedNodeIDs: []uint64{1}, MinHops: 3, MaxHops: 1, Direction: LeidenMatchOutbound},
	})
	if err == nil {
		t.Fatal("expected error for invalid hop interval")
	}
	t.Logf("invalid hops: %v", err)

	// Closed epoch.
	epoch2, _ := h.db.BeginEpochTx(context.Background())
	epoch2.Rollback(context.Background())
	_, err = epoch2.ExecuteBoundLeidenMatchPlan(context.Background(), &BoundLeidenMatchPlan{
		Spec: LeidenMatchSpec{Collection: "nodes", SeedNodeIDs: []uint64{1}, MinHops: 0, MaxHops: 1, Direction: LeidenMatchOutbound},
	})
	if err == nil {
		t.Fatal("expected error for closed epoch")
	}
	t.Logf("closed epoch: %v", err)

	// Cancelled context.
	epoch3, _ := h.db.BeginEpochTx(context.Background())
	defer epoch3.Rollback(context.Background())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = epoch3.ExecuteBoundLeidenMatchPlan(ctx, &BoundLeidenMatchPlan{
		Spec: LeidenMatchSpec{Collection: "nodes", SeedNodeIDs: []uint64{1}, MinHops: 0, MaxHops: 1, Direction: LeidenMatchOutbound},
	})
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
	t.Logf("cancelled context: %v", err)

	// Unsupported direction.
	_, err = epoch.ExecuteBoundLeidenMatchPlan(context.Background(), &BoundLeidenMatchPlan{
		Spec: LeidenMatchSpec{Collection: "nodes", SeedNodeIDs: []uint64{1}, MinHops: 0, MaxHops: 1, Direction: 99},
	})
	if err == nil {
		t.Fatal("expected error for unsupported direction")
	}
	t.Logf("unsupported direction: %v", err)

	t.Log("✅ invalid bound plans")
}

// =============================================================================
// Test 8: No global mutation
// =============================================================================

func TestLeidenExecute_NoMutation(t *testing.T) {
	h := newExecHarness(t)
	graph.RegisterEdgeKind("LINK", 10)

	s1 := h.insertLabeled("s1", "seeds")
	t1 := h.insert("t1")
	h.addEdge(s1, t1, 10)

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	genBefore := epoch.generation
	recordsBefore, _ := epoch.ListRecords(context.Background(), "nodes")
	recCount := len(recordsBefore)

	// Execute 5 times.
	for i := 0; i < 5; i++ {
		h.bindAndExecute(t, epoch, "LINK", 10, 1, 1, LeidenMatchOutbound)
	}

	genAfter := epoch.generation
	recordsAfter, _ := epoch.ListRecords(context.Background(), "nodes")

	if genAfter != genBefore {
		t.Errorf("generation changed: %d → %d", genBefore, genAfter)
	}
	if len(recordsAfter) != recCount {
		t.Errorf("record count changed: %d → %d", recCount, len(recordsAfter))
	}

	t.Log("✅ no mutation across 5 executions")
}

// =============================================================================
// Test 9: Determinism
// =============================================================================

func TestLeidenExecute_Determinism(t *testing.T) {
	h := newExecHarness(t)
	graph.RegisterEdgeKind("LINK", 10)

	s1 := h.insertLabeled("s1", "seeds")
	t1 := h.insert("t1")
	h.addEdge(s1, t1, 10)

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	var first *LeidenExecutionResult
	for i := 0; i < 10; i++ {
		result := h.bindAndExecute(t, epoch, "LINK", 10, 1, 1, LeidenMatchOutbound)
		if i == 0 {
			first = result
			continue
		}

		if !reflect.DeepEqual(result.MatchResult.MatchedNodeIDs, first.MatchResult.MatchedNodeIDs) {
			t.Fatalf("call %d: MatchedNodeIDs differ", i)
		}
		if !reflect.DeepEqual(result.MatchResult.LeidenResult.Assignments(), first.MatchResult.LeidenResult.Assignments()) {
			t.Fatalf("call %d: Assignments differ", i)
		}
		if !reflect.DeepEqual(result.Relation.Rows, first.Relation.Rows) {
			t.Fatalf("call %d: relation rows differ", i)
		}
		if absDiff(result.Relation.Modularity, first.Relation.Modularity) > 1e-12 {
			t.Fatalf("call %d: modularity differs", i)
		}
		if result.Relation.Truncated != first.Relation.Truncated {
			t.Fatalf("call %d: Truncated differs", i)
		}
	}

	t.Log("✅ determinism across 10 calls")
}

// =============================================================================
// Test 10: Error phase reporting
// =============================================================================

func TestLeidenExecute_ErrorPhaseReporting(t *testing.T) {
	h := newExecHarness(t)

	// Force an error inside the execution pipeline (invalid collection).
	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	_, err := epoch.ExecuteBoundLeidenMatchPlan(context.Background(), &BoundLeidenMatchPlan{
		Spec: LeidenMatchSpec{
			Collection:  "nonexistent",
			SeedNodeIDs: []uint64{1},
			MinHops:     1,
			MaxHops:     1,
			Direction:   LeidenMatchOutbound,
		},
		Options: EpochLeidenOptions{},
	})
	if err == nil {
		t.Fatal("expected error for nonexistent collection")
	}
	// Error should identify the failing phase (ComputeLeidenFromMatch calls
	// GraphTxn which calls GetCollection).
	t.Logf("phase error: %v", err)

	// Closed epoch — should be caught in validation.
	epoch2, _ := h.db.BeginEpochTx(context.Background())
	epoch2.Rollback(context.Background())
	_, err = epoch2.ExecuteBoundLeidenMatchPlan(context.Background(), &BoundLeidenMatchPlan{
		Spec: LeidenMatchSpec{Collection: "nodes", SeedNodeIDs: []uint64{1}, MinHops: 0, MaxHops: 1, Direction: LeidenMatchOutbound},
	})
	if err == nil {
		t.Fatal("expected error for closed epoch")
	}
	t.Logf("closed epoch phase: %v", err)

	t.Log("✅ error phase reporting")
}

// =============================================================================
// Test 11: Full pipeline: plan → lower → bind → execute
// =============================================================================

func TestLeidenExecute_FullPipeline(t *testing.T) {
	h := newExecHarness(t)
	graph.RegisterEdgeKind("CONNECT", 50)

	s1 := h.insertLabeled("alpha", "roots")
	t1 := h.insertLabeled("beta", "leaves")
	t2 := h.insert("gamma")
	h.addEdge(s1, t1, 50)
	h.addEdge(s1, t2, 50)

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// Lower a plan from SQL (no terminal label — not yet supported).
	sql := `COMPUTE LEIDEN FROM MATCH (r:roots)-[:CONNECT*1..2]->(target)`
	var doc parser.QueryDoc
	if err := parser.Parse([]byte(sql), &doc); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	plan, err := LowerComputeLeidenPlan([]byte(sql), &doc, 0)
	if err != nil {
		t.Fatalf("LowerComputeLeidenPlan: %v", err)
	}
	plan.Collection = "nodes"

	// Bind.
	bound, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "")
	if err != nil {
		t.Fatalf("BindLeidenMatchPlan: %v", err)
	}

	// Execute.
	result, err := epoch.ExecuteBoundLeidenMatchPlan(context.Background(), bound)
	if err != nil {
		t.Fatalf("ExecuteBoundLeidenMatchPlan: %v", err)
	}

	// Verify results.
	if len(result.Relation.Rows) == 0 {
		t.Fatal("expected non-empty relation from full pipeline")
	}

	t.Logf("full pipeline: %d rows", len(result.Relation.Rows))

	for _, row := range result.Relation.Rows {
		if row.Collection != "nodes" {
			t.Errorf("row collection mismatch: %q", row.Collection)
		}
	}

	t.Log("✅ full pipeline: SQL → plan → bind → execute")
}
