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

type bindHarness struct {
	db  *Database
	gr  Graph
	col *Collection
}

func newBindHarness(t *testing.T) *bindHarness {
	t.Helper()
	db, err := Open(WithStoragePath(t.TempDir() + "/leiden_bind.libravdb"))
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

	return &bindHarness{db: db, gr: gr, col: col}
}

func (h *bindHarness) insert(id string) uint64 {
	if err := h.col.Insert(context.Background(), id, []float32{1, 0, 0}, nil); err != nil {
		panic(err)
	}
	nid, err := h.db.GetNodeID(context.Background(), "nodes", id)
	if err != nil {
		panic(err)
	}
	return nid
}

func (h *bindHarness) insertLabeled(id, label string) {
	nid := h.insert(id)
	h.gr.RegisterVertexLabel(nid, label)
}

// lowerPlan parses SQL and lowers to a plan, suitable for binding tests.
func lowerPlan(t *testing.T, sql string) *LeidenMatchPlan {
	t.Helper()
	var doc parser.QueryDoc
	if err := parser.Parse([]byte(sql), &doc); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(doc.ComputeLeidenStmts) == 0 {
		t.Fatalf("no ComputeLeidenStmts")
	}
	plan, err := LowerComputeLeidenPlan([]byte(sql), &doc, 0)
	if err != nil {
		t.Fatalf("LowerComputeLeidenPlan: %v", err)
	}
	return plan
}

// =============================================================================
// Test 1: Basic binding
// =============================================================================

func TestLeidenBind_BasicBinding(t *testing.T) {
	h := newBindHarness(t)
	graph.RegisterEdgeKind("CONNECTED_TO", 50)

	h.insertLabeled("s1", "seeds")
	h.insertLabeled("s2", "seeds")
	h.insert("target")

	epoch, err := h.db.BeginEpochTx(context.Background())
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}
	defer epoch.Rollback(context.Background())

	plan := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:CONNECTED_TO*1..3]->(target)`)
	plan.Collection = "nodes"

	bound, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "")
	if err != nil {
		t.Fatalf("BindLeidenMatchPlan: %v", err)
	}

	if bound.Spec.Collection != "nodes" {
		t.Errorf("Collection: want 'nodes', got %q", bound.Spec.Collection)
	}

	// Seeds must be sorted, deduplicated, and all labeled.
	if len(bound.Spec.SeedNodeIDs) != 2 {
		t.Fatalf("SeedNodeIDs: want 2 seeds, got %d: %v", len(bound.Spec.SeedNodeIDs), bound.Spec.SeedNodeIDs)
	}
	if bound.Spec.SeedNodeIDs[0] >= bound.Spec.SeedNodeIDs[1] {
		t.Error("SeedNodeIDs must be sorted ascending")
	}

	// Edge kind resolved correctly.
	if len(bound.Spec.EdgeKinds) != 1 || bound.Spec.EdgeKinds[0] != 30 {
		t.Errorf("EdgeKinds: want [30], got %v", bound.Spec.EdgeKinds)
	}

	// Direction and hops copied.
	if bound.Spec.Direction != LeidenMatchOutbound {
		t.Errorf("Direction: want outbound")
	}
	if bound.Spec.MinHops != 1 || bound.Spec.MaxHops != 3 {
		t.Errorf("Hops: want [1,3], got [%d,%d]", bound.Spec.MinHops, bound.Spec.MaxHops)
	}

	// Options.Seeds match Spec.SeedNodeIDs.
	if !reflect.DeepEqual(bound.Options.Seeds, bound.Spec.SeedNodeIDs) {
		t.Errorf("Options.Seeds must match Spec.SeedNodeIDs")
	}

	t.Logf("bound %d seeds, edge kind %d", len(bound.Spec.SeedNodeIDs), bound.Spec.EdgeKinds[0])
	t.Log("✅ basic binding")
}

// =============================================================================
// Test 2: Snapshot isolation
// =============================================================================

func TestLeidenBind_SnapshotIsolation(t *testing.T) {
	h := newBindHarness(t)
	graph.RegisterEdgeKind("LINK", 10)

	h.insertLabeled("pre_t0", "seeds")

	t0 := time.Now().UTC()
	time.Sleep(10 * time.Millisecond)

	// Post-t0: insert and label a seed that must not be visible.
	postNID := h.insert("post_t0")
	h.gr.RegisterVertexLabel(postNID, "seeds")

	epoch, err := h.db.BeginEpochTxAt(context.Background(), t0)
	if err != nil {
		t.Fatalf("BeginEpochTxAt: %v", err)
	}
	defer epoch.Rollback(context.Background())

	plan := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)`)
	plan.Collection = "nodes"

	bound, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "")
	if err != nil {
		t.Fatalf("BindLeidenMatchPlan: %v", err)
	}

	if len(bound.Spec.SeedNodeIDs) != 1 {
		t.Fatalf("expected 1 pre-t0 seed, got %d", len(bound.Spec.SeedNodeIDs))
	}
	for _, nid := range bound.Spec.SeedNodeIDs {
		if nid == postNID {
			t.Fatal("post-t0 seed must be excluded")
		}
	}

	t.Log("✅ snapshot isolation: post-t0 seed excluded")
}

// =============================================================================
// Test 3: Epoch delete visibility
// =============================================================================

func TestLeidenBind_DeleteVisibility(t *testing.T) {
	h := newBindHarness(t)
	graph.RegisterEdgeKind("LINK", 10)

	h.insertLabeled("to_delete", "seeds")
	h.insertLabeled("keep", "seeds")

	epoch, err := h.db.BeginEpochTx(context.Background())
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}
	defer epoch.Rollback(context.Background())

	// Stage a delete.
	if err := epoch.Delete(context.Background(), "nodes", "to_delete"); err != nil {
		t.Fatalf("staged Delete: %v", err)
	}

	plan := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)`)
	plan.Collection = "nodes"

	bound, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "")
	if err != nil {
		t.Fatalf("BindLeidenMatchPlan: %v", err)
	}

	if len(bound.Spec.SeedNodeIDs) != 1 {
		t.Fatalf("expected 1 visible seed (deleted excluded), got %d", len(bound.Spec.SeedNodeIDs))
	}
	t.Log("✅ deleted seed excluded from binding")
}

// =============================================================================
// Test 4: Staged graph overlay compatibility
// =============================================================================

func TestLeidenBind_StagedGraphOverlay(t *testing.T) {
	h := newBindHarness(t)
	graph.RegisterEdgeKind("LINK", 10)

	h.insertLabeled("s", "seeds")
	h.insert("t")

	epoch, err := h.db.BeginEpochTx(context.Background())
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}
	defer epoch.Rollback(context.Background())

	plan := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)`)
	plan.Collection = "nodes"

	// First bind.
	first, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "")
	if err != nil {
		t.Fatalf("first BindLeidenMatchPlan: %v", err)
	}

	// Stage graph edges.
	sNID := first.Spec.SeedNodeIDs[0]
	tNID, err := epoch.LookupNodeID(context.Background(), "nodes", "t")
	if err != nil {
		t.Fatalf("LookupNodeID t: %v", err)
	}
	if err := epoch.AddGraphEdge("nodes", sNID, tNID, 1.0, 10); err != nil {
		t.Fatalf("AddGraphEdge: %v", err)
	}

	// Second bind — must remain identical.
	second, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "")
	if err != nil {
		t.Fatalf("second BindLeidenMatchPlan: %v", err)
	}

	if !reflect.DeepEqual(first, second) {
		t.Fatal("binding must be deterministic regardless of staged graph edges")
	}

	t.Log("✅ staged graph edges do not affect binding determinism")
}

// =============================================================================
// Test 5: Edge-kind resolution
// =============================================================================

func TestLeidenBind_EdgeKindResolution(t *testing.T) {
	h := newBindHarness(t)
	graph.RegisterEdgeKind("KNOWN", 42)

	h.insertLabeled("s", "seeds")
	h.insert("t")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// Known edge kind.
	plan := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:KNOWN]->(target)`)
	plan.Collection = "nodes"
	bound, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "")
	if err != nil {
		t.Fatalf("known edge kind: %v", err)
	}
	if len(bound.Spec.EdgeKinds) != 1 || bound.Spec.EdgeKinds[0] != 42 {
		t.Errorf("known edge kind: want [42], got %v", bound.Spec.EdgeKinds)
	}

	// Unknown edge kind.
	plan2 := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:UNKNOWN]->(target)`)
	plan2.Collection = "nodes"
	_, err = epoch.BindLeidenMatchPlan(context.Background(), plan2, "")
	if err == nil {
		t.Fatal("expected error for unknown edge kind")
	}
	t.Logf("unknown edge kind correctly rejected: %v", err)

	// Empty edge kind → nil filter (all kinds).
	plan3 := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[*1..3]->(target)`)
	plan3.Collection = "nodes"
	bound3, err := epoch.BindLeidenMatchPlan(context.Background(), plan3, "")
	if err != nil {
		t.Fatalf("empty edge kind: %v", err)
	}
	if bound3.Spec.EdgeKinds != nil {
		t.Errorf("empty edge kind: want nil, got %v", bound3.Spec.EdgeKinds)
	}

	t.Log("✅ edge-kind resolution")
}

// =============================================================================
// Test 6: Collection validation
// =============================================================================

func TestLeidenBind_CollectionValidation(t *testing.T) {
	h := newBindHarness(t)
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("s", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	plan := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)`)

	// Empty collection.
	_, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "")
	if err == nil {
		t.Fatal("expected error for empty collection")
	}
	t.Logf("empty collection: %v", err)

	// Nonexistent collection.
	_, err = epoch.BindLeidenMatchPlan(context.Background(), plan, "ghost")
	if err == nil {
		t.Fatal("expected error for nonexistent collection")
	}
	t.Logf("nonexistent collection: %v", err)

	// Conflicting collections.
	plan2 := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)`)
	plan2.Collection = "nodes"
	_, err = epoch.BindLeidenMatchPlan(context.Background(), plan2, "other")
	if err == nil {
		t.Fatal("expected error for collection mismatch")
	}
	t.Logf("collection mismatch: %v", err)

	// Collection without graph.
	db2, _ := Open(WithStoragePath(t.TempDir() + "/no_graph.libravdb"))
	defer db2.Drop(context.Background())
	db2.CreateCollection(context.Background(), "no_graph", WithDimension(3))
	epoch2, _ := db2.BeginEpochTx(context.Background())
	defer epoch2.Rollback(context.Background())
	plan3 := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)`)
	plan3.Collection = "no_graph"
	_, err = epoch2.BindLeidenMatchPlan(context.Background(), plan3, "")
	if err == nil {
		t.Fatal("expected error for collection without graph")
	}
	t.Logf("no graph: %v", err)

	t.Log("✅ collection validation")
}

// =============================================================================
// Test 7: Seed validation
// =============================================================================

func TestLeidenBind_SeedValidation(t *testing.T) {
	h := newBindHarness(t)
	graph.RegisterEdgeKind("LINK", 10)

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// Nil plan.
	_, err := epoch.BindLeidenMatchPlan(context.Background(), nil, "nodes")
	if err == nil {
		t.Fatal("expected error for nil plan")
	}
	t.Logf("nil plan: %v", err)

	// Unlabeled seed.
	plan := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s)-[:LINK]->(target)`)
	plan.Collection = "nodes"
	_, err = epoch.BindLeidenMatchPlan(context.Background(), plan, "")
	if err == nil {
		t.Fatal("expected error for unlabeled seed")
	}
	t.Logf("unlabeled seed: %v", err)

	// No visible labeled seeds.
	h.insert("orphan") // not labeled
	plan2 := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)`)
	plan2.Collection = "nodes"
	_, err = epoch.BindLeidenMatchPlan(context.Background(), plan2, "")
	if err == nil {
		t.Fatal("expected error for no visible labeled seeds")
	}
	t.Logf("no visible seeds: %v", err)

	// Closed epoch.
	epoch2, _ := h.db.BeginEpochTx(context.Background())
	epoch2.Rollback(context.Background())
	plan3 := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)`)
	plan3.Collection = "nodes"
	_, err = epoch2.BindLeidenMatchPlan(context.Background(), plan3, "")
	if err == nil {
		t.Fatal("expected error for closed epoch")
	}
	t.Logf("closed epoch: %v", err)

	// Cancelled context.
	epoch3, _ := h.db.BeginEpochTx(context.Background())
	defer epoch3.Rollback(context.Background())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = epoch3.BindLeidenMatchPlan(ctx, plan3, "")
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
	t.Logf("cancelled context: %v", err)

	t.Log("✅ seed validation")
}

// =============================================================================
// Test 8: Terminal-label policy
// =============================================================================

func TestLeidenBind_TerminalLabelPolicy(t *testing.T) {
	h := newBindHarness(t)
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("s", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// Plan with terminal label must error.
	plan := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target:Terminals)`)
	plan.Collection = "nodes"
	_, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "")
	if err == nil {
		t.Fatal("expected error for unsupported terminal label")
	}
	t.Logf("terminal label: %v", err)

	// Plan without terminal label must succeed.
	plan2 := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)`)
	plan2.Collection = "nodes"
	_, err = epoch.BindLeidenMatchPlan(context.Background(), plan2, "")
	if err != nil {
		t.Fatalf("plan without terminal label must succeed: %v", err)
	}

	t.Log("✅ terminal-label policy")
}

// =============================================================================
// Test 9: Defensive ownership
// =============================================================================

func TestLeidenBind_DefensiveOwnership(t *testing.T) {
	h := newBindHarness(t)
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("s1", "seeds")
	h.insertLabeled("s2", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	plan := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)`)
	plan.Collection = "nodes"

	bound, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "")
	if err != nil {
		t.Fatalf("BindLeidenMatchPlan: %v", err)
	}

	// Mutate returned slices.
	bound.Spec.SeedNodeIDs[0] = 99999
	bound.Options.Seeds[0] = 88888
	bound.Spec.EdgeKinds[0] = 255

	// Plan must remain unchanged.
	if plan.SeedAlias != "s" {
		t.Error("plan.SeedAlias mutated")
	}
	if plan.EdgeKind != "LINK" {
		t.Error("plan.EdgeKind mutated")
	}

	// Bind again: must produce identical output to the pre-mutation bind.
	bound2, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "")
	if err != nil {
		t.Fatalf("second BindLeidenMatchPlan: %v", err)
	}

	// Reconstruct expected seed IDs (they should match original, not mutated).
	if len(bound2.Spec.SeedNodeIDs) != 2 {
		t.Fatalf("second bind: want 2 seeds, got %d", len(bound2.Spec.SeedNodeIDs))
	}
	if bound2.Spec.SeedNodeIDs[0] == 99999 || bound2.Spec.SeedNodeIDs[1] == 99999 {
		t.Error("mutation leaked into second binding")
	}
	if len(bound2.Spec.EdgeKinds) != 1 || bound2.Spec.EdgeKinds[0] != 10 {
		t.Errorf("EdgeKinds mutated: want [10], got %v", bound2.Spec.EdgeKinds)
	}

	t.Log("✅ defensive ownership")
}

// =============================================================================
// Test 10: Determinism
// =============================================================================

func TestLeidenBind_Determinism(t *testing.T) {
	h := newBindHarness(t)
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("a", "seeds")
	h.insertLabeled("b", "seeds")
	h.insertLabeled("c", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	plan := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*2..4]->(target)`)
	plan.Collection = "nodes"

	var first *BoundLeidenMatchPlan
	for i := 0; i < 10; i++ {
		bound, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "")
		if err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
		if i == 0 {
			first = bound
			continue
		}
		if !reflect.DeepEqual(bound, first) {
			t.Fatalf("call %d: result differs from first call", i)
		}
	}

	t.Log("✅ determinism across 10 calls")
}

// =============================================================================
// Test 11: No mutation of epoch state
// =============================================================================

func TestLeidenBind_NoMutation(t *testing.T) {
	h := newBindHarness(t)
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("s", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	genBefore := epoch.generation
	recordsBefore, _ := epoch.ListRecords(context.Background(), "nodes")
	recCount := len(recordsBefore)

	plan := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*1..3]->(target)`)
	plan.Collection = "nodes"

	for i := 0; i < 5; i++ {
		_, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "")
		if err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
	}

	genAfter := epoch.generation
	recordsAfter, _ := epoch.ListRecords(context.Background(), "nodes")

	if genAfter != genBefore {
		t.Errorf("generation changed: %d → %d", genBefore, genAfter)
	}
	if len(recordsAfter) != recCount {
		t.Errorf("record count changed: %d → %d", recCount, len(recordsAfter))
	}

	t.Log("✅ binding does not mutate epoch state")
}

// =============================================================================
// Test 12: Explicit collection argument
// =============================================================================

func TestLeidenBind_ExplicitCollection(t *testing.T) {
	h := newBindHarness(t)
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("s", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// Plan has no collection set; pass explicitly.
	plan := lowerPlan(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)`)
	// plan.Collection is ""

	bound, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "nodes")
	if err != nil {
		t.Fatalf("BindLeidenMatchPlan with explicit collection: %v", err)
	}
	if bound.Spec.Collection != "nodes" {
		t.Errorf("Collection: want 'nodes', got %q", bound.Spec.Collection)
	}
	t.Log("✅ explicit collection argument")
}
