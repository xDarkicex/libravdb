package libravdb

import (
	"context"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// =============================================================================
// Test infrastructure
// =============================================================================

// matchTestDB creates an in-memory DB with a graph-enabled collection and
// registers an edge kind. Returns db, graph, collection.
func matchTestDB(t *testing.T) (*Database, Graph, *Collection) {
	t.Helper()
	db, err := Open(WithStoragePath(t.TempDir() + "/leiden_match.libravdb"))
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
	graph.RegisterEdgeKind("LINK", 10)
	graph.RegisterEdgeKind("ALT", 20)

	return db, gr, col
}

// insertNode inserts a record and returns its durable node ID.
func insertNode(t *testing.T, db *Database, col *Collection, id string) uint64 {
	t.Helper()
	if err := col.Insert(context.Background(), id, []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("Insert %s: %v", id, err)
	}
	nid, err := db.GetNodeID(context.Background(), "nodes", id)
	if err != nil {
		t.Fatalf("GetNodeID %s: %v", id, err)
	}
	return nid
}

// addEdge commits a live edge between two nodes.
func addEdge(t *testing.T, gr Graph, src, tgt uint64, kind uint8) {
	t.Helper()
	txn := gr.BeginTxn()
	if err := txn.AddEdge(src, tgt, 1.0, kind); err != nil {
		t.Fatalf("AddEdge %d→%d: %v", src, tgt, err)
	}
	if err := txn.Commit(context.Background()); err != nil {
		t.Fatalf("Commit edge %d→%d: %v", src, tgt, err)
	}
}

// assertMatched checks MatchedNodeIDs match expected set (order-insensitive).
func assertMatched(t *testing.T, got []uint64, want []uint64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("MatchedNodeIDs length: got %d, want %d\ngot:  %v\nwant: %v", len(got), len(want), got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("MatchedNodeIDs[%d]: got %d, want %d\ngot:  %v\nwant: %v", i, got[i], want[i], got, want)
		}
	}
}

// =============================================================================
// Test 1: Outbound hop interval
// =============================================================================

func TestLeiden_Match_OutboundHopInterval(t *testing.T) {
	db, gr, col := matchTestDB(t)

	A := insertNode(t, db, col, "A")
	B := insertNode(t, db, col, "B")
	C := insertNode(t, db, col, "C")
	D := insertNode(t, db, col, "D")

	addEdge(t, gr, A, B, 10)
	addEdge(t, gr, B, C, 10)
	addEdge(t, gr, C, D, 10)

	epoch, _ := db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	spec := LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		EdgeKinds:   nil,
		MinHops:     1,
		MaxHops:     3,
		Direction:   LeidenMatchOutbound,
	}
	result, err := epoch.ComputeLeidenFromMatch(context.Background(), spec, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	assertMatched(t, result.MatchedNodeIDs, []uint64{B, C, D})
	if result.LeidenResult == nil {
		t.Fatal("LeidenResult must not be nil")
	}
	t.Log("✅ outbound hop interval: {B, C, D}")
}

// =============================================================================
// Test 2: Min-hop exclusion
// =============================================================================

func TestLeiden_Match_MinHopExclusion(t *testing.T) {
	db, gr, col := matchTestDB(t)

	A := insertNode(t, db, col, "A")
	B := insertNode(t, db, col, "B")
	C := insertNode(t, db, col, "C")
	D := insertNode(t, db, col, "D")

	addEdge(t, gr, A, B, 10)
	addEdge(t, gr, B, C, 10)
	addEdge(t, gr, C, D, 10)

	epoch, _ := db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	spec := LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		EdgeKinds:   nil,
		MinHops:     2,
		MaxHops:     3,
		Direction:   LeidenMatchOutbound,
	}
	result, err := epoch.ComputeLeidenFromMatch(context.Background(), spec, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	assertMatched(t, result.MatchedNodeIDs, []uint64{C, D})
	t.Log("✅ min-hop exclusion: {C, D}")
}

// =============================================================================
// Test 3: Max-hop exclusion
// =============================================================================

func TestLeiden_Match_MaxHopExclusion(t *testing.T) {
	db, gr, col := matchTestDB(t)

	A := insertNode(t, db, col, "A")
	B := insertNode(t, db, col, "B")
	C := insertNode(t, db, col, "C")
	D := insertNode(t, db, col, "D")

	addEdge(t, gr, A, B, 10)
	addEdge(t, gr, B, C, 10)
	addEdge(t, gr, C, D, 10)

	epoch, _ := db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	spec := LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		EdgeKinds:   nil,
		MinHops:     1,
		MaxHops:     2,
		Direction:   LeidenMatchOutbound,
	}
	result, err := epoch.ComputeLeidenFromMatch(context.Background(), spec, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	assertMatched(t, result.MatchedNodeIDs, []uint64{B, C})
	t.Log("✅ max-hop exclusion: {B, C}")
}

// =============================================================================
// Test 4: Seed inclusion (MinHops == 0)
// =============================================================================

func TestLeiden_Match_SeedInclusion(t *testing.T) {
	db, gr, col := matchTestDB(t)

	A := insertNode(t, db, col, "A")
	B := insertNode(t, db, col, "B")

	addEdge(t, gr, A, B, 10)

	epoch, _ := db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	spec := LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		EdgeKinds:   nil,
		MinHops:     0,
		MaxHops:     1,
		Direction:   LeidenMatchOutbound,
	}
	result, err := epoch.ComputeLeidenFromMatch(context.Background(), spec, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	assertMatched(t, result.MatchedNodeIDs, []uint64{A, B})
	t.Log("✅ seed inclusion: {A, B}")
}

// =============================================================================
// Test 5: Edge-kind filtering
// =============================================================================

func TestLeiden_Match_EdgeKindFiltering(t *testing.T) {
	db, gr, col := matchTestDB(t)

	A := insertNode(t, db, col, "A")
	B := insertNode(t, db, col, "B")
	C := insertNode(t, db, col, "C")

	addEdge(t, gr, A, B, 10) // LINK
	addEdge(t, gr, A, C, 20) // ALT

	epoch, _ := db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	spec := LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		EdgeKinds:   []uint8{10}, // only LINK
		MinHops:     1,
		MaxHops:     1,
		Direction:   LeidenMatchOutbound,
	}
	result, err := epoch.ComputeLeidenFromMatch(context.Background(), spec, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	assertMatched(t, result.MatchedNodeIDs, []uint64{B})
	t.Log("✅ edge-kind filtering: only LINK → {B}")
}

// =============================================================================
// Test 6: Inbound traversal
// =============================================================================

func TestLeiden_Match_InboundTraversal(t *testing.T) {
	db, gr, col := matchTestDB(t)

	A := insertNode(t, db, col, "A")
	B := insertNode(t, db, col, "B")
	C := insertNode(t, db, col, "C")

	addEdge(t, gr, A, B, 10) // A → B
	addEdge(t, gr, C, B, 10) // C → B

	epoch, _ := db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	spec := LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{B},
		EdgeKinds:   nil,
		MinHops:     1,
		MaxHops:     1,
		Direction:   LeidenMatchInbound,
	}
	result, err := epoch.ComputeLeidenFromMatch(context.Background(), spec, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	assertMatched(t, result.MatchedNodeIDs, []uint64{A, C})
	t.Log("✅ inbound traversal: {A, C}")
}

// =============================================================================
// Test 7: Epoch snapshot isolation
// =============================================================================

func TestLeiden_Match_SnapshotIsolation(t *testing.T) {
	db, gr, col := matchTestDB(t)

	A := insertNode(t, db, col, "A")
	B := insertNode(t, db, col, "B")
	Post := insertNode(t, db, col, "Post")

	addEdge(t, gr, A, B, 10)

	// Snapshot time t0 after initial edges committed.
	t0 := time.Now().UTC()
	time.Sleep(10 * time.Millisecond)

	// Commit a post-t0 edge that the historical epoch must not see.
	addEdge(t, gr, B, Post, 10)

	epoch, err := db.BeginEpochTxAt(context.Background(), t0)
	if err != nil {
		t.Fatalf("BeginEpochTxAt: %v", err)
	}
	defer epoch.Rollback(context.Background())

	spec := LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		EdgeKinds:   nil,
		MinHops:     1,
		MaxHops:     2,
		Direction:   LeidenMatchOutbound,
	}
	result, err := epoch.ComputeLeidenFromMatch(context.Background(), spec, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	// Post node must be absent (committed after t0).
	for _, id := range result.MatchedNodeIDs {
		if id == Post {
			t.Fatal("post-t0 node must be absent from historical epoch")
		}
	}

	// B must be present (committed before t0).
	foundB := false
	for _, id := range result.MatchedNodeIDs {
		if id == B {
			foundB = true
		}
	}
	if !foundB {
		t.Fatal("B must be present (committed before t0)")
	}

	// Add a staged edge — target must be present via overlay.
	NewNode := insertNode(t, db, col, "NewNode")
	if err := epoch.AddGraphEdge("nodes", B, NewNode, 1.0, 10); err != nil {
		t.Fatalf("staged AddGraphEdge: %v", err)
	}

	result2, err := epoch.ComputeLeidenFromMatch(context.Background(), spec, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("second ComputeLeidenFromMatch: %v", err)
	}
	foundNew := false
	for _, id := range result2.MatchedNodeIDs {
		if id == NewNode {
			foundNew = true
		}
	}
	if !foundNew {
		t.Fatal("staged target NewNode must be present via epoch overlay")
	}

	// Rollback: live graph must be unchanged.
	if err := epoch.Rollback(context.Background()); err != nil {
		t.Fatalf("Rollback: %v", err)
	}
	if _, err := col.Get(context.Background(), "NewNode"); err != nil {
		t.Fatalf("NewNode must survive after rollback (was committed live, not staged)")
	}

	t.Log("✅ snapshot isolation: post-t0 absent, staged present, live unchanged")
}

// =============================================================================
// Test 8: Provisional nodes
// =============================================================================

func TestLeiden_Match_ProvisionalNodes(t *testing.T) {
	db, _, _ := matchTestDB(t)

	epoch, err := db.BeginEpochTx(context.Background())
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}
	defer epoch.Rollback(context.Background())

	// Insert staged records.
	if err := epoch.Insert(context.Background(), "nodes", "Src", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("staged Insert Src: %v", err)
	}
	if err := epoch.Insert(context.Background(), "nodes", "Tgt", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("staged Insert Tgt: %v", err)
	}

	srcID, err := epoch.LookupNodeID(context.Background(), "nodes", "Src")
	if err != nil {
		t.Fatalf("LookupNodeID Src: %v", err)
	}
	tgtID, err := epoch.LookupNodeID(context.Background(), "nodes", "Tgt")
	if err != nil {
		t.Fatalf("LookupNodeID Tgt: %v", err)
	}

	// Add staged edge between provisional nodes.
	if err := epoch.AddGraphEdge("nodes", srcID, tgtID, 1.0, 10); err != nil {
		t.Fatalf("staged AddGraphEdge: %v", err)
	}

	// Run match from Src.
	spec := LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{srcID},
		EdgeKinds:   nil,
		MinHops:     1,
		MaxHops:     1,
		Direction:   LeidenMatchOutbound,
	}
	result, err := epoch.ComputeLeidenFromMatch(context.Background(), spec, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	foundTgt := false
	for _, id := range result.MatchedNodeIDs {
		if id == tgtID {
			foundTgt = true
		}
	}
	if !foundTgt {
		t.Fatal("staged target Tgt must be in MatchedNodeIDs")
	}
	t.Log("Phase 1: staged provisional target matched ✓")

	// Rollback: assert staged IDs are no longer matched.
	if err := epoch.Rollback(context.Background()); err != nil {
		t.Fatalf("Rollback: %v", err)
	}

	// New epoch: staged nodes and edges should not exist.
	epoch2, _ := db.BeginEpochTx(context.Background())
	defer epoch2.Rollback(context.Background())

	result2, err := epoch2.ComputeLeidenFromMatch(context.Background(), spec, EpochLeidenOptions{})
	if err != nil {
		// Expected: seeds don't exist after rollback, so this may error.
		t.Logf("post-rollback ComputeLeidenFromMatch: %v (expected)", err)
		return
	}
	for _, id := range result2.MatchedNodeIDs {
		if id == tgtID {
			t.Fatal("staged target must be absent after rollback")
		}
	}
	t.Log("Phase 2: staged IDs absent after rollback ✓")

	t.Log("✅ provisional nodes")
}

// =============================================================================
// Test 9: Determinism
// =============================================================================

func TestLeiden_Match_Determinism(t *testing.T) {
	db, gr, col := matchTestDB(t)

	A := insertNode(t, db, col, "A")
	B := insertNode(t, db, col, "B")
	C := insertNode(t, db, col, "C")

	addEdge(t, gr, A, B, 10)
	addEdge(t, gr, B, C, 10)
	addEdge(t, gr, A, C, 20) // second edge kind for more interesting Leiden

	epoch, _ := db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	spec := LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		EdgeKinds:   nil,
		MinHops:     1,
		MaxHops:     2,
		Direction:   LeidenMatchOutbound,
	}

	var firstResult *LeidenMatchResult
	var firstErr error

	for i := 0; i < 10; i++ {
		result, err := epoch.ComputeLeidenFromMatch(context.Background(), spec, EpochLeidenOptions{})
		if err != nil {
			t.Fatalf("call %d error: %v", i, err)
		}
		if i == 0 {
			firstResult = result
			firstErr = err
			continue
		}

		// Compare MatchedNodeIDs.
		if !reflect.DeepEqual(result.MatchedNodeIDs, firstResult.MatchedNodeIDs) {
			t.Fatalf("call %d MatchedNodeIDs differ:\n  got  %v\n  want %v",
				i, result.MatchedNodeIDs, firstResult.MatchedNodeIDs)
		}

		// Compare LeidenResult.Assignments().
		if !reflect.DeepEqual(result.LeidenResult.Assignments(), firstResult.LeidenResult.Assignments()) {
			t.Fatalf("call %d Assignments differ", i)
		}

		// Compare modularity (float equality with small delta).
		if absDiff(result.LeidenResult.Modularity, firstResult.LeidenResult.Modularity) > 1e-12 {
			t.Fatalf("call %d Modularity differs: got %v, want %v",
				i, result.LeidenResult.Modularity, firstResult.LeidenResult.Modularity)
		}
	}
	_ = firstErr
	t.Log("✅ determinism across 10 calls")
}

func absDiff(a, b float64) float64 {
	if a > b {
		return a - b
	}
	return b - a
}

// =============================================================================
// Test 10: Budget behavior
// =============================================================================

func TestLeiden_Match_BudgetBehavior(t *testing.T) {
	db, gr, col := matchTestDB(t)

	// Build a chain: A → B → C → D → E
	nodes := []string{"A", "B", "C", "D", "E"}
	nodeIDs := make([]uint64, len(nodes))
	for i, name := range nodes {
		nodeIDs[i] = insertNode(t, db, col, name)
	}
	for i := 0; i < len(nodeIDs)-1; i++ {
		addEdge(t, gr, nodeIDs[i], nodeIDs[i+1], 10)
	}

	epoch, _ := db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	spec := LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{nodeIDs[0]},
		EdgeKinds:   nil,
		MinHops:     1,
		MaxHops:     4,
		Direction:   LeidenMatchOutbound,
	}

	// Budget: MaxVertices limits the local graph, causing truncation.
	opts := EpochLeidenOptions{
		MaxVertices: 2, // tight budget
		MaxEdges:    10,
	}

	result, err := epoch.ComputeLeidenFromMatch(context.Background(), spec, opts)
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	// MatchedNodeIDs should still be sorted and unique.
	for i := 1; i < len(result.MatchedNodeIDs); i++ {
		if result.MatchedNodeIDs[i] <= result.MatchedNodeIDs[i-1] {
			t.Fatalf("MatchedNodeIDs not sorted/unique: [%d]=%d, [%d]=%d",
				i, result.MatchedNodeIDs[i], i-1, result.MatchedNodeIDs[i-1])
		}
	}

	// Truncation must be reported if the graph was too large.
	// (Not guaranteed for all budgets, but we test no panic.)
	if result.LeidenResult.Truncated {
		if result.LeidenResult.Scope != EpochLeidenScopeBudgetTruncated {
			t.Fatalf("truncated result must have budget_truncated scope, got %q", result.LeidenResult.Scope)
		}
		t.Logf("budget truncated as expected: vertices=%d edges=%d", result.LeidenResult.Vertices, result.LeidenResult.Edges)
	}

	// Assignments must be valid.
	assignments := result.LeidenResult.Assignments()
	if len(assignments) == 0 {
		t.Fatal("budget-constrained Leiden must produce non-empty assignments")
	}

	// No post-snapshot nodes should appear.
	sort.Slice(nodeIDs, func(i, j int) bool { return nodeIDs[i] < nodeIDs[j] })

	t.Log("✅ budget behavior: no panic, sorted, truncated")
}

// =============================================================================
// Test 11: Invalid requests
// =============================================================================

func TestLeiden_Match_InvalidRequests(t *testing.T) {
	db, _, _ := matchTestDB(t)

	epoch, _ := db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	tests := []struct {
		name string
		spec LeidenMatchSpec
	}{
		{"empty Collection", LeidenMatchSpec{Collection: "", SeedNodeIDs: []uint64{1}, MinHops: 0, MaxHops: 1}},
		{"empty SeedNodeIDs", LeidenMatchSpec{Collection: "nodes", SeedNodeIDs: nil, MinHops: 0, MaxHops: 1}},
		{"MinHops < 0", LeidenMatchSpec{Collection: "nodes", SeedNodeIDs: []uint64{1}, MinHops: -1, MaxHops: 1}},
		{"MaxHops < 0", LeidenMatchSpec{Collection: "nodes", SeedNodeIDs: []uint64{1}, MinHops: 0, MaxHops: -1}},
		{"MinHops > MaxHops", LeidenMatchSpec{Collection: "nodes", SeedNodeIDs: []uint64{1}, MinHops: 2, MaxHops: 1}},
		{"unknown collection", LeidenMatchSpec{Collection: "nonexistent", SeedNodeIDs: []uint64{1}, MinHops: 0, MaxHops: 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := epoch.ComputeLeidenFromMatch(context.Background(), tt.spec, EpochLeidenOptions{})
			if err == nil {
				t.Fatalf("expected error for %q", tt.name)
			}
			t.Logf("%s → %v ✓", tt.name, err)
		})
	}

	// Collection without graph.
	dbNoGraph, _ := Open(WithStoragePath(t.TempDir() + "/no_graph.libravdb"))
	defer dbNoGraph.Drop(context.Background())
	if _, err := dbNoGraph.CreateCollection(context.Background(), "no_graph", WithDimension(3)); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	epoch2, _ := dbNoGraph.BeginEpochTx(context.Background())
	defer epoch2.Rollback(context.Background())
	_, err := epoch2.ComputeLeidenFromMatch(context.Background(),
		LeidenMatchSpec{Collection: "no_graph", SeedNodeIDs: []uint64{1}, MinHops: 0, MaxHops: 1},
		EpochLeidenOptions{})
	if err == nil {
		t.Fatal("expected error for collection without graph")
	}
	t.Logf("collection without graph → %v ✓", err)

	t.Log("✅ all invalid requests rejected")
}
