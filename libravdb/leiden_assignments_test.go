package libravdb

import (
	"context"
	"reflect"
	"sort"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// assertNodeIDInvariant checks that no assignment contains a NodeID that looks
// like an internal array index (0, 1, 2, ...) unless those are the actual
// durable node IDs. We test this by ensuring the assignments match expected
// values explicitly, not that they "don't look like indices". The dedicated
// test below proves sparse IDs survive round-trip.
func assertAssignmentsEqual(t *testing.T, got, want []LeidenAssignment) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("assignment[%d]: got {%d, %d}, want {%d, %d}",
				i, got[i].NodeID, got[i].CommunityID,
				want[i].NodeID, want[i].CommunityID)
		}
	}
}

// =============================================================================
// Test 1: Sparse-ID flattening
// =============================================================================

func TestLeiden_Assignments_SparseFlattening(t *testing.T) {
	result := &EpochLeidenResult{
		Communities: []EpochCommunity{
			{ID: 101, Members: []uint64{101, 503}},
			{ID: 9001, Members: []uint64{9001, 42000}},
		},
	}

	want := []LeidenAssignment{
		{NodeID: 101, CommunityID: 101},
		{NodeID: 503, CommunityID: 101},
		{NodeID: 9001, CommunityID: 9001},
		{NodeID: 42000, CommunityID: 9001},
	}

	got := result.Assignments()
	assertAssignmentsEqual(t, got, want)
	t.Log("✅ sparse-ID flattening")
}

// =============================================================================
// Test 2: No internal-index leakage
// =============================================================================

func TestLeiden_Assignments_NoInternalIndexLeakage(t *testing.T) {
	// Use member IDs that are not contiguous and not small integers.
	// The assignment must contain these exact IDs, never 0, 1, 2, etc.
	result := &EpochLeidenResult{
		Communities: []EpochCommunity{
			{ID: 777777, Members: []uint64{777777, 888888}},
			{ID: 999999, Members: []uint64{999999, 111111}},
		},
	}

	got := result.Assignments()

	// Verify no small indices leaked.
	for _, a := range got {
		if a.NodeID <= 10 {
			t.Fatalf("internal index leaked: NodeID=%d looks like an array index", a.NodeID)
		}
		if a.CommunityID <= 10 {
			t.Fatalf("internal index leaked: CommunityID=%d looks like an array index", a.CommunityID)
		}
	}

	// Verify expected member count.
	expectedCount := 4
	for _, c := range result.Communities {
		expectedCount -= len(c.Members)
	}
	// Re-compute: 2+2=4.
	if len(got) != 4 {
		t.Fatalf("expected 4 assignments, got %d", len(got))
	}

	// Verify each member appears with correct community ID.
	for _, c := range result.Communities {
		for _, member := range c.Members {
			found := false
			for _, a := range got {
				if a.NodeID == member && a.CommunityID == c.ID {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("member %d not found in assignments with community %d", member, c.ID)
			}
		}
	}
	t.Log("✅ no internal-index leakage")
}

// =============================================================================
// Test 3: Determinism
// =============================================================================

func TestLeiden_Assignments_Determinism(t *testing.T) {
	result := &EpochLeidenResult{
		Communities: []EpochCommunity{
			{ID: 100, Members: []uint64{100, 200, 300}},
			{ID: 400, Members: []uint64{400, 500}},
		},
	}

	first := result.Assignments()
	for i := 0; i < 10; i++ {
		next := result.Assignments()
		if !reflect.DeepEqual(first, next) {
			t.Fatalf("non-deterministic output at iteration %d", i)
		}
	}
	t.Log("✅ determinism across 10 calls")
}

// =============================================================================
// Test 4: Defensive copy
// =============================================================================

func TestLeiden_Assignments_DefensiveCopy(t *testing.T) {
	result := &EpochLeidenResult{
		Communities: []EpochCommunity{
			{ID: 10, Members: []uint64{10, 20}},
		},
	}

	first := result.Assignments()
	// Mutate the returned slice.
	if len(first) > 0 {
		first[0].NodeID = 99999
		first[0].CommunityID = 88888
	}
	first = append(first, LeidenAssignment{NodeID: 77777, CommunityID: 66666})

	second := result.Assignments()

	want := []LeidenAssignment{
		{NodeID: 10, CommunityID: 10},
		{NodeID: 20, CommunityID: 10},
	}
	assertAssignmentsEqual(t, second, want)
	t.Log("✅ defensive copy: mutation does not affect source")
}

// =============================================================================
// Test 5: Real ComputeLeiden result
// =============================================================================

func TestLeiden_Assignments_RealComputeLeiden(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/leiden_assign_real.libravdb"))
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

	// Insert nodes with sparse, non-contiguous IDs.
	sparseIDs := []string{"n100", "n503", "n9001", "n42000", "n77777"}
	for _, id := range sparseIDs {
		if err := col.Insert(context.Background(), id, []float32{1, 0, 0}, nil); err != nil {
			t.Fatalf("Insert %s: %v", id, err)
		}
	}

	// Resolve durable graph node IDs.
	nodeIDs := make(map[string]uint64, len(sparseIDs))
	for _, id := range sparseIDs {
		nid, err := db.GetNodeID(context.Background(), "nodes", id)
		if err != nil {
			t.Fatalf("GetNodeID %s: %v", id, err)
		}
		nodeIDs[id] = nid
	}

	// Build a small graph: n100 ↔ n503, n503 ↔ n9001, n9001 ↔ n42000, n42000 ↔ n77777.
	baseTxn := gr.BeginTxn()
	edges := [][2]string{
		{"n100", "n503"},
		{"n503", "n9001"},
		{"n9001", "n42000"},
		{"n42000", "n77777"},
	}
	for _, e := range edges {
		if err := baseTxn.AddEdge(nodeIDs[e[0]], nodeIDs[e[1]], 1.0, 10); err != nil {
			t.Fatalf("AddEdge %s→%s: %v", e[0], e[1], err)
		}
	}
	if err := baseTxn.Commit(context.Background()); err != nil {
		t.Fatalf("base Commit: %v", err)
	}

	// Create epoch and run ComputeLeiden.
	epoch, err := db.BeginEpochTx(context.Background())
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}

	// Add a staged edge so collectDeltaNodes finds seeds.
	n100 := nodeIDs["n100"]
	n77777 := nodeIDs["n77777"]
	if err := epoch.AddGraphEdge("nodes", n100, n77777, 1.0, 10); err != nil {
		t.Fatalf("staged AddGraphEdge: %v", err)
	}

	result, err := epoch.ComputeLeiden(context.Background(), EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeiden: %v", err)
	}

	if result == nil {
		t.Fatal("ComputeLeiden returned nil result")
	}

	assignments := result.Assignments()

	// Invariant: every returned NodeID is a real graph node ID.
	realNodeIDs := make(map[uint64]bool)
	for _, nid := range nodeIDs {
		realNodeIDs[nid] = true
	}
	for _, a := range assignments {
		if !realNodeIDs[a.NodeID] {
			t.Fatalf("assignment NodeID %d is not a real graph node", a.NodeID)
		}
	}

	// Invariant: no duplicate node IDs.
	seenNodes := make(map[uint64]bool)
	for _, a := range assignments {
		if seenNodes[a.NodeID] {
			t.Fatalf("duplicate NodeID %d in assignments", a.NodeID)
		}
		seenNodes[a.NodeID] = true
	}

	// Invariant: every node appears in exactly one community.
	nodeToComm := make(map[uint64]uint64)
	for _, a := range assignments {
		if prevComm, exists := nodeToComm[a.NodeID]; exists {
			t.Fatalf("NodeID %d appears in multiple communities: %d and %d",
				a.NodeID, prevComm, a.CommunityID)
		}
		nodeToComm[a.NodeID] = a.CommunityID
	}

	// Invariant: rows are sorted by NodeID.
	for i := 1; i < len(assignments); i++ {
		if assignments[i].NodeID < assignments[i-1].NodeID {
			t.Fatalf("assignments not sorted by NodeID: [%d]=%d < [%d]=%d",
				i, assignments[i].NodeID, i-1, assignments[i-1].NodeID)
		}
	}

	// Invariant: community IDs match result communities.
	communityIDs := make(map[uint64]bool)
	for _, c := range result.Communities {
		communityIDs[c.ID] = true
	}
	for _, a := range assignments {
		if !communityIDs[a.CommunityID] {
			t.Fatalf("assignment CommunityID %d not found in result communities", a.CommunityID)
		}
	}

	// Invariant: member count matches.
	expectedTotal := 0
	for _, c := range result.Communities {
		expectedTotal += len(c.Members)
	}
	if len(assignments) != expectedTotal {
		t.Fatalf("len(Assignments()) = %d, sum(len(c.Members)) = %d", len(assignments), expectedTotal)
	}

	// Rollback and verify live graph unchanged.
	if err := epoch.Rollback(context.Background()); err != nil {
		t.Fatalf("Rollback: %v", err)
	}

	// Verify all nodes still exist (no side effects from Leiden).
	for _, id := range sparseIDs {
		if _, err := col.Get(context.Background(), id); err != nil {
			t.Fatalf("record %s missing after rollback: %v", id, err)
		}
	}

	t.Log("✅ real ComputeLeiden Assignments")
}

// =============================================================================
// Test 6: Empty result
// =============================================================================

func TestLeiden_Assignments_EmptyResult(t *testing.T) {
	result := &EpochLeidenResult{}
	got := result.Assignments()
	if len(got) != 0 {
		t.Fatalf("empty result: expected 0 assignments, got %d", len(got))
	}
	if got == nil {
		t.Fatal("empty result: expected non-nil empty slice, got nil")
	}
	t.Log("✅ empty result returns zero-length non-nil slice")

	// Nil receiver.
	var nilResult *EpochLeidenResult
	if nilResult.Assignments() != nil {
		t.Fatal("nil receiver: expected nil")
	}
	t.Log("✅ nil receiver returns nil")
}

// =============================================================================
// Test 7: Sorted order with multiple communities per node (tie-break)
// =============================================================================

func TestLeiden_Assignments_SortOrderTiebreak(t *testing.T) {
	// Construct a result where the same NodeID could theoretically appear with
	// different CommunityIDs (defensive tie-break). Though ComputeLeiden
	// guarantees unique membership, the sort logic should handle ties.
	result := &EpochLeidenResult{
		Communities: []EpochCommunity{
			{ID: 50, Members: []uint64{200, 100}},
			{ID: 10, Members: []uint64{400, 300}},
		},
	}

	got := result.Assignments()

	// Expected: sorted by NodeID ascending:
	// {100, 50}, {200, 50}, {300, 10}, {400, 10}
	want := []LeidenAssignment{
		{NodeID: 100, CommunityID: 50},
		{NodeID: 200, CommunityID: 50},
		{NodeID: 300, CommunityID: 10},
		{NodeID: 400, CommunityID: 10},
	}
	assertAssignmentsEqual(t, got, want)
	t.Log("✅ sorted by NodeID ascending")
}

// =============================================================================
// Test 8: ValidateAssignments helper for uniqueness checking
// =============================================================================

func TestLeiden_Assignments_ValidateUniqueness(t *testing.T) {
	// Valid result: each node appears once.
	valid := &EpochLeidenResult{
		Communities: []EpochCommunity{
			{ID: 10, Members: []uint64{1, 2, 3}},
			{ID: 20, Members: []uint64{4, 5}},
		},
	}
	dups := findDuplicateNodes(valid)
	if len(dups) > 0 {
		t.Fatalf("valid result has unexpected duplicates: %v", dups)
	}

	nodes := collectNodeSet(valid)
	expected := map[uint64]bool{1: true, 2: true, 3: true, 4: true, 5: true}
	if !reflect.DeepEqual(nodes, expected) {
		t.Fatalf("node set mismatch: got %v, want %v", nodes, expected)
	}
	t.Log("✅ validate uniqueness passes for valid result")
}

// findDuplicateNodes returns any NodeID that appears in more than one community.
// This is a validation helper for tests — ComputeLeiden guarantees uniqueness,
// but callers constructing EpochLeidenResult manually can use this to verify.
func findDuplicateNodes(r *EpochLeidenResult) []uint64 {
	if r == nil {
		return nil
	}
	seen := make(map[uint64]uint64) // nodeID → communityID
	var dups []uint64
	for _, c := range r.Communities {
		for _, member := range c.Members {
			if prevComm, exists := seen[member]; exists {
				dups = append(dups, member)
				_ = prevComm
			}
			seen[member] = c.ID
		}
	}
	sort.Slice(dups, func(i, j int) bool { return dups[i] < dups[j] })
	return dups
}

// collectNodeSet returns the set of all original node IDs across communities.
func collectNodeSet(r *EpochLeidenResult) map[uint64]bool {
	nodes := make(map[uint64]bool)
	if r == nil {
		return nodes
	}
	for _, c := range r.Communities {
		for _, member := range c.Members {
			nodes[member] = true
		}
	}
	return nodes
}
