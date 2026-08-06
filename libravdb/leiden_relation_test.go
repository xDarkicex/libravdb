package libravdb

import (
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// =============================================================================
// Test infrastructure
// =============================================================================

type relTestHarness struct {
	db  *Database
	gr  Graph
	col *Collection
}

func newRelTestHarness(t *testing.T) *relTestHarness {
	t.Helper()
	db, err := Open(WithStoragePath(t.TempDir() + "/leiden_rel.libravdb"))
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

	return &relTestHarness{db: db, gr: gr, col: col}
}

func (h *relTestHarness) insert(id string) uint64 {
	if err := h.col.Insert(context.Background(), id, []float32{1, 0, 0}, nil); err != nil {
		panic(err)
	}
	nid, err := h.db.GetNodeID(context.Background(), "nodes", id)
	if err != nil {
		panic(err)
	}
	return nid
}

func (h *relTestHarness) addEdge(src, tgt uint64, kind uint8) {
	txn := h.gr.BeginTxn()
	if err := txn.AddEdge(src, tgt, 1.0, kind); err != nil {
		panic(err)
	}
	if err := txn.Commit(context.Background()); err != nil {
		panic(err)
	}
}

// assertRelationRows checks rows match expected values exactly.
func assertRelationRows(t *testing.T, got, want []LeidenRelationRow) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("row count: got %d, want %d\ngot:  %v\nwant: %v", len(got), len(want), got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("row[%d]: got {%d %d %q %q}, want {%d %d %q %q}",
				i,
				got[i].NodeID, got[i].CommunityID, got[i].Collection, got[i].RecordID,
				want[i].NodeID, want[i].CommunityID, want[i].Collection, want[i].RecordID,
			)
		}
	}
}

// =============================================================================
// Test 1: Basic materialization
// =============================================================================

func TestLeiden_Relation_BasicMaterialization(t *testing.T) {
	h := newRelTestHarness(t)
	A := h.insert("A")
	B := h.insert("B")
	C := h.insert("C")
	h.addEdge(A, B, 10)
	h.addEdge(B, C, 10)

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	mr, err := epoch.ComputeLeidenFromMatch(context.Background(), LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		MinHops:     1,
		MaxHops:     2,
		Direction:   LeidenMatchOutbound,
	}, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	rel, err := epoch.MaterializeLeidenRelation(context.Background(), mr)
	if err != nil {
		t.Fatalf("MaterializeLeidenRelation: %v", err)
	}

	if len(rel.Rows) == 0 {
		t.Fatal("expected non-empty rows")
	}

	// Every row must have non-zero NodeID, non-empty Collection and RecordID.
	for i, row := range rel.Rows {
		if row.NodeID == 0 {
			t.Errorf("row[%d]: NodeID is zero", i)
		}
		if row.CommunityID == 0 {
			t.Errorf("row[%d]: CommunityID is zero", i)
		}
		if row.Collection != "nodes" {
			t.Errorf("row[%d]: Collection = %q, want %q", i, row.Collection, "nodes")
		}
		if row.RecordID == "" {
			t.Errorf("row[%d]: RecordID is empty", i)
		}
	}

	// Propagated diagnostics.
	if !rel.Truncated && rel.Scope == EpochLeidenScopeBudgetTruncated {
		t.Error("Truncated=false but Scope=budget_truncated")
	}

	t.Logf("materialized %d rows, modularity=%v", len(rel.Rows), rel.Modularity)
	t.Log("✅ basic materialization")
}

// =============================================================================
// Test 2: MATCH filtering — only matched nodes appear
// =============================================================================

func TestLeiden_Relation_MatchFiltering(t *testing.T) {
	h := newRelTestHarness(t)
	A := h.insert("A")   // seed
	h.insert("I1")       // intermediate node 1 (hop 1)
	h.insert("I2")       // intermediate node 2 (hop 1)
	T1 := h.insert("T1") // terminal (hop 2)
	T2 := h.insert("T2") // terminal (hop 2)

	I1 := h.db.nodeIDForRecord("nodes", "I1")
	I2 := h.db.nodeIDForRecord("nodes", "I2")

	// A → I1 → T1,  A → I2 → T2
	h.addEdge(A, I1, 10)
	h.addEdge(I1, T1, 10)
	h.addEdge(A, I2, 10)
	h.addEdge(I2, T2, 10)

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// MinHops=2: only terminals should be matched, not intermediates.
	mr, err := epoch.ComputeLeidenFromMatch(context.Background(), LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		MinHops:     2,
		MaxHops:     2,
		Direction:   LeidenMatchOutbound,
	}, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	rel, err := epoch.MaterializeLeidenRelation(context.Background(), mr)
	if err != nil {
		t.Fatalf("MaterializeLeidenRelation: %v", err)
	}

	recordIDs := make(map[string]bool)
	for _, row := range rel.Rows {
		recordIDs[row.RecordID] = true
	}

	if recordIDs["I1"] || recordIDs["I2"] {
		t.Fatal("intermediate nodes must not appear in relation rows (MinHops=2)")
	}
	if !recordIDs["T1"] {
		t.Fatal("T1 must appear (hop 2, matched)")
	}
	if !recordIDs["T2"] {
		t.Fatal("T2 must appear (hop 2, matched)")
	}

	t.Log("✅ MATCH filtering: only terminal nodes appear")
}

// =============================================================================
// Test 3: Sparse IDs preserved (durable node IDs match GetNodeID)
// =============================================================================

func TestLeiden_Relation_SparseIDs(t *testing.T) {
	h := newRelTestHarness(t)

	// Use distinct record IDs. Durable node IDs assigned by the graph may be
	// small integers — that's normal. The invariant is that NodeID in the
	// relation equals the durable node ID returned by GetNodeID, never a
	// slice position or an internal array index.
	ids := []string{"r1", "r2", "r3", "r4"}
	nodes := make(map[string]uint64, len(ids))
	for _, id := range ids {
		nodes[id] = h.insert(id)
	}

	// r1 → r2 → r3 → r4
	h.addEdge(nodes["r1"], nodes["r2"], 10)
	h.addEdge(nodes["r2"], nodes["r3"], 10)
	h.addEdge(nodes["r3"], nodes["r4"], 10)

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	mr, err := epoch.ComputeLeidenFromMatch(context.Background(), LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{nodes["r1"]},
		MinHops:     1,
		MaxHops:     3,
		Direction:   LeidenMatchOutbound,
	}, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	rel, err := epoch.MaterializeLeidenRelation(context.Background(), mr)
	if err != nil {
		t.Fatalf("MaterializeLeidenRelation: %v", err)
	}

	// Build a reverse map: durable node ID → record ID.
	nodeToRecord := make(map[uint64]string)
	for _, id := range ids {
		nid, err := h.db.GetNodeID(context.Background(), "nodes", id)
		if err != nil {
			t.Fatalf("GetNodeID %s: %v", id, err)
		}
		nodeToRecord[nid] = id
	}

	// Every relation row's NodeID must match the durable node ID for that
	// record (not a slice index or internal offset).
	for _, row := range rel.Rows {
		expectedRecord, ok := nodeToRecord[row.NodeID]
		if !ok {
			t.Fatalf("NodeID %d does not correspond to any known record", row.NodeID)
		}
		if expectedRecord != row.RecordID {
			t.Fatalf("NodeID %d → RecordID %q, but GetNodeID says it should be %q",
				row.NodeID, row.RecordID, expectedRecord)
		}
	}

	// Verify matched records appear exactly once.
	seen := make(map[string]bool)
	for _, row := range rel.Rows {
		if seen[row.RecordID] {
			t.Fatalf("duplicate RecordID: %q", row.RecordID)
		}
		seen[row.RecordID] = true
	}

	// All three terminal nodes (r2, r3, r4) must be present.
	expectedIDs := map[string]bool{"r2": true, "r3": true, "r4": true}
	for _, row := range rel.Rows {
		delete(expectedIDs, row.RecordID)
	}
	if len(expectedIDs) > 0 {
		t.Fatalf("missing records in relation: %v", expectedIDs)
	}

	t.Log("✅ durable node IDs match GetNodeID, no internal indices leak")
}

// =============================================================================
// Test 4: Epoch staged record appears
// =============================================================================

func TestLeiden_Relation_StagedRecord(t *testing.T) {
	h := newRelTestHarness(t)
	Src := h.insert("Src")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// Stage a terminal record and edge.
	if err := epoch.Insert(context.Background(), "nodes", "StagedTgt", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("staged Insert: %v", err)
	}
	tgtID, err := epoch.LookupNodeID(context.Background(), "nodes", "StagedTgt")
	if err != nil {
		t.Fatalf("LookupNodeID: %v", err)
	}
	if err := epoch.AddGraphEdge("nodes", Src, tgtID, 1.0, 10); err != nil {
		t.Fatalf("staged AddGraphEdge: %v", err)
	}

	mr, err := epoch.ComputeLeidenFromMatch(context.Background(), LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{Src},
		MinHops:     1,
		MaxHops:     1,
		Direction:   LeidenMatchOutbound,
	}, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	rel, err := epoch.MaterializeLeidenRelation(context.Background(), mr)
	if err != nil {
		t.Fatalf("MaterializeLeidenRelation: %v", err)
	}

	found := false
	for _, row := range rel.Rows {
		if row.RecordID == "StagedTgt" && row.NodeID == tgtID {
			found = true
		}
	}
	if !found {
		t.Fatal("staged terminal must appear with provisional node ID and record ID")
	}

	t.Log("✅ epoch staged record appears in relation")
}

// =============================================================================
// Test 5: Snapshot isolation — post-t0 nodes excluded
// =============================================================================

func TestLeiden_Relation_SnapshotIsolation(t *testing.T) {
	h := newRelTestHarness(t)
	A := h.insert("A")
	B := h.insert("B")
	h.addEdge(A, B, 10)

	t0 := time.Now().UTC()
	time.Sleep(10 * time.Millisecond)

	// Post-t0: commit new node and edge.
	Post := h.insert("Post")
	h.addEdge(B, Post, 10)

	epoch, err := h.db.BeginEpochTxAt(context.Background(), t0)
	if err != nil {
		t.Fatalf("BeginEpochTxAt: %v", err)
	}
	defer epoch.Rollback(context.Background())

	mr, err := epoch.ComputeLeidenFromMatch(context.Background(), LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		MinHops:     1,
		MaxHops:     2,
		Direction:   LeidenMatchOutbound,
	}, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	rel, err := epoch.MaterializeLeidenRelation(context.Background(), mr)
	if err != nil {
		t.Fatalf("MaterializeLeidenRelation: %v", err)
	}

	// B must be present (committed before t0).
	foundB := false
	for _, row := range rel.Rows {
		if row.RecordID == "Post" {
			t.Fatal("post-t0 record must be absent from historical epoch")
		}
		if row.RecordID == "B" {
			foundB = true
		}
	}
	if !foundB {
		t.Fatal("B must be present (committed before t0)")
	}

	t.Log("✅ snapshot isolation: post-t0 record absent")
}

// =============================================================================
// Test 6: Staged delete and rollback
// =============================================================================

func TestLeiden_Relation_StagedDeleteAndRollback(t *testing.T) {
	h := newRelTestHarness(t)
	A := h.insert("A")
	B := h.insert("B")
	C := h.insert("C")
	h.addEdge(A, B, 10)
	h.addEdge(B, C, 10)

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// Stage a deletion of C (reachable terminal).
	if err := epoch.Delete(context.Background(), "nodes", "C"); err != nil {
		t.Fatalf("staged Delete C: %v", err)
	}

	// Materialize: C must be absent (deleted in epoch).
	mr, err := epoch.ComputeLeidenFromMatch(context.Background(), LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		MinHops:     1,
		MaxHops:     2,
		Direction:   LeidenMatchOutbound,
	}, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	rel, err := epoch.MaterializeLeidenRelation(context.Background(), mr)
	if err != nil {
		t.Fatalf("MaterializeLeidenRelation: %v", err)
	}

	for _, row := range rel.Rows {
		if row.RecordID == "C" {
			t.Fatal("deleted record C must be absent from relation")
		}
	}
	t.Log("Phase 1: staged delete → C absent ✓")

	// Savepoint, then rollback.
	if err := epoch.Savepoint("sp"); err != nil {
		t.Fatalf("Savepoint: %v", err)
	}
	// Delete B as well.
	if err := epoch.Delete(context.Background(), "nodes", "B"); err != nil {
		t.Fatalf("staged Delete B: %v", err)
	}

	// Rollback to savepoint: B restored, C still deleted.
	if err := epoch.RollbackTo("sp"); err != nil {
		t.Fatalf("RollbackTo: %v", err)
	}

	mr2, err := epoch.ComputeLeidenFromMatch(context.Background(), LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		MinHops:     1,
		MaxHops:     2,
		Direction:   LeidenMatchOutbound,
	}, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("second ComputeLeidenFromMatch: %v", err)
	}

	rel2, err := epoch.MaterializeLeidenRelation(context.Background(), mr2)
	if err != nil {
		t.Fatalf("second MaterializeLeidenRelation: %v", err)
	}

	hasB, hasC := false, false
	for _, row := range rel2.Rows {
		if row.RecordID == "B" {
			hasB = true
		}
		if row.RecordID == "C" {
			hasC = true
		}
	}
	if !hasB {
		t.Fatal("B must be restored after rollback")
	}
	if hasC {
		t.Fatal("C must remain deleted (deletion was before savepoint)")
	}
	t.Log("Phase 2: rollback restores B, C still absent ✓")

	t.Log("✅ staged delete and rollback")
}

// =============================================================================
// Test 7: Provisional cleanup
// =============================================================================

func TestLeiden_Relation_ProvisionalCleanup(t *testing.T) {
	h := newRelTestHarness(t)
	Src := h.insert("Src")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	if err := epoch.Savepoint("base"); err != nil {
		t.Fatalf("Savepoint base: %v", err)
	}

	// Stage terminal record and edge.
	if err := epoch.Insert(context.Background(), "nodes", "ProvisionalTgt", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("staged Insert: %v", err)
	}
	tgtID, err := epoch.LookupNodeID(context.Background(), "nodes", "ProvisionalTgt")
	if err != nil {
		t.Fatalf("LookupNodeID: %v", err)
	}
	if err := epoch.AddGraphEdge("nodes", Src, tgtID, 1.0, 10); err != nil {
		t.Fatalf("staged AddGraphEdge: %v", err)
	}

	// Materialize — must see provisional target.
	mr, err := epoch.ComputeLeidenFromMatch(context.Background(), LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{Src},
		MinHops:     1,
		MaxHops:     1,
		Direction:   LeidenMatchOutbound,
	}, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	rel, err := epoch.MaterializeLeidenRelation(context.Background(), mr)
	if err != nil {
		t.Fatalf("MaterializeLeidenRelation: %v", err)
	}
	found := false
	for _, row := range rel.Rows {
		if row.RecordID == "ProvisionalTgt" {
			found = true
		}
	}
	if !found {
		t.Fatal("provisional target must be in relation before rollback")
	}
	t.Log("Phase 1: provisional target visible ✓")

	// Rollback to base — wipe the staged insert+edge.
	if err := epoch.RollbackTo("base"); err != nil {
		t.Fatalf("RollbackTo base: %v", err)
	}

	// Re-materialize — no stale provisional IDs.
	// After rollback, the staged edge is gone, so no nodes are reachable
	// from Src. ComputeLeidenFromMatch may error (no seeds with edges) or
	// return empty matched nodes. Either way, the provisional target must
	// not appear.
	mr2, err := epoch.ComputeLeidenFromMatch(context.Background(), LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{Src},
		MinHops:     1,
		MaxHops:     1,
		Direction:   LeidenMatchOutbound,
	}, EpochLeidenOptions{})
	if err == nil && mr2 != nil {
		rel2, err := epoch.MaterializeLeidenRelation(context.Background(), mr2)
		if err != nil {
			t.Fatalf("MaterializeLeidenRelation after rollback: %v", err)
		}
		for _, row := range rel2.Rows {
			if row.RecordID == "ProvisionalTgt" || row.NodeID == tgtID {
				t.Fatal("provisional target must be absent after rollback")
			}
		}
	} else if err != nil {
		t.Logf("ComputeLeidenFromMatch after rollback (expected, no edges): %v", err)
	}
	t.Log("Phase 2: provisional target absent after rollback ✓")

	t.Log("✅ provisional cleanup")
}

// =============================================================================
// Test 8: Assignment validation
// =============================================================================

func TestLeiden_Relation_AssignmentValidation(t *testing.T) {
	h := newRelTestHarness(t)
	h.insert("X")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	t.Run("nil match result", func(t *testing.T) {
		_, err := epoch.MaterializeLeidenRelation(context.Background(), nil)
		if err == nil {
			t.Fatal("expected error for nil match result")
		}
		t.Logf("nil match result: %v ✓", err)
	})

	t.Run("nil leiden result", func(t *testing.T) {
		_, err := epoch.MaterializeLeidenRelation(context.Background(), &LeidenMatchResult{
			Collection:   "nodes",
			LeidenResult: nil,
		})
		if err == nil {
			t.Fatal("expected error for nil LeidenResult")
		}
		t.Logf("nil LeidenResult: %v ✓", err)
	})

	t.Run("empty collection", func(t *testing.T) {
		_, err := epoch.MaterializeLeidenRelation(context.Background(), &LeidenMatchResult{
			Collection:   "",
			LeidenResult: &EpochLeidenResult{},
		})
		if err == nil {
			t.Fatal("expected error for empty collection")
		}
		t.Logf("empty collection: %v ✓", err)
	})

	t.Run("duplicate assignment", func(t *testing.T) {
		mr := &LeidenMatchResult{
			Collection: "nodes",
			LeidenResult: &EpochLeidenResult{
				Communities: []EpochCommunity{
					{ID: 10, Members: []uint64{100}},
					{ID: 20, Members: []uint64{100}}, // duplicate node 100
				},
			},
			MatchedNodeIDs: []uint64{100},
		}
		_, err := epoch.MaterializeLeidenRelation(context.Background(), mr)
		if err == nil {
			t.Fatal("expected error for duplicate assignment")
		}
		t.Logf("duplicate assignment: %v ✓", err)
	})
	t.Log("✅ assignment validation")
}

// =============================================================================
// Test 9: Defensive copy
// =============================================================================

func TestLeiden_Relation_DefensiveCopy(t *testing.T) {
	h := newRelTestHarness(t)
	A := h.insert("A")
	B := h.insert("B")
	h.addEdge(A, B, 10)

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	mr, err := epoch.ComputeLeidenFromMatch(context.Background(), LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		MinHops:     1,
		MaxHops:     1,
		Direction:   LeidenMatchOutbound,
	}, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	first, err := epoch.MaterializeLeidenRelation(context.Background(), mr)
	if err != nil {
		t.Fatalf("first MaterializeLeidenRelation: %v", err)
	}

	// Mutate returned rows.
	first.Rows = append(first.Rows, LeidenRelationRow{NodeID: 99999, CommunityID: 88888, Collection: "hacked", RecordID: "hacked"})
	if len(first.Rows) > 0 {
		first.Rows[0].NodeID = 0
		first.Rows[0].RecordID = "mutated"
	}

	// Re-materialize: must match original.
	second, err := epoch.MaterializeLeidenRelation(context.Background(), mr)
	if err != nil {
		t.Fatalf("second MaterializeLeidenRelation: %v", err)
	}

	if len(second.Rows) != 1 {
		t.Fatalf("mutated first call affected second result: got %d rows", len(second.Rows))
	}
	if second.Rows[0].RecordID == "mutated" || second.Rows[0].RecordID == "hacked" {
		t.Fatal("mutation leaked into second materialization")
	}

	t.Log("✅ defensive copy")
}

// =============================================================================
// Test 10: Determinism
// =============================================================================

func TestLeiden_Relation_Determinism(t *testing.T) {
	h := newRelTestHarness(t)
	A := h.insert("A")
	B := h.insert("B")
	C := h.insert("C")
	h.addEdge(A, B, 10)
	h.addEdge(B, C, 10)

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	spec := LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		MinHops:     1,
		MaxHops:     2,
		Direction:   LeidenMatchOutbound,
	}

	var first *LeidenRelation
	for i := 0; i < 10; i++ {
		mr, err := epoch.ComputeLeidenFromMatch(context.Background(), spec, EpochLeidenOptions{})
		if err != nil {
			t.Fatalf("ComputeLeidenFromMatch call %d: %v", i, err)
		}
		rel, err := epoch.MaterializeLeidenRelation(context.Background(), mr)
		if err != nil {
			t.Fatalf("MaterializeLeidenRelation call %d: %v", i, err)
		}
		if i == 0 {
			first = rel
			continue
		}

		if !reflect.DeepEqual(rel.Rows, first.Rows) {
			t.Fatalf("call %d: rows differ from first call\nfirst: %v\nthis:  %v", i, first.Rows, rel.Rows)
		}
		if rel.Truncated != first.Truncated {
			t.Fatalf("call %d: Truncated differs", i)
		}
		if rel.Scope != first.Scope {
			t.Fatalf("call %d: Scope differs", i)
		}
		if absDiff(rel.Modularity, first.Modularity) > 1e-12 {
			t.Fatalf("call %d: Modularity differs: %v vs %v", i, rel.Modularity, first.Modularity)
		}
	}

	t.Log("✅ determinism across 10 calls")
}

// =============================================================================
// Test 11: Closed epoch error
// =============================================================================

func TestLeiden_Relation_ClosedEpoch(t *testing.T) {
	h := newRelTestHarness(t)
	h.insert("X")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	epoch.Rollback(context.Background()) // close it

	_, err := epoch.MaterializeLeidenRelation(context.Background(), &LeidenMatchResult{
		Collection:   "nodes",
		LeidenResult: &EpochLeidenResult{},
	})
	if err == nil {
		t.Fatal("expected error for closed epoch")
	}
	t.Logf("closed epoch: %v ✓", err)
	t.Log("✅ closed epoch error")
}

// =============================================================================
// Test 12: Sort order verification
// =============================================================================

func TestLeiden_Relation_SortOrder(t *testing.T) {
	h := newRelTestHarness(t)
	A := h.insert("A")
	B := h.insert("B")
	C := h.insert("C")
	D := h.insert("D")
	h.addEdge(A, B, 10)
	h.addEdge(A, C, 10)
	h.addEdge(A, D, 10)

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	mr, err := epoch.ComputeLeidenFromMatch(context.Background(), LeidenMatchSpec{
		Collection:  "nodes",
		SeedNodeIDs: []uint64{A},
		MinHops:     1,
		MaxHops:     1,
		Direction:   LeidenMatchOutbound,
	}, EpochLeidenOptions{})
	if err != nil {
		t.Fatalf("ComputeLeidenFromMatch: %v", err)
	}

	rel, err := epoch.MaterializeLeidenRelation(context.Background(), mr)
	if err != nil {
		t.Fatalf("MaterializeLeidenRelation: %v", err)
	}

	// Verify rows are sorted by NodeID ASC, then CommunityID ASC, then RecordID ASC.
	for i := 1; i < len(rel.Rows); i++ {
		prev, cur := rel.Rows[i-1], rel.Rows[i]
		if cur.NodeID < prev.NodeID {
			t.Fatalf("sort violation at row %d: NodeID %d < %d", i, cur.NodeID, prev.NodeID)
		}
		if cur.NodeID == prev.NodeID && cur.CommunityID < prev.CommunityID {
			t.Fatalf("sort violation at row %d: CommunityID %d < %d (same NodeID)", i, cur.CommunityID, prev.CommunityID)
		}
		if cur.NodeID == prev.NodeID && cur.CommunityID == prev.CommunityID && cur.RecordID < prev.RecordID {
			t.Fatalf("sort violation at row %d: RecordID %q < %q", i, cur.RecordID, prev.RecordID)
		}
	}

	// Verify no duplicate rows.
	seen := make(map[LeidenRelationRow]bool)
	for _, row := range rel.Rows {
		if seen[row] {
			t.Fatalf("duplicate row: %+v", row)
		}
		seen[row] = true
	}

	t.Log("✅ deterministic sort order, no duplicates")
}

// =============================================================================
// Test 13: Cross-collection guard
// =============================================================================

func TestLeiden_Relation_CrossCollectionGuard(t *testing.T) {
	h := newRelTestHarness(t)
	h.insert("A")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// Manually construct a match result with a wrong-collection node ID
	// that resolves to a different collection.
	// Get the node ID for A in "nodes" collection.
	nodeA, err := epoch.LookupNodeID(context.Background(), "nodes", "A")
	if err != nil {
		t.Fatalf("LookupNodeID A: %v", err)
	}

	// Create another collection.
	db2 := h.db
	if _, err := db2.CreateCollection(context.Background(), "other", WithDimension(3), WithGraph(h.gr)); err != nil {
		// May fail if graph already attached — try creating without graph.
		if _, err := db2.CreateCollection(context.Background(), "other", WithDimension(3)); err != nil {
			t.Fatalf("CreateCollection other: %v", err)
		}
	}

	// Try to materialize with wrong collection — node A resolves to "nodes",
	// but we claim collection "other".
	_, err = epoch.MaterializeLeidenRelation(context.Background(), &LeidenMatchResult{
		Collection: "other",
		LeidenResult: &EpochLeidenResult{
			Communities: []EpochCommunity{{ID: 10, Members: []uint64{nodeA}}},
		},
		MatchedNodeIDs: []uint64{nodeA},
	})
	if err == nil {
		t.Fatal("expected error for cross-collection node")
	}
	t.Logf("cross-collection guard: %v ✓", err)

	t.Log("✅ cross-collection guard")
}

func init() {
	// Ensure edge kinds are registered for test runs that may execute concurrently.
	graph.RegisterEdgeKind("LINK", 10)
	graph.RegisterEdgeKind("ALT", 20)
}

// nodeIDForRecord is a test helper to look up node IDs without needing Database.
func (db *Database) nodeIDForRecord(collection, id string) uint64 {
	nid, _ := db.GetNodeID(context.Background(), collection, id)
	return nid
}
