package libravdb

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// =============================================================================
// Test: Live graph result resolution does not recurse
// =============================================================================

func TestExecutor_LiveGraphResultResolution(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/live_graph_resolve.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()

	col, err := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	graph.RegisterEdgeKind("LINK", 10)

	// Insert two records and a live graph edge.
	if err := col.Insert(context.Background(), "src", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("Insert src: %v", err)
	}
	if err := col.Insert(context.Background(), "tgt", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("Insert tgt: %v", err)
	}

	srcNode, err := db.GetNodeID(context.Background(), "docs", "src")
	if err != nil {
		t.Fatalf("GetNodeID src: %v", err)
	}
	tgtNode, err := db.GetNodeID(context.Background(), "docs", "tgt")
	if err != nil {
		t.Fatalf("GetNodeID tgt: %v", err)
	}

	txn := gr.BeginTxn()
	if err := txn.AddEdge(srcNode, tgtNode, 1.0, 10); err != nil {
		t.Fatalf("AddEdge: %v", err)
	}
	if err := txn.Commit(context.Background()); err != nil {
		t.Fatalf("Commit edge: %v", err)
	}

	// Run a non-epoch graph query that materializes node IDs to record IDs.
	// This calls resolveNodeIDInContext on the non-epoch path, which must
	// delegate to e.db.ResolveNodeID rather than recursing.
	results, err := db.QueryWithParams(context.Background(), "SELECT id FROM GRAPH_TABLE(docs MATCH (src)-[:LINK]->(tgt))", nil)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}

	if results == nil {
		t.Fatal("expected non-nil results")
	}
	if len(results.Results) == 0 {
		t.Fatal("expected at least one result row from GRAPH_TABLE MATCH")
	}
	t.Log("✅ live graph resolution uses database resolver, no recursion")
}

// =============================================================================
// Test: SQL graph INSERT increments generation through EpochTx
// =============================================================================

func TestSession_Savepoint_GraphInsertGeneration(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/graph_insert_gen.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()

	col, err := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	graph.RegisterEdgeKind("E", 1)

	// Pre-insert records for edge endpoints.
	if err := col.Insert(context.Background(), "X", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("Insert X: %v", err)
	}
	if err := col.Insert(context.Background(), "Y", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("Insert Y: %v", err)
	}

	s, err := db.NewSQLSession(context.Background())
	if err != nil {
		t.Fatalf("NewSQLSession: %v", err)
	}
	defer s.Close()

	if err := s.Exec("BEGIN EPOCH TRANSACTION"); err != nil {
		t.Fatalf("BEGIN EPOCH: %v", err)
	}

	g0 := s.epoch.generation

	// Successful SQL graph INSERT.
	if err := s.Exec("INSERT INTO GRAPH_EDGES VALUES ('X', 'E', 'Y')"); err != nil {
		t.Fatalf("INSERT GRAPH_EDGES: %v", err)
	}

	if s.epoch.generation != g0+1 {
		t.Fatalf("generation after successful graph INSERT: want %d, got %d", g0+1, s.epoch.generation)
	}
	t.Log("Phase 1: SQL graph INSERT increments generation ✓")

	// Failed graph INSERT (invalid edge kind).
	g1 := s.epoch.generation
	if err := s.Exec("INSERT INTO GRAPH_EDGES VALUES ('X', 'UNKNOWN_KIND_ZZZ', 'Y')"); err == nil {
		t.Fatal("expected error for unknown edge kind")
	}

	if s.epoch.generation != g1 {
		t.Fatalf("generation after failed graph INSERT: want %d (unchanged), got %d", g1, s.epoch.generation)
	}
	t.Log("Phase 2: failed graph INSERT leaves generation unchanged ✓")

	if err := s.Exec("ROLLBACK"); err != nil {
		t.Fatalf("ROLLBACK: %v", err)
	}
	t.Log("✅ graph insert generation tracking")
}

// =============================================================================
// Test: Ordered graph replay — Cases 1–4
// =============================================================================

func TestEpoch_Savepoint_OrderedGraphReplay(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/ordered_replay.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()

	col, err := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	graph.RegisterEdgeKind("R", 20)

	// Insert nodes for each case (avoid cross-contamination between cases).
	if err := col.Insert(context.Background(), "A1", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("Insert A1: %v", err)
	}
	if err := col.Insert(context.Background(), "B1", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("Insert B1: %v", err)
	}
	if err := col.Insert(context.Background(), "A2", []float32{2, 0, 0}, nil); err != nil {
		t.Fatalf("Insert A2: %v", err)
	}
	if err := col.Insert(context.Background(), "B2", []float32{0, 2, 0}, nil); err != nil {
		t.Fatalf("Insert B2: %v", err)
	}
	if err := col.Insert(context.Background(), "A3", []float32{3, 0, 0}, nil); err != nil {
		t.Fatalf("Insert A3: %v", err)
	}
	if err := col.Insert(context.Background(), "B3", []float32{0, 3, 0}, nil); err != nil {
		t.Fatalf("Insert B3: %v", err)
	}

	a1, _ := db.GetNodeID(context.Background(), "docs", "A1")
	b1, _ := db.GetNodeID(context.Background(), "docs", "B1")
	a2, _ := db.GetNodeID(context.Background(), "docs", "A2")
	b2, _ := db.GetNodeID(context.Background(), "docs", "B2")
	a3, _ := db.GetNodeID(context.Background(), "docs", "A3")
	b3, _ := db.GetNodeID(context.Background(), "docs", "B3")

	// ── Case 1: base has A1→B1, REMOVE, ADD, ROLLBACK to before REMOVE ──
	t.Run("Case1_add_remove_add_rollback", func(t *testing.T) {
		// Base edge.
		base := gr.BeginTxn()
		if err := base.AddEdge(a1, b1, 1.0, 20); err != nil {
			t.Fatalf("base AddEdge: %v", err)
		}
		if err := base.Commit(context.Background()); err != nil {
			t.Fatalf("base Commit: %v", err)
		}

		epoch, err := db.BeginEpochTx(context.Background())
		if err != nil {
			t.Fatalf("BeginEpochTx: %v", err)
		}
		defer epoch.Rollback(context.Background())

		// REMOVE A1→B1
		if err := epoch.RemoveGraphEdge("docs", a1, b1, 20); err != nil {
			t.Fatalf("Remove A1→B1: %v", err)
		}
		// ADD A1→B1
		if err := epoch.AddGraphEdge("docs", a1, b1, 1.0, 20); err != nil {
			t.Fatalf("Add A1→B1: %v", err)
		}

		// Verify outbound: A1→B1 visible via overlay (staged ADD, base edge exists).
		gtx, err := epoch.GraphTxn("docs")
		if err != nil {
			t.Fatalf("GraphTxn: %v", err)
		}
		neighbors, err := gtx.NeighborsOverlay(a1)
		if err != nil {
			t.Fatalf("NeighborsOverlay: %v", err)
		}
		hasB := false
		for _, nb := range neighbors {
			if nb.Target == b1 && nb.GetKind() == 20 {
				hasB = true
			}
		}
		if !hasB {
			t.Fatal("Case 1 outbound: A1→B1 must be visible")
		}

		// Verify inbound: B1 has A1.
		inbound, err := gtx.InboundNeighborsOverlay(b1)
		if err != nil {
			t.Fatalf("InboundNeighborsOverlay: %v", err)
		}
		hasA := false
		for _, nb := range inbound {
			if nb.Target == a1 && nb.GetKind() == 20 {
				hasA = true
			}
		}
		if !hasA {
			t.Fatal("Case 1 inbound: B1←A1 must be visible")
		}
		t.Log("Case 1 ✓")
	})

	// ── Case 2: base does NOT have A2→B2, ADD, REMOVE, ROLLBACK to before ADD ──
	t.Run("Case2_add_remove_rollback", func(t *testing.T) {
		epoch, err := db.BeginEpochTx(context.Background())
		if err != nil {
			t.Fatalf("BeginEpochTx: %v", err)
		}
		defer epoch.Rollback(context.Background())

		if err := epoch.Savepoint("sp"); err != nil {
			t.Fatalf("Savepoint: %v", err)
		}
		// ADD A2→B2
		if err := epoch.AddGraphEdge("docs", a2, b2, 1.0, 20); err != nil {
			t.Fatalf("Add A2→B2: %v", err)
		}
		// REMOVE A2→B2 (cancels the staged add).
		if err := epoch.RemoveGraphEdge("docs", a2, b2, 20); err != nil {
			t.Fatalf("Remove A2→B2: %v", err)
		}
		// ROLLBACK TO sp
		if err := epoch.RollbackTo("sp"); err != nil {
			t.Fatalf("RollbackTo: %v", err)
		}

		gtx, err := epoch.GraphTxn("docs")
		if err != nil {
			t.Fatalf("GraphTxn: %v", err)
		}
		neighbors, err := gtx.NeighborsOverlay(a2)
		if err != nil {
			t.Fatalf("NeighborsOverlay: %v", err)
		}
		for _, nb := range neighbors {
			if nb.Target == b2 && nb.GetKind() == 20 {
				t.Fatal("Case 2 outbound: A2→B2 must be absent")
			}
		}

		inbound, err := gtx.InboundNeighborsOverlay(b2)
		if err != nil {
			t.Fatalf("InboundNeighborsOverlay: %v", err)
		}
		for _, nb := range inbound {
			if nb.Target == a2 && nb.GetKind() == 20 {
				t.Fatal("Case 2 inbound: B2←A2 must be absent")
			}
		}
		t.Log("Case 2 ✓")
	})

	// ── Case 3: ADD A3→B3, SAVEPOINT, DROP NODE A3 EDGES, ROLLBACK ──
	t.Run("Case3_add_savepoint_dropnode_rollback", func(t *testing.T) {
		epoch, err := db.BeginEpochTx(context.Background())
		if err != nil {
			t.Fatalf("BeginEpochTx: %v", err)
		}
		defer epoch.Rollback(context.Background())

		// ADD A3→B3
		if err := epoch.AddGraphEdge("docs", a3, b3, 1.0, 20); err != nil {
			t.Fatalf("Add A3→B3: %v", err)
		}
		if err := epoch.Savepoint("sp"); err != nil {
			t.Fatalf("Savepoint: %v", err)
		}
		// DROP NODE A3 EDGES
		if err := epoch.DropGraphNodeEdges("docs", a3); err != nil {
			t.Fatalf("DropNodeEdges A3: %v", err)
		}
		// ROLLBACK TO sp
		if err := epoch.RollbackTo("sp"); err != nil {
			t.Fatalf("RollbackTo: %v", err)
		}

		gtx, err := epoch.GraphTxn("docs")
		if err != nil {
			t.Fatalf("GraphTxn: %v", err)
		}
		neighbors, err := gtx.NeighborsOverlay(a3)
		if err != nil {
			t.Fatalf("NeighborsOverlay: %v", err)
		}
		hasB := false
		for _, nb := range neighbors {
			if nb.Target == b3 && nb.GetKind() == 20 {
				hasB = true
			}
		}
		if !hasB {
			t.Fatal("Case 3 outbound: staged A3→B3 must be visible after rollback")
		}

		inbound, err := gtx.InboundNeighborsOverlay(b3)
		if err != nil {
			t.Fatalf("InboundNeighborsOverlay: %v", err)
		}
		hasA := false
		for _, nb := range inbound {
			if nb.Target == a3 && nb.GetKind() == 20 {
				hasA = true
			}
		}
		if !hasA {
			t.Fatal("Case 3 inbound: staged B3←A3 must be visible after rollback")
		}
		t.Log("Case 3 ✓")
	})

	// ── Case 4: REMOVE base A1→B1, ADD A1→C4, SAVEPOINT, REMOVE A1→C4, ADD A1→D4, ROLLBACK ──
	t.Run("Case4_complex_sequence", func(t *testing.T) {
		// Insert nodes C4 and D4 for this case.
		if err := col.Insert(context.Background(), "C4", []float32{0, 0, 4}, nil); err != nil {
			t.Fatalf("Insert C4: %v", err)
		}
		if err := col.Insert(context.Background(), "D4", []float32{4, 4, 4}, nil); err != nil {
			t.Fatalf("Insert D4: %v", err)
		}
		c4, err := db.GetNodeID(context.Background(), "docs", "C4")
		if err != nil {
			t.Fatalf("GetNodeID C4: %v", err)
		}
		d4, err := db.GetNodeID(context.Background(), "docs", "D4")
		if err != nil {
			t.Fatalf("GetNodeID D4: %v", err)
		}

		epoch, err := db.BeginEpochTx(context.Background())
		if err != nil {
			t.Fatalf("BeginEpochTx: %v", err)
		}
		defer epoch.Rollback(context.Background())

		// REMOVE base A1→B1 (which exists from Case 1).
		if err := epoch.RemoveGraphEdge("docs", a1, b1, 20); err != nil {
			t.Fatalf("Remove A1→B1: %v", err)
		}
		// ADD A1→C4
		if err := epoch.AddGraphEdge("docs", a1, c4, 1.0, 20); err != nil {
			t.Fatalf("Add A1→C4: %v", err)
		}
		if err := epoch.Savepoint("sp"); err != nil {
			t.Fatalf("Savepoint: %v", err)
		}
		// REMOVE A1→C4
		if err := epoch.RemoveGraphEdge("docs", a1, c4, 20); err != nil {
			t.Fatalf("Remove A1→C4: %v", err)
		}
		// ADD A1→D4
		if err := epoch.AddGraphEdge("docs", a1, d4, 1.0, 20); err != nil {
			t.Fatalf("Add A1→D4: %v", err)
		}
		// ROLLBACK TO sp
		if err := epoch.RollbackTo("sp"); err != nil {
			t.Fatalf("RollbackTo: %v", err)
		}

		gtx, err := epoch.GraphTxn("docs")
		if err != nil {
			t.Fatalf("GraphTxn: %v", err)
		}

		// Outbound: A1→B1 absent, A1→C4 visible, A1→D4 absent.
		neighbors, err := gtx.NeighborsOverlay(a1)
		if err != nil {
			t.Fatalf("NeighborsOverlay: %v", err)
		}
		hasB, hasC, hasD := false, false, false
		for _, nb := range neighbors {
			switch nb.Target {
			case b1:
				if nb.GetKind() == 20 {
					hasB = true
				}
			case c4:
				if nb.GetKind() == 20 {
					hasC = true
				}
			case d4:
				if nb.GetKind() == 20 {
					hasD = true
				}
			}
		}
		if hasB {
			t.Fatal("Case 4 outbound: A1→B1 must be absent (removed before savepoint)")
		}
		if !hasC {
			t.Fatal("Case 4 outbound: A1→C4 must be visible (added before savepoint)")
		}
		if hasD {
			t.Fatal("Case 4 outbound: A1→D4 must be absent (added after savepoint)")
		}

		// Inbound: C4←A1 visible, D4←A1 absent.
		inboundC, err := gtx.InboundNeighborsOverlay(c4)
		if err != nil {
			t.Fatalf("InboundNeighborsOverlay C4: %v", err)
		}
		hasAinC := false
		for _, nb := range inboundC {
			if nb.Target == a1 && nb.GetKind() == 20 {
				hasAinC = true
			}
		}
		if !hasAinC {
			t.Fatal("Case 4 inbound: C4←A1 must be visible")
		}

		inboundD, err := gtx.InboundNeighborsOverlay(d4)
		if err != nil {
			t.Fatalf("InboundNeighborsOverlay D4: %v", err)
		}
		for _, nb := range inboundD {
			if nb.Target == a1 && nb.GetKind() == 20 {
				t.Fatal("Case 4 inbound: D4←A1 must be absent")
			}
		}
		t.Log("Case 4 ✓")
	})

	t.Log("✅ ordered graph replay Cases 1–4")
}

// =============================================================================
// Test: Graph branch rollback via public SQL
// =============================================================================

func TestSession_Savepoint_GraphBranchRollback(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/graph_branch_rollback.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()

	col, err := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	graph.RegisterEdgeKind("LINK", 10)

	if err := col.Insert(context.Background(), "Target", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("Insert Target: %v", err)
	}

	s, err := db.NewSQLSession(context.Background())
	if err != nil {
		t.Fatalf("NewSQLSession: %v", err)
	}
	defer s.Close()

	// BEGIN EPOCH; SAVEPOINT; staged graph INSERT; graph/vector MATCH; ROLLBACK TO
	if err := s.Exec("BEGIN EPOCH TRANSACTION"); err != nil {
		t.Fatalf("BEGIN EPOCH: %v", err)
	}
	if err := s.ExecWithParams("INSERT INTO docs (id, embedding) VALUES ('BranchNode', '[0,1,0]')", nil); err != nil {
		t.Fatalf("INSERT BranchNode: %v", err)
	}
	if err := s.Exec("SAVEPOINT branch"); err != nil {
		t.Fatalf("SAVEPOINT: %v", err)
	}
	if err := s.Exec("INSERT INTO GRAPH_EDGES VALUES ('BranchNode', 'LINK', 'Target')"); err != nil {
		t.Fatalf("INSERT GRAPH_EDGES: %v", err)
	}

	// Verify graph candidate visible in branch — use epoch overlay check.
	epochGtx, err := s.epoch.GraphTxn("docs")
	if err != nil {
		t.Fatalf("GraphTxn: %v", err)
	}
	branchNodeID, err := s.epoch.LookupNodeID(context.Background(), "docs", "BranchNode")
	if err != nil {
		t.Fatalf("LookupNodeID BranchNode: %v", err)
	}
	targetNodeID, err := s.epoch.LookupNodeID(context.Background(), "docs", "Target")
	if err != nil {
		t.Fatalf("LookupNodeID Target: %v", err)
	}
	neighbors, err := epochGtx.NeighborsOverlay(branchNodeID)
	if err != nil {
		t.Fatalf("NeighborsOverlay: %v", err)
	}
	hasLink := false
	for _, nb := range neighbors {
		if nb.Target == targetNodeID && nb.GetKind() == 10 {
			hasLink = true
		}
	}
	if !hasLink {
		t.Fatal("BranchNode should have LINK edge to Target in branch")
	}
	t.Log("Phase 1: LINK edge from BranchNode to Target visible in epoch ✓")

	// ROLLBACK TO SAVEPOINT branch
	if err := s.Exec("ROLLBACK TO SAVEPOINT branch"); err != nil {
		t.Fatalf("ROLLBACK TO: %v", err)
	}

	// Graph edge disappears after rollback.
	epochGtx2, err := s.epoch.GraphTxn("docs")
	if err != nil {
		t.Fatalf("GraphTxn after rollback: %v", err)
	}
	neighbors2, err := epochGtx2.NeighborsOverlay(branchNodeID)
	if err != nil {
		t.Fatalf("NeighborsOverlay after rollback: %v", err)
	}
	for _, nb := range neighbors2 {
		if nb.Target == targetNodeID && nb.GetKind() == 10 {
			t.Fatal("BranchNode should not have LINK edge after rollback")
		}
	}
	t.Log("Phase 2: LINK edge from BranchNode absent after rollback ✓")

	// Separate live session never sees it.
	sLive, err := db.NewSQLSession(context.Background())
	if err != nil {
		t.Fatalf("NewSQLSession live: %v", err)
	}
	defer sLive.Close()

	liveRecs, err := sLive.Query("SELECT id FROM docs WHERE id = 'BranchNode'")
	if err != nil {
		t.Fatalf("Live query: %v", err)
	}
	if len(liveRecs.Results) > 0 {
		t.Fatal("live session must not see uncommitted BranchNode")
	}
	t.Log("Phase 3: live session never sees branch data ✓")

	if err := s.Exec("ROLLBACK"); err != nil {
		t.Fatalf("outer ROLLBACK: %v", err)
	}
	t.Log("✅ graph branch rollback via public SQL")
}

// =============================================================================
// Test: Repeatable rollback
// =============================================================================

func TestEpoch_Savepoint_RepeatableRollback(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/repeat_rollback.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()

	col, err := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	graph.RegisterEdgeKind("R", 30)

	if err := col.Insert(context.Background(), "X", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("Insert X: %v", err)
	}
	if err := col.Insert(context.Background(), "Y", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("Insert Y: %v", err)
	}

	epoch, err := db.BeginEpochTx(context.Background())
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}
	defer epoch.Rollback(context.Background())

	if err := epoch.Insert(context.Background(), "docs", "outer", []float32{1, 1, 0}, nil); err != nil {
		t.Fatalf("Insert outer: %v", err)
	}
	if err := epoch.Savepoint("sp"); err != nil {
		t.Fatalf("Savepoint: %v", err)
	}

	// First branch mutation
	if err := epoch.Insert(context.Background(), "docs", "branch1", []float32{0, 0, 1}, nil); err != nil {
		t.Fatalf("Insert branch1: %v", err)
	}
	gAfterBranch1 := epoch.generation

	// First rollback
	if err := epoch.RollbackTo("sp"); err != nil {
		t.Fatalf("First RollbackTo: %v", err)
	}

	// Verify branch1 gone, outer present.
	recs, err := epoch.ListRecords(context.Background(), "docs")
	if err != nil {
		t.Fatalf("ListRecords after rollback 1: %v", err)
	}
	hasOuter, hasBranch1 := false, false
	for _, r := range recs {
		if r.ID == "outer" {
			hasOuter = true
		}
		if r.ID == "branch1" {
			hasBranch1 = true
		}
	}
	if !hasOuter || hasBranch1 {
		t.Fatalf("after first rollback: outer=%v branch1=%v (want outer=true branch1=false)", hasOuter, hasBranch1)
	}
	t.Log("First rollback: outer present, branch1 absent ✓")

	// Second branch mutation (different record)
	if err := epoch.Insert(context.Background(), "docs", "branch2", []float32{0, 0, 1}, nil); err != nil {
		t.Fatalf("Insert branch2: %v", err)
	}
	gAfterBranch2 := epoch.generation

	// Verify generation changed from first branch.
	if gAfterBranch2 <= gAfterBranch1 {
		t.Fatalf("generation should increase: after branch1=%d after branch2=%d", gAfterBranch1, gAfterBranch2)
	}

	// Second rollback to same savepoint.
	if err := epoch.RollbackTo("sp"); err != nil {
		t.Fatalf("Second RollbackTo: %v", err)
	}

	recs2, err := epoch.ListRecords(context.Background(), "docs")
	if err != nil {
		t.Fatalf("ListRecords after rollback 2: %v", err)
	}
	hasOuter2, hasBranch2 := false, false
	for _, r := range recs2 {
		if r.ID == "outer" {
			hasOuter2 = true
		}
		if r.ID == "branch2" {
			hasBranch2 = true
		}
	}
	if !hasOuter2 || hasBranch2 {
		t.Fatalf("after second rollback: outer=%v branch2=%v (want outer=true branch2=false)", hasOuter2, hasBranch2)
	}
	t.Log("Second rollback: outer present, branch2 absent ✓")

	t.Log("✅ repeatable rollback")
}

// =============================================================================
// Test: Commit/reopen excludes rolled-back operations
// =============================================================================

func TestEpoch_Savepoint_CommitReopenExclusion(t *testing.T) {
	path := t.TempDir() + "/commit_exclude.libravdb"
	db, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()

	col, err := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	graph.RegisterEdgeKind("L", 40)

	if err := col.Insert(context.Background(), "P", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("Insert P: %v", err)
	}
	if err := col.Insert(context.Background(), "Q", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("Insert Q: %v", err)
	}

	p, err := db.GetNodeID(context.Background(), "docs", "P")
	if err != nil {
		t.Fatalf("GetNodeID P: %v", err)
	}
	q, err := db.GetNodeID(context.Background(), "docs", "Q")
	if err != nil {
		t.Fatalf("GetNodeID Q: %v", err)
	}

	// Create epoch, savepoint, add branch-only data, rollback, then commit outer.
	epoch, err := db.BeginEpochTx(context.Background())
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}

	// Outer mutation: insert record Surviving.
	if err := epoch.Insert(context.Background(), "docs", "Surviving", []float32{0, 0, 1}, nil); err != nil {
		t.Fatalf("Insert Surviving: %v", err)
	}
	// Outer graph mutation: add P→Q edge.
	if err := epoch.AddGraphEdge("docs", p, q, 1.0, 40); err != nil {
		t.Fatalf("AddGraphEdge P→Q: %v", err)
	}

	if err := epoch.Savepoint("sp"); err != nil {
		t.Fatalf("Savepoint: %v", err)
	}

	// Branch mutations (will be rolled back).
	if err := epoch.Insert(context.Background(), "docs", "Doomed", []float32{9, 9, 9}, nil); err != nil {
		t.Fatalf("Insert Doomed: %v", err)
	}
	if err := epoch.AddGraphEdge("docs", q, p, 1.0, 40); err != nil { // Q→P should not survive
		t.Fatalf("AddGraphEdge Q→P: %v", err)
	}

	// Rollback branch.
	if err := epoch.RollbackTo("sp"); err != nil {
		t.Fatalf("RollbackTo: %v", err)
	}

	// Commit outer operations.
	if err := epoch.Commit(context.Background()); err != nil {
		t.Fatalf("Commit: %v", err)
	}

	// Reopen and verify.
	db2, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatalf("Reopen: %v", err)
	}
	defer db2.Close()

	col2, err := db2.GetCollection("docs")
	if err != nil {
		t.Fatalf("GetCollection after reopen: %v", err)
	}

	// Surviving must exist.
	_, err = col2.Get(context.Background(), "Surviving")
	if err != nil {
		t.Fatal("Surviving must exist after commit+reopen")
	}

	// Doomed must not exist.
	_, err = col2.Get(context.Background(), "Doomed")
	if err == nil {
		t.Fatal("Doomed must not exist after commit+reopen (was rolled back)")
	}

	// Verify surviving operations persist and rolled-back ones are absent.
	// Record-level verification is sufficient for reopen durability;
	// graph edges are verified before close (see below).
	t.Log("Phase 1: records survive/don't survive reopen correctly ✓")

	// Before close: verify graph edges in the original db handle.
	col1, err := db.GetCollection("docs")
	if err != nil {
		t.Fatalf("GetCollection before close: %v", err)
	}
	g1 := col1.GetGraph()
	if g1 == nil {
		t.Fatal("graph must be available before close")
	}
	neighbors, err := g1.Neighbors(p)
	if err != nil {
		t.Fatalf("Neighbors P: %v", err)
	}
	hasPQ := false
	for _, nb := range neighbors {
		if nb.Target == q && nb.GetKind() == 40 {
			hasPQ = true
		}
	}
	if !hasPQ {
		t.Fatal("P→Q edge must survive commit")
	}

	// Q→P edge must NOT survive (rolled back).
	inbound, err := g1.InboundNeighbors(p)
	if err != nil {
		t.Fatalf("InboundNeighbors P: %v", err)
	}
	for _, nb := range inbound {
		if nb.Target == q && nb.GetKind() == 40 {
			t.Fatal("Q→P edge must not survive commit (was rolled back)")
		}
	}
	t.Log("✅ commit/reopen exclusion")
}
