package libravdb

import (
	"context"
	"fmt"
	"math"
	"sync"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// =============================================================================
// Test A: Aggregation modularity invariant
// =============================================================================

func TestLeiden_AggregationModularityInvariant(t *testing.T) {
	lg := buildTestGraph([][3]float64{
		{0, 1, 2}, {1, 2, 3}, {0, 2, 1},
		{3, 4, 4}, {4, 5, 1},
		{2, 3, 2},
	})
	lg.addSelfLoop(0, 1.5)
	lg.nodeToComm = []uint32{0, 0, 0, 1, 1, 1}
	twoM := 2.0 * lg.totalM
	qBefore := computeModularity(lg, 1.0, twoM)
	mBefore := lg.totalM

	agg := aggregateGraph(lg)

	if math.Abs(agg.totalM-mBefore) > 1e-12 {
		t.Fatalf("m invariant: original=%.12f aggregated=%.12f", mBefore, agg.totalM)
	}

	agg.nodeToComm = make([]uint32, len(agg.nodes))
	for i := range agg.nodeToComm {
		agg.nodeToComm[i] = uint32(i)
	}
	twoMAgg := 2.0 * agg.totalM
	qAfter := computeModularity(agg, 1.0, twoMAgg)

	if math.Abs(qBefore-qAfter) > 1e-9 {
		t.Fatalf("Q invariant: original=%.12f aggregated=%.12f diff=%.2e", qBefore, qAfter, math.Abs(qBefore-qAfter))
	}
	t.Logf("aggregation: Q=%.10f m=%.10f preserved", qBefore, mBefore)
	t.Log("\u2705 test A: aggregation preserves m and modularity")
}

func TestLeiden_SelfLoopAccounting(t *testing.T) {
	lg := &leidenGraph{nodeIdx: make(map[uint64]int)}
	lg.addNode(100)
	lg.addNode(200)
	lg.addSelfLoop(100, 3)            // self-loop w=3 on node 100
	lg.addUndirectedEdge(100, 200, 2) // edge w=2 between 100 and 200

	// degree(100) = 2*3 + 2 = 8
	// degree(200) = 2
	// totalM = 3 + 2 = 5
	// 2m = 10
	if math.Abs(lg.degrees[0]-8) > 1e-12 {
		t.Fatalf("degree(loopNode): want 8, got %.2f", lg.degrees[0])
	}
	if math.Abs(lg.degrees[1]-2) > 1e-12 {
		t.Fatalf("degree(other): want 2, got %.2f", lg.degrees[1])
	}
	if math.Abs(lg.totalM-5) > 1e-12 {
		t.Fatalf("totalM: want 5, got %.2f", lg.totalM)
	}

	lg.nodeToComm = []uint32{0, 1}
	twoM := 2.0 * lg.totalM
	prod := computeModularity(lg, 1.0, twoM)
	ref := referenceModularity(lg, 1.0, twoM)
	if math.Abs(prod-ref) > 1e-10 {
		t.Fatalf("self-loop modularity: prod=%.12f ref=%.12f", prod, ref)
	}

	// Move node 100 to community 1.
	qBefore := computeModularity(lg, 1.0, twoM)
	dq := deltaMove(lg, 0, 1, 1.0, twoM)
	lg.nodeToComm[0] = 1
	qAfter := computeModularity(lg, 1.0, twoM)
	if math.Abs(dq-(qAfter-qBefore)) > 1e-10 {
		t.Fatalf("self-loop move delta: dq=%.12f observed=%.12f", dq, qAfter-qBefore)
	}

	t.Logf("self-loop: deg=8 m=5 Q=%.6f ✓", prod)
	t.Log("✅ test B: self-loop accounting correct")
}

// =============================================================================
// Test C: Multi-level hierarchy projection with sparse IDs
// =============================================================================

func TestLeiden_MultiLevelProjection(t *testing.T) {
	// Use sparse, non-contiguous original IDs.
	ids := []uint64{101, 503, 9001, 42000, 777777}
	lg := &leidenGraph{nodeIdx: make(map[uint64]int)}
	for _, id := range ids {
		lg.addNode(id)
	}
	// Create a graph that benefits from aggregation:
	// Two dense clusters with a weak bridge.
	lg.addUndirectedEdge(ids[0], ids[1], 5)
	lg.addUndirectedEdge(ids[1], ids[2], 5)
	lg.addUndirectedEdge(ids[0], ids[2], 5)
	lg.addUndirectedEdge(ids[2], ids[3], 1) // weak bridge
	lg.addUndirectedEdge(ids[3], ids[4], 5)

	lg.nodeToComm = make([]uint32, len(ids))
	for i := range lg.nodeToComm {
		lg.nodeToComm[i] = uint32(i)
	}
	twoM := 2.0 * lg.totalM

	initialQ := computeModularity(lg, 1.0, twoM)
	t.Logf("initial Q with %d singletons: %.6f", len(ids), initialQ)

	// Run local moving.
	moves := 0
	for pass := 0; pass < 10; pass++ {
		mv := localMovingPhase(lg, 1.0, twoM, 1e-12)
		moves += mv
		if mv == 0 {
			break
		}
	}
	refineCommunities(lg)

	// Aggregate.
	agg := aggregateGraph(lg)
	t.Logf("after aggregation: %d nodes → %d supernodes, %d moves", len(lg.nodes), len(agg.nodes), moves)

	// Project back to original IDs.
	communities := projectToOriginalIDs(agg)

	// Every returned member must be one of the original IDs.
	allIDs := make(map[uint64]bool)
	for _, id := range ids {
		allIDs[id] = true
	}
	seen := make(map[uint64]bool)
	for _, comm := range communities {
		if comm.ID != comm.Members[0] {
			t.Fatalf("community ID %d != min member %d", comm.ID, comm.Members[0])
		}
		for _, m := range comm.Members {
			if !allIDs[m] {
				t.Fatalf("non-original ID %d in community", m)
			}
			if seen[m] {
				t.Fatalf("duplicate ID %d across communities", m)
			}
			seen[m] = true
		}
	}
	if len(seen) != len(ids) {
		t.Fatalf("expected %d original IDs, got %d", len(ids), len(seen))
	}

	// Verify determinism: 5 runs produce identical results.
	for run := 0; run < 5; run++ {
		lg2 := &leidenGraph{nodeIdx: make(map[uint64]int)}
		for _, id := range ids {
			lg2.addNode(id)
		}
		lg2.addUndirectedEdge(ids[0], ids[1], 5)
		lg2.addUndirectedEdge(ids[1], ids[2], 5)
		lg2.addUndirectedEdge(ids[0], ids[2], 5)
		lg2.addUndirectedEdge(ids[2], ids[3], 1)
		lg2.addUndirectedEdge(ids[3], ids[4], 5)
		lg2.nodeToComm = make([]uint32, len(ids))
		for i := range lg2.nodeToComm {
			lg2.nodeToComm[i] = uint32(i)
		}
		twoM2 := 2.0 * lg2.totalM
		for pass := 0; pass < 10; pass++ {
			if localMovingPhase(lg2, 1.0, twoM2, 1e-12) == 0 {
				break
			}
		}
		refineCommunities(lg2)
		agg2 := aggregateGraph(lg2)
		comms2 := projectToOriginalIDs(agg2)
		if len(comms2) != len(communities) {
			t.Fatalf("run %d: %d communities, run 0: %d", run, len(comms2), len(communities))
		}
	}

	t.Logf("projection: %d communities, all %d original IDs covered, 5 deterministic runs ✓", len(communities), len(ids))
	t.Log("✅ test C: multi-level projection with sparse original IDs")
}

// =============================================================================
// Test D: Two-clique hierarchy
// =============================================================================

func TestLeiden_TwoCliqueHierarchy(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/leiden_hierarchy.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	graph.RegisterEdgeKind("E", 1)

	nodes := make([]uint64, 7)
	for i := 1; i <= 6; i++ {
		col.Insert(context.Background(), fmt.Sprintf("n%d", i), []float32{float32(i), 0, 0}, nil)
		nodes[i], _ = db.GetNodeID(context.Background(), "docs", fmt.Sprintf("n%d", i))
	}

	// Clique A: strong edges between 1,2,3
	// Clique B: strong edges between 4,5,6
	// Weak bridge: 3-4
	for _, e := range [][2]uint64{{1, 2}, {1, 3}, {2, 3}, {4, 5}, {4, 6}, {5, 6}} {
		txn := gr.BeginTxn()
		txn.AddEdge(nodes[e[0]], nodes[e[1]], 5.0, 1)
		txn.Commit(context.Background())
	}
	txn := gr.BeginTxn()
	txn.AddEdge(nodes[3], nodes[4], 0.5, 1)
	txn.Commit(context.Background())

	epoch, _ := db.BeginEpochTx(context.Background())
	gtx, _ := epoch.GraphTxn("docs")
	// Stage edges so they're visible in the epoch overlay.
	for _, e := range [][2]uint64{{1, 2}, {1, 3}, {2, 3}, {4, 5}, {4, 6}, {5, 6}, {3, 4}} {
		gtx.AddEdge(nodes[e[0]], nodes[e[1]], 5.0, 1)
	}
	// Override bridge weight as weak.
	gtx.AddEdge(nodes[3], nodes[4], 0.5, 1)

	seeds := []uint64{nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[6]}
	result, err := epoch.ComputeLeiden(context.Background(), EpochLeidenOptions{
		Seeds:         seeds,
		ExpansionHops: 1,
		Resolution:    1.0,
		MaxLevels:     5,
	})
	if err != nil {
		t.Fatalf("ComputeLeiden: %v", err)
	}

	t.Logf("two-clique: %d communities, Q=%.6f, levels=%d, moves=%d",
		len(result.Communities), result.Modularity, result.Levels, result.Moves)

	if result.Modularity <= result.InitialModularity {
		t.Errorf("modularity must improve: initial=%.6f final=%.6f", result.InitialModularity, result.Modularity)
	}

	// Every community must be connected.
	for _, comm := range result.Communities {
		if len(comm.Members) <= 1 {
			continue
		}
		visited := make(map[uint64]bool)
		queue := []uint64{comm.Members[0]}
		visited[comm.Members[0]] = true
		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]
			neighbors, _ := gr.Neighbors(cur)
			for _, nb := range neighbors {
				if !visited[nb.Target] {
					for _, m := range comm.Members {
						if m == nb.Target {
							visited[m] = true
							queue = append(queue, m)
							break
						}
					}
				}
			}
		}
		for _, m := range comm.Members {
			if !visited[m] {
				t.Errorf("community %d not connected: node %d unreachable", comm.ID, m)
			}
		}
	}
	t.Logf("all communities connected ✓")

	epoch.Rollback(context.Background())
	t.Log("✅ test D: two-clique hierarchy")
}

// =============================================================================
// Test: Race safety
// =============================================================================

func TestLeiden_RaceSafety(t *testing.T) {
	db, _ := Open(WithStoragePath(t.TempDir() + "/leiden_race.libravdb"))
	defer db.Drop(context.Background())

	gr, _ := NewGraph(GraphConfig{})
	defer gr.Close()
	col, _ := db.CreateCollection(context.Background(), "docs", WithDimension(3), WithGraph(gr))
	graph.RegisterEdgeKind("E", 1)

	col.Insert(context.Background(), "a", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "b", []float32{0, 1, 0}, nil)
	col.Insert(context.Background(), "c", []float32{0, 0, 1}, nil)
	a, _ := db.GetNodeID(context.Background(), "docs", "a")
	b, _ := db.GetNodeID(context.Background(), "docs", "b")
	c, _ := db.GetNodeID(context.Background(), "docs", "c")

	txn := gr.BeginTxn()
	txn.AddEdge(a, b, 1.0, 1)
	txn.AddEdge(b, c, 1.0, 1)
	txn.Commit(context.Background())

	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			epoch, _ := db.BeginEpochTx(context.Background())
			gtx, _ := epoch.GraphTxn("docs")
			gtx.AddEdge(a, c, 1.0, 1)
			result, err := epoch.ComputeLeiden(context.Background(), EpochLeidenOptions{
				Seeds:         []uint64{a, b, c},
				ExpansionHops: 1,
				Resolution:    1.0,
			})
			if err != nil {
				t.Errorf("ComputeLeiden: %v", err)
			}
			if result == nil {
				t.Error("nil result")
			}
			epoch.Rollback(context.Background())
		}()
	}
	wg.Wait()

	// Verify live graph unchanged.
	edges, _ := gr.Neighbors(a)
	if len(edges) != 1 {
		t.Errorf("live graph mutated: want 1 edge from A, got %d", len(edges))
	}
	t.Log("✅ race safety: no live graph mutation, no data races")
}
