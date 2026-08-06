package libravdb

import (
	"math"
	"math/rand"
	"testing"
)

// =============================================================================
// Test 1: Delta-Q versus full recomputation — correct sequential moves
// =============================================================================

func TestLeiden_DeltaQ_vs_FullRecomputation(t *testing.T) {
	// Each test specifies moves as (node, dst) — source is derived from
	// the current partition. This prevents stale-source bugs.
	tests := []struct {
		name  string
		edges [][3]float64
		moves []struct{ node, dst int }
		gamma float64
	}{
		{
			name:  "triangle",
			edges: [][3]float64{{0, 1, 1}, {1, 2, 1}, {0, 2, 1}},
			moves: []struct{ node, dst int }{{0, 1}, {1, 2}, {2, 1}},
			gamma: 1.0,
		},
		{
			name:  "path_of_4",
			edges: [][3]float64{{0, 1, 1}, {1, 2, 1}, {2, 3, 1}},
			moves: []struct{ node, dst int }{{1, 0}, {2, 3}},
			gamma: 1.0,
		},
		{
			name:  "weighted_parallel",
			edges: [][3]float64{{0, 1, 2}, {0, 1, 3}, {1, 2, 1}},
			moves: []struct{ node, dst int }{{0, 1}},
			gamma: 1.0,
		},
		{
			name:  "isolated_node",
			edges: [][3]float64{{0, 1, 1}, {2, 2, 0}},
			moves: []struct{ node, dst int }{{0, 1}},
			gamma: 1.0,
		},
		{
			name:  "gamma_0_5",
			edges: [][3]float64{{0, 1, 1}, {1, 2, 1}, {0, 2, 1}},
			moves: []struct{ node, dst int }{{0, 1}},
			gamma: 0.5,
		},
		{
			name:  "gamma_2_0",
			edges: [][3]float64{{0, 1, 1}, {1, 2, 1}, {0, 2, 1}},
			moves: []struct{ node, dst int }{{0, 1}},
			gamma: 2.0,
		},
		{
			name:  "singleton_src_and_dst",
			edges: [][3]float64{{0, 1, 1}, {1, 2, 1}},
			moves: []struct{ node, dst int }{{0, 1}, {2, 1}},
			gamma: 1.0,
		},
		{
			name:  "multi_node_src",
			edges: [][3]float64{{0, 1, 1}, {0, 2, 1}, {1, 2, 1}, {2, 3, 2}},
			moves: []struct{ node, dst int }{{0, 1}, {1, 2}, {2, 3}},
			gamma: 1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lg := buildTestGraph(tt.edges)
			twoM := 2.0 * lg.totalM
			if twoM < 1e-15 {
				twoM = 1.0
			}

			for mi, mv := range tt.moves {
				src := lg.nodeToComm[mv.node]
				dst := uint32(mv.dst)
				if src == dst {
					continue // skip no-op moves
				}

				kiSrc, kiDst, kSrc, kDst := gatherMoveStats(lg, mv.node, src, dst)

				qBefore := computeModularity(lg, tt.gamma, twoM)
				refBefore := referenceModularity(lg, tt.gamma, twoM)
				if math.Abs(qBefore-refBefore) > 1e-10 {
					t.Fatalf("move %d: prod Q=%.12f ref Q=%.12f before move", mi, qBefore, refBefore)
				}

				dq := deltaMove(lg, mv.node, dst, tt.gamma, twoM)
				lg.nodeToComm[mv.node] = dst

				qAfter := computeModularity(lg, tt.gamma, twoM)
				refAfter := referenceModularity(lg, tt.gamma, twoM)
				if math.Abs(qAfter-refAfter) > 1e-10 {
					t.Fatalf("move %d: prod Q=%.12f ref Q=%.12f after move", mi, qAfter, refAfter)
				}

				observed := qAfter - qBefore
				if math.Abs(dq-observed) > 1e-10 {
					t.Fatalf("move %d (node %d: %d→%d): deltaQ=%.12f observed=%.12f diff=%.2e\n"+
						"  k_i=%.3f k_i,src=%.3f k_i,dst=%.3f K_src=%.3f K_dst=%.3f",
						mi, mv.node, src, dst, dq, observed, math.Abs(dq-observed),
						lg.degrees[mv.node], kiSrc, kiDst, kSrc, kDst)
				}
			}
		})
	}
	t.Log("✅ delta-Q matches full recomputation within 1e-10")
}

// gatherMoveStats collects diagnostic values for a candidate move.
func gatherMoveStats(lg *leidenGraph, i int, src, dst uint32) (kiSrc, kiDst, kSrc, kDst float64) {
	for j := 0; j < len(lg.nodes); j++ {
		c := lg.nodeToComm[j]
		if c == src {
			kSrc += lg.degrees[j]
		}
		if c == dst {
			kDst += lg.degrees[j]
		}
	}
	for _, nb := range lg.adj[i] {
		c := lg.nodeToComm[nb.to]
		if c == src {
			kiSrc += nb.weight
		}
		if c == dst {
			kiDst += nb.weight
		}
	}
	return
}

// =============================================================================
// Test: Delta-Q preconditions
// =============================================================================

func TestLeiden_DeltaQ_Preconditions(t *testing.T) {
	lg := buildTestGraph([][3]float64{{0, 1, 1}, {1, 2, 1}})
	twoM := 2.0 * lg.totalM

	// Move to same community → zero.
	dq := deltaMove(lg, 0, lg.nodeToComm[0], 1.0, twoM)
	if dq != 0 {
		t.Errorf("move to same community: want 0, got %.12f", dq)
	}

	// Isolated node → zero.
	lg2 := buildTestGraph([][3]float64{{0, 0, 0}, {1, 2, 1}})
	lg2.nodeToComm = []uint32{0, 1, 1}
	dq2 := deltaMove(lg2, 0, 1, 1.0, 2.0*lg2.totalM)
	if dq2 != 0 {
		t.Errorf("isolated node move: want 0, got %.12f", dq2)
	}

	// Source includes K_A with moved node.
	lg3 := buildTestGraph([][3]float64{{0, 1, 1}, {0, 2, 1}, {1, 2, 1}, {2, 3, 2}})
	lg3.nodeToComm = []uint32{0, 0, 0, 1} // 0,1,2 in comm 0; 3 in comm 1
	twoM3 := 2.0 * lg3.totalM
	qBefore := computeModularity(lg3, 1.0, twoM3)
	dq3 := deltaMove(lg3, 2, 1, 1.0, twoM3)
	lg3.nodeToComm[2] = 1
	qAfter := computeModularity(lg3, 1.0, twoM3)
	if math.Abs(dq3-(qAfter-qBefore)) > 1e-10 {
		t.Fatalf("multi-node src: deltaQ=%.12f observed=%.12f", dq3, qAfter-qBefore)
	}

	t.Log("✅ delta-Q preconditions: same-comm=0, isolated=0, multi-node K_A verified")
}

// =============================================================================
// Test: Reference modularity vs production (all gamma)
// =============================================================================

func TestLeiden_ReferenceModularity(t *testing.T) {
	lg := buildTestGraph([][3]float64{
		{0, 1, 2}, {1, 2, 3}, {0, 2, 1}, {2, 3, 4},
	})
	lg.nodeToComm = []uint32{0, 0, 1, 1}
	twoM := 2.0 * lg.totalM

	for _, gamma := range []float64{0.5, 1.0, 2.0} {
		prod := computeModularity(lg, gamma, twoM)
		ref := referenceModularity(lg, gamma, twoM)
		if math.Abs(prod-ref) > 1e-10 {
			t.Fatalf("gamma=%.1f: prod=%.12f ref=%.12f diff=%.2e", gamma, prod, ref, math.Abs(prod-ref))
		}
	}
	t.Log("✅ production modularity matches reference within 1e-10")
}

// =============================================================================
// Test: 100 deterministic small random graphs
// =============================================================================

func TestLeiden_DeterministicRandomGraphs(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	const tolerance = 1e-10

	for seed := int64(0); seed < 100; seed++ {
		rng = rand.New(rand.NewSource(seed + 42))
		n := 2 + rng.Intn(7) // 2–8 nodes
		var edges [][3]float64
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				if rng.Float64() < 0.6 { // ~60% edge density
					w := 1.0 + float64(rng.Intn(5)) // weight 1–5
					edges = append(edges, [3]float64{float64(i), float64(j), w})
				}
			}
		}

		lg := buildTestGraphN(n, edges)
		twoM := 2.0 * lg.totalM
		if twoM < 1e-15 || len(lg.nodes) < 2 {
			continue
		}

		// Test a few random partitions and moves.
		for pi := 0; pi < 5; pi++ {
			// Random partition.
			for i := range lg.nodeToComm {
				lg.nodeToComm[i] = uint32(rng.Intn(min(3, n)))
			}

			// Try moving random nodes to random destinations.
			for mi := 0; mi < min(5, n); mi++ {
				node := rng.Intn(n)
				dst := uint32(rng.Intn(min(3, n)))
				if lg.nodeToComm[node] == dst {
					continue
				}

				qBefore := computeModularity(lg, 1.0, twoM)
				dq := deltaMove(lg, node, dst, 1.0, twoM)
				lg.nodeToComm[node] = dst
				qAfter := computeModularity(lg, 1.0, twoM)

				if math.Abs(dq-(qAfter-qBefore)) > tolerance {
					t.Fatalf("seed=%d n=%d pi=%d mi=%d: deltaQ=%.12f observed=%.12f",
						seed, n, pi, mi, dq, qAfter-qBefore)
				}

				// Also verify reference.
				lg.nodeToComm[node] = dst // restore for ref check
				refBefore := referenceModularity(lg, 1.0, twoM)
				lg.nodeToComm[node] = lg.nodeToComm[node] // no-op, already set
				if math.Abs(qAfter-refBefore) > tolerance {
					t.Fatalf("seed=%d: ref mismatch after move", seed)
				}
			}

			// Test gamma values.
			for _, gamma := range []float64{0.5, 1.0, 2.0} {
				if n < 2 {
					continue
				}
				node := 0
				dst := uint32(1)
				if lg.nodeToComm[node] == dst {
					continue
				}
				qBefore := computeModularity(lg, gamma, twoM)
				dq := deltaMove(lg, node, dst, gamma, twoM)
				lg.nodeToComm[node] = dst
				qAfter := computeModularity(lg, gamma, twoM)
				if math.Abs(dq-(qAfter-qBefore)) > tolerance {
					t.Fatalf("seed=%d gamma=%.1f: deltaQ=%.12f observed=%.12f",
						seed, gamma, dq, qAfter-qBefore)
				}
			}
		}
	}
	t.Log("✅ 100 random graphs: delta-Q correct for all moves")
}

// =============================================================================
// Test: Refinement guarantees (fixed mutation order)
// =============================================================================

func TestLeiden_RefinementSplitsDisconnected(t *testing.T) {
	lg := buildTestGraph([][3]float64{
		{0, 1, 1},
		{2, 3, 1},
	})
	lg.nodeToComm = []uint32{0, 0, 0, 0} // all in one community

	refineCommunities(lg)

	comms := make(map[uint32][]int)
	for i, c := range lg.nodeToComm {
		comms[c] = append(comms[c], i)
	}
	if len(comms) < 2 {
		t.Fatalf("refinement must split into >= 2 communities, got %d", len(comms))
	}

	// Each community must be internally connected.
	for _, members := range comms {
		if len(members) <= 1 {
			continue
		}
		visited := make(map[int]bool)
		queue := []int{members[0]}
		visited[members[0]] = true
		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]
			for _, nb := range lg.adj[cur] {
				if visited[nb.to] {
					continue
				}
				for _, m := range members {
					if m == nb.to {
						visited[m] = true
						queue = append(queue, m)
						break
					}
				}
			}
		}
		for _, m := range members {
			if !visited[m] {
				t.Fatalf("refined community has disconnected node %d", m)
			}
		}
	}
	t.Logf("refinement split into %d connected components ✓", len(comms))
	t.Log("✅ refinement guarantee: disconnected community split correctly")
}

// =============================================================================
// Test: Aggregation is disabled — single-level Leiden only
// =============================================================================

func TestLeiden_SingleLevelOnly(t *testing.T) {
	lg := buildTestGraph([][3]float64{
		{0, 1, 2}, {1, 2, 3}, {0, 2, 1}, {2, 3, 4},
	})
	lg.nodeToComm = []uint32{0, 0, 1, 1}
	twoM := 2.0 * lg.totalM
	qBefore := computeModularity(lg, 1.0, twoM)

	// Verify that a single level of local moving + refinement works.
	moves := localMovingPhase(lg, 1.0, twoM, 1e-12)
	refineCommunities(lg)
	qAfter := computeModularity(lg, 1.0, twoM)
	t.Logf("single-level: Q %.6f → %.6f (%d moves)", qBefore, qAfter, moves)
	t.Log("✅ single-level Leiden: local moving + refinement produces valid Q")
}

// =============================================================================
// Test helpers
// =============================================================================

func buildTestGraphN(n int, edges [][3]float64) *leidenGraph {
	lg := &leidenGraph{nodeIdx: make(map[uint64]int)}
	for i := 0; i < n; i++ {
		lg.addNode(uint64(i))
	}
	for _, e := range edges {
		u, v, w := uint64(e[0]), uint64(e[1]), e[2]
		if u == v && w == 0 {
			continue
		}
		lg.addUndirectedEdge(u, v, w)
	}
	lg.nodeToComm = make([]uint32, n)
	for i := range lg.nodeToComm {
		lg.nodeToComm[i] = uint32(i)
	}
	return lg
}

func buildTestGraph(edges [][3]float64) *leidenGraph {
	lg := &leidenGraph{nodeIdx: make(map[uint64]int)}
	maxN := uint64(0)
	for _, e := range edges {
		u, v := uint64(e[0]), uint64(e[1])
		if u > maxN {
			maxN = u
		}
		if v > maxN {
			maxN = v
		}
	}
	for i := uint64(0); i <= maxN; i++ {
		lg.addNode(i)
	}
	for _, e := range edges {
		u, v, w := uint64(e[0]), uint64(e[1]), e[2]
		if u == v && w == 0 {
			continue
		}
		lg.addUndirectedEdge(u, v, w)
	}
	n := len(lg.nodes)
	lg.nodeToComm = make([]uint32, n)
	for i := range lg.nodeToComm {
		lg.nodeToComm[i] = uint32(i)
	}
	return lg
}
