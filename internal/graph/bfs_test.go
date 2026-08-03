package graph

import (
	"testing"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

func TestBFS_Reachability(t *testing.T) {
	// Property 11: BFS Reachability Completeness
	properties := gopter.NewProperties(nil)

	properties.Property("BFS visits each reachable node exactly once", prop.ForAll(
		func(edges []uint64) bool {
			if len(edges) == 0 {
				return true
			}
			cfg := testConfig()
			store, err := NewGraph(cfg)
			if err != nil {
				return false
			}
			defer store.Close()

			txn := &Txn{ID: 1}
			for i := range edges {
				// Chain: 1 -> 2 -> ... -> N
				if err := store.AddEdge(txn, uint64(i+1), uint64(i+2), 1.0, 0); err != nil {
					return false
				}
			}

			bitset, err := store.GetBitset()
			if err != nil {
				t.Fatalf("GetBitset failed: %v", err)
			}
			frontier, err := store.GetFrontierBuf()
			if err != nil {
				t.Fatalf("GetFrontierBuf failed: %v", err)
			}
			defer store.PutBitset(bitset)
			defer store.PutFrontierBuf(frontier)

			visitedCount := 0
			err = store.BFS(1, 100, func(nodeID uint64, band int, step int) bool {
				visitedCount++
				return true
			}, bitset, frontier)

			if err != nil {
				return false
			}

			return visitedCount == len(edges)+1
		},
		gen.SliceOfN(10, gen.UInt64()),
	))

	properties.TestingRun(t)
}

func TestBFS_EarlyTermination(t *testing.T) {
	// Property 12: BFS Early Termination
	properties := gopter.NewProperties(nil)

	properties.Property("BFS stops immediately when VisitAction returns false", prop.ForAll(
		func(stopAt int) bool {
			cfg := testConfig()
			store, err := NewGraph(cfg)
			if err != nil {
				return false
			}
			defer store.Close()

			txn := &Txn{ID: 1}
			for i := 1; i <= 20; i++ {
				if err := store.AddEdge(txn, uint64(i), uint64(i+1), 1.0, 0); err != nil {
					return false
				}
			}

			bitset, err := store.GetBitset()
			if err != nil {
				t.Fatalf("GetBitset failed: %v", err)
			}
			frontier, err := store.GetFrontierBuf()
			if err != nil {
				t.Fatalf("GetFrontierBuf failed: %v", err)
			}
			defer store.PutBitset(bitset)
			defer store.PutFrontierBuf(frontier)

			visitedCount := 0
			err = store.BFS(1, 100, func(nodeID uint64, band int, step int) bool {
				visitedCount++
				return visitedCount != stopAt
			}, bitset, frontier)

			if err != nil {
				return false
			}

			if stopAt > 0 && stopAt <= 20 {
				return visitedCount == stopAt
			}
			return true
		},
		gen.IntRange(1, 15),
	))

	properties.TestingRun(t)
}

func TestBFS_ZeroAllocations(t *testing.T) {
	cfg := testConfig()
	store, err := NewGraph(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	txn := &Txn{ID: 1}
	for i := 1; i <= 10; i++ {
		if err := store.AddEdge(txn, uint64(i), uint64(i+1), 1.0, 0); err != nil {
			t.Fatal(err)
		}
	}

	bitset, err := store.GetBitset()
	if err != nil {
		t.Fatalf("GetBitset failed: %v", err)
	}
	frontier, err := store.GetFrontierBuf()
	if err != nil {
		t.Fatalf("GetFrontierBuf failed: %v", err)
	}
	defer store.PutBitset(bitset)
	defer store.PutFrontierBuf(frontier)

	visit := func(nodeID uint64, band int, step int) bool {
		return true
	}

	allocs := testing.AllocsPerRun(10, func() {
		store.BFS(1, 100, visit, bitset, frontier)
	})

	if allocs > 0 {
		t.Errorf("Expected 0 allocations, got %f", allocs)
	}
}

// TestBFSPattern_MultiBandChaining verifies that BFSPattern correctly chains
// multiple edge bands with different kind filters. A 3-node chain
// s -(kind1*1..3)-> m -(kind2)-> d must visit s, m, d.
// A node reachable only via kind2 from a non-chain node must not be visited.
func TestBFSPattern_MultiBandChaining(t *testing.T) {
	cfg := testConfig()
	store, err := NewGraph(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	// Node IDs: 1=s, 2=m, 3=d, 4=orphan
	// s --kind1--> m --kind2--> d
	// orphan --kind2--> m (wrong direction for kind1 chain)
	txn := &Txn{ID: 1}
	if err := store.AddEdge(txn, 1, 2, 1.0, 1); err != nil {
		t.Fatal(err)
	}
	if err := store.AddEdge(txn, 2, 3, 1.0, 2); err != nil {
		t.Fatal(err)
	}
	if err := store.AddEdge(txn, 4, 2, 1.0, 2); err != nil {
		t.Fatal(err)
	}

	bitset, err := store.GetBitset()
	if err != nil {
		t.Fatalf("GetBitset: %v", err)
	}
	frontier, err := store.GetFrontierBuf()
	if err != nil {
		t.Fatalf("GetFrontierBuf: %v", err)
	}
	defer store.PutBitset(bitset)
	defer store.PutFrontierBuf(frontier)

	// Pattern: (s)-[:DEPENDS_ON*1..3]->(api)-[:DOCUMENTED_BY]->(doc)
	edges := []EdgePlan{
		{Dir: 1, Min: 1, Max: 3, KindSet: NewKindSet(1)},
		{Dir: 1, Min: 1, Max: 1, KindSet: NewKindSet(2)},
	}
	visited := make(map[uint64]map[int]bool) // nodeID -> bands reached
	err = store.BFSPattern(1, edges, 4, func(nodeID uint64, band int, step int) bool {
		if visited[nodeID] == nil {
			visited[nodeID] = make(map[int]bool)
		}
		visited[nodeID][band] = true
		return true
	}, bitset, frontier)

	if err != nil {
		t.Fatal(err)
	}

	t.Logf("visited: %v", visited)

	// s (node 1) must be visited in band 0
	if !visited[1][0] {
		t.Error("s (node 1) must be visited in band 0 (seed)")
	}
	// m (node 2) must be visited in band 0 (intermediate) AND band 1 (transition)
	if !visited[2][0] {
		t.Error("m (node 2) must be visited in band 0 (via kind1 edge from s)")
	}
	if !visited[2][1] {
		t.Error("m (node 2) must be visited in band 1 (transition from band 0)")
	}
	// d (node 3) must be visited in band 1 (final target)
	if !visited[3][1] {
		t.Error("d (node 3) must be visited in band 1 (via kind2 edge from m)")
	}
	// orphan (node 4) must NOT be visited at all
	if visited[4] != nil {
		t.Error("orphan (node 4) must NOT be visited (not on the chain)")
	}
}

// TestBFSPattern_ZeroHopStart verifies that ->* (Min=0) on the first band
// allows the seed itself as a valid 0-hop source that transitions to band 1.
func TestBFSPattern_ZeroHopStart(t *testing.T) {
	cfg := testConfig()
	store, err := NewGraph(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	// m --kind2--> d  (only the second band matters)
	txn := &Txn{ID: 1}
	if err := store.AddEdge(txn, 2, 3, 1.0, 2); err != nil {
		t.Fatal(err)
	}

	bitset, err := store.GetBitset()
	if err != nil {
		t.Fatalf("GetBitset: %v", err)
	}
	frontier, err := store.GetFrontierBuf()
	if err != nil {
		t.Fatalf("GetFrontierBuf: %v", err)
	}
	defer store.PutBitset(bitset)
	defer store.PutFrontierBuf(frontier)

	// Pattern: (s)-[:ANY*]->(m)-[:DOCUMENTED_BY]->(d)
	// Start from node 2 (m) which should match as 0-hop in band 0, then expand band 1
	edges := []EdgePlan{
		{Dir: 1, Min: 0, Max: 100, KindSet: KindSet{}}, // * (any, 0-or-more)
		{Dir: 1, Min: 1, Max: 1, KindSet: NewKindSet(2)},
	}

	visited := make(map[uint64]bool)
	err = store.BFSPattern(2, edges, 101, func(nodeID uint64, band int, step int) bool {
		visited[nodeID] = true
		return true
	}, bitset, frontier)

	if err != nil {
		t.Fatal(err)
	}

	t.Logf("visited: %v", visited)

	if !visited[2] {
		t.Error("m (node 2) must be visited (seed, 0-hop in band 0)")
	}
	if !visited[3] {
		t.Error("d (node 3) must be visited (band 1 expansion from m)")
	}
}
