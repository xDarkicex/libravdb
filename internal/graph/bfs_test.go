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
				store.AddEdge(txn, uint64(i+1), uint64(i+2), 1.0, 0)
			}

			bitset := store.GetBitset()
			frontier := store.GetFrontierBuf()
			defer store.PutBitset(bitset)
			defer store.PutFrontierBuf(frontier)

			visitedCount := 0
			err = store.BFS(1, 100, func(nodeID uint64, depth int) bool {
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
				store.AddEdge(txn, uint64(i), uint64(i+1), 1.0, 0)
			}

			bitset := store.GetBitset()
			frontier := store.GetFrontierBuf()
			defer store.PutBitset(bitset)
			defer store.PutFrontierBuf(frontier)

			visitedCount := 0
			err = store.BFS(1, 100, func(nodeID uint64, depth int) bool {
				visitedCount++
				if visitedCount == stopAt {
					return false
				}
				return true
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
		store.AddEdge(txn, uint64(i), uint64(i+1), 1.0, 0)
	}

	bitset := store.GetBitset()
	frontier := store.GetFrontierBuf()
	defer store.PutBitset(bitset)
	defer store.PutFrontierBuf(frontier)

	visit := func(nodeID uint64, depth int) bool {
		return true
	}

	allocs := testing.AllocsPerRun(10, func() {
		store.BFS(1, 100, visit, bitset, frontier)
	})

	if allocs > 0 {
		t.Errorf("Expected 0 allocations, got %f", allocs)
	}
}
