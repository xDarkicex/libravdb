package graph

import (
	"testing"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

func testConfig() GraphConfig {
	return GraphConfig{
		EdgeSlots:        120000,
		EdgeSlotSize:     80,
		EdgeShards:       4,
		PageSlots:        4096,
		PageShards:       4,
		BitsetPoolSize:   8,
		FrontierPoolSize: 8,
		ArenaPages:       16,
	}
}

func TestEdgeTable_InlineFirst8(t *testing.T) {
	// Property 4: Inline-First-8 Layout Optimization
	properties := gopter.NewProperties(nil)

	properties.Property("≤8 edges store inline", prop.ForAll(
		func(edgeCount uint16) bool {
			cfg := testConfig()
			store, err := NewGraph(cfg)
			if err != nil {
				return false
			}
			defer store.Close()

			txn := &Txn{ID: 1}
			for i := uint16(0); i < edgeCount; i++ {
				if err := store.AddEdge(txn, 1, uint64(i), 1.0, 0); err != nil {
					return false
				}
			}

			gs := store.(*graphStore)
			slot := gs.index.Lookup(1)
			if slot == 0 {
				return edgeCount == 0
			}
			page := gs.pageReg.Get(slot)

			// Verify inline storage: Overflow must be 0 if count <= 8
			if edgeCount <= 8 && page.Header.Overflow != 0 {
				return false
			}

			return true
		},
		gen.UInt16Range(0, 8),
	))

	properties.TestingRun(t)
}

func TestEdgeTable_OverflowActivation(t *testing.T) {
	// Property 5: Overflow Chain Activation
	properties := gopter.NewProperties(nil)

	properties.Property(">250 edges activate overflow chain", prop.ForAll(
		func(edgeCount uint16) bool {
			cfg := testConfig()
			store, err := NewGraph(cfg)
			if err != nil {
				return false
			}
			defer store.Close()

			txn := &Txn{ID: 1}
			for i := uint16(0); i < edgeCount; i++ {
				if err := store.AddEdge(txn, 1, uint64(i), 1.0, 0); err != nil {
					return false
				}
			}

			gs := store.(*graphStore)
			slot := gs.index.Lookup(1)
			if slot == 0 {
				return false
			}
			page := gs.pageReg.Get(slot)

			// Verify overflow chain
			if edgeCount > 250 && page.Header.Overflow == 0 {
				return false
			}

			return true
		},
		gen.UInt16Range(251, 300),
	))

	properties.TestingRun(t)
}

func TestGraph_GenerationMonotonicity(t *testing.T) {
	// Property 6: Generation Monotonicity
	properties := gopter.NewProperties(nil)

	properties.Property("Generation increases monotonically", prop.ForAll(
		func(operations []uint16) bool {
			cfg := testConfig()
			store, err := NewGraph(cfg)
			if err != nil {
				return false
			}
			defer store.Close()

			gs := store.(*graphStore)
			txn := &Txn{ID: 1}

			lastGen := uint32(0)
			for i := range operations {
				if err := store.AddEdge(txn, 1, uint64(i), 1.0, 0); err != nil {
					return false
				}
				slot := gs.index.Lookup(1)
				page := gs.pageReg.Get(slot)
				gen := page.Header.Generation
				if gen <= lastGen {
					return false
				}
				lastGen = gen
			}

			return true
		},
		gen.SliceOfN(10, gen.UInt16()),
	))

	properties.TestingRun(t)
}

func TestGraph_NeighborsCompleteness(t *testing.T) {
	// Property 8: Neighbors Completeness
	properties := gopter.NewProperties(nil)

	properties.Property("Neighbors returns all added edges", prop.ForAll(
		func(targets []uint64) bool {
			cfg := testConfig()
			store, err := NewGraph(cfg)
			if err != nil {
				return false
			}
			defer store.Close()

			txn := &Txn{ID: 1}
			expected := make(map[uint64]bool)
			for _, tgt := range targets {
				if err := store.AddEdge(txn, 1, tgt, 1.0, 0); err != nil {
					return false
				}
				expected[tgt] = true
			}

			neighbors, err := store.Neighbors(1)
			if err != nil {
				return false
			}

			if len(neighbors) != len(targets) {
				return false
			}

			for _, e := range neighbors {
				if !expected[e.Target] {
					return false
				}
			}

			return true
		},
		gen.SliceOfN(20, gen.UInt64()),
	))

	properties.TestingRun(t)
}

func TestGraph_DegreeConsistency(t *testing.T) {
	// Property 10: Degree-Neighbors Consistency
	properties := gopter.NewProperties(nil)

	properties.Property("Degree == len(Neighbors)", prop.ForAll(
		func(targets []uint64) bool {
			cfg := testConfig()
			store, err := NewGraph(cfg)
			if err != nil {
				return false
			}
			defer store.Close()

			txn := &Txn{ID: 1}
			for _, tgt := range targets {
				if err := store.AddEdge(txn, 1, tgt, 1.0, 0); err != nil {
					return false
				}
			}

			neighbors, _ := store.Neighbors(1)
			degree, _ := store.Degree(1)

			return degree == len(neighbors)
		},
		gen.SliceOfN(15, gen.UInt64()),
	))

	properties.TestingRun(t)
}
