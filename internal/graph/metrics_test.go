package graph

import (
	"testing"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

func TestMetrics_Correctness(t *testing.T) {
	// Property 18: Metric Counter Correctness
	properties := gopter.NewProperties(nil)

	properties.Property("Each operation increments its counter exactly", prop.ForAll(
		func(edges uint16) bool {
			cfg := testConfig()
			store, err := NewGraph(cfg)
			if err != nil {
				return false
			}
			defer store.Close()

			txn := &Txn{ID: 1}
			for i := uint16(0); i < edges; i++ {
				store.AddEdge(txn, 1, uint64(i+2), 1.0, 0)
			}

			stats := store.Stats()

			if stats.EdgesAdded != uint64(edges) {
				return false
			}

			if edges > 0 && stats.PagesAllocated == 0 {
				return false
			}

			return true
		},
		gen.UInt16Range(1, 300),
	))

	properties.TestingRun(t)
}
