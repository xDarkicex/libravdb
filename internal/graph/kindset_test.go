package graph

import (
	"testing"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

func TestKindSet_FilteringCorrectness(t *testing.T) {
	// Validates: Requirement 1.4
	// Property 1: KindSet Filtering Correctness

	properties := gopter.NewProperties(nil)

	properties.Property("Set and Has consistency", prop.ForAll(
		func(kinds []uint8) bool {
			ks := NewKindSet(kinds...)

			// Build a map for exact verification
			expected := make(map[uint8]bool)
			for _, k := range kinds {
				expected[k] = true
			}

			// Verify all 256 possible byte values
			for i := 0; i < 256; i++ {
				k := uint8(i)
				if ks.Has(k) != expected[k] {
					return false
				}
			}
			return true
		},
		gen.SliceOf(gen.UInt8()),
	))

	properties.TestingRun(t)
}
