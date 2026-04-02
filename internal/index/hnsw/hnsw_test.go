package hnsw

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/xDarkicex/libravdb/internal/util"
)

func TestAppendUniqueLinkPreventsDuplicates(t *testing.T) {
	node := &Node{
		Links: [][]uint32{{1, 2}},
	}

	if added := appendUniqueLink(node, 8, 0, 2); added {
		t.Fatal("expected duplicate link append to be skipped")
	}

	if len(node.Links[0]) != 2 {
		t.Fatalf("expected link count to remain 2, got %d", len(node.Links[0]))
	}

	if added := appendUniqueLink(node, 8, 0, 3); !added {
		t.Fatal("expected unique link to be appended")
	}

	if len(node.Links[0]) != 3 {
		t.Fatalf("expected link count to become 3, got %d", len(node.Links[0]))
	}
}

func TestHNSWDeterministicFixedSeedSearchResults(t *testing.T) {
	buildAndSearch := func() []string {
		t.Helper()

		config := &Config{
			Dimension:      64,
			M:              16,
			EfConstruction: 100,
			EfSearch:       50,
			ML:             1.0,
			Metric:         util.L2Distance,
			RandomSeed:     42,
		}

		index, err := NewHNSW(config)
		if err != nil {
			t.Fatalf("failed to create HNSW index: %v", err)
		}
		defer index.Close()

		vectors := generateTestVectors(200, 64)
		for i, vec := range vectors {
			err := index.Insert(context.Background(), &VectorEntry{
				ID:       fmt.Sprintf("vec_%d", i),
				Vector:   vec,
				Metadata: map[string]interface{}{"index": i},
			})
			if err != nil {
				t.Fatalf("failed to insert vector %d: %v", i, err)
			}
		}

		results, err := index.Search(context.Background(), vectors[0], 10)
		if err != nil {
			t.Fatalf("failed to search deterministic index: %v", err)
		}

		ids := make([]string, len(results))
		for i, result := range results {
			ids[i] = result.ID
		}
		return ids
	}

	first := buildAndSearch()
	second := buildAndSearch()
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("expected deterministic results with fixed seed, got %v and %v", first, second)
	}
}
