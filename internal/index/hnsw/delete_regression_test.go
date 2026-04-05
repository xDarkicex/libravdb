package hnsw

import (
	"context"
	"fmt"
	"testing"

	"github.com/xDarkicex/libravdb/internal/util"
)

func TestDeleteRepairsAsymmetricIncomingLinksBeforePrune(t *testing.T) {
	ctx := context.Background()
	index, err := NewHNSW(&Config{
		Dimension:      4,
		M:              2,
		EfConstruction: 16,
		EfSearch:       8,
		ML:             1,
		Metric:         util.L2Distance,
		RandomSeed:     42,
	})
	if err != nil {
		t.Fatalf("new hnsw: %v", err)
	}
	defer index.Close()

	for i := 0; i < 8; i++ {
		vec := []float32{float32(i), float32(i + 1), float32(i + 2), float32(i + 3)}
		if err := index.Insert(ctx, &VectorEntry{ID: fmt.Sprintf("vec_%d", i), Vector: vec}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	node0 := index.idToIndex["vec_0"]
	node1 := index.idToIndex["vec_1"]
	staleLinks := make([]uint32, 0, 7)
	for i := 1; i < 8; i++ {
		staleLinks = append(staleLinks, index.idToIndex[fmt.Sprintf("vec_%d", i)])
	}

	index.nodes[node0].Links[0] = staleLinks
	index.nodes[node1].Links[0] = index.nodes[node1].Links[0][:0]
	index.neighborSelector = NewNeighborSelector(index.config.M, 2.0)

	if err := index.Delete(ctx, "vec_1"); err != nil {
		t.Fatalf("delete vec_1: %v", err)
	}

	if err := index.neighborSelector.PruneConnections(node0, 0, index); err != nil {
		t.Fatalf("prune after delete: %v", err)
	}

	for _, linkID := range index.nodes[node0].Links[0] {
		if linkID == node1 {
			t.Fatalf("stale link to deleted node survived pruning: %+v", index.nodes[node0].Links[0])
		}
	}
}
