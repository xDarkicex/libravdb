package hnsw

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"unsafe"

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

	node0, _ := index.idToIndex.Get(hashID("vec_0"))
	node1, _ := index.idToIndex.Get(hashID("vec_1"))
	var staleLinks [7]uint32
	staleLen := 0
	for i := 1; i < 8; i++ {
		nodeI, _ := index.idToIndex.Get(hashID(fmt.Sprintf("vec_%d", i)))
		staleLinks[staleLen] = nodeI.Ordinal
		staleLen++
	}

	staleSlice := unsafe.Slice(index.nodes.Get(node0.Ordinal).Links[0], staleLen+1)
	for i := 0; i < staleLen; i++ {
		staleSlice[i] = staleLinks[i]
	}
	staleSlice[staleLen] = SentinelNodeID
	atomic.StoreUint32(&index.nodes.Get(node0.Ordinal).LinkCounts[0], uint32(staleLen))

	node1Slice := unsafe.Slice(index.nodes.Get(node1.Ordinal).Links[0], 1)
	node1Slice[0] = SentinelNodeID
	atomic.StoreUint32(&index.nodes.Get(node1.Ordinal).LinkCounts[0], 0)
	index.neighborSelector = NewNeighborSelector(index.config.M, 2.0)

	if err := index.Delete(ctx, "vec_1"); err != nil {
		t.Fatalf("delete vec_1: %v", err)
	}

	if err := index.neighborSelector.PruneConnections(node0.Ordinal, 0, index); err != nil {
		t.Fatalf("prune after delete: %v", err)
	}

	links := index.getNodeLinks(index.nodes.Get(node0.Ordinal), 0)
	for _, link := range links {
		if link == node1.Ordinal {
			t.Fatalf("stale link to deleted node survived pruning: %+v", index.nodes.Get(node0.Ordinal).Links[0])
		}
	}
}
