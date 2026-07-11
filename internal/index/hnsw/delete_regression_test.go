package hnsw

import (
	"context"
	"fmt"
	"runtime"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/util"
)

func TestLinkMutationRejectsNodeUnpublishedWhileWaiting(t *testing.T) {
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

	for i := 0; i < 2; i++ {
		vec := []float32{float32(i), float32(i + 1), float32(i + 2), float32(i + 3)}
		if err := index.Insert(ctx, &VectorEntry{ID: fmt.Sprintf("vec_%d", i), Vector: vec}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	target := index.nodes.Get(0)
	if target == nil || target.Links[0] == nil {
		t.Fatal("target node has no level-zero link storage")
	}
	if !index.acquirePruneLock(target) {
		t.Fatal("failed to acquire target mutation lock")
	}

	done := make(chan bool, 1)
	go func() {
		done <- index.neighborSelector.connectLinkWithHeuristic(0, 1, 0, index)
	}()

	// Give the writer time to capture target before it blocks on PruneLock.
	for range 100 {
		runtime.Gosched()
	}
	index.nodes.Set(0, nil)
	index.releasePruneLock(target)

	if accepted := <-done; accepted {
		t.Fatal("stale writer accepted a link for an unpublished node")
	}
}

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
