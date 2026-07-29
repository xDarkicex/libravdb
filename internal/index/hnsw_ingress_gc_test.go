package index

import (
	"context"
	"fmt"
	"runtime"
	"sync/atomic"
	"testing"

	"github.com/xDarkicex/libravdb/internal/index/hnsw"
	"github.com/xDarkicex/libravdb/internal/util"
)

// Compile-time proof that the public index ingress records can be passed to
// HNSW without an adapter allocation or an unsafe off-heap pointer graph.
var _ []*hnsw.VectorEntry = []*VectorEntry(nil)

func TestHNSWBatchIngressSurvivesConcurrentGC(t *testing.T) {
	idx, err := NewHNSW(&HNSWConfig{
		Dimension:      32,
		M:              8,
		EfConstruction: 32,
		EfSearch:       16,
		ML:             1,
		Metric:         util.L2Distance,
		RawVectorStore: hnsw.RawVectorStoreSlabby,
		RawStoreCap:    4096,
		IDMapCapacity:  4096,
	})
	if err != nil {
		t.Fatalf("NewHNSW: %v", err)
	}
	defer idx.Close()

	const count = 2048
	entries := make([]*VectorEntry, count)
	for i := range entries {
		vector := make([]float32, 32)
		for j := range vector {
			vector[j] = float32((i + 1) * (j + 1))
		}
		entries[i] = &VectorEntry{
			ID:       fmt.Sprintf("gc-entry-%d", i),
			Ordinal:  uint32(i),
			Vector:   vector,
			Metadata: map[string]interface{}{"ordinal": i},
			Version:  1,
		}
	}

	var stop atomic.Bool
	done := make(chan struct{})
	go func() {
		defer close(done)
		for !stop.Load() {
			runtime.GC()
		}
	}()

	err = idx.BatchInsert(context.Background(), entries)
	stop.Store(true)
	<-done
	if err != nil {
		t.Fatalf("BatchInsert: %v", err)
	}
	if got := idx.Size(); got != count {
		t.Fatalf("size = %d, want %d", got, count)
	}
}
