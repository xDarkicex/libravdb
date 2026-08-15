package index

import (
	"context"
	"fmt"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

func TestIVFPQBatchIngressSurvivesConcurrentGC(t *testing.T) {
	idx, err := NewIVFPQ(&IVFPQConfig{
		Dimension: 32,
		NClusters: 4,
		NProbes:   4,
		Metric:    util.L2Distance,
		Quantization: &quant.QuantizationConfig{
			Type:       quant.ProductQuantization,
			Codebooks:  2,
			Bits:       4,
			TrainRatio: 1,
		},
	})
	if err != nil {
		t.Fatalf("NewIVFPQ: %v", err)
	}
	defer idx.Close()

	// Need to train the index first
	trainVectors := make([][]float32, 128)
	for i := range trainVectors {
		trainVectors[i] = make([]float32, 32)
		for j := range trainVectors[i] {
			trainVectors[i][j] = float32(i + j)
		}
	}
	if trainer, ok := idx.(interface {
		Train(context.Context, [][]float32) error
	}); ok {
		if err := trainer.Train(context.Background(), trainVectors); err != nil {
			t.Fatalf("Train: %v", err)
		}
	} else {
		t.Fatalf("Index does not support Train()")
	}

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
			time.Sleep(time.Millisecond)
		}
	}()
	defer func() {
		stop.Store(true)
		<-done
	}()

	err = idx.BatchInsert(context.Background(), entries)
	if err != nil {
		t.Fatalf("BatchInsert: %v", err)
	}

	// Trigger UAF overwrite by allocating same slots
	entries2 := make([]*VectorEntry, count)
	for i := range entries2 {
		entries2[i] = &VectorEntry{
			ID:       fmt.Sprintf("OVERWRITE-%d", i),
			Ordinal:  uint32(i + count),
			Vector:   entries[i].Vector, // same vector so we can find them
			Metadata: map[string]interface{}{"ordinal": i + count},
			Version:  2,
		}
	}
	err = idx.BatchInsert(context.Background(), entries2)
	if err != nil {
		t.Fatalf("BatchInsert 2: %v", err)
	}

	// The index may retain ordinals and PQ codes, but never IDs, vectors, or
	// metadata. The caller-owned objects are no longer live after this point.
	query := entries[0].Vector

	// Force another GC and sleep to ensure caller-owned data is collectible.
	runtime.GC()
	time.Sleep(10 * time.Millisecond)

	if got := idx.Size(); got != count*2 {
		t.Fatalf("size = %d, want %d", got, count*2)
	}

	// Ask for the entire corpus so the test validates every retained ordinal,
	// independent of approximate-PQ tie ordering.
	results, err := idx.Search(context.Background(), query, count*2, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	found := make(map[uint32]bool, len(results))
	for _, res := range results {
		found[res.Ordinal] = true
	}

	for ordinal := uint32(0); ordinal < count*2; ordinal++ {
		if !found[ordinal] {
			t.Fatalf("retained ordinal %d was overwritten or lost", ordinal)
		}
	}
}
