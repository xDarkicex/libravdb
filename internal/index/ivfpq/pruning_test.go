package ivfpq

import (
	"context"
	"math/rand"
	"testing"
	"fmt"
	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

// mockThresholdFilter implements both index.GraphFilter and index.ThresholdFilter
type mockThresholdFilter struct {
	simThreshold float32
}

func (m *mockThresholdFilter) Test(idx uint64) bool {
	return true
}

func (m *mockThresholdFilter) Threshold() float32 {
	return m.simThreshold
}

func TestIVFPartitionPruningExecution(t *testing.T) {
	config := &Config{
		Dimension:     4,
		NClusters:     4,
		NProbes:       4,
		Metric:        util.L2Distance,
		MaxIterations: 20,
		Tolerance:     1e-4,
		RandomSeed:    42,
		Quantization:  &quant.QuantizationConfig{
			Type:       quant.ProductQuantization,
			Codebooks:  2,
			Bits:       8,
			TrainRatio: 1.0,
			CacheSize:  100,
		},
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer idx.Close()

	ctx := context.Background()

	// 1. Insert structured clusters of points to force clear separation
	centers := [][]float32{
		{10, 10, 10, 10},
		{20, 20, 20, 20},
		{-10, -10, -10, -10},
		{-20, -20, -20, -20},
	}

	var trainingVectors [][]float32
	var entries []*VectorEntry

	id := 0
	for _, center := range centers {
		for i := 0; i < 25; i++ { // 100 points total
			vec := []float32{
				center[0] + rand.Float32() - 0.5,
				center[1] + rand.Float32() - 0.5,
				center[2] + rand.Float32() - 0.5,
				center[3] + rand.Float32() - 0.5,
			}
			trainingVectors = append(trainingVectors, vec)
			entries = append(entries, &VectorEntry{
				Ordinal: uint32(id),
				Vector:  vec,
				ID:      fmt.Sprintf("%d", id),
			})
			id++
		}
	}

	err = idx.Train(ctx, trainingVectors)
	if err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	for _, entry := range entries {
		err = idx.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("Insert failed: %v", err)
		}
	}

	// 2. Query exactly at cluster 1 center
	query := centers[0]

	// 3. Search without threshold
	resNoThreshold, err := idx.Search(ctx, query, 10, nil)
	if err != nil {
		t.Fatalf("Search without threshold failed: %v", err)
	}

	if len(resNoThreshold) == 0 {
		t.Fatal("Expected results without threshold, got 0")
	}

	// 4. Determine a strict similarity threshold
	simThreshold := float32(1.0 / 11.0)
	filter := &mockThresholdFilter{simThreshold: simThreshold}

	// 5. Search with threshold
	resWithThreshold, err := idx.Search(ctx, query, 10, filter)
	if err != nil {
		t.Fatalf("Search with threshold failed: %v", err)
	}

	var expected []*SearchResult
	for _, r := range resNoThreshold {
		sim := float32(1.0 / (1.0 + r.Score))
		if sim > simThreshold {
			expected = append(expected, r)
		}
	}

	if len(expected) == 0 {
		t.Fatal("Threshold is too strict, all results pruned")
	}

	if len(resWithThreshold) != len(expected) {
		t.Fatalf("Expected %d results with threshold, got %d", len(expected), len(resWithThreshold))
	}

	for i, exp := range expected {
		if resWithThreshold[i].Ordinal != exp.Ordinal {
			t.Errorf("Result mismatch at %d: expected ordinal %d, got %d", i, exp.Ordinal, resWithThreshold[i].Ordinal)
		}
	}
}
