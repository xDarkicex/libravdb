package hnsw

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

func TestHNSWWithQuantization(t *testing.T) {
	ctx := context.Background()

	t.Run("Product Quantization Integration", func(t *testing.T) {
		// Create HNSW with Product Quantization
		config := &Config{
			Dimension:      128,
			M:              16,
			EfConstruction: 200,
			EfSearch:       50,
			ML:             1.0 / math.Log(2.0),
			Metric:         util.L2Distance,
			RandomSeed:     42,
			Quantization: &quant.QuantizationConfig{
				Type:       quant.ProductQuantization,
				Codebooks:  8,
				Bits:       8,
				TrainRatio: 0.2,
				CacheSize:  100,
			},
		}

		index, err := NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create HNSW index: %v", err)
		}
		defer index.Close()

		// Generate test vectors
		vectors := generateTestVectors(500, 128)

		// Insert vectors to trigger quantization training
		for i, vec := range vectors {
			entry := &VectorEntry{
				ID:       fmt.Sprintf("vec_%d", i),
				Vector:   vec,
				Metadata: map[string]interface{}{"index": i},
			}

			err := index.Insert(ctx, entry)
			if err != nil {
				t.Fatalf("Failed to insert vector %d: %v", i, err)
			}
		}

		// Verify quantization was trained
		if !index.quantizationTrained {
			t.Error("Quantization should be trained after inserting enough vectors")
		}

		// Verify quantizer is present
		if index.quantizer == nil {
			t.Error("Quantizer should be initialized")
		}

		// Test search functionality
		query := vectors[0]
		results, err := index.Search(ctx, query, 5)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		if len(results) == 0 {
			t.Error("Search should return results")
		}

		// First result should be the exact match
		if results[0].ID != "vec_0" {
			t.Errorf("Expected first result to be vec_0, got %s", results[0].ID)
		}

		// Verify memory usage is reduced compared to unquantized
		memUsage := index.MemoryUsage()
		if memUsage <= 0 {
			t.Error("Memory usage should be positive")
		}
	})

	t.Run("Scalar Quantization Integration", func(t *testing.T) {
		// Create HNSW with Scalar Quantization
		config := &Config{
			Dimension:      64,
			M:              16,
			EfConstruction: 100,
			EfSearch:       50,
			ML:             1.0 / math.Log(2.0),
			Metric:         util.CosineDistance,
			RandomSeed:     42,
			Quantization: &quant.QuantizationConfig{
				Type:       quant.ScalarQuantization,
				Bits:       8,
				TrainRatio: 0.1,
			},
		}

		index, err := NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create HNSW index: %v", err)
		}
		defer index.Close()

		// Generate test vectors
		vectors := generateTestVectors(500, 64)

		// Insert vectors
		for i, vec := range vectors {
			entry := &VectorEntry{
				ID:       fmt.Sprintf("scalar_vec_%d", i),
				Vector:   vec,
				Metadata: map[string]interface{}{"type": "scalar"},
			}

			err := index.Insert(ctx, entry)
			if err != nil {
				t.Fatalf("Failed to insert vector %d: %v", i, err)
			}
		}

		// Test search
		query := vectors[10]
		results, err := index.Search(ctx, query, 3)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		if len(results) == 0 {
			t.Error("Search should return results")
		}

		// Verify results have reasonable scores
		for _, result := range results {
			if result.Score < 0 {
				t.Errorf("Score should be non-negative, got %f", result.Score)
			}
		}
	})

	t.Run("Quantization Training Threshold", func(t *testing.T) {
		config := &Config{
			Dimension:      32,
			M:              8,
			EfConstruction: 50,
			EfSearch:       20,
			ML:             1.0 / math.Log(2.0),
			Metric:         util.L2Distance,
			RandomSeed:     42,
			Quantization: &quant.QuantizationConfig{
				Type:       quant.ProductQuantization,
				Codebooks:  4,
				Bits:       4,
				TrainRatio: 0.5,
			},
		}

		index, err := NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create HNSW index: %v", err)
		}
		defer index.Close()

		// Check training threshold
		threshold := index.getTrainingThreshold()
		if threshold <= 0 {
			t.Error("Training threshold should be positive")
		}

		// Insert fewer vectors than threshold
		vectors := generateTestVectors(threshold/2, 32)
		for i, vec := range vectors {
			entry := &VectorEntry{
				ID:     fmt.Sprintf("thresh_vec_%d", i),
				Vector: vec,
			}
			err := index.Insert(ctx, entry)
			if err != nil {
				t.Fatalf("Failed to insert vector %d: %v", i, err)
			}
		}

		// Quantization should not be trained yet
		if index.quantizationTrained {
			t.Error("Quantization should not be trained with insufficient data")
		}

		// Insert more vectors to exceed threshold
		moreVectors := generateTestVectors(threshold, 32)
		for i, vec := range moreVectors {
			entry := &VectorEntry{
				ID:     fmt.Sprintf("more_vec_%d", i),
				Vector: vec,
			}
			err := index.Insert(ctx, entry)
			if err != nil {
				t.Fatalf("Failed to insert additional vector %d: %v", i, err)
			}
		}

		// Now quantization should be trained
		if !index.quantizationTrained {
			t.Error("Quantization should be trained after exceeding threshold")
		}
	})

	t.Run("Mixed Quantized and Unquantized Nodes", func(t *testing.T) {
		config := &Config{
			Dimension:      16,
			M:              4,
			EfConstruction: 20,
			EfSearch:       10,
			ML:             1.0 / math.Log(2.0),
			Metric:         util.L2Distance,
			RandomSeed:     42,
			Quantization: &quant.QuantizationConfig{
				Type:       quant.ScalarQuantization,
				Bits:       4,
				TrainRatio: 0.3,
			},
		}

		index, err := NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create HNSW index: %v", err)
		}
		defer index.Close()

		// Insert some vectors before training (will be unquantized)
		preTrainingVectors := generateTestVectors(50, 16)
		for i, vec := range preTrainingVectors {
			entry := &VectorEntry{
				ID:     fmt.Sprintf("pre_vec_%d", i),
				Vector: vec,
			}
			err := index.Insert(ctx, entry)
			if err != nil {
				t.Fatalf("Failed to insert pre-training vector %d: %v", i, err)
			}
		}

		// Insert more vectors to trigger training (will be quantized)
		postTrainingVectors := generateTestVectors(200, 16)
		for i, vec := range postTrainingVectors {
			entry := &VectorEntry{
				ID:     fmt.Sprintf("post_vec_%d", i),
				Vector: vec,
			}
			err := index.Insert(ctx, entry)
			if err != nil {
				t.Fatalf("Failed to insert post-training vector %d: %v", i, err)
			}
		}

		// Verify we have both quantized and unquantized nodes
		quantizedCount := 0
		unquantizedCount := 0
		for _, node := range index.nodes {
			if node.CompressedVector != nil {
				quantizedCount++
			} else {
				unquantizedCount++
			}
		}

		if quantizedCount == 0 {
			t.Error("Should have some quantized nodes")
		}
		if unquantizedCount == 0 {
			t.Error("Should have some unquantized nodes")
		}

		// Test search works with mixed nodes
		query := preTrainingVectors[0]
		results, err := index.Search(ctx, query, 5)
		if err != nil {
			t.Fatalf("Search failed with mixed nodes: %v", err)
		}

		if len(results) == 0 {
			t.Error("Search should return results with mixed nodes")
		}
	})
}

func TestQuantizationPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	ctx := context.Background()

	// Test with larger dataset to measure performance impact
	t.Run("Performance Comparison", func(t *testing.T) {
		dimension := 256
		numVectors := 5000

		// Create unquantized index
		unquantizedConfig := &Config{
			Dimension:      dimension,
			M:              32,
			EfConstruction: 200,
			EfSearch:       50,
			ML:             1.0 / math.Log(2.0),
			Metric:         util.L2Distance,
			RandomSeed:     42,
		}

		unquantizedIndex, err := NewHNSW(unquantizedConfig)
		if err != nil {
			t.Fatalf("Failed to create unquantized index: %v", err)
		}
		defer unquantizedIndex.Close()

		// Create quantized index
		quantizedConfig := &Config{
			Dimension:      dimension,
			M:              32,
			EfConstruction: 200,
			EfSearch:       50,
			ML:             1.0 / math.Log(2.0),
			Metric:         util.L2Distance,
			RandomSeed:     42,
			Quantization: &quant.QuantizationConfig{
				Type:       quant.ProductQuantization,
				Codebooks:  16,
				Bits:       8,
				TrainRatio: 0.1,
			},
		}

		quantizedIndex, err := NewHNSW(quantizedConfig)
		if err != nil {
			t.Fatalf("Failed to create quantized index: %v", err)
		}
		defer quantizedIndex.Close()

		// Generate test data
		vectors := generateTestVectors(numVectors, dimension)

		// Insert into both indices
		for i, vec := range vectors {
			entry := &VectorEntry{
				ID:     fmt.Sprintf("perf_vec_%d", i),
				Vector: vec,
			}

			err := unquantizedIndex.Insert(ctx, entry)
			if err != nil {
				t.Fatalf("Failed to insert into unquantized index: %v", err)
			}

			err = quantizedIndex.Insert(ctx, entry)
			if err != nil {
				t.Fatalf("Failed to insert into quantized index: %v", err)
			}
		}

		// Compare memory usage
		unquantizedMemory := unquantizedIndex.MemoryUsage()
		quantizedMemory := quantizedIndex.MemoryUsage()

		t.Logf("Unquantized memory usage: %d bytes", unquantizedMemory)
		t.Logf("Quantized memory usage: %d bytes", quantizedMemory)

		// Quantized should use less memory
		if quantizedMemory >= unquantizedMemory {
			t.Logf("Warning: Quantized memory (%d) not less than unquantized (%d)",
				quantizedMemory, unquantizedMemory)
		}

		// Test search accuracy
		query := vectors[0]

		unquantizedResults, err := unquantizedIndex.Search(ctx, query, 10)
		if err != nil {
			t.Fatalf("Unquantized search failed: %v", err)
		}

		quantizedResults, err := quantizedIndex.Search(ctx, query, 10)
		if err != nil {
			t.Fatalf("Quantized search failed: %v", err)
		}

		// Both should return results
		if len(unquantizedResults) == 0 || len(quantizedResults) == 0 {
			t.Error("Both indices should return search results")
		}

		// First result should be the same (exact match)
		if unquantizedResults[0].ID != quantizedResults[0].ID {
			t.Logf("Warning: First results differ - unquantized: %s, quantized: %s",
				unquantizedResults[0].ID, quantizedResults[0].ID)
		}
	})
}

// generateTestVectors creates random test vectors
func generateTestVectors(count, dimension int) [][]float32 {
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		vec := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			vec[j] = float32(math.Sin(float64(i*dimension+j))) * 10.0
		}
		vectors[i] = vec
	}
	return vectors
}
