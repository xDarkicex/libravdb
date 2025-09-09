package hnsw

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

func TestSimpleQuantizationIntegration(t *testing.T) {
	ctx := context.Background()

	t.Run("Basic Quantization Setup", func(t *testing.T) {
		// Create HNSW with Scalar Quantization (simpler than PQ)
		config := &Config{
			Dimension:      32,
			M:              8,
			EfConstruction: 50,
			EfSearch:       20,
			ML:             1.0 / math.Log(2.0),
			Metric:         util.L2Distance,
			RandomSeed:     42,
			Quantization: &quant.QuantizationConfig{
				Type:       quant.ScalarQuantization,
				Bits:       8,
				TrainRatio: 0.2,
			},
		}

		index, err := NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create HNSW index: %v", err)
		}
		defer index.Close()

		// Verify quantizer is initialized
		if index.quantizer == nil {
			t.Error("Quantizer should be initialized")
		}

		// Check training threshold
		threshold := index.getTrainingThreshold()
		t.Logf("Training threshold: %d", threshold)
		if threshold <= 0 {
			t.Error("Training threshold should be positive")
		}

		// Insert a few vectors (less than threshold)
		for i := 0; i < 10; i++ {
			vec := make([]float32, 32)
			for j := range vec {
				vec[j] = float32(i + j)
			}

			entry := &VectorEntry{
				ID:     fmt.Sprintf("vec_%d", i),
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

		// All nodes should have original vectors (not quantized)
		for _, node := range index.nodes {
			if node.CompressedVector != nil {
				t.Error("Nodes should not be quantized before training")
			}
			if node.Vector == nil {
				t.Error("Nodes should have original vectors before training")
			}
		}
	})

	t.Run("Quantization Training Trigger", func(t *testing.T) {
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
				TrainRatio: 0.5,
			},
		}

		index, err := NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create HNSW index: %v", err)
		}
		defer index.Close()

		threshold := index.getTrainingThreshold()
		t.Logf("Training threshold: %d", threshold)

		// Insert enough vectors to trigger training
		for i := 0; i < threshold+10; i++ {
			vec := make([]float32, 16)
			for j := range vec {
				vec[j] = float32(math.Sin(float64(i*16+j) * 0.1))
			}

			entry := &VectorEntry{
				ID:     fmt.Sprintf("train_vec_%d", i),
				Vector: vec,
			}

			err := index.Insert(ctx, entry)
			if err != nil {
				t.Fatalf("Failed to insert vector %d: %v", i, err)
			}

			// Check if training happened
			if i >= threshold && index.quantizationTrained {
				t.Logf("Training triggered after %d vectors", i+1)
				break
			}
		}

		// Quantization should be trained now
		if !index.quantizationTrained {
			t.Error("Quantization should be trained after inserting enough vectors")
		}

		// Some nodes should be quantized (those inserted after training)
		quantizedCount := 0
		for _, node := range index.nodes {
			if node.CompressedVector != nil {
				quantizedCount++
			}
		}

		if quantizedCount == 0 {
			t.Error("Some nodes should be quantized after training")
		}

		t.Logf("Quantized nodes: %d out of %d", quantizedCount, len(index.nodes))
	})

	t.Run("Search with Mixed Nodes", func(t *testing.T) {
		config := &Config{
			Dimension:      8,
			M:              4,
			EfConstruction: 10,
			EfSearch:       5,
			ML:             1.0 / math.Log(2.0),
			Metric:         util.L2Distance,
			RandomSeed:     42,
			Quantization: &quant.QuantizationConfig{
				Type:       quant.ScalarQuantization,
				Bits:       8,
				TrainRatio: 0.3,
			},
		}

		index, err := NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create HNSW index: %v", err)
		}
		defer index.Close()

		// Insert vectors to trigger training
		vectors := make([][]float32, 100)
		for i := 0; i < 100; i++ {
			vec := make([]float32, 8)
			for j := range vec {
				vec[j] = float32(i + j)
			}
			vectors[i] = vec

			entry := &VectorEntry{
				ID:     fmt.Sprintf("mixed_vec_%d", i),
				Vector: vec,
			}

			err := index.Insert(ctx, entry)
			if err != nil {
				t.Fatalf("Failed to insert vector %d: %v", i, err)
			}
		}

		// Test search
		query := vectors[0]
		results, err := index.Search(ctx, query, 3)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		if len(results) == 0 {
			t.Error("Search should return results")
		}

		// First result should be the exact match
		if results[0].ID != "mixed_vec_0" {
			t.Errorf("Expected first result to be mixed_vec_0, got %s", results[0].ID)
		}

		// Verify all results have valid vectors
		for i, result := range results {
			if result.Vector == nil {
				t.Errorf("Result %d should have a vector", i)
			}
			if len(result.Vector) != 8 {
				t.Errorf("Result %d vector should have dimension 8, got %d", i, len(result.Vector))
			}
		}
	})
}

func TestQuantizationConfiguration(t *testing.T) {
	t.Run("Invalid Quantization Config", func(t *testing.T) {
		config := &Config{
			Dimension:      32,
			M:              8,
			EfConstruction: 50,
			EfSearch:       20,
			ML:             1.0 / math.Log(2.0),
			Metric:         util.L2Distance,
			RandomSeed:     42,
			Quantization: &quant.QuantizationConfig{
				Type:       quant.ScalarQuantization,
				Bits:       0, // Invalid
				TrainRatio: 0.1,
			},
		}

		_, err := NewHNSW(config)
		if err == nil {
			t.Error("Should fail with invalid quantization config")
		}
	})

	t.Run("No Quantization Config", func(t *testing.T) {
		config := &Config{
			Dimension:      32,
			M:              8,
			EfConstruction: 50,
			EfSearch:       20,
			ML:             1.0 / math.Log(2.0),
			Metric:         util.L2Distance,
			RandomSeed:     42,
			// No quantization config
		}

		index, err := NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create HNSW index: %v", err)
		}
		defer index.Close()

		// Quantizer should be nil
		if index.quantizer != nil {
			t.Error("Quantizer should be nil when no quantization config is provided")
		}

		// Training should be false
		if index.quantizationTrained {
			t.Error("Quantization trained should be false when no quantization config")
		}
	})
}
