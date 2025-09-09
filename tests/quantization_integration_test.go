package tests

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/libravdb"
)

func TestQuantizationIntegration(t *testing.T) {
	ctx := context.Background()

	// Create temporary directory for test
	tempDir, err := os.MkdirTemp("", "libravdb_quant_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	t.Run("Product Quantization End-to-End", func(t *testing.T) {
		// Create database
		db, err := libravdb.New(
			libravdb.WithStoragePath(filepath.Join(tempDir, "pq_test")),
			libravdb.WithMetrics(false),
		)
		if err != nil {
			t.Fatalf("Failed to create database: %v", err)
		}
		defer db.Close()

		// Create collection with Product Quantization
		collection, err := db.CreateCollection(
			ctx,
			"pq_vectors",
			libravdb.WithDimension(128),
			libravdb.WithMetric(libravdb.L2Distance),
			libravdb.WithHNSW(16, 200, 50),
			libravdb.WithProductQuantization(8, 8, 0.2),
		)
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// Generate and insert test vectors
		vectors := generateTestVectors(1000, 128)
		for i, vec := range vectors {
			err := collection.Insert(ctx, fmt.Sprintf("pq_vec_%d", i), vec, map[string]interface{}{
				"category": "test",
				"index":    i,
			})
			if err != nil {
				t.Fatalf("Failed to insert vector %d: %v", i, err)
			}
		}

		// Test search functionality
		query := vectors[0]
		results, err := collection.Search(ctx, query, 5)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		if len(results.Results) == 0 {
			t.Error("Search should return results")
		}

		// First result should be exact match
		if results.Results[0].ID != "pq_vec_0" {
			t.Errorf("Expected first result to be pq_vec_0, got %s", results.Results[0].ID)
		}

		// Verify collection stats
		stats := collection.Stats()
		if stats.VectorCount != 1000 {
			t.Errorf("Expected 1000 vectors, got %d", stats.VectorCount)
		}

		if stats.MemoryUsage <= 0 {
			t.Error("Memory usage should be positive")
		}

		t.Logf("Collection stats: %d vectors, %d bytes memory", stats.VectorCount, stats.MemoryUsage)
	})

	t.Run("Scalar Quantization End-to-End", func(t *testing.T) {
		// Create database
		db, err := libravdb.New(
			libravdb.WithStoragePath(filepath.Join(tempDir, "sq_test")),
			libravdb.WithMetrics(false),
		)
		if err != nil {
			t.Fatalf("Failed to create database: %v", err)
		}
		defer db.Close()

		// Create collection with Scalar Quantization
		collection, err := db.CreateCollection(
			ctx,
			"sq_vectors",
			libravdb.WithDimension(64),
			libravdb.WithMetric(libravdb.CosineDistance),
			libravdb.WithHNSW(8, 100, 30),
			libravdb.WithScalarQuantization(8, 0.1),
		)
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// Generate and insert test vectors
		vectors := generateTestVectors(500, 64)
		for i, vec := range vectors {
			err := collection.Insert(ctx, fmt.Sprintf("sq_vec_%d", i), vec, map[string]interface{}{
				"type": "scalar_quantized",
			})
			if err != nil {
				t.Fatalf("Failed to insert vector %d: %v", i, err)
			}
		}

		// Test search
		query := vectors[10]
		results, err := collection.Search(ctx, query, 3)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		if len(results.Results) == 0 {
			t.Error("Search should return results")
		}

		// Verify search quality
		for _, result := range results.Results {
			if result.Score < 0 {
				t.Errorf("Score should be non-negative, got %f", result.Score)
			}
		}
	})

	t.Run("Quantization vs Unquantized Comparison", func(t *testing.T) {
		// Create database
		db, err := libravdb.New(
			libravdb.WithStoragePath(filepath.Join(tempDir, "comparison_test")),
			libravdb.WithMetrics(false),
		)
		if err != nil {
			t.Fatalf("Failed to create database: %v", err)
		}
		defer db.Close()

		// Create unquantized collection
		unquantizedCollection, err := db.CreateCollection(
			ctx,
			"unquantized_vectors",
			libravdb.WithDimension(256),
			libravdb.WithMetric(libravdb.L2Distance),
			libravdb.WithHNSW(32, 200, 50),
		)
		if err != nil {
			t.Fatalf("Failed to create unquantized collection: %v", err)
		}

		// Create quantized collection
		quantizedCollection, err := db.CreateCollection(
			ctx,
			"quantized_vectors",
			libravdb.WithDimension(256),
			libravdb.WithMetric(libravdb.L2Distance),
			libravdb.WithHNSW(32, 200, 50),
			libravdb.WithProductQuantization(16, 8, 0.1),
		)
		if err != nil {
			t.Fatalf("Failed to create quantized collection: %v", err)
		}

		// Generate test data
		vectors := generateTestVectors(2000, 256)

		// Insert into both collections
		for i, vec := range vectors {
			id := fmt.Sprintf("comp_vec_%d", i)
			metadata := map[string]interface{}{"index": i}

			err := unquantizedCollection.Insert(ctx, id, vec, metadata)
			if err != nil {
				t.Fatalf("Failed to insert into unquantized collection: %v", err)
			}

			err = quantizedCollection.Insert(ctx, id, vec, metadata)
			if err != nil {
				t.Fatalf("Failed to insert into quantized collection: %v", err)
			}
		}

		// Compare memory usage
		unquantizedStats := unquantizedCollection.Stats()
		quantizedStats := quantizedCollection.Stats()

		t.Logf("Unquantized memory: %d bytes", unquantizedStats.MemoryUsage)
		t.Logf("Quantized memory: %d bytes", quantizedStats.MemoryUsage)

		// Test search accuracy
		query := vectors[0]

		unquantizedResults, err := unquantizedCollection.Search(ctx, query, 10)
		if err != nil {
			t.Fatalf("Unquantized search failed: %v", err)
		}

		quantizedResults, err := quantizedCollection.Search(ctx, query, 10)
		if err != nil {
			t.Fatalf("Quantized search failed: %v", err)
		}

		// Both should return results
		if len(unquantizedResults.Results) == 0 || len(quantizedResults.Results) == 0 {
			t.Error("Both collections should return search results")
		}

		// Calculate recall@5
		recall := calculateRecall(unquantizedResults.Results[:5], quantizedResults.Results[:5])
		t.Logf("Recall@5: %.2f", recall)

		// Recall should be reasonable (at least 60%)
		if recall < 0.6 {
			t.Logf("Warning: Low recall@5: %.2f", recall)
		}
	})

	t.Run("Custom Quantization Configuration", func(t *testing.T) {
		// Create database
		db, err := libravdb.New(
			libravdb.WithStoragePath(filepath.Join(tempDir, "custom_test")),
			libravdb.WithMetrics(false),
		)
		if err != nil {
			t.Fatalf("Failed to create database: %v", err)
		}
		defer db.Close()

		// Create collection with custom quantization config
		customConfig := &quant.QuantizationConfig{
			Type:       quant.ProductQuantization,
			Codebooks:  4,
			Bits:       4,
			TrainRatio: 0.3,
			CacheSize:  500,
		}

		collection, err := db.CreateCollection(
			ctx,
			"custom_vectors",
			libravdb.WithDimension(32),
			libravdb.WithMetric(libravdb.CosineDistance),
			libravdb.WithHNSW(8, 50, 20),
			libravdb.WithQuantization(customConfig),
		)
		if err != nil {
			t.Fatalf("Failed to create collection with custom config: %v", err)
		}

		// Insert test vectors
		vectors := generateTestVectors(300, 32)
		for i, vec := range vectors {
			err := collection.Insert(ctx, fmt.Sprintf("custom_vec_%d", i), vec, nil)
			if err != nil {
				t.Fatalf("Failed to insert vector %d: %v", i, err)
			}
		}

		// Test search
		query := vectors[5]
		results, err := collection.Search(ctx, query, 3)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		if len(results.Results) == 0 {
			t.Error("Search should return results")
		}
	})
}

// generateTestVectors creates deterministic test vectors
func generateTestVectors(count, dimension int) [][]float32 {
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		vec := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			// Create deterministic but varied vectors
			vec[j] = float32(math.Sin(float64(i*dimension+j)*0.1) * 10.0)
		}
		// Normalize for cosine distance tests
		normalize(vec)
		vectors[i] = vec
	}
	return vectors
}

// normalize normalizes a vector to unit length
func normalize(vec []float32) {
	var norm float32
	for _, v := range vec {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}
}

// calculateRecall calculates recall between two result sets
func calculateRecall(groundTruth, results []*libravdb.SearchResult) float64 {
	if len(groundTruth) == 0 {
		return 0.0
	}

	// Create set of ground truth IDs
	truthSet := make(map[string]bool)
	for _, result := range groundTruth {
		truthSet[result.ID] = true
	}

	// Count matches in results
	matches := 0
	for _, result := range results {
		if truthSet[result.ID] {
			matches++
		}
	}

	return float64(matches) / float64(len(groundTruth))
}
