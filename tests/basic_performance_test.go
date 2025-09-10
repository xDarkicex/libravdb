package tests

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/libravdb"
)

// TestBasicPerformance validates that basic operations work within reasonable time limits
func TestBasicPerformance(t *testing.T) {
	ctx := context.Background()

	// Create temporary directory for test
	tempDir, err := os.MkdirTemp("", "libravdb_basic_perf_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	t.Run("Small Collection Performance", func(t *testing.T) {
		// Create database
		db, err := libravdb.New(
			libravdb.WithStoragePath(filepath.Join(tempDir, "small_test")),
			libravdb.WithMetrics(false),
		)
		if err != nil {
			t.Fatalf("Failed to create database: %v", err)
		}
		defer db.Close()

		// Create collection
		collection, err := db.CreateCollection(
			ctx,
			"small_vectors",
			libravdb.WithDimension(64),
			libravdb.WithMetric(libravdb.L2Distance),
			libravdb.WithHNSW(8, 50, 20),
		)
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// Test insertion performance with small dataset
		vectorCount := 25
		insertStart := time.Now()

		for i := range vectorCount {
			vector := make([]float32, 64)
			for j := range 64 {
				vector[j] = float32(i*64 + j)
			}

			err := collection.Insert(ctx, fmt.Sprintf("vec_%d", i), vector, map[string]interface{}{
				"index": i,
				"batch": i / 5,
			})
			if err != nil {
				t.Fatalf("Failed to insert vector %d: %v", i, err)
			}
		}

		insertDuration := time.Since(insertStart)
		t.Logf("Inserted %d vectors in %v (%.2f ops/sec)",
			vectorCount, insertDuration,
			float64(vectorCount)/insertDuration.Seconds())

		// Test search performance
		query := make([]float32, 64)
		for i := range 64 {
			query[i] = float32(i)
		}

		searchStart := time.Now()
		results, err := collection.Search(ctx, query, 5)
		searchDuration := time.Since(searchStart)

		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		if len(results.Results) == 0 {
			t.Error("Search should return results")
		}

		t.Logf("Search completed in %v", searchDuration)

		// Verify reasonable performance
		if insertDuration > 5*time.Second {
			t.Errorf("Insert performance too slow: %v", insertDuration)
		}

		if searchDuration > 100*time.Millisecond {
			t.Errorf("Search performance too slow: %v", searchDuration)
		}

		// Verify collection stats
		stats := collection.Stats()
		if stats.VectorCount != vectorCount {
			t.Errorf("Expected %d vectors, got %d", vectorCount, stats.VectorCount)
		}

		t.Logf("Collection stats: %d vectors, %d bytes memory",
			stats.VectorCount, stats.MemoryUsage)
	})

	t.Run("Flat Index Performance", func(t *testing.T) {
		// Create database
		db, err := libravdb.New(
			libravdb.WithStoragePath(filepath.Join(tempDir, "flat_test")),
			libravdb.WithMetrics(false),
		)
		if err != nil {
			t.Fatalf("Failed to create database: %v", err)
		}
		defer db.Close()

		// Create collection with Flat index (better for small datasets)
		collection, err := db.CreateCollection(
			ctx,
			"flat_vectors",
			libravdb.WithDimension(32),
			libravdb.WithMetric(libravdb.CosineDistance),
			libravdb.WithFlat(), // Use flat index for exact search
		)
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// Test with slightly larger dataset for flat index
		vectorCount := 100
		insertStart := time.Now()

		for i := range vectorCount {
			vector := make([]float32, 32)
			for j := range 32 {
				vector[j] = float32(i*32 + j)
			}

			err := collection.Insert(ctx, fmt.Sprintf("flat_vec_%d", i), vector, map[string]interface{}{
				"type": "flat_test",
			})
			if err != nil {
				t.Fatalf("Failed to insert vector %d: %v", i, err)
			}
		}

		insertDuration := time.Since(insertStart)
		t.Logf("Flat index: Inserted %d vectors in %v (%.2f ops/sec)",
			vectorCount, insertDuration,
			float64(vectorCount)/insertDuration.Seconds())

		// Test search
		query := make([]float32, 32)
		for i := range 32 {
			query[i] = float32(i)
		}

		searchStart := time.Now()
		results, err := collection.Search(ctx, query, 10)
		searchDuration := time.Since(searchStart)

		if err != nil {
			t.Fatalf("Flat search failed: %v", err)
		}

		if len(results.Results) == 0 {
			t.Error("Flat search should return results")
		}

		t.Logf("Flat search completed in %v", searchDuration)

		// Flat index should be fast for small datasets
		if searchDuration > 50*time.Millisecond {
			t.Errorf("Flat search performance too slow: %v", searchDuration)
		}
	})

	t.Run("Batch Operations Performance", func(t *testing.T) {
		// Create database
		db, err := libravdb.New(
			libravdb.WithStoragePath(filepath.Join(tempDir, "batch_test")),
			libravdb.WithMetrics(false),
		)
		if err != nil {
			t.Fatalf("Failed to create database: %v", err)
		}
		defer db.Close()

		// Create collection
		collection, err := db.CreateCollection(
			ctx,
			"batch_vectors",
			libravdb.WithDimension(16),
			libravdb.WithMetric(libravdb.L2Distance),
			libravdb.WithFlat(), // Use flat for predictable performance
		)
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// Prepare batch data
		batchSize := 20
		vectors := make([][]float32, batchSize)
		ids := make([]string, batchSize)
		metadata := make([]map[string]interface{}, batchSize)

		for i := range batchSize {
			vectors[i] = make([]float32, 16)
			for j := range 16 {
				vectors[i][j] = float32(i*16 + j)
			}
			ids[i] = fmt.Sprintf("batch_vec_%d", i)
			metadata[i] = map[string]interface{}{
				"batch_id": i,
			}
		}

		// Test batch insert performance (insert individually for now)
		batchStart := time.Now()
		for i := range batchSize {
			err = collection.Insert(ctx, ids[i], vectors[i], metadata[i])
			if err != nil {
				t.Fatalf("Failed to insert vector %d in batch: %v", i, err)
			}
		}
		batchDuration := time.Since(batchStart)

		t.Logf("Batch insert: %d vectors in %v (%.2f ops/sec)",
			batchSize, batchDuration,
			float64(batchSize)/batchDuration.Seconds())

		// Verify all vectors were inserted
		stats := collection.Stats()
		if stats.VectorCount != batchSize {
			t.Errorf("Expected %d vectors, got %d", batchSize, stats.VectorCount)
		}

		// Test search after batch insert
		query := vectors[0]
		results, err := collection.Search(ctx, query, 5)
		if err != nil {
			t.Fatalf("Search after batch insert failed: %v", err)
		}

		if len(results.Results) == 0 {
			t.Error("Search after batch insert should return results")
		}

		// First result should be exact match
		if results.Results[0].ID != ids[0] {
			t.Errorf("Expected first result to be %s, got %s", ids[0], results.Results[0].ID)
		}
	})
}
