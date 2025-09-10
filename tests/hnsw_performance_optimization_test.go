package tests

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/index/hnsw"
	"github.com/xDarkicex/libravdb/internal/util"
)

// TestHNSWPerformanceOptimizations validates the performance improvements
func TestHNSWPerformanceOptimizations(t *testing.T) {
	ctx := context.Background()

	// Create temporary directory for test
	tempDir, err := os.MkdirTemp("", "hnsw_perf_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	t.Run("Large Dataset Performance", func(t *testing.T) {
		// Test with 500 vectors (previously problematic size)
		vectorCount := 500
		dimension := 128

		// Create optimized HNSW index
		config := &hnsw.Config{
			Dimension:      dimension,
			M:              16,
			EfConstruction: 100,
			EfSearch:       50,
			ML:             1.0 / 0.693147, // 1/ln(2)
			Metric:         util.L2Distance,
			RandomSeed:     42,
		}

		index, err := hnsw.NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create HNSW index: %v", err)
		}
		defer index.Close()

		// Generate test vectors
		vectors := generateOptimizedTestVectors(vectorCount, dimension, 42)

		// Test individual insertion performance
		t.Run("Individual Insertion", func(t *testing.T) {
			insertStart := time.Now()

			for i, vector := range vectors {
				entry := &hnsw.VectorEntry{
					ID:       fmt.Sprintf("vec_%d", i),
					Vector:   vector,
					Metadata: map[string]interface{}{"index": i},
				}

				err := index.Insert(ctx, entry)
				if err != nil {
					t.Fatalf("Failed to insert vector %d: %v", i, err)
				}

				// Progress reporting for large datasets
				if (i+1)%100 == 0 {
					t.Logf("Inserted %d/%d vectors", i+1, vectorCount)
				}
			}

			insertDuration := time.Since(insertStart)
			insertRate := float64(vectorCount) / insertDuration.Seconds()

			t.Logf("Individual insertion: %d vectors in %v (%.2f ops/sec)",
				vectorCount, insertDuration, insertRate)

			// Performance should be reasonable for 500 vectors
			if insertRate < 10 { // At least 10 ops/sec
				t.Errorf("Insert performance too slow: %.2f ops/sec", insertRate)
			}
		})

		// Test search performance with populated index
		t.Run("Search Performance", func(t *testing.T) {
			queryVector := vectors[0] // Use first vector as query

			// Warm up
			_, err := index.Search(ctx, queryVector, 10)
			if err != nil {
				t.Fatalf("Warmup search failed: %v", err)
			}

			// Measure search performance
			searchStart := time.Now()
			results, err := index.Search(ctx, queryVector, 10)
			searchDuration := time.Since(searchStart)

			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			if len(results) == 0 {
				t.Error("Search should return results")
			}

			t.Logf("Search completed in %v for %d vectors", searchDuration, vectorCount)

			// Search should be fast even with 500 vectors
			if searchDuration > 50*time.Millisecond {
				t.Errorf("Search performance too slow: %v", searchDuration)
			}

			// Verify accuracy - first result should be exact match
			if results[0].ID != "vec_0" {
				t.Errorf("Expected first result to be vec_0, got %s", results[0].ID)
			}
		})

		// Test memory usage
		t.Run("Memory Usage", func(t *testing.T) {
			memUsage := index.MemoryUsage()
			t.Logf("Memory usage for %d vectors: %d bytes (%.2f MB)",
				vectorCount, memUsage, float64(memUsage)/(1024*1024))

			// Memory usage should be reasonable (account for HNSW graph structure)
			// HNSW uses additional memory for graph links, metadata, and internal structures
			expectedMaxMemory := int64(vectorCount * dimension * 16) // Realistic estimate for HNSW
			if memUsage > expectedMaxMemory {
				t.Errorf("Memory usage too high: %d bytes (expected < %d)",
					memUsage, expectedMaxMemory)
			}

			// Log memory efficiency
			vectorDataSize := int64(vectorCount * dimension * 4) // Just the vector data
			overhead := float64(memUsage-vectorDataSize) / float64(vectorDataSize) * 100
			t.Logf("Memory overhead: %.1f%% (vector data: %d bytes, total: %d bytes)",
				overhead, vectorDataSize, memUsage)
		})
	})

	t.Run("Batch Insertion Performance", func(t *testing.T) {
		// Test batch insertion with 200 vectors
		vectorCount := 200
		dimension := 64

		config := &hnsw.Config{
			Dimension:      dimension,
			M:              8,
			EfConstruction: 50,
			EfSearch:       30,
			ML:             1.0 / 0.693147,
			Metric:         util.L2Distance,
			RandomSeed:     123,
		}

		index, err := hnsw.NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create HNSW index: %v", err)
		}
		defer index.Close()

		// Generate test vectors
		vectors := generateOptimizedTestVectors(vectorCount, dimension, 123)

		// Prepare batch entries
		entries := make([]*hnsw.VectorEntry, vectorCount)
		for i, vector := range vectors {
			entries[i] = &hnsw.VectorEntry{
				ID:       fmt.Sprintf("batch_vec_%d", i),
				Vector:   vector,
				Metadata: map[string]interface{}{"batch": i / 20},
			}
		}

		// Test batch insertion
		batchStart := time.Now()
		err = index.BatchInsert(ctx, entries)
		batchDuration := time.Since(batchStart)

		if err != nil {
			t.Fatalf("Batch insertion failed: %v", err)
		}

		batchRate := float64(vectorCount) / batchDuration.Seconds()
		t.Logf("Batch insertion: %d vectors in %v (%.2f ops/sec)",
			vectorCount, batchDuration, batchRate)

		// Batch insertion should be faster than individual insertion
		if batchRate < 20 { // At least 20 ops/sec for batch
			t.Errorf("Batch insert performance too slow: %.2f ops/sec", batchRate)
		}

		// Verify all vectors were inserted
		if index.Size() != vectorCount {
			t.Errorf("Expected %d vectors, got %d", vectorCount, index.Size())
		}

		// Test search after batch insertion
		queryVector := vectors[vectorCount/2]
		results, err := index.Search(ctx, queryVector, 5)
		if err != nil {
			t.Fatalf("Search after batch insertion failed: %v", err)
		}

		if len(results) == 0 {
			t.Error("Search after batch insertion should return results")
		}
	})

	t.Run("Neighbor Selection Optimization", func(t *testing.T) {
		// Test that neighbor selection is efficient with many candidates
		vectorCount := 100
		dimension := 32

		config := &hnsw.Config{
			Dimension:      dimension,
			M:              16,  // Higher M to test neighbor selection
			EfConstruction: 200, // High ef to generate many candidates
			EfSearch:       100,
			ML:             1.0 / 0.693147,
			Metric:         util.L2Distance,
			RandomSeed:     456,
		}

		index, err := hnsw.NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create HNSW index: %v", err)
		}
		defer index.Close()

		// Generate clustered vectors to stress neighbor selection
		vectors := generateClusteredVectors(vectorCount, dimension, 5, 456)

		insertStart := time.Now()
		for i, vector := range vectors {
			entry := &hnsw.VectorEntry{
				ID:       fmt.Sprintf("cluster_vec_%d", i),
				Vector:   vector,
				Metadata: map[string]interface{}{"cluster": i / 20},
			}

			err := index.Insert(ctx, entry)
			if err != nil {
				t.Fatalf("Failed to insert clustered vector %d: %v", i, err)
			}
		}
		insertDuration := time.Since(insertStart)

		insertRate := float64(vectorCount) / insertDuration.Seconds()
		t.Logf("Clustered insertion: %d vectors in %v (%.2f ops/sec)",
			vectorCount, insertDuration, insertRate)

		// Even with clustered data, performance should be reasonable
		if insertRate < 15 { // At least 15 ops/sec
			t.Errorf("Clustered insert performance too slow: %.2f ops/sec", insertRate)
		}

		// Test search quality
		queryVector := vectors[0]
		results, err := index.Search(ctx, queryVector, 10)
		if err != nil {
			t.Fatalf("Search on clustered data failed: %v", err)
		}

		if len(results) < 5 {
			t.Errorf("Expected at least 5 results, got %d", len(results))
		}

		// Results should be sorted by distance
		for i := 1; i < len(results); i++ {
			if results[i].Score < results[i-1].Score {
				t.Errorf("Results not sorted by distance: %f < %f at position %d",
					results[i].Score, results[i-1].Score, i)
			}
		}
	})
}

// generateOptimizedTestVectors creates random test vectors
func generateOptimizedTestVectors(count, dimension int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	vectors := make([][]float32, count)

	for i := 0; i < count; i++ {
		vector := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			vector[j] = rng.Float32()*2 - 1 // Random values between -1 and 1
		}
		vectors[i] = vector
	}

	return vectors
}

// generateClusteredVectors creates vectors in clusters to test neighbor selection
func generateClusteredVectors(count, dimension, numClusters int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	vectors := make([][]float32, count)

	// Generate cluster centers
	centers := make([][]float32, numClusters)
	for i := 0; i < numClusters; i++ {
		center := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			center[j] = rng.Float32()*10 - 5 // Centers between -5 and 5
		}
		centers[i] = center
	}

	// Generate vectors around cluster centers
	for i := 0; i < count; i++ {
		clusterIdx := i % numClusters
		center := centers[clusterIdx]

		vector := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			// Add small random offset to cluster center
			offset := (rng.Float32()*2 - 1) * 0.5 // Small offset
			vector[j] = center[j] + offset
		}
		vectors[i] = vector
	}

	return vectors
}

// BenchmarkHNSWOptimizations provides benchmarks for the optimizations
func BenchmarkHNSWOptimizations(b *testing.B) {
	ctx := context.Background()
	dimension := 128

	config := &hnsw.Config{
		Dimension:      dimension,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0 / 0.693147,
		Metric:         util.L2Distance,
		RandomSeed:     42,
	}

	b.Run("Insert", func(b *testing.B) {
		index, err := hnsw.NewHNSW(config)
		if err != nil {
			b.Fatalf("Failed to create HNSW index: %v", err)
		}
		defer index.Close()

		vectors := generateOptimizedTestVectors(b.N, dimension, 42)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			entry := &hnsw.VectorEntry{
				ID:     fmt.Sprintf("bench_vec_%d", i),
				Vector: vectors[i],
			}
			err := index.Insert(ctx, entry)
			if err != nil {
				b.Fatalf("Insert failed: %v", err)
			}
		}
	})

	b.Run("BatchInsert", func(b *testing.B) {
		index, err := hnsw.NewHNSW(config)
		if err != nil {
			b.Fatalf("Failed to create HNSW index: %v", err)
		}
		defer index.Close()

		vectors := generateOptimizedTestVectors(b.N, dimension, 42)
		entries := make([]*hnsw.VectorEntry, b.N)
		for i := 0; i < b.N; i++ {
			entries[i] = &hnsw.VectorEntry{
				ID:     fmt.Sprintf("batch_bench_vec_%d", i),
				Vector: vectors[i],
			}
		}

		b.ResetTimer()
		err = index.BatchInsert(ctx, entries)
		if err != nil {
			b.Fatalf("BatchInsert failed: %v", err)
		}
	})

	b.Run("Search", func(b *testing.B) {
		index, err := hnsw.NewHNSW(config)
		if err != nil {
			b.Fatalf("Failed to create HNSW index: %v", err)
		}
		defer index.Close()

		// Pre-populate index
		vectors := generateOptimizedTestVectors(1000, dimension, 42)
		for i, vector := range vectors {
			entry := &hnsw.VectorEntry{
				ID:     fmt.Sprintf("search_vec_%d", i),
				Vector: vector,
			}
			err := index.Insert(ctx, entry)
			if err != nil {
				b.Fatalf("Failed to populate index: %v", err)
			}
		}

		queryVector := vectors[0]

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := index.Search(ctx, queryVector, 10)
			if err != nil {
				b.Fatalf("Search failed: %v", err)
			}
		}
	})
}
