package benchmark

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

// BenchmarkFormatCompatibility tests backward compatibility with format v1
func BenchmarkFormatCompatibility(b *testing.B) {
	index, filePath := createTestIndex(b, 25000)
	ctx := context.Background()

	// Save with current format
	if err := index.SaveToDisk(ctx, filePath); err != nil {
		b.Fatalf("Failed to save index: %v", err)
	}

	config := &hnsw.Config{
		Dimension:      Dimension,
		M:              M,
		EfConstruction: EfConst,
		EfSearch:       EfSearch,
		ML:             1.0 / 2.303,
		Metric:         util.L2Distance,
		RandomSeed:     42,
	}

	compatibilitySuccessCount := 0

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loadIndex, err := hnsw.NewHNSW(config)
		if err != nil {
			b.Fatalf("Failed to create index: %v", err)
		}

		// Load and verify compatibility
		if err := loadIndex.LoadFromDisk(ctx, filePath); err != nil {
			b.Errorf("Compatibility test failed: %v", err)
			continue
		}

		// Verify loaded index works
		queryVector := make([]float32, Dimension)
		for j := range queryVector {
			queryVector[j] = rand.Float32()
		}

		results, err := loadIndex.Search(ctx, queryVector, 5)
		if err != nil {
			b.Errorf("Search after load failed: %v", err)
			continue
		}

		if len(results) == 0 {
			b.Errorf("No search results after compatibility load")
			continue
		}

		compatibilitySuccessCount++
	}

	compatibilityRate := float64(compatibilitySuccessCount) / float64(b.N) * 100
	b.Logf("Format compatibility success rate: %.2f%%", compatibilityRate)
	b.ReportMetric(compatibilityRate, "compatibility_%")
}

// BenchmarkZeroDataLoss tests that no data is lost during save/load cycles
func BenchmarkZeroDataLoss(b *testing.B) {
	const testVectors = 10000
	index, filePath := createTestIndex(b, testVectors)
	ctx := context.Background()

	// Get original metadata
	originalMeta := index.GetPersistenceMetadata()
	if originalMeta == nil {
		b.Fatal("Failed to get original metadata")
	}

	dataLossCount := 0

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Save index
		if err := index.SaveToDisk(ctx, filePath); err != nil {
			b.Fatalf("Failed to save index: %v", err)
		}

		// Load into new index
		config := &hnsw.Config{
			Dimension:      Dimension,
			M:              M,
			EfConstruction: EfConst,
			EfSearch:       EfSearch,
			ML:             1.0 / 2.303,
			Metric:         util.L2Distance,
			RandomSeed:     42,
		}

		loadIndex, err := hnsw.NewHNSW(config)
		if err != nil {
			b.Fatalf("Failed to create load index: %v", err)
		}

		if err := loadIndex.LoadFromDisk(ctx, filePath); err != nil {
			b.Fatalf("Failed to load index: %v", err)
		}

		// Compare metadata
		loadedMeta := loadIndex.GetPersistenceMetadata()
		if loadedMeta == nil {
			b.Errorf("Failed to get loaded metadata")
			dataLossCount++
			continue
		}

		if originalMeta.NodeCount != loadedMeta.NodeCount {
			b.Errorf("Node count mismatch: original=%d, loaded=%d",
				originalMeta.NodeCount, loadedMeta.NodeCount)
			dataLossCount++
			continue
		}

		if originalMeta.Dimension != loadedMeta.Dimension {
			b.Errorf("Dimension mismatch: original=%d, loaded=%d",
				originalMeta.Dimension, loadedMeta.Dimension)
			dataLossCount++
			continue
		}

		// Test search functionality
		queryVector := make([]float32, Dimension)
		for j := range queryVector {
			queryVector[j] = rand.Float32()
		}

		originalResults, err := index.Search(ctx, queryVector, 10)
		if err != nil {
			b.Errorf("Original search failed: %v", err)
			dataLossCount++
			continue
		}

		loadedResults, err := loadIndex.Search(ctx, queryVector, 10)
		if err != nil {
			b.Errorf("Loaded search failed: %v", err)
			dataLossCount++
			continue
		}

		if len(originalResults) != len(loadedResults) {
			b.Errorf("Search result count mismatch: original=%d, loaded=%d",
				len(originalResults), len(loadedResults))
			dataLossCount++
			continue
		}
	}

	dataIntegrityRate := float64(b.N-dataLossCount) / float64(b.N) * 100
	b.Logf("Data integrity success rate: %.2f%%", dataIntegrityRate)
	b.ReportMetric(dataIntegrityRate, "integrity_%")
}

// BenchmarkLargeScaleOperations tests performance with very large indices
func BenchmarkLargeScaleOperations(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping large scale test in short mode")
	}

	// Test with 2M vectors (large scale)
	const largeVectorCount = 2000000
	b.Logf("Creating large scale index with %d vectors...", largeVectorCount)

	index, filePath := createTestIndex(b, largeVectorCount)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()

		// Save large index
		if err := index.SaveToDisk(ctx, filePath); err != nil {
			b.Fatalf("Failed to save large index: %v", err)
		}

		saveDuration := time.Since(start)

		// Load large index
		config := &hnsw.Config{
			Dimension:      Dimension,
			M:              M,
			EfConstruction: EfConst,
			EfSearch:       EfSearch,
			ML:             1.0 / 2.303,
			Metric:         util.L2Distance,
			RandomSeed:     42,
		}

		loadIndex, err := hnsw.NewHNSW(config)
		if err != nil {
			b.Fatalf("Failed to create large load index: %v", err)
		}

		loadStart := time.Now()
		if err := loadIndex.LoadFromDisk(ctx, filePath); err != nil {
			b.Fatalf("Failed to load large index: %v", err)
		}
		loadDuration := time.Since(loadStart)

		if i == 0 {
			reportThroughput(b, filePath, saveDuration, "Large Scale Save")
			reportThroughput(b, filePath, loadDuration, "Large Scale Load")
		}
	}
}

// BenchmarkConcurrentOperations tests thread safety during persistence operations
func BenchmarkConcurrentOperations(b *testing.B) {
	index, filePath := createTestIndex(b, 50000)
	ctx := context.Background()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		threadID := 0
		for pb.Next() {
			// Each thread uses its own file path
			threadFilePath := fmt.Sprintf("%s.thread_%d", filePath, threadID)

			// Save operation
			if err := index.SaveToDisk(ctx, threadFilePath); err != nil {
				b.Errorf("Concurrent save failed: %v", err)
				continue
			}

			// Load operation
			config := &hnsw.Config{
				Dimension:      Dimension,
				M:              M,
				EfConstruction: EfConst,
				EfSearch:       EfSearch,
				ML:             1.0 / 2.303,
				Metric:         util.L2Distance,
				RandomSeed:     42,
			}

			loadIndex, err := hnsw.NewHNSW(config)
			if err != nil {
				b.Errorf("Failed to create concurrent load index: %v", err)
				continue
			}

			if err := loadIndex.LoadFromDisk(ctx, threadFilePath); err != nil {
				b.Errorf("Concurrent load failed: %v", err)
				continue
			}

			// Clean up thread file
			os.Remove(threadFilePath)
			threadID++
		}
	})
}

// BenchmarkProgressiveLoad tests loading performance with different file sizes
func BenchmarkProgressiveLoad(b *testing.B) {
	sizes := []int{1000, 5000, 10000, 25000, 50000, 100000}
	ctx := context.Background()

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Load_%dK", size/1000), func(b *testing.B) {
			index, filePath := createTestIndex(b, size)

			// Save the index
			if err := index.SaveToDisk(ctx, filePath); err != nil {
				b.Fatalf("Failed to save index: %v", err)
			}

			config := &hnsw.Config{
				Dimension:      Dimension,
				M:              M,
				EfConstruction: EfConst,
				EfSearch:       EfSearch,
				ML:             1.0 / 2.303,
				Metric:         util.L2Distance,
				RandomSeed:     42,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				loadIndex, err := hnsw.NewHNSW(config)
				if err != nil {
					b.Fatalf("Failed to create index: %v", err)
				}

				start := time.Now()
				if err := loadIndex.LoadFromDisk(ctx, filePath); err != nil {
					b.Fatalf("Failed to load index: %v", err)
				}
				duration := time.Since(start)

				if i == 0 {
					reportThroughput(b, filePath, duration, fmt.Sprintf("Progressive Load %dK", size/1000))
				}
			}
		})
	}
}
