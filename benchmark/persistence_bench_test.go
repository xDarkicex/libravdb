package benchmark

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/index/hnsw"
	"github.com/xDarkicex/libravdb/internal/util"
)

// Benchmark configuration
const (
	Dimension = 128 // Common embedding dimension
	M         = 16
	EfConst   = 200
	EfSearch  = 50
)

// Helper function to create test index with specified number of vectors
func createTestIndex(b *testing.B, vectorCount int) (*hnsw.Index, string) {
	config := &hnsw.Config{
		Dimension:      Dimension,
		M:              M,
		EfConstruction: EfConst,
		EfSearch:       EfSearch,
		ML:             1.0 / 2.303,
		Metric:         util.L2Distance,
		RandomSeed:     42,
	}

	index, err := hnsw.NewHNSW(config)
	if err != nil {
		b.Fatalf("Failed to create HNSW index: %v", err)
	}

	// Generate and insert test vectors
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	b.ResetTimer()
	insertStart := time.Now()

	for i := 0; i < vectorCount; i++ {
		vector := make([]float32, Dimension)
		for j := range vector {
			vector[j] = rng.Float32()*2 - 1 // Random values between -1 and 1
		}

		entry := &hnsw.VectorEntry{
			ID:       fmt.Sprintf("vec_%d", i),
			Vector:   vector,
			Metadata: map[string]interface{}{"index": i},
		}

		if err := index.Insert(ctx, entry); err != nil {
			b.Fatalf("Failed to insert vector %d: %v", i, err)
		}

		// Progress reporting for large datasets
		if vectorCount >= 100000 && (i+1)%10000 == 0 {
			b.Logf("Inserted %d/%d vectors", i+1, vectorCount)
		}
	}

	insertDuration := time.Since(insertStart)
	b.Logf("Index creation took %v for %d vectors", insertDuration, vectorCount)

	// Create temporary file path
	tempDir := b.TempDir()
	filePath := filepath.Join(tempDir, "test_index.bin")

	b.ResetTimer()
	return index, filePath
}

// Helper function to get file size and calculate throughput
func reportThroughput(b *testing.B, filePath string, duration time.Duration, operation string) {
	if stat, err := os.Stat(filePath); err == nil {
		sizeBytes := stat.Size()
		sizeMB := float64(sizeBytes) / (1024 * 1024)
		throughputMBps := sizeMB / duration.Seconds()

		b.Logf("%s: %.2f MB in %v (%.2f MB/s)", operation, sizeMB, duration, throughputMBps)

		// Report as custom metrics
		b.ReportMetric(throughputMBps, "MB/s")
		b.ReportMetric(sizeMB, "MB")
	}
}

// Save benchmarks
func BenchmarkSave10K(b *testing.B) {
	index, filePath := createTestIndex(b, 10000)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		if err := index.SaveToDisk(ctx, filePath); err != nil {
			b.Fatalf("Failed to save index: %v", err)
		}
		duration := time.Since(start)

		if i == 0 { // Report throughput on first iteration
			reportThroughput(b, filePath, duration, "Save 10K")
		}
	}
}

func BenchmarkSave100K(b *testing.B) {
	index, filePath := createTestIndex(b, 100000)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		if err := index.SaveToDisk(ctx, filePath); err != nil {
			b.Fatalf("Failed to save index: %v", err)
		}
		duration := time.Since(start)

		if i == 0 {
			reportThroughput(b, filePath, duration, "Save 100K")
		}
	}
}

func BenchmarkSave1M(b *testing.B) {
	index, filePath := createTestIndex(b, 1000000)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		if err := index.SaveToDisk(ctx, filePath); err != nil {
			b.Fatalf("Failed to save index: %v", err)
		}
		duration := time.Since(start)

		if i == 0 {
			reportThroughput(b, filePath, duration, "Save 1M")
		}
	}
}

// Load benchmarks
func BenchmarkLoad10K(b *testing.B) {
	index, filePath := createTestIndex(b, 10000)
	ctx := context.Background()

	// Save the index first
	if err := index.SaveToDisk(ctx, filePath); err != nil {
		b.Fatalf("Failed to save index for load test: %v", err)
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
			b.Fatalf("Failed to create index for loading: %v", err)
		}

		start := time.Now()
		if err := loadIndex.LoadFromDisk(ctx, filePath); err != nil {
			b.Fatalf("Failed to load index: %v", err)
		}
		duration := time.Since(start)

		if i == 0 {
			reportThroughput(b, filePath, duration, "Load 10K")
		}
	}
}

func BenchmarkLoad100K(b *testing.B) {
	index, filePath := createTestIndex(b, 100000)
	ctx := context.Background()

	if err := index.SaveToDisk(ctx, filePath); err != nil {
		b.Fatalf("Failed to save index for load test: %v", err)
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
			b.Fatalf("Failed to create index for loading: %v", err)
		}

		start := time.Now()
		if err := loadIndex.LoadFromDisk(ctx, filePath); err != nil {
			b.Fatalf("Failed to load index: %v", err)
		}
		duration := time.Since(start)

		if i == 0 {
			reportThroughput(b, filePath, duration, "Load 100K")
		}
	}
}

func BenchmarkLoad1M(b *testing.B) {
	index, filePath := createTestIndex(b, 1000000)
	ctx := context.Background()

	if err := index.SaveToDisk(ctx, filePath); err != nil {
		b.Fatalf("Failed to save index for load test: %v", err)
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
			b.Fatalf("Failed to create index for loading: %v", err)
		}

		start := time.Now()
		if err := loadIndex.LoadFromDisk(ctx, filePath); err != nil {
			b.Fatalf("Failed to load index: %v", err)
		}
		duration := time.Since(start)

		if i == 0 {
			reportThroughput(b, filePath, duration, "Load 1M")
		}
	}
}
