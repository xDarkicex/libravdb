package benchmark

import (
	"context"
	"fmt"
	"hash/crc32"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/index/hnsw"
	"github.com/xDarkicex/libravdb/internal/util"
)

// BenchmarkMemoryUsageDuringSave measures memory overhead during save operations
func BenchmarkMemoryUsageDuringSave(b *testing.B) {
	index, filePath := createTestIndex(b, 100000)
	ctx := context.Background()

	// Force garbage collection before measurement
	runtime.GC()
	runtime.GC()

	var memBefore, memAfter runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := index.SaveToDisk(ctx, filePath); err != nil {
			b.Fatalf("Failed to save index: %v", err)
		}
	}

	runtime.ReadMemStats(&memAfter)

	memUsedMB := float64(memAfter.Alloc-memBefore.Alloc) / (1024 * 1024)
	b.Logf("Memory overhead during save: %.2f MB", memUsedMB)
	b.ReportMetric(memUsedMB, "MB_overhead")
}

// BenchmarkMemoryUsageDuringLoad measures memory overhead during load operations
func BenchmarkMemoryUsageDuringLoad(b *testing.B) {
	index, filePath := createTestIndex(b, 100000)
	ctx := context.Background()

	// Save the index first
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

	runtime.GC()
	runtime.GC()

	var memBefore, memAfter runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loadIndex, err := hnsw.NewHNSW(config)
		if err != nil {
			b.Fatalf("Failed to create index: %v", err)
		}

		if err := loadIndex.LoadFromDisk(ctx, filePath); err != nil {
			b.Fatalf("Failed to load index: %v", err)
		}
	}

	runtime.ReadMemStats(&memAfter)

	memUsedMB := float64(memAfter.Alloc-memBefore.Alloc) / (1024 * 1024)
	b.Logf("Memory overhead during load: %.2f MB", memUsedMB)
	b.ReportMetric(memUsedMB, "MB_overhead")
}

// BenchmarkCRC32Validation tests corruption detection performance
func BenchmarkCRC32Validation(b *testing.B) {
	index, filePath := createTestIndex(b, 50000)
	ctx := context.Background()

	// Save the index
	if err := index.SaveToDisk(ctx, filePath); err != nil {
		b.Fatalf("Failed to save index: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Read file and calculate CRC32
		file, err := os.Open(filePath)
		if err != nil {
			b.Fatalf("Failed to open file: %v", err)
		}

		hash := crc32.NewIEEE()
		if _, err := io.Copy(hash, file); err != nil {
			file.Close()
			b.Fatalf("Failed to calculate CRC32: %v", err)
		}
		file.Close()

		crc := hash.Sum32()
		if crc == 0 {
			b.Fatal("Invalid CRC32 calculated")
		}
	}
}

// BenchmarkAtomicWriteReliability tests atomic write operations
func BenchmarkAtomicWriteReliability(b *testing.B) {
	index, _ := createTestIndex(b, 10000)
	ctx := context.Background()
	tempDir := b.TempDir()

	successCount := 0
	var mu sync.Mutex

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		threadID := 0
		for pb.Next() {
			filePath := filepath.Join(tempDir, fmt.Sprintf("test_%d_%d.bin", threadID, time.Now().UnixNano()))

			if err := index.SaveToDisk(ctx, filePath); err != nil {
				b.Errorf("Atomic write failed: %v", err)
				continue
			}

			// Verify file exists and is valid
			if _, err := os.Stat(filePath); err != nil {
				b.Errorf("File not created atomically: %v", err)
				continue
			}

			mu.Lock()
			successCount++
			mu.Unlock()
			threadID++
		}
	})

	successRate := float64(successCount) / float64(b.N) * 100
	b.Logf("Atomic write success rate: %.2f%%", successRate)
	b.ReportMetric(successRate, "success_rate_%")
}

// BenchmarkSearchDuringSave measures search performance impact during save operations
func BenchmarkSearchDuringSave(b *testing.B) {
	index, filePath := createTestIndex(b, 50000)
	ctx := context.Background()

	// Create query vector
	queryVector := make([]float32, Dimension)
	rng := rand.New(rand.NewSource(42))
	for i := range queryVector {
		queryVector[i] = rng.Float32()*2 - 1
	}

	// Measure baseline search performance
	var baselineLatency time.Duration
	for i := 0; i < 10; i++ {
		start := time.Now()
		_, err := index.Search(ctx, queryVector, 10)
		if err != nil {
			b.Fatalf("Baseline search failed: %v", err)
		}
		baselineLatency += time.Since(start)
	}
	baselineLatency /= 10

	// Measure search performance during save operations
	var searchLatencies []time.Duration
	var wg sync.WaitGroup
	var mu sync.Mutex

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Start save operation in background
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := index.SaveToDisk(ctx, filePath); err != nil {
				b.Errorf("Save failed: %v", err)
			}
		}()

		// Perform searches during save
		for j := 0; j < 5; j++ {
			start := time.Now()
			_, err := index.Search(ctx, queryVector, 10)
			if err != nil {
				b.Errorf("Search during save failed: %v", err)
				continue
			}
			latency := time.Since(start)

			mu.Lock()
			searchLatencies = append(searchLatencies, latency)
			mu.Unlock()
		}

		wg.Wait()
	}

	// Calculate average latency during save
	var totalLatency time.Duration
	for _, lat := range searchLatencies {
		totalLatency += lat
	}
	avgLatencyDuringSave := totalLatency / time.Duration(len(searchLatencies))

	// Calculate impact percentage
	impactPercent := float64(avgLatencyDuringSave-baselineLatency) / float64(baselineLatency) * 100

	b.Logf("Baseline search latency: %v", baselineLatency)
	b.Logf("Search latency during save: %v", avgLatencyDuringSave)
	b.Logf("Performance impact: %.2f%%", impactPercent)

	b.ReportMetric(float64(baselineLatency.Microseconds()), "baseline_μs")
	b.ReportMetric(float64(avgLatencyDuringSave.Microseconds()), "during_save_μs")
	b.ReportMetric(impactPercent, "impact_%")
}

// BenchmarkRecoveryFromCorruption tests recovery capabilities
func BenchmarkRecoveryFromCorruption(b *testing.B) {
	index, filePath := createTestIndex(b, 10000)
	ctx := context.Background()

	// Save valid index
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

	recoverySuccessCount := 0

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Create corrupted copy
		corruptedPath := filePath + ".corrupted"

		// Copy file
		src, err := os.Open(filePath)
		if err != nil {
			b.Fatalf("Failed to open source: %v", err)
		}

		dst, err := os.Create(corruptedPath)
		if err != nil {
			src.Close()
			b.Fatalf("Failed to create corrupted copy: %v", err)
		}

		if _, err := io.Copy(dst, src); err != nil {
			src.Close()
			dst.Close()
			b.Fatalf("Failed to copy file: %v", err)
		}
		src.Close()
		dst.Close()

		// Corrupt the file (modify a few bytes in the middle)
		file, err := os.OpenFile(corruptedPath, os.O_WRONLY, 0644)
		if err != nil {
			b.Fatalf("Failed to open for corruption: %v", err)
		}

		file.Seek(1024, 0)                         // Seek to middle of file
		file.Write([]byte{0xFF, 0xFF, 0xFF, 0xFF}) // Write corruption
		file.Close()

		// Try to load corrupted file
		loadIndex, err := hnsw.NewHNSW(config)
		if err != nil {
			b.Fatalf("Failed to create index: %v", err)
		}

		err = loadIndex.LoadFromDisk(ctx, corruptedPath)
		if err != nil {
			// Expected - corruption should be detected
			recoverySuccessCount++
		} else {
			b.Logf("Warning: Corruption not detected in iteration %d", i)
		}

		os.Remove(corruptedPath)
	}

	detectionRate := float64(recoverySuccessCount) / float64(b.N) * 100
	b.Logf("Corruption detection rate: %.2f%%", detectionRate)
	b.ReportMetric(detectionRate, "detection_rate_%")
}
