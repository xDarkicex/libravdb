package ivfpq

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/index/hnsw"
	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

// BenchmarkResult holds performance benchmark results
type BenchmarkResult struct {
	IndexType        string
	DatasetSize      int
	Dimension        int
	InsertTimeMs     int64
	SearchTimeMs     int64
	MemoryUsageBytes int64
	Accuracy         float64
	Throughput       float64 // Operations per second
}

// TestIVFPQvsHNSWPerformance compares IVF-PQ and HNSW performance on various dataset sizes
func TestIVFPQvsHNSWPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	ctx := context.Background()
	dimension := 128

	// Test different dataset sizes
	testSizes := []int{1000, 5000, 10000, 25000}

	for _, size := range testSizes {
		t.Run(fmt.Sprintf("Dataset_%d", size), func(t *testing.T) {
			// Generate test dataset
			vectors := generateLargeTestDataset(size, dimension, 42)
			queries := generateLargeTestDataset(100, dimension, 123) // 100 test queries

			// Test IVF-PQ with different configurations
			ivfpqConfigs := []struct {
				name   string
				config *Config
			}{
				{
					name:   "IVF-PQ_Default",
					config: DefaultConfig(dimension),
				},
				{
					name:   "IVF-PQ_AutoTuned",
					config: AutoTuneConfig(dimension, size, 100), // 100MB memory target
				},
				{
					name: "IVF-PQ_HighAccuracy",
					config: &Config{
						Dimension: dimension,
						NClusters: int(math.Sqrt(float64(size))),
						NProbes:   int(math.Sqrt(float64(size)) / 2),
						Metric:    util.L2Distance,
						Quantization: &quant.QuantizationConfig{
							Type:       quant.ProductQuantization,
							Codebooks:  dimension / 4,
							Bits:       8,
							TrainRatio: 0.2,
							CacheSize:  5000,
						},
						MaxIterations: 100,
						Tolerance:     1e-4,
						RandomSeed:    42,
					},
				},
				{
					name: "IVF-PQ_HighSpeed",
					config: &Config{
						Dimension: dimension,
						NClusters: int(math.Sqrt(float64(size)) / 2),
						NProbes:   max(1, int(math.Sqrt(float64(size))/4)),
						Metric:    util.L2Distance,
						Quantization: &quant.QuantizationConfig{
							Type:       quant.ProductQuantization,
							Codebooks:  dimension / 8,
							Bits:       4,
							TrainRatio: 0.1,
							CacheSize:  1000,
						},
						MaxIterations: 50,
						Tolerance:     1e-3,
						RandomSeed:    42,
					},
				},
			}

			// Test HNSW configuration
			hnswConfig := &hnsw.Config{
				Dimension:      dimension,
				M:              16,
				EfConstruction: 200,
				EfSearch:       50,
				Metric:         util.L2Distance,
				RandomSeed:     42,
			}

			var results []BenchmarkResult

			// Benchmark IVF-PQ configurations
			for _, cfg := range ivfpqConfigs {
				result := benchmarkIVFPQ(t, ctx, cfg.config, vectors, queries)
				result.IndexType = cfg.name
				result.DatasetSize = size
				result.Dimension = dimension
				results = append(results, result)
			}

			// Benchmark HNSW
			hnswResult := benchmarkHNSW(t, ctx, hnswConfig, vectors, queries)
			hnswResult.IndexType = "HNSW"
			hnswResult.DatasetSize = size
			hnswResult.Dimension = dimension
			results = append(results, hnswResult)

			// Print comparison results
			t.Logf("\n=== Performance Comparison for %d vectors ===", size)
			t.Logf("%-20s %-12s %-12s %-15s %-10s %-12s",
				"Index Type", "Insert(ms)", "Search(ms)", "Memory(MB)", "Accuracy", "Throughput")
			t.Logf("%-20s %-12s %-12s %-15s %-10s %-12s",
				"----------", "----------", "----------", "-----------", "--------", "----------")

			for _, result := range results {
				memoryMB := float64(result.MemoryUsageBytes) / (1024 * 1024)
				t.Logf("%-20s %-12d %-12d %-15.2f %-10.3f %-12.1f",
					result.IndexType,
					result.InsertTimeMs,
					result.SearchTimeMs,
					memoryMB,
					result.Accuracy,
					result.Throughput)
			}

			// Analyze results
			analyzePerformanceResults(t, results)
		})
	}
}

// benchmarkIVFPQ benchmarks IVF-PQ index performance
func benchmarkIVFPQ(t *testing.T, ctx context.Context, config *Config, vectors [][]float32, queries [][]float32) BenchmarkResult {
	// Create and train index
	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("Failed to create IVF-PQ index: %v", err)
	}
	defer idx.Close()

	// Training phase
	trainStart := time.Now()
	err = idx.Train(ctx, vectors[:min(len(vectors), 1000)]) // Use subset for training
	if err != nil {
		t.Fatalf("Failed to train IVF-PQ index: %v", err)
	}
	trainTime := time.Since(trainStart)

	// Insert phase
	insertStart := time.Now()
	for i, vec := range vectors {
		entry := &VectorEntry{
			ID:     fmt.Sprintf("vec_%d", i),
			Vector: vec,
		}
		err := idx.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}
	}
	insertTime := time.Since(insertStart)

	// Search phase
	searchStart := time.Now()
	totalAccuracy := 0.0
	k := 10

	for _, query := range queries {
		results, err := idx.Search(ctx, query, k)
		if err != nil {
			t.Fatalf("Failed to search: %v", err)
		}

		// Calculate accuracy (simplified - based on result count)
		accuracy := float64(len(results)) / float64(k)
		totalAccuracy += accuracy
	}
	searchTime := time.Since(searchStart)

	avgAccuracy := totalAccuracy / float64(len(queries))
	throughput := float64(len(queries)) / searchTime.Seconds()

	return BenchmarkResult{
		InsertTimeMs:     (insertTime + trainTime).Milliseconds(),
		SearchTimeMs:     searchTime.Milliseconds(),
		MemoryUsageBytes: idx.MemoryUsage(),
		Accuracy:         avgAccuracy,
		Throughput:       throughput,
	}
}

// benchmarkHNSW benchmarks HNSW index performance
func benchmarkHNSW(t *testing.T, ctx context.Context, config *hnsw.Config, vectors [][]float32, queries [][]float32) BenchmarkResult {
	// Create HNSW index
	idx, err := hnsw.NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}
	defer idx.Close()

	// Insert phase
	insertStart := time.Now()
	for i, vec := range vectors {
		entry := &hnsw.VectorEntry{
			ID:     fmt.Sprintf("vec_%d", i),
			Vector: vec,
		}
		err := idx.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}
	}
	insertTime := time.Since(insertStart)

	// Search phase
	searchStart := time.Now()
	totalAccuracy := 0.0
	k := 10

	for _, query := range queries {
		results, err := idx.Search(ctx, query, k)
		if err != nil {
			t.Fatalf("Failed to search: %v", err)
		}

		// Calculate accuracy (simplified - based on result count)
		accuracy := float64(len(results)) / float64(k)
		totalAccuracy += accuracy
	}
	searchTime := time.Since(searchStart)

	avgAccuracy := totalAccuracy / float64(len(queries))
	throughput := float64(len(queries)) / searchTime.Seconds()

	return BenchmarkResult{
		InsertTimeMs:     insertTime.Milliseconds(),
		SearchTimeMs:     searchTime.Milliseconds(),
		MemoryUsageBytes: idx.MemoryUsage(),
		Accuracy:         avgAccuracy,
		Throughput:       throughput,
	}
}

// TestQuantizationEffectiveness tests the effectiveness of different quantization settings
func TestQuantizationEffectiveness(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping quantization effectiveness test in short mode")
	}

	ctx := context.Background()
	dimension := 64
	datasetSize := 5000

	vectors := generateLargeTestDataset(datasetSize, dimension, 42)
	queries := generateLargeTestDataset(50, dimension, 123)

	// Test different quantization configurations
	quantConfigs := []struct {
		name   string
		config *quant.QuantizationConfig
	}{
		{
			name:   "No_Quantization",
			config: nil,
		},
		{
			name: "PQ_4bit",
			config: &quant.QuantizationConfig{
				Type:       quant.ProductQuantization,
				Codebooks:  8,
				Bits:       4,
				TrainRatio: 0.1,
				CacheSize:  1000,
			},
		},
		{
			name: "PQ_6bit",
			config: &quant.QuantizationConfig{
				Type:       quant.ProductQuantization,
				Codebooks:  8,
				Bits:       6,
				TrainRatio: 0.1,
				CacheSize:  1000,
			},
		},
		{
			name: "PQ_8bit",
			config: &quant.QuantizationConfig{
				Type:       quant.ProductQuantization,
				Codebooks:  8,
				Bits:       8,
				TrainRatio: 0.1,
				CacheSize:  1000,
			},
		},
		{
			name: "Scalar_4bit",
			config: &quant.QuantizationConfig{
				Type:       quant.ScalarQuantization,
				Bits:       4,
				TrainRatio: 0.1,
			},
		},
		{
			name: "Scalar_8bit",
			config: &quant.QuantizationConfig{
				Type:       quant.ScalarQuantization,
				Bits:       8,
				TrainRatio: 0.1,
			},
		},
	}

	t.Logf("\n=== Quantization Effectiveness Comparison ===")
	t.Logf("%-15s %-12s %-15s %-10s %-12s %-15s",
		"Quantization", "Memory(MB)", "Compression", "Accuracy", "Search(ms)", "Ratio")
	t.Logf("%-15s %-12s %-15s %-10s %-12s %-15s",
		"-----------", "----------", "-----------", "--------", "----------", "-----")

	baselineMemory := int64(0)

	for i, qConfig := range quantConfigs {
		// Create IVF-PQ config
		config := &Config{
			Dimension:     dimension,
			NClusters:     64,
			NProbes:       8,
			Metric:        util.L2Distance,
			Quantization:  qConfig.config,
			MaxIterations: 50,
			Tolerance:     1e-4,
			RandomSeed:    42,
		}

		result := benchmarkIVFPQ(t, ctx, config, vectors, queries)

		if i == 0 {
			baselineMemory = result.MemoryUsageBytes
		}

		memoryMB := float64(result.MemoryUsageBytes) / (1024 * 1024)
		compressionRatio := float64(baselineMemory) / float64(result.MemoryUsageBytes)

		t.Logf("%-15s %-12.2f %-15.2fx %-10.3f %-12d %-15.2f",
			qConfig.name,
			memoryMB,
			compressionRatio,
			result.Accuracy,
			result.SearchTimeMs,
			result.Throughput)
	}
}

// TestAdaptiveSearchOptimization tests the adaptive search optimization
func TestAdaptiveSearchOptimization(t *testing.T) {
	ctx := context.Background()
	dimension := 32
	datasetSize := 1000

	vectors := generateLargeTestDataset(datasetSize, dimension, 42)
	queries := generateLargeTestDataset(200, dimension, 123)

	config := &Config{
		Dimension:     dimension,
		NClusters:     32,
		NProbes:       8,
		Metric:        util.L2Distance,
		Quantization:  quant.DefaultConfig(quant.ProductQuantization),
		MaxIterations: 50,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("Failed to create IVF-PQ index: %v", err)
	}
	defer idx.Close()

	// Train and populate index
	err = idx.Train(ctx, vectors[:200])
	if err != nil {
		t.Fatalf("Failed to train index: %v", err)
	}

	for i, vec := range vectors {
		entry := &VectorEntry{
			ID:     fmt.Sprintf("vec_%d", i),
			Vector: vec,
		}
		err := idx.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("Failed to insert vector: %v", err)
		}
	}

	// Test without adaptive search
	t.Log("Testing without adaptive search...")
	startTime := time.Now()
	for _, query := range queries[:100] {
		_, err := idx.Search(ctx, query, 5)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
	}
	nonAdaptiveTime := time.Since(startTime)

	// Enable adaptive search
	t.Log("Testing with adaptive search...")
	idx.EnableAdaptiveSearch()

	startTime = time.Now()
	for _, query := range queries[100:] {
		_, err := idx.Search(ctx, query, 5)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
	}
	adaptiveTime := time.Since(startTime)

	// Get final statistics
	stats := idx.GetSearchStats()

	t.Logf("Non-adaptive search time: %v", nonAdaptiveTime)
	t.Logf("Adaptive search time: %v", adaptiveTime)
	t.Logf("Final adaptive probe count: %d (original: %d)", stats.currentProbes, config.NProbes)
	t.Logf("Total adaptive searches: %d", stats.totalSearches)

	if stats.totalSearches > 0 {
		avgLatency := float64(stats.totalLatencyMs) / float64(stats.totalSearches)
		avgAccuracy := stats.accuracySum / float64(stats.totalSearches)
		t.Logf("Average latency: %.2f ms", avgLatency)
		t.Logf("Average accuracy: %.3f", avgAccuracy)
	}
}

// analyzePerformanceResults analyzes and reports on benchmark results
func analyzePerformanceResults(t *testing.T, results []BenchmarkResult) {
	if len(results) == 0 {
		return
	}

	t.Log("\n=== Performance Analysis ===")

	// Find best performers in each category
	var bestInsert, bestSearch, bestMemory, bestAccuracy, bestThroughput BenchmarkResult
	bestInsert = results[0]
	bestSearch = results[0]
	bestMemory = results[0]
	bestAccuracy = results[0]
	bestThroughput = results[0]

	for _, result := range results {
		if result.InsertTimeMs < bestInsert.InsertTimeMs {
			bestInsert = result
		}
		if result.SearchTimeMs < bestSearch.SearchTimeMs {
			bestSearch = result
		}
		if result.MemoryUsageBytes < bestMemory.MemoryUsageBytes {
			bestMemory = result
		}
		if result.Accuracy > bestAccuracy.Accuracy {
			bestAccuracy = result
		}
		if result.Throughput > bestThroughput.Throughput {
			bestThroughput = result
		}
	}

	t.Logf("Best Insert Performance: %s (%.2f ms)", bestInsert.IndexType, float64(bestInsert.InsertTimeMs))
	t.Logf("Best Search Performance: %s (%.2f ms)", bestSearch.IndexType, float64(bestSearch.SearchTimeMs))
	t.Logf("Best Memory Efficiency: %s (%.2f MB)", bestMemory.IndexType, float64(bestMemory.MemoryUsageBytes)/(1024*1024))
	t.Logf("Best Accuracy: %s (%.3f)", bestAccuracy.IndexType, bestAccuracy.Accuracy)
	t.Logf("Best Throughput: %s (%.1f ops/sec)", bestThroughput.IndexType, bestThroughput.Throughput)

	// Calculate relative performance
	t.Log("\n=== Relative Performance vs HNSW ===")
	var hnswResult *BenchmarkResult
	for i := range results {
		if results[i].IndexType == "HNSW" {
			hnswResult = &results[i]
			break
		}
	}

	if hnswResult != nil {
		for _, result := range results {
			if result.IndexType == "HNSW" {
				continue
			}

			insertRatio := float64(hnswResult.InsertTimeMs) / float64(result.InsertTimeMs)
			searchRatio := float64(hnswResult.SearchTimeMs) / float64(result.SearchTimeMs)
			memoryRatio := float64(hnswResult.MemoryUsageBytes) / float64(result.MemoryUsageBytes)
			accuracyRatio := result.Accuracy / hnswResult.Accuracy
			throughputRatio := result.Throughput / hnswResult.Throughput

			t.Logf("%s vs HNSW:", result.IndexType)
			t.Logf("  Insert: %.2fx %s", insertRatio, getPerformanceIndicator(insertRatio))
			t.Logf("  Search: %.2fx %s", searchRatio, getPerformanceIndicator(searchRatio))
			t.Logf("  Memory: %.2fx %s", memoryRatio, getMemoryIndicator(memoryRatio))
			t.Logf("  Accuracy: %.2fx %s", accuracyRatio, getPerformanceIndicator(accuracyRatio))
			t.Logf("  Throughput: %.2fx %s", throughputRatio, getPerformanceIndicator(throughputRatio))
		}
	}
}

// Helper functions
func getPerformanceIndicator(ratio float64) string {
	if ratio > 1.1 {
		return "faster"
	} else if ratio < 0.9 {
		return "slower"
	}
	return "similar"
}

func getMemoryIndicator(ratio float64) string {
	if ratio > 1.1 {
		return "more efficient"
	} else if ratio < 0.9 {
		return "less efficient"
	}
	return "similar"
}

// generateLargeTestDataset generates a large test dataset for benchmarking
func generateLargeTestDataset(count, dimension int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	vectors := make([][]float32, count)

	for i := 0; i < count; i++ {
		vector := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			vector[j] = float32(rng.NormFloat64()) // Use normal distribution for more realistic data
		}
		vectors[i] = vector
	}

	return vectors
}
