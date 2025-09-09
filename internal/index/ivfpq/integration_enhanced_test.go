package ivfpq

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

// TestEnhancedIVFPQIntegration tests all the enhanced features working together
func TestEnhancedIVFPQIntegration(t *testing.T) {
	ctx := context.Background()
	dimension := 64
	datasetSize := 2000

	// Generate test dataset
	vectors := generateLargeTestDataset(datasetSize, dimension, 42)
	queries := generateLargeTestDataset(50, dimension, 123)

	// Test auto-tuned configuration
	config := AutoTuneConfig(dimension, datasetSize, 100) // 100MB memory target

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("Failed to create IVF-PQ index: %v", err)
	}
	defer idx.Close()

	t.Logf("Auto-tuned config: %d clusters, %d probes, %d codebooks, %d bits",
		config.NClusters, config.NProbes, config.Quantization.Codebooks, config.Quantization.Bits)

	// Enable adaptive search
	idx.EnableAdaptiveSearch()

	// Train the index
	trainStart := time.Now()
	err = idx.Train(ctx, vectors[:500]) // Use subset for training
	if err != nil {
		t.Fatalf("Failed to train index: %v", err)
	}
	trainTime := time.Since(trainStart)
	t.Logf("Training completed in %v", trainTime)

	// Verify quantizer is trained
	if idx.quantizer == nil || !idx.quantizer.IsTrained() {
		t.Errorf("Quantizer should be trained after index training")
	}

	// Insert vectors
	insertStart := time.Now()
	for i, vec := range vectors {
		entry := &VectorEntry{
			ID:     fmt.Sprintf("vec_%d", i),
			Vector: vec,
			Metadata: map[string]interface{}{
				"category": i % 10,
				"batch":    i / 100,
			},
		}
		err := idx.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}
	}
	insertTime := time.Since(insertStart)
	t.Logf("Inserted %d vectors in %v", datasetSize, insertTime)

	if idx.Size() != datasetSize {
		t.Errorf("Expected size %d, got %d", datasetSize, idx.Size())
	}

	// Test search performance and accuracy
	searchStart := time.Now()
	totalResults := 0
	k := 10

	for i, query := range queries {
		results, err := idx.Search(ctx, query, k)
		if err != nil {
			t.Fatalf("Failed to search query %d: %v", i, err)
		}
		totalResults += len(results)

		// Verify results are reasonable
		if len(results) == 0 {
			t.Errorf("Query %d returned no results", i)
		}

		// Verify results are sorted by score
		for j := 1; j < len(results); j++ {
			if results[j-1].Score > results[j].Score {
				t.Errorf("Results not sorted by score for query %d", i)
			}
		}
	}
	searchTime := time.Since(searchStart)

	avgSearchTime := searchTime / time.Duration(len(queries))
	t.Logf("Completed %d searches in %v (avg: %v per search)", len(queries), searchTime, avgSearchTime)
	t.Logf("Total results found: %d", totalResults)

	// Test adaptive search statistics
	stats := idx.GetSearchStats()
	if stats.totalSearches == 0 {
		t.Errorf("Expected search statistics to be recorded")
	}

	t.Logf("Adaptive search stats: %d searches, avg latency: %.2f ms, avg accuracy: %.3f",
		stats.totalSearches,
		float64(stats.totalLatencyMs)/float64(stats.totalSearches),
		stats.accuracySum/float64(stats.totalSearches))

	// Test memory usage
	memUsage := idx.MemoryUsage()
	memUsageMB := float64(memUsage) / (1024 * 1024)
	t.Logf("Memory usage: %.2f MB", memUsageMB)

	// Verify compression ratio if quantization is enabled
	if idx.quantizer != nil {
		compressionRatio := idx.quantizer.CompressionRatio()
		t.Logf("Compression ratio: %.2fx", compressionRatio)

		if compressionRatio < 2.0 {
			t.Errorf("Expected compression ratio >= 2.0, got %.2f", compressionRatio)
		}
	}

	// Test cluster distribution
	clusterInfo := idx.GetClusterInfo()
	nonEmptyClusters := 0
	totalVectors := 0

	for i, info := range clusterInfo {
		if info.Size > 0 {
			nonEmptyClusters++
			totalVectors += info.Size
		}
		t.Logf("Cluster %d: %d vectors", i, info.Size)
	}

	if totalVectors != datasetSize {
		t.Errorf("Total vectors in clusters (%d) doesn't match dataset size (%d)", totalVectors, datasetSize)
	}

	t.Logf("Cluster distribution: %d/%d clusters non-empty", nonEmptyClusters, len(clusterInfo))

	// Test delete functionality
	deleteCount := 100
	for i := 0; i < deleteCount; i++ {
		err := idx.Delete(ctx, fmt.Sprintf("vec_%d", i))
		if err != nil {
			t.Errorf("Failed to delete vector %d: %v", i, err)
		}
	}

	if idx.Size() != datasetSize-deleteCount {
		t.Errorf("Expected size %d after deletions, got %d", datasetSize-deleteCount, idx.Size())
	}

	// Verify deleted vectors are not found in search
	deletedQuery := vectors[0] // This vector was deleted
	results, err := idx.Search(ctx, deletedQuery, k)
	if err != nil {
		t.Fatalf("Failed to search for deleted vector: %v", err)
	}

	// Check that the deleted vector is not in the results
	for _, result := range results {
		if result.ID == "vec_0" {
			t.Errorf("Deleted vector found in search results")
		}
	}

	t.Log("Enhanced IVF-PQ integration test completed successfully")
}

// TestAutoTuningEffectiveness compares auto-tuned vs default configurations
func TestAutoTuningEffectiveness(t *testing.T) {
	ctx := context.Background()
	dimension := 32
	datasetSize := 1000

	vectors := generateLargeTestDataset(datasetSize, dimension, 42)
	queries := generateLargeTestDataset(20, dimension, 123)

	// Test default configuration
	defaultConfig := DefaultConfig(dimension)
	defaultIdx, err := NewIVFPQ(defaultConfig)
	if err != nil {
		t.Fatalf("Failed to create default index: %v", err)
	}
	defer defaultIdx.Close()

	// Test auto-tuned configuration
	autoConfig := AutoTuneConfig(dimension, datasetSize, 50) // 50MB target
	autoIdx, err := NewIVFPQ(autoConfig)
	if err != nil {
		t.Fatalf("Failed to create auto-tuned index: %v", err)
	}
	defer autoIdx.Close()

	// Train both indices
	err = defaultIdx.Train(ctx, vectors[:200])
	if err != nil {
		t.Fatalf("Failed to train default index: %v", err)
	}

	err = autoIdx.Train(ctx, vectors[:200])
	if err != nil {
		t.Fatalf("Failed to train auto-tuned index: %v", err)
	}

	// Insert data into both indices
	for i, vec := range vectors {
		entry := &VectorEntry{
			ID:     fmt.Sprintf("vec_%d", i),
			Vector: vec,
		}

		err := defaultIdx.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("Failed to insert into default index: %v", err)
		}

		err = autoIdx.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("Failed to insert into auto-tuned index: %v", err)
		}
	}

	// Compare performance
	k := 5

	// Test default configuration
	defaultStart := time.Now()
	defaultResults := 0
	for _, query := range queries {
		results, err := defaultIdx.Search(ctx, query, k)
		if err != nil {
			t.Fatalf("Default search failed: %v", err)
		}
		defaultResults += len(results)
	}
	defaultTime := time.Since(defaultStart)

	// Test auto-tuned configuration
	autoStart := time.Now()
	autoResults := 0
	for _, query := range queries {
		results, err := autoIdx.Search(ctx, query, k)
		if err != nil {
			t.Fatalf("Auto-tuned search failed: %v", err)
		}
		autoResults += len(results)
	}
	autoTime := time.Since(autoStart)

	// Compare results
	t.Logf("Default config: %d clusters, %d probes", defaultConfig.NClusters, defaultConfig.NProbes)
	t.Logf("Auto-tuned config: %d clusters, %d probes", autoConfig.NClusters, autoConfig.NProbes)
	t.Logf("Default search time: %v, results: %d", defaultTime, defaultResults)
	t.Logf("Auto-tuned search time: %v, results: %d", autoTime, autoResults)
	t.Logf("Default memory: %.2f MB", float64(defaultIdx.MemoryUsage())/(1024*1024))
	t.Logf("Auto-tuned memory: %.2f MB", float64(autoIdx.MemoryUsage())/(1024*1024))

	// Both should find similar numbers of results
	if absInt(defaultResults-autoResults) > len(queries) {
		t.Errorf("Result counts differ significantly: default=%d, auto=%d", defaultResults, autoResults)
	}
}

// TestQuantizationIntegration tests the integration between coarse and fine quantization
func TestQuantizationIntegration(t *testing.T) {
	ctx := context.Background()
	dimension := 16 // Small dimension for easier testing
	datasetSize := 500

	vectors := generateLargeTestDataset(datasetSize, dimension, 42)

	// Create configuration with specific quantization settings
	config := &Config{
		Dimension: dimension,
		NClusters: 8,
		NProbes:   4,
		Metric:    util.L2Distance,
		Quantization: &quant.QuantizationConfig{
			Type:       quant.ProductQuantization,
			Codebooks:  4, // 4 subspaces
			Bits:       4, // 16 centroids per subspace
			TrainRatio: 0.2,
			CacheSize:  100,
		},
		MaxIterations: 50,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer idx.Close()

	// Train the index
	err = idx.Train(ctx, vectors)
	if err != nil {
		t.Fatalf("Failed to train index: %v", err)
	}

	// Verify both coarse and fine quantizers are trained
	if !idx.IsTrained() {
		t.Errorf("Index should be trained")
	}

	if idx.quantizer == nil || !idx.quantizer.IsTrained() {
		t.Errorf("Fine quantizer should be trained")
	}

	// Test compression and decompression
	testVector := vectors[0]
	compressed, err := idx.quantizer.Compress(testVector)
	if err != nil {
		t.Fatalf("Failed to compress vector: %v", err)
	}

	decompressed, err := idx.quantizer.Decompress(compressed)
	if err != nil {
		t.Fatalf("Failed to decompress vector: %v", err)
	}

	if len(decompressed) != dimension {
		t.Errorf("Decompressed vector has wrong dimension: %d vs %d", len(decompressed), dimension)
	}

	// Calculate reconstruction error
	mse := float32(0)
	for i := 0; i < dimension; i++ {
		diff := testVector[i] - decompressed[i]
		mse += diff * diff
	}
	mse /= float32(dimension)

	t.Logf("Compression: %d bytes -> %d bytes", dimension*4, len(compressed))
	t.Logf("Reconstruction MSE: %.6f", mse)

	// MSE should be reasonable for 4-bit quantization
	if mse > 1.0 {
		t.Errorf("Reconstruction error too high: %.6f", mse)
	}

	// Insert vectors and test search with quantization
	for i, vec := range vectors[:100] { // Insert subset for faster test
		entry := &VectorEntry{
			ID:     fmt.Sprintf("vec_%d", i),
			Vector: vec,
		}
		err := idx.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("Failed to insert vector: %v", err)
		}
	}

	// Test search accuracy with quantization
	query := vectors[0]
	results, err := idx.Search(ctx, query, 5)
	if err != nil {
		t.Fatalf("Failed to search: %v", err)
	}

	if len(results) == 0 {
		t.Errorf("No search results found")
	}

	// The exact vector should be found (ID "vec_0")
	found := false
	for _, result := range results {
		if result.ID == "vec_0" {
			found = true
			t.Logf("Exact match found with score: %.6f", result.Score)
			break
		}
	}

	if !found {
		t.Errorf("Exact match not found in search results")
	}

	t.Log("Quantization integration test completed successfully")
}

// Helper functions
func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
