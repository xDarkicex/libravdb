package hnsw

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/memory"
	"github.com/xDarkicex/libravdb/internal/util"
)

func TestHNSW_MemoryMapping(t *testing.T) {
	// Create temporary directory for memory mapping
	tmpDir, err := os.MkdirTemp("", "hnsw_mmap_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create HNSW index
	config := &Config{
		Dimension:      4,
		M:              16,
		EfConstruction: 200,
		EfSearch:       50,
		ML:             1.0 / 2.0,
		Metric:         util.L2Distance,
		RandomSeed:     42,
	}

	index, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}

	// Add some test vectors
	testVectors := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 3.0, 4.0, 5.0},
		{3.0, 4.0, 5.0, 6.0},
		{4.0, 5.0, 6.0, 7.0},
		{5.0, 6.0, 7.0, 8.0},
	}

	ctx := context.Background()
	for i, vector := range testVectors {
		entry := &VectorEntry{
			ID:       string(rune('a' + i)),
			Vector:   vector,
			Metadata: map[string]interface{}{"index": i},
		}

		err := index.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}
	}

	// Test MemoryMappable interface implementation
	if !index.CanMemoryMap() {
		t.Error("Expected index to be memory mappable")
	}

	originalSize := index.EstimateSize()
	if originalSize <= 0 {
		t.Error("Expected positive estimated size")
	}

	originalMemUsage := index.MemoryUsage()
	if originalMemUsage <= 0 {
		t.Error("Expected positive memory usage")
	}

	// Test that index is not initially memory mapped
	if index.IsMemoryMapped() {
		t.Error("Expected index to not be memory mapped initially")
	}

	if index.MemoryMappedSize() != 0 {
		t.Error("Expected zero memory mapped size initially")
	}

	// Enable memory mapping
	err = index.EnableMemoryMapping(tmpDir)
	if err != nil {
		t.Fatalf("Failed to enable memory mapping: %v", err)
	}

	// Test that index is now memory mapped
	if !index.IsMemoryMapped() {
		t.Error("Expected index to be memory mapped after enabling")
	}

	mappedSize := index.MemoryMappedSize()
	if mappedSize <= 0 {
		t.Error("Expected positive memory mapped size")
	}

	// Memory usage should be significantly reduced
	newMemUsage := index.MemoryUsage()
	if newMemUsage >= originalMemUsage {
		t.Errorf("Expected reduced memory usage after mapping: original=%d, new=%d",
			originalMemUsage, newMemUsage)
	}

	// Note: In a full implementation, search would need to handle loading data from the mapped file
	// For now, we just test that the mapping state is correct

	// Disable memory mapping
	err = index.DisableMemoryMapping()
	if err != nil {
		t.Fatalf("Failed to disable memory mapping: %v", err)
	}

	// Test that index is no longer memory mapped
	if index.IsMemoryMapped() {
		t.Error("Expected index to not be memory mapped after disabling")
	}

	if index.MemoryMappedSize() != 0 {
		t.Error("Expected zero memory mapped size after disabling")
	}

	// Memory usage should be back to normal levels
	finalMemUsage := index.MemoryUsage()
	if finalMemUsage < originalMemUsage/2 {
		t.Errorf("Expected memory usage to return to normal levels: original=%d, final=%d",
			originalMemUsage, finalMemUsage)
	}

	// Test that search still works after disabling mapping
	query := []float32{1.5, 2.5, 3.5, 4.5}
	results, err := index.Search(ctx, query, 3)
	if err != nil {
		t.Fatalf("Failed to search after disabling memory mapping: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected search results after disabling memory mapping")
	}
}

func TestHNSW_MemoryMappingErrors(t *testing.T) {
	// Create HNSW index
	config := &Config{
		Dimension:      4,
		M:              16,
		EfConstruction: 200,
		EfSearch:       50,
		ML:             1.0 / 2.0,
		Metric:         util.L2Distance,
		RandomSeed:     42,
	}

	index, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}

	// Test enabling memory mapping on empty index
	err = index.EnableMemoryMapping("/tmp")
	if err == nil {
		t.Error("Expected error when enabling memory mapping on empty index")
	}

	// Add a vector
	ctx := context.Background()
	entry := &VectorEntry{
		ID:     "test",
		Vector: []float32{1.0, 2.0, 3.0, 4.0},
	}

	err = index.Insert(ctx, entry)
	if err != nil {
		t.Fatalf("Failed to insert vector: %v", err)
	}

	// Test enabling memory mapping with invalid path
	err = index.EnableMemoryMapping("/invalid/path/that/does/not/exist")
	if err == nil {
		t.Error("Expected error when enabling memory mapping with invalid path")
	}

	// Test double enabling
	tmpDir, err := os.MkdirTemp("", "hnsw_mmap_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	err = index.EnableMemoryMapping(tmpDir)
	if err != nil {
		t.Fatalf("Failed to enable memory mapping: %v", err)
	}

	err = index.EnableMemoryMapping(tmpDir)
	if err == nil {
		t.Error("Expected error when enabling memory mapping twice")
	}

	// Test disabling when not mapped
	index2, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create second HNSW index: %v", err)
	}

	err = index2.DisableMemoryMapping()
	if err == nil {
		t.Error("Expected error when disabling memory mapping on non-mapped index")
	}
}

func TestHNSW_MemoryMappingEmptyIndex(t *testing.T) {
	// Create HNSW index
	config := &Config{
		Dimension:      4,
		M:              16,
		EfConstruction: 200,
		EfSearch:       50,
		ML:             1.0 / 2.0,
		Metric:         util.L2Distance,
		RandomSeed:     42,
	}

	index, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}

	// Empty index should not be memory mappable
	if index.CanMemoryMap() {
		t.Error("Expected empty index to not be memory mappable")
	}

	if index.EstimateSize() != 0 {
		t.Error("Expected zero estimated size for empty index")
	}
}

func TestHNSW_MemoryMappingWithQuantization(t *testing.T) {
	// Create temporary directory for memory mapping
	tmpDir, err := os.MkdirTemp("", "hnsw_mmap_quant_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create HNSW index with quantization
	config := &Config{
		Dimension:      4,
		M:              16,
		EfConstruction: 200,
		EfSearch:       50,
		ML:             1.0 / 2.0,
		Metric:         util.L2Distance,
		RandomSeed:     42,
		// Note: Quantization config would be added here in a real implementation
	}

	index, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}

	// Add test vectors
	ctx := context.Background()
	for i := range 10 {
		entry := &VectorEntry{
			ID:     string(rune('a' + i)),
			Vector: []float32{float32(i), float32(i + 1), float32(i + 2), float32(i + 3)},
		}

		err := index.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}
	}

	// Test memory mapping with quantized index
	originalMemUsage := index.MemoryUsage()

	err = index.EnableMemoryMapping(tmpDir)
	if err != nil {
		t.Fatalf("Failed to enable memory mapping: %v", err)
	}

	// Memory usage should be reduced even with quantization
	newMemUsage := index.MemoryUsage()
	if newMemUsage >= originalMemUsage {
		t.Errorf("Expected reduced memory usage with quantized mapping: original=%d, new=%d",
			originalMemUsage, newMemUsage)
	}

	// Note: In a full implementation, search would work with memory-mapped data
	// For now, we just verify the memory mapping state is correct
	if !index.IsMemoryMapped() {
		t.Error("Expected index to be memory mapped")
	}
}

func TestHNSW_AutomaticMemoryMappingIntegration(t *testing.T) {
	// Create temporary directory for memory mapping
	tmpDir, err := os.MkdirTemp("", "hnsw_auto_mmap_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create HNSW index
	config := &Config{
		Dimension:      128, // Larger dimension for more memory usage
		M:              16,
		EfConstruction: 200,
		EfSearch:       50,
		ML:             1.0 / 2.0,
		Metric:         util.L2Distance,
		RandomSeed:     42,
	}

	index, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}

	// Add many vectors to make the index large enough for automatic mapping
	ctx := context.Background()
	vectorCount := 1000 // Large enough to trigger automatic mapping

	for i := range vectorCount {
		vector := make([]float32, config.Dimension)
		for j := range config.Dimension {
			vector[j] = float32(i*config.Dimension + j)
		}

		entry := &VectorEntry{
			ID:       fmt.Sprintf("vec_%d", i),
			Vector:   vector,
			Metadata: map[string]any{"index": i},
		}

		err := index.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}
	}

	// Create memory manager with automatic mapping enabled
	memConfig := memory.MemoryConfig{
		MaxMemory:       50 * 1024 * 1024, // 50MB limit
		EnableMMap:      true,
		MMapThreshold:   1024 * 1024, // 1MB threshold
		MMapPath:        tmpDir,
		MonitorInterval: 100 * time.Millisecond,
	}

	memManager := memory.NewManager(memConfig)

	// Register the index with the memory manager
	err = memManager.RegisterMemoryMappable("hnsw_index", index)
	if err != nil {
		t.Fatalf("Failed to register index with memory manager: %v", err)
	}

	// Start memory monitoring
	monitorCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err = memManager.Start(monitorCtx)
	if err != nil {
		t.Fatalf("Failed to start memory manager: %v", err)
	}
	defer memManager.Stop()

	// Wait for automatic memory mapping to trigger
	time.Sleep(300 * time.Millisecond)

	// Check if the index was automatically memory mapped
	if index.EstimateSize() >= memConfig.MMapThreshold && !index.IsMemoryMapped() {
		t.Error("Expected large index to be automatically memory mapped")
	}

	// Test search functionality still works
	query := make([]float32, config.Dimension)
	for i := range config.Dimension {
		query[i] = float32(i)
	}

	results, err := index.Search(ctx, query, 5)
	if err != nil {
		t.Fatalf("Failed to search after automatic memory mapping: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected search results after automatic memory mapping")
	}
}

func TestHNSW_MemoryPressureResponse(t *testing.T) {
	// Create temporary directory for memory mapping
	tmpDir, err := os.MkdirTemp("", "hnsw_pressure_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create multiple HNSW indices
	indices := make([]*Index, 3)
	for i := range 3 {
		config := &Config{
			Dimension:      64,
			M:              16,
			EfConstruction: 200,
			EfSearch:       50,
			ML:             1.0 / 2.0,
			Metric:         util.L2Distance,
			RandomSeed:     int64(42 + i),
		}

		index, err := NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create HNSW index %d: %v", i, err)
		}
		indices[i] = index

		// Add vectors to each index
		ctx := context.Background()
		for j := range 100 {
			vector := make([]float32, config.Dimension)
			for k := range config.Dimension {
				vector[k] = float32(i*1000 + j*config.Dimension + k)
			}

			entry := &VectorEntry{
				ID:     fmt.Sprintf("idx%d_vec%d", i, j),
				Vector: vector,
			}

			err := index.Insert(ctx, entry)
			if err != nil {
				t.Fatalf("Failed to insert vector %d to index %d: %v", j, i, err)
			}
		}
	}

	// Create memory manager with low memory limit to trigger pressure
	memConfig := memory.MemoryConfig{
		MaxMemory:   1 * 1024 * 1024, // 1MB - very low limit to force pressure
		EnableMMap:  true,
		MMapPath:    tmpDir,
		EnableGC:    true,
		GCThreshold: 0.7,
		PressureThresholds: map[memory.MemoryPressureLevel]float64{
			memory.LowPressure:      0.6,
			memory.ModeratePressure: 0.7,
			memory.HighPressure:     0.8,
			memory.CriticalPressure: 0.9,
		},
	}

	memManager := memory.NewManager(memConfig)

	// Register all indices
	for i, index := range indices {
		err = memManager.RegisterMemoryMappable(fmt.Sprintf("index_%d", i), index)
		if err != nil {
			t.Fatalf("Failed to register index %d: %v", i, err)
		}
	}

	// Track memory pressure events
	pressureCount := 0
	memManager.OnMemoryPressure(func(usage memory.MemoryUsage) {
		pressureCount++
		t.Logf("Memory pressure detected: %d/%d bytes used", usage.Total, usage.Limit)
	})

	// Check initial memory usage and index sizes
	initialUsage := memManager.GetUsage()
	t.Logf("Initial memory usage: Total=%d, Limit=%d", initialUsage.Total, initialUsage.Limit)

	for i, index := range indices {
		size := index.EstimateSize()
		canMap := index.CanMemoryMap()
		t.Logf("Index %d: size=%d, canMap=%v", i, size, canMap)
	}

	// Trigger memory pressure handling
	err = memManager.HandleMemoryLimitExceeded()
	if err != nil {
		t.Logf("Memory limit handling returned error (may be expected): %v", err)
	}

	// Check final memory usage
	finalUsage := memManager.GetUsage()
	t.Logf("Final memory usage: Total=%d, Limit=%d", finalUsage.Total, finalUsage.Limit)

	// Check that at least some indices were memory mapped in response to pressure
	mappedCount := 0
	for _, index := range indices {
		if index.IsMemoryMapped() {
			mappedCount++
		}
	}

	if mappedCount == 0 {
		t.Error("Expected at least one index to be memory mapped under memory pressure")
	}

	t.Logf("Memory mapped %d out of %d indices under pressure", mappedCount, len(indices))
}
