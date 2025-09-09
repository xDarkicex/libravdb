package hnsw

import (
	"context"
	"os"
	"testing"

	"github.com/xDarkicex/libravdb/internal/util"
)

func TestHNSW_MemoryMappingSimple(t *testing.T) {
	// Create temporary directory for memory mapping
	tmpDir, err := os.MkdirTemp("", "hnsw_mmap_simple_test")
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

	// Add one test vector
	ctx := context.Background()
	entry := &VectorEntry{
		ID:     "test",
		Vector: []float32{1.0, 2.0, 3.0, 4.0},
	}

	err = index.Insert(ctx, entry)
	if err != nil {
		t.Fatalf("Failed to insert vector: %v", err)
	}

	// Test basic MemoryMappable interface
	if !index.CanMemoryMap() {
		t.Error("Expected index to be memory mappable")
	}

	if index.EstimateSize() <= 0 {
		t.Error("Expected positive estimated size")
	}

	if index.IsMemoryMapped() {
		t.Error("Expected index to not be memory mapped initially")
	}

	// Test enabling memory mapping
	t.Logf("Attempting to enable memory mapping...")
	err = index.EnableMemoryMapping(tmpDir)
	if err != nil {
		t.Fatalf("Failed to enable memory mapping: %v", err)
	}

	// Test that it's now mapped
	if !index.IsMemoryMapped() {
		t.Error("Expected index to be memory mapped after enabling")
	}

	t.Logf("Memory mapping test completed successfully")
}
