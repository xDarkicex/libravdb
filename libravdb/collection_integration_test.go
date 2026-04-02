package libravdb

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/memory"
	"github.com/xDarkicex/libravdb/internal/quant"
)

func TestEnhancedCollectionConfiguration(t *testing.T) {
	// Create a temporary directory for testing
	tmpDir := t.TempDir()

	ctx := context.Background()

	// Create database with enhanced configuration
	db, err := New(
		WithStoragePath(filepath.Join(tmpDir, "enhanced.libravdb")),
		WithMetrics(true),
	)
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Test creating a collection with all new configuration options
	collection, err := db.CreateCollection(ctx, "test_enhanced",
		WithDimension(128),
		WithMetric(CosineDistance),
		WithHNSW(16, 200, 50),

		// NEW: Quantization configuration
		WithProductQuantization(8, 8, 0.1),

		// NEW: Memory management
		WithMemoryLimit(100*1024*1024), // 100MB limit
		WithCachePolicy(LRUCache),
		WithMemoryMapping(true),
		WithMemoryConfig(&memory.MemoryConfig{
			MaxMemory:       100 * 1024 * 1024,
			MonitorInterval: 1 * time.Second,
			EnableGC:        true,
			GCThreshold:     0.8,
			EnableMMap:      true,
			MMapThreshold:   10 * 1024 * 1024, // 10MB
		}),

		// NEW: Metadata schema
		WithMetadataSchema(MetadataSchema{
			"category": StringField,
			"price":    FloatField,
			"tags":     StringArrayField,
			"active":   BoolField,
		}),
		WithIndexedFields("category", "price"),

		// NEW: Batch configuration
		WithBatchConfig(BatchConfig{
			ChunkSize:       500,
			MaxConcurrency:  2,
			FailFast:        false,
			TimeoutPerChunk: 10 * time.Second,
		}),
	)
	if err != nil {
		t.Fatalf("Failed to create enhanced collection: %v", err)
	}

	// Verify configuration was applied correctly
	stats := collection.Stats()
	if stats.Dimension != 128 {
		t.Errorf("Expected dimension 128, got %d", stats.Dimension)
	}

	// Test inserting vectors with metadata that matches the schema

	testVectors := []struct {
		id       string
		vector   []float32
		metadata map[string]interface{}
	}{
		{
			id:     "vec1",
			vector: make([]float32, 128),
			metadata: map[string]interface{}{
				"category": "electronics",
				"price":    99.99,
				"tags":     []string{"gadget", "tech"},
				"active":   true,
			},
		},
		{
			id:     "vec2",
			vector: make([]float32, 128),
			metadata: map[string]interface{}{
				"category": "books",
				"price":    19.99,
				"tags":     []string{"fiction", "novel"},
				"active":   true,
			},
		},
	}

	// Initialize vectors with some values
	for i := range testVectors {
		for j := range testVectors[i].vector {
			testVectors[i].vector[j] = float32(i*j) * 0.1
		}
	}

	// Insert test vectors
	for _, tv := range testVectors {
		err := collection.Insert(ctx, tv.id, tv.vector, tv.metadata)
		if err != nil {
			t.Errorf("Failed to insert vector %s: %v", tv.id, err)
		}
	}

	// Test search functionality
	queryVector := make([]float32, 128)
	for i := range queryVector {
		queryVector[i] = 0.1
	}

	results, err := collection.Search(ctx, queryVector, 2)
	if err != nil {
		t.Errorf("Failed to search: %v", err)
	}

	if len(results.Results) == 0 {
		t.Error("Expected search results, got none")
	}

	// Verify metadata is preserved
	for _, result := range results.Results {
		if result.Metadata == nil {
			t.Error("Expected metadata in search results")
		}
		if category, ok := result.Metadata["category"]; !ok {
			t.Error("Expected category in metadata")
		} else if category != "electronics" && category != "books" {
			t.Errorf("Unexpected category value: %v", category)
		}
	}

	t.Logf("Enhanced collection test completed successfully")
	t.Logf("Collection stats: %+v", stats)
	t.Logf("Search took: %v", results.Took)
}

func TestBackwardCompatibleConfiguration(t *testing.T) {
	// Test that old-style configuration still works
	tmpDir := t.TempDir()

	db, err := New(WithStoragePath(filepath.Join(tmpDir, "compat.libravdb")))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Create collection with only old configuration options
	collection, err := db.CreateCollection(context.Background(), "test_compat",
		WithDimension(64),
		WithMetric(L2Distance),
		WithHNSW(32, 200, 50),
		WithIndexPersistence(true),
	)
	if err != nil {
		t.Fatalf("Failed to create backward compatible collection: %v", err)
	}

	// Should work exactly as before
	ctx := context.Background()
	vector := make([]float32, 64)
	for i := range vector {
		vector[i] = float32(i) * 0.1
	}

	err = collection.Insert(ctx, "test", vector, map[string]interface{}{
		"old_style": "metadata",
	})
	if err != nil {
		t.Errorf("Failed to insert with backward compatible config: %v", err)
	}

	results, err := collection.Search(ctx, vector, 1)
	if err != nil {
		t.Errorf("Failed to search with backward compatible config: %v", err)
	}

	if len(results.Results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results.Results))
	}

	t.Log("Backward compatibility test passed")
}

func TestConfigurationValidationIntegration(t *testing.T) {
	tmpDir := t.TempDir()

	db, err := New(WithStoragePath(filepath.Join(tmpDir, "validation.libravdb")))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Test that invalid configurations are properly rejected
	_, err = db.CreateCollection(ctx, "invalid_config",
		WithDimension(-1), // Invalid dimension
	)
	if err == nil {
		t.Error("Expected error for invalid dimension, got none")
	}

	// Test that invalid quantization config is rejected
	_, err = db.CreateCollection(ctx, "invalid_quant",
		WithDimension(128),
		WithQuantization(&quant.QuantizationConfig{
			Type:       quant.ProductQuantization,
			Codebooks:  0, // Invalid
			Bits:       8,
			TrainRatio: 0.1,
		}),
	)
	if err == nil {
		t.Error("Expected error for invalid quantization config, got none")
	}

	// Test that invalid memory limit is rejected
	_, err = db.CreateCollection(ctx, "invalid_memory",
		WithDimension(128),
		WithMemoryLimit(-1), // Invalid
	)
	if err == nil {
		t.Error("Expected error for invalid memory limit, got none")
	}

	t.Log("Configuration validation integration test passed")
}
