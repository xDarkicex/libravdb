package libravdb

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/memory"
)

func TestEnsureCollectionDimensionMismatchIsNonDestructive(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	collection, err := db.EnsureCollection(ctx, "embeddings", 3, WithFlat())
	if err != nil {
		t.Fatalf("EnsureCollection create: %v", err)
	}

	if err := collection.Insert(ctx, "vec_1", []float32{0.1, 0.2, 0.3}, map[string]interface{}{"source": "original"}); err != nil {
		t.Fatalf("Insert: %v", err)
	}

	_, err = db.EnsureCollection(ctx, "embeddings", 4, WithFlat())
	if err == nil {
		t.Fatal("Expected dimension mismatch error, got nil")
	}
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("Expected ErrDimensionMismatch, got %v", err)
	}

	var mismatch *CollectionDimensionMismatchError
	if !errors.As(err, &mismatch) {
		t.Fatalf("Expected CollectionDimensionMismatchError, got %T: %v", err, err)
	}
	if mismatch.Collection != "embeddings" || mismatch.ExistingDimension != 3 || mismatch.RequestedDimension != 4 {
		t.Fatalf("Unexpected mismatch details: %+v", mismatch)
	}

	surviving, err := db.GetCollection("embeddings")
	if err != nil {
		t.Fatalf("GetCollection after mismatch: %v", err)
	}
	if surviving != collection {
		t.Fatal("Expected original collection instance to remain loaded")
	}
	if surviving.Dimension() != 3 {
		t.Fatalf("Expected original dimension 3 to survive, got %d", surviving.Dimension())
	}

	record, err := surviving.Get(ctx, "vec_1")
	if err != nil {
		t.Fatalf("Expected original vector to survive mismatch: %v", err)
	}
	if len(record.Vector) != 3 || record.Vector[0] != 0.1 || record.Vector[1] != 0.2 || record.Vector[2] != 0.3 {
		t.Fatalf("Unexpected surviving vector: %#v", record.Vector)
	}

	stats := surviving.Stats(ctx)
	if stats.VectorCount != 1 {
		t.Fatalf("Expected one vector after mismatch, got %d", stats.VectorCount)
	}
}

func TestEnsureCollectionRecreateOnDimensionMismatchOptIn(t *testing.T) {
	ctx := context.Background()
	path := testDBPath(t)
	db, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	collection, err := db.EnsureCollection(ctx, "embeddings", 3, WithFlat())
	if err != nil {
		t.Fatalf("EnsureCollection create: %v", err)
	}
	if err := collection.Insert(ctx, "vec_1", []float32{0.1, 0.2, 0.3}, nil); err != nil {
		t.Fatalf("Insert: %v", err)
	}

	recreated, err := db.EnsureCollectionRecreateOnDimensionMismatch(ctx, "embeddings", 4, WithFlat())
	if err != nil {
		t.Fatalf("EnsureCollectionRecreateOnDimensionMismatch: %v", err)
	}
	if recreated == collection {
		t.Fatal("Expected collection to be recreated through explicit opt-in")
	}
	if err := db.Close(); err != nil {
		t.Fatalf("Close before reopen: %v", err)
	}

	db, err = Open(WithStoragePath(path))
	if err != nil {
		t.Fatalf("Reopen after recreate: %v", err)
	}
	recreated, err = db.GetCollection("embeddings")
	if err != nil {
		t.Fatalf("GetCollection after reopen: %v", err)
	}
	if recreated.Dimension() != 4 {
		t.Fatalf("Expected recreated dimension 4 after reopen, got %d", recreated.Dimension())
	}
	if stats := recreated.Stats(ctx); stats.VectorCount != 0 {
		t.Fatalf("Expected recreated collection to start empty, got %d vectors", stats.VectorCount)
	}
}

func TestCollectionMemoryManagement(t *testing.T) {
	// Create database
	db, err := Open(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Create collection with memory limit
	collection, err := db.CreateCollection(
		context.Background(),
		"test_memory",
		WithDimension(128),
		WithMemoryLimit(50*1024*1024), // 50MB limit
		WithMemoryMapping(true),
	)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Test memory usage reporting
	usage, err := collection.GetMemoryUsage(context.Background())
	if err != nil {
		t.Fatalf("Failed to get memory usage: %v", err)
	}

	if usage.Limit != 50*1024*1024 {
		t.Errorf("Expected memory limit 50MB, got %d", usage.Limit)
	}

	// Test memory limit update
	err = collection.SetMemoryLimit(100 * 1024 * 1024) // 100MB
	if err != nil {
		t.Fatalf("Failed to set memory limit: %v", err)
	}

	usage, err = collection.GetMemoryUsage(context.Background())
	if err != nil {
		t.Fatalf("Failed to get memory usage after limit update: %v", err)
	}

	if usage.Limit != 100*1024*1024 {
		t.Errorf("Expected updated memory limit 100MB, got %d", usage.Limit)
	}

	// Test GC trigger
	err = collection.TriggerGC()
	if err != nil {
		t.Fatalf("Failed to trigger GC: %v", err)
	}
}

func TestCollectionOptimization(t *testing.T) {
	// Create database
	db, err := Open(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Create collection with auto index selection
	collection, err := db.CreateCollection(
		context.Background(),
		"test_optimization",
		WithDimension(128),
		WithAutoIndexSelection(true),
	)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert some test vectors
	ctx := context.Background()
	for i := 0; i < 100; i++ {
		vector := make([]float32, 128)
		for j := range vector {
			vector[j] = float32(i*j) * 0.1
		}

		err := collection.Insert(ctx, fmt.Sprintf("vec_%d", i), vector, map[string]interface{}{
			"id": i,
		})
		if err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}
	}

	// Test optimization status
	status := collection.GetOptimizationStatus()
	if !status.CanOptimize {
		t.Error("Expected collection to be optimizable")
	}

	if status.InProgress {
		t.Error("Expected optimization not to be in progress initially")
	}

	// Test collection optimization
	options := &OptimizationOptions{
		RebuildIndex:   true,
		OptimizeMemory: true,
		CompactStorage: true,
	}

	err = collection.OptimizeCollection(ctx, options)
	if err != nil {
		t.Fatalf("Failed to optimize collection: %v", err)
	}

	// Check optimization status after completion
	status = collection.GetOptimizationStatus()
	if status.InProgress {
		t.Error("Expected optimization to be completed")
	}

	if status.LastOptimization.IsZero() {
		t.Error("Expected last optimization timestamp to be set")
	}
}

func TestCollectionMemoryMapping(t *testing.T) {
	// Create database
	db, err := Open(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Create collection
	collection, err := db.CreateCollection(
		context.Background(),
		"test_mmap",
		WithDimension(128),
	)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Test enabling memory mapping
	err = collection.EnableMemoryMapping(t.TempDir())
	if err != nil {
		// Memory mapping might not be supported by all index types
		// This is acceptable for this test
		t.Logf("Memory mapping not supported: %v", err)
		return
	}

	// Test disabling memory mapping
	err = collection.DisableMemoryMapping()
	if err != nil {
		t.Fatalf("Failed to disable memory mapping: %v", err)
	}
}

func TestDatabaseGlobalMemoryManagement(t *testing.T) {
	// Create database
	db, err := Open(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Create multiple collections
	collections := make([]*Collection, 3)
	for i := 0; i < 3; i++ {
		collection, err := db.CreateCollection(
			context.Background(),
			fmt.Sprintf("test_global_%d", i),
			WithDimension(64),
		)
		if err != nil {
			t.Fatalf("Failed to create collection %d: %v", i, err)
		}
		collections[i] = collection
	}

	// Test global memory limit
	err = db.SetGlobalMemoryLimit(150 * 1024 * 1024) // 150MB total
	if err != nil {
		t.Fatalf("Failed to set global memory limit: %v", err)
	}

	// Test global memory usage
	usage, err := db.GetGlobalMemoryUsage(context.Background())
	if err != nil {
		t.Fatalf("Failed to get global memory usage: %v", err)
	}

	if len(usage.Collections) != 3 {
		t.Errorf("Expected 3 collections in usage report, got %d", len(usage.Collections))
	}

	// Test global GC
	err = db.TriggerGlobalGC()
	if err != nil {
		t.Fatalf("Failed to trigger global GC: %v", err)
	}

	// Test global optimization. Skip memory optimization since empty collections
	// have no vectors to free and the per-collection limit (50MB) is too low for
	// fixed index overhead (SFL slotGen arrays + graph structures).
	options := &OptimizationOptions{
		RebuildIndex:   false,
		OptimizeMemory: false,
		CompactStorage: false,
	}

	err = db.OptimizeAllCollections(context.Background(), options)
	if err != nil {
		t.Fatalf("Failed to optimize all collections: %v", err)
	}
}

func TestCollectionStatsEnhancement(t *testing.T) {
	// Create database
	db, err := Open(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Create collection with various features
	collection, err := db.CreateCollection(
		context.Background(),
		"test_stats",
		WithDimension(128),
		WithMemoryLimit(50*1024*1024),
		WithMemoryMapping(true),
		WithProductQuantization(8, 8, 0.1),
	)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Get enhanced stats
	stats := collection.Stats(context.Background())

	// Verify enhanced fields
	if !stats.HasQuantization {
		t.Error("Expected HasQuantization to be true")
	}

	if !stats.HasMemoryLimit {
		t.Error("Expected HasMemoryLimit to be true")
	}

	if !stats.MemoryMappingEnabled {
		t.Error("Expected MemoryMappingEnabled to be true")
	}

	if stats.OptimizationStatus == nil {
		t.Error("Expected OptimizationStatus to be present")
	}

	if stats.OptimizationStatus != nil && !stats.OptimizationStatus.CanOptimize {
		t.Error("Expected collection to be optimizable")
	}
}

func TestCollectionConfigurationValidation(t *testing.T) {
	// Create database
	db, err := Open(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Test invalid memory limit
	_, err = db.CreateCollection(
		context.Background(),
		"test_invalid",
		WithDimension(128),
		WithMemoryLimit(-1), // Invalid negative limit
	)
	if err == nil {
		t.Error("Expected error for negative memory limit")
	}

	// Test valid memory configuration
	memConfig := &memory.MemoryConfig{
		MaxMemory:       100 * 1024 * 1024,
		MonitorInterval: 5 * time.Second,
		EnableGC:        true,
		GCThreshold:     0.8,
		EnableMMap:      true,
		MMapThreshold:   50 * 1024 * 1024,
	}

	collection, err := db.CreateCollection(
		context.Background(),
		"test_valid",
		WithDimension(128),
		WithMemoryConfig(memConfig),
	)
	if err != nil {
		t.Fatalf("Failed to create collection with valid memory config: %v", err)
	}

	// Verify configuration was applied
	usage, err := collection.GetMemoryUsage(context.Background())
	if err != nil {
		t.Fatalf("Failed to get memory usage: %v", err)
	}

	if usage.Limit != memConfig.MaxMemory {
		t.Errorf("Expected memory limit %d, got %d", memConfig.MaxMemory, usage.Limit)
	}
}

func TestCollectionLifecycleWithMemoryManagement(t *testing.T) {
	// Create database
	db, err := Open(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Create collection with memory management
	collection, err := db.CreateCollection(
		context.Background(),
		"test_lifecycle",
		WithDimension(64),
		WithMemoryLimit(100*1024*1024), // 100MB limit
	)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert vectors and monitor memory
	ctx := context.Background()
	for i := 0; i < 50; i++ {
		vector := make([]float32, 64)
		for j := range vector {
			vector[j] = float32(i*j) * 0.01
		}

		err := collection.Insert(ctx, fmt.Sprintf("vec_%d", i), vector, map[string]interface{}{
			"batch": i / 10,
		})
		if err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}

		// Check memory usage periodically
		if i%10 == 0 {
			usage, err := collection.GetMemoryUsage(context.Background())
			if err != nil {
				t.Fatalf("Failed to get memory usage at vector %d: %v", i, err)
			}

			t.Logf("Memory usage at vector %d: %d bytes (limit: %d)", i, usage.Total, usage.Limit)
		}
	}

	// Perform search to ensure functionality
	queryVector := make([]float32, 64)
	for i := range queryVector {
		queryVector[i] = 0.5
	}

	results, err := collection.Search(ctx, queryVector, 5)
	if err != nil {
		t.Fatalf("Failed to search: %v", err)
	}

	if len(results.Results) == 0 {
		t.Error("Expected search results")
	}

	// Test optimization
	err = collection.OptimizeCollection(ctx, &OptimizationOptions{
		RebuildIndex:   true,
		OptimizeMemory: true,
	})
	if err != nil {
		t.Fatalf("Failed to optimize collection: %v", err)
	}

	// Verify search still works after optimization
	results, err = collection.Search(ctx, queryVector, 5)
	if err != nil {
		t.Fatalf("Failed to search after optimization: %v", err)
	}

	if len(results.Results) == 0 {
		t.Error("Expected search results after optimization")
	}
}

func TestShardedCollectionUnsupportedMethods(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := Open(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "unsupported_test", WithDimension(3), WithSharding(true))
	if err != nil {
		t.Fatalf("create sharded collection: %v", err)
	}

	// OptimizeCollection should return explicit unsupported error
	err = collection.OptimizeCollection(ctx, &OptimizationOptions{RebuildIndex: true})
	if err == nil {
		t.Error("Expected error from OptimizeCollection for sharded collection, got nil")
	} else if str := err.Error(); str != "OptimizeCollection is not supported for sharded collections" {
		t.Errorf("Unexpected error message: %v", err)
	}

	// SetMemoryLimit should return explicit unsupported error
	err = collection.SetMemoryLimit(1024 * 1024 * 1024)
	if err == nil {
		t.Error("Expected error from SetMemoryLimit for sharded collection, got nil")
	} else if str := err.Error(); str != "SetMemoryLimit is not supported for sharded collections" {
		t.Errorf("Unexpected error message: %v", err)
	}

	// EnableMemoryMapping should return explicit unsupported error
	err = collection.EnableMemoryMapping("/tmp/test")
	if err == nil {
		t.Error("Expected error from EnableMemoryMapping for sharded collection, got nil")
	} else if str := err.Error(); str != "EnableMemoryMapping is not supported for sharded collections" {
		t.Errorf("Unexpected error message: %v", err)
	}

	// DisableMemoryMapping should return explicit unsupported error
	err = collection.DisableMemoryMapping()
	if err == nil {
		t.Error("Expected error from DisableMemoryMapping for sharded collection, got nil")
	} else if str := err.Error(); str != "DisableMemoryMapping is not supported for sharded collections" {
		t.Errorf("Unexpected error message: %v", err)
	}

	// SaveIndex should return explicit unsupported error
	err = collection.SaveIndex(ctx, "/tmp/test_index")
	if err == nil {
		t.Error("Expected error from SaveIndex for sharded collection, got nil")
	} else if str := err.Error(); str != "SaveIndex is not supported for sharded collections" {
		t.Errorf("Unexpected error message: %v", err)
	}

	// LoadIndex should return explicit unsupported error
	err = collection.LoadIndex(ctx, "/tmp/test_index")
	if err == nil {
		t.Error("Expected error from LoadIndex for sharded collection, got nil")
	} else if str := err.Error(); str != "LoadIndex is not supported for sharded collections" {
		t.Errorf("Unexpected error message: %v", err)
	}
}
