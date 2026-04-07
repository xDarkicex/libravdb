package libravdb

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"
)

// TestRollbackEdgeCases tests various edge cases for rollback functionality
func TestRollbackEdgeCases(t *testing.T) {
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	t.Run("EmptyBatchRollback", func(t *testing.T) {
		collection, err := db.CreateCollection(ctx, "empty_rollback_test", WithDimension(3))
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// Test empty batch insert
		emptyBatch := collection.NewBatchInsert([]*VectorEntry{}, &BatchOptions{
			EnableRollback: true,
			FailFast:       true,
		})
		result, err := emptyBatch.Execute(ctx)
		if err != nil {
			t.Fatalf("Empty batch execution failed: %v", err)
		}
		if result.Successful != 0 {
			t.Errorf("Expected 0 successful, got %d", result.Successful)
		}
		if result.Failed != 0 {
			t.Errorf("Expected 0 failed, got %d", result.Failed)
		}
		if result.RollbackRequired {
			t.Errorf("Expected RollbackRequired to be false, got true")
		}
	})

	t.Run("SingleItemRollback", func(t *testing.T) {
		collection, err := db.CreateCollection(ctx, "single_rollback_test", WithDimension(3))
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// Insert one valid item and one invalid
		vectors := []*VectorEntry{
			{ID: "valid", Vector: []float32{1.0, 2.0, 3.0}},
			{ID: "invalid", Vector: []float32{1.0, 2.0}}, // Wrong dimension
		}

		batch := collection.NewBatchInsert(vectors, &BatchOptions{
			EnableRollback: true,
			FailFast:       true,
		})
		result, err := batch.Execute(ctx)
		if err == nil {
			t.Fatal("Expected error due to invalid vector, but got none")
		}
		if !result.RollbackRequired {
			t.Errorf("Expected RollbackRequired to be true, got false")
		}

		// Verify rollback worked - no vectors should be present
		searchResults, err := collection.Search(ctx, []float32{1.0, 2.0, 3.0}, 10)
		if err != nil {
			// Empty index is acceptable
			if searchResults != nil && len(searchResults.Results) != 0 {
				t.Errorf("Expected 0 results, got %d", len(searchResults.Results))
			}
		} else {
			if len(searchResults.Results) != 0 {
				t.Errorf("Expected 0 results, got %d", len(searchResults.Results))
			}
		}
	})

	t.Run("DuplicateIDRollback", func(t *testing.T) {
		collection, err := db.CreateCollection(ctx, "duplicate_rollback_test", WithDimension(3))
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// First insert a vector
		err = collection.Insert(ctx, "dup", []float32{1.0, 2.0, 3.0}, nil)
		if err != nil {
			t.Fatalf("Failed to insert vector: %v", err)
		}

		// Try to insert duplicate in batch
		vectors := []*VectorEntry{
			{ID: "dup", Vector: []float32{4.0, 5.0, 6.0}}, // Duplicate ID
			{ID: "new", Vector: []float32{7.0, 8.0, 9.0}},
		}

		batch := collection.NewBatchInsert(vectors, &BatchOptions{
			EnableRollback: true,
			FailFast:       true,
		})
		result, err := batch.Execute(ctx)
		if err == nil {
			t.Fatal("Expected error due to duplicate ID, but got none")
		}
		if !result.RollbackRequired {
			t.Errorf("Expected RollbackRequired to be true, got false")
		}

		// Verify original vector still exists and new one doesn't
		searchResults, err := collection.Search(ctx, []float32{1.0, 2.0, 3.0}, 10)
		if err != nil {
			t.Fatalf("Failed to search: %v", err)
		}
		if len(searchResults.Results) != 1 {
			t.Errorf("Expected 1 result, got %d", len(searchResults.Results))
		}
		if searchResults.Results[0].ID != "dup" {
			t.Errorf("Expected ID 'dup', got '%s'", searchResults.Results[0].ID)
		}
	})
}

// TestRollbackStress tests rollback under stress conditions
func TestRollbackStress(t *testing.T) {
	requireStressMode(t)

	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	t.Run("LargeBatchRollback", func(t *testing.T) {
		collection, err := db.CreateCollection(ctx, "large_rollback_test", WithDimension(128), WithFlat())
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// Create large batch with one invalid item at the end
		const batchSize = 5000
		vectors := make([]*VectorEntry, batchSize+1)

		for i := 0; i < batchSize; i++ {
			vector := make([]float32, 128)
			for j := range vector {
				vector[j] = rand.Float32()
			}
			vectors[i] = &VectorEntry{
				ID:     fmt.Sprintf("vec_%d", i),
				Vector: vector,
			}
		}

		// Add invalid vector at the end
		vectors[batchSize] = &VectorEntry{
			ID:     "invalid",
			Vector: []float32{1.0, 2.0}, // Wrong dimension
		}

		start := time.Now()
		batch := collection.NewBatchInsert(vectors, &BatchOptions{
			EnableRollback: true,
			FailFast:       true,
			ChunkSize:      1000,
		})
		result, err := batch.Execute(ctx)
		duration := time.Since(start)

		if err == nil {
			t.Fatal("Expected error due to invalid vector, but got none")
		}
		if !result.RollbackRequired {
			t.Errorf("Expected RollbackRequired to be true, got false")
		}
		t.Logf("Large batch rollback took %v", duration)

		// Verify no vectors were inserted
		searchResults, err := collection.Search(ctx, make([]float32, 128), 10)
		if err == nil {
			if len(searchResults.Results) != 0 {
				t.Errorf("Expected 0 results, got %d", len(searchResults.Results))
			}
		}
	})

	t.Run("ConcurrentRollback", func(t *testing.T) {
		collection, err := db.CreateCollection(ctx, "concurrent_rollback_test", WithDimension(3))
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		const numGoroutines = 10
		const itemsPerBatch = 100

		var wg sync.WaitGroup
		errorChan := make(chan error, numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(goroutineID int) {
				defer wg.Done()

				// Create batch with mix of valid and invalid items
				vectors := make([]*VectorEntry, itemsPerBatch)
				for j := 0; j < itemsPerBatch; j++ {
					if j == itemsPerBatch-1 {
						// Make last item invalid
						vectors[j] = &VectorEntry{
							ID:     fmt.Sprintf("g%d_invalid_%d", goroutineID, j),
							Vector: []float32{1.0, 2.0}, // Wrong dimension
						}
					} else {
						vectors[j] = &VectorEntry{
							ID:     fmt.Sprintf("g%d_vec_%d", goroutineID, j),
							Vector: []float32{float32(goroutineID), float32(j), 3.0},
						}
					}
				}

				batch := collection.NewBatchInsert(vectors, &BatchOptions{
					EnableRollback: true,
					FailFast:       true,
				})
				_, err := batch.Execute(ctx)
				if err != nil {
					errorChan <- err
				}
			}(i)
		}

		wg.Wait()
		close(errorChan)

		// Should have received errors from all goroutines
		errorCount := 0
		for range errorChan {
			errorCount++
		}
		if errorCount != numGoroutines {
			t.Errorf("Expected %d errors, got %d", numGoroutines, errorCount)
		}

		// Verify collection is still empty due to rollbacks
		searchResults, err := collection.Search(ctx, []float32{1.0, 1.0, 1.0}, 100)
		if err == nil {
			if len(searchResults.Results) != 0 {
				t.Errorf("Expected 0 results, got %d", len(searchResults.Results))
			}
		}
	})
}

// TestRollbackFailureScenarios tests various rollback failure scenarios
func TestRollbackFailureScenarios(t *testing.T) {
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	t.Run("RollbackFailureHandling", func(t *testing.T) {
		collection, err := db.CreateCollection(ctx, "rollback_failure_test", WithDimension(3))
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// Insert some initial data
		initialVectors := []*VectorEntry{
			{ID: "initial1", Vector: []float32{1.0, 2.0, 3.0}},
			{ID: "initial2", Vector: []float32{4.0, 5.0, 6.0}},
		}

		batch := collection.NewBatchInsert(initialVectors)
		_, err = batch.Execute(ctx)
		if err != nil {
			t.Fatalf("Failed to execute initial batch: %v", err)
		}

		// Now try a batch that will succeed partially then fail
		// This is harder to test without mocking, so we'll test the structure
		vectors := []*VectorEntry{
			{ID: "new1", Vector: []float32{7.0, 8.0, 9.0}},
			{ID: "new2", Vector: []float32{10.0, 11.0, 12.0}},
			{ID: "invalid", Vector: []float32{1.0, 2.0}}, // Wrong dimension
		}

		batch = collection.NewBatchInsert(vectors, &BatchOptions{
			EnableRollback: true,
			FailFast:       true,
		})
		result, err := batch.Execute(ctx)
		if err == nil {
			t.Fatal("Expected error due to invalid vector, but got none")
		}
		if !result.RollbackRequired {
			t.Errorf("Expected RollbackRequired to be true, got false")
		}

		// Verify rollback error is properly reported
		if result.RollbackError != nil {
			t.Logf("Rollback error: %v", result.RollbackError)
		}

		// Verify only initial vectors remain
		searchResults, err := collection.Search(ctx, []float32{1.0, 1.0, 1.0}, 10)
		if err != nil {
			t.Fatalf("Failed to search: %v", err)
		}
		if len(searchResults.Results) != 2 {
			t.Errorf("Expected 2 results, got %d", len(searchResults.Results))
		}
	})
}

// TestRollbackDataConsistency tests data consistency after rollback operations
func TestRollbackDataConsistency(t *testing.T) {
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	t.Run("UpdateRollbackConsistency", func(t *testing.T) {
		collection, err := db.CreateCollection(ctx, "update_consistency_test", WithDimension(3))
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// Insert initial data with metadata
		initialEntry := &VectorEntry{
			ID:       "consistency_test",
			Vector:   []float32{1.0, 2.0, 3.0},
			Metadata: map[string]interface{}{"version": 1, "status": "original"},
		}

		err = collection.Insert(ctx, initialEntry.ID, initialEntry.Vector, initialEntry.Metadata)
		if err != nil {
			t.Fatalf("Failed to insert vector: %v", err)
		}

		// Attempt batch update that will fail
		updates := []*VectorUpdate{
			{
				ID:       "consistency_test",
				Vector:   []float32{1.1, 2.1, 3.1},
				Metadata: map[string]interface{}{"version": 2, "status": "updated"},
			},
			{
				ID:       "nonexistent",
				Vector:   []float32{4.0, 5.0, 6.0},
				Metadata: map[string]interface{}{"version": 1},
			},
		}

		batch := collection.NewBatchUpdate(updates, &BatchOptions{
			EnableRollback: true,
			FailFast:       true,
		})
		result, err := batch.Execute(ctx)
		if err == nil {
			t.Fatal("Expected error due to nonexistent ID, but got none")
		}
		if !result.RollbackRequired {
			t.Errorf("Expected RollbackRequired to be true, got false")
		}

		// Verify data consistency - original data should be intact
		searchResults, err := collection.Search(ctx, []float32{1.0, 2.0, 3.0}, 1)
		if err != nil {
			t.Fatalf("Failed to search: %v", err)
		}
		if len(searchResults.Results) != 1 {
			t.Errorf("Expected 1 result, got %d", len(searchResults.Results))
		}

		resultEntry := searchResults.Results[0]
		if resultEntry.ID != "consistency_test" {
			t.Errorf("Expected ID 'consistency_test', got '%s'", resultEntry.ID)
		}
		if resultEntry.Metadata["version"] != 1 && resultEntry.Metadata["version"] != float32(1) {
			t.Errorf("Expected version 1, got %v", resultEntry.Metadata["version"])
		}
		if resultEntry.Metadata["status"] != "original" {
			t.Errorf("Expected status 'original', got '%v'", resultEntry.Metadata["status"])
		}
	})

	t.Run("DeleteRollbackConsistency", func(t *testing.T) {
		collection, err := db.CreateCollection(ctx, "delete_consistency_test", WithDimension(3))
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// Insert test data
		entries := []*VectorEntry{
			{ID: "delete1", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"type": "test"}},
			{ID: "delete2", Vector: []float32{4.0, 5.0, 6.0}, Metadata: map[string]interface{}{"type": "test"}},
			{ID: "delete3", Vector: []float32{7.0, 8.0, 9.0}, Metadata: map[string]interface{}{"type": "test"}},
		}

		for _, entry := range entries {
			err = collection.Insert(ctx, entry.ID, entry.Vector, entry.Metadata)
			if err != nil {
				t.Fatalf("Failed to insert vector: %v", err)
			}
		}

		// Attempt batch delete that will fail
		idsToDelete := []string{"delete1", "delete2", "nonexistent"}

		batch := collection.NewBatchDelete(idsToDelete, &BatchOptions{
			EnableRollback: true,
			FailFast:       true,
		})
		result, err := batch.Execute(ctx)
		if err == nil {
			t.Fatal("Expected error due to nonexistent ID, but got none")
		}
		if !result.RollbackRequired {
			t.Errorf("Expected RollbackRequired to be true, got false")
		}

		// Verify all original data is still present
		searchResults, err := collection.Search(ctx, []float32{1.0, 1.0, 1.0}, 10)
		if err != nil {
			t.Fatalf("Failed to search: %v", err)
		}
		if len(searchResults.Results) != 3 {
			t.Errorf("Expected 3 results, got %d", len(searchResults.Results))
		}

		// Verify all IDs are present
		foundIDs := make(map[string]bool)
		for _, result := range searchResults.Results {
			foundIDs[result.ID] = true
			if result.Metadata["type"] != "test" {
				t.Errorf("Expected type 'test', got '%v'", result.Metadata["type"])
			}
		}

		if !foundIDs["delete1"] {
			t.Errorf("Expected to find delete1")
		}
		if !foundIDs["delete2"] {
			t.Errorf("Expected to find delete2")
		}
		if !foundIDs["delete3"] {
			t.Errorf("Expected to find delete3")
		}
	})
}

// TestRollbackPerformance benchmarks rollback performance
func TestRollbackPerformance(t *testing.T) {
	requireStressMode(t)

	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	t.Run("RollbackPerformanceBenchmark", func(t *testing.T) {
		collection, err := db.CreateCollection(ctx, "performance_test", WithDimension(128))
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// Test different batch sizes
		batchSizes := []int{100, 500, 1000, 5000}

		for _, size := range batchSizes {
			t.Run(fmt.Sprintf("BatchSize_%d", size), func(t *testing.T) {
				// Create batch with one invalid item
				vectors := make([]*VectorEntry, size+1)
				for i := 0; i < size; i++ {
					vector := make([]float32, 128)
					for j := range vector {
						vector[j] = rand.Float32()
					}
					vectors[i] = &VectorEntry{
						ID:     fmt.Sprintf("perf_%d_%d", size, i),
						Vector: vector,
					}
				}
				vectors[size] = &VectorEntry{
					ID:     "invalid",
					Vector: []float32{1.0, 2.0}, // Wrong dimension
				}

				start := time.Now()
				batch := collection.NewBatchInsert(vectors, &BatchOptions{
					EnableRollback: true,
					FailFast:       true,
					ChunkSize:      1000,
				})
				result, err := batch.Execute(ctx)
				duration := time.Since(start)

				if err == nil {
					t.Fatal("Expected error due to invalid vector, but got none")
				}
				if !result.RollbackRequired {
					t.Errorf("Expected RollbackRequired to be true, got false")
				}

				t.Logf("Batch size %d: rollback took %v", size, duration)
				t.Logf("Successful: %d, Failed: %d", result.Successful, result.Failed)
			})
		}
	})
}

// TestRollbackIntegration tests rollback with other system components
func TestRollbackIntegration(t *testing.T) {
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	t.Run("RollbackWithTimeout", func(t *testing.T) {
		collection, err := db.CreateCollection(ctx, "timeout_rollback_test", WithDimension(3))
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// Create a batch that might take time
		vectors := make([]*VectorEntry, 1000)
		for i := 0; i < 999; i++ {
			vectors[i] = &VectorEntry{
				ID:     fmt.Sprintf("timeout_%d", i),
				Vector: []float32{float32(i), float32(i + 1), float32(i + 2)},
			}
		}
		vectors[999] = &VectorEntry{
			ID:     "invalid",
			Vector: []float32{1.0, 2.0}, // Wrong dimension
		}

		// Test with very short timeout
		batch := collection.NewBatchInsert(vectors, &BatchOptions{
			EnableRollback: true,
			FailFast:       true,
			Timeout:        1 * time.Millisecond, // Very short timeout
		})
		result, err := batch.Execute(ctx)

		// Should either timeout or fail due to invalid vector
		if err == nil && result.Failed == 0 {
			t.Errorf("Expected either an error or some failed operations")
		}
		if result.RollbackRequired {
			t.Log("Rollback was triggered")
		}
	})

	t.Run("RollbackMemoryUsage", func(t *testing.T) {
		collection, err := db.CreateCollection(ctx, "memory_rollback_test", WithDimension(128), WithFlat())
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		// Create large vectors to test memory usage
		const numVectors = 1000
		vectors := make([]*VectorEntry, numVectors+1)

		for i := 0; i < numVectors; i++ {
			vector := make([]float32, 128)
			for j := range vector {
				vector[j] = rand.Float32()
			}
			vectors[i] = &VectorEntry{
				ID:     fmt.Sprintf("memory_%d", i),
				Vector: vector,
				Metadata: map[string]interface{}{
					"large_data": make([]byte, 1024), // 1KB of metadata per vector
				},
			}
		}
		vectors[numVectors] = &VectorEntry{
			ID:     "invalid",
			Vector: []float32{1.0, 2.0}, // Wrong dimension
		}

		batch := collection.NewBatchInsert(vectors, &BatchOptions{
			EnableRollback: true,
			FailFast:       true,
			ChunkSize:      100,
		})

		// This should complete without memory issues
		result, err := batch.Execute(ctx)
		if err == nil {
			t.Fatal("Expected error due to invalid vector, but got none")
		}
		if !result.RollbackRequired {
			t.Errorf("Expected RollbackRequired to be true, got false")
		}

		// Verify memory was properly cleaned up (no vectors inserted)
		searchResults, err := collection.Search(ctx, make([]float32, 128), 10)
		if err == nil {
			if len(searchResults.Results) != 0 {
				t.Errorf("Expected 0 results, got %d", len(searchResults.Results))
			}
		}
	})
}
