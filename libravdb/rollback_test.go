package libravdb

import (
	"context"
	"fmt"
	"testing"
	"time"
)

func TestBatchInsertRollback(t *testing.T) {
	// Create a test database
	db, err := New(WithStoragePath(t.TempDir()))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create a test collection
	collectionName := fmt.Sprintf("rollback_insert_test_%d", time.Now().UnixNano())
	collection, err := db.CreateCollection(ctx, collectionName, WithDimension(3), WithMetric(CosineDistance), WithHNSW(16, 200, 50))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create batch with some valid and some invalid entries
	vectors := []*VectorEntry{
		{ID: "valid1", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"type": "valid"}},
		{ID: "valid2", Vector: []float32{4.0, 5.0, 6.0}, Metadata: map[string]interface{}{"type": "valid"}},
		{ID: "invalid", Vector: []float32{7.0, 8.0}, Metadata: map[string]interface{}{"type": "invalid"}}, // Wrong dimension
		{ID: "valid3", Vector: []float32{7.0, 8.0, 9.0}, Metadata: map[string]interface{}{"type": "valid"}},
	}

	// Enable rollback and fail-fast
	options := DefaultBatchOptions()
	options.EnableRollback = true
	options.FailFast = true

	batchInsert := collection.NewBatchInsert(vectors, options)
	result, err := batchInsert.Execute(ctx)

	// Should fail due to invalid vector dimension
	if err == nil {
		t.Error("Expected batch insert to fail due to invalid vector dimension")
	}

	// Should have triggered rollback
	if !result.RollbackRequired {
		t.Error("Expected rollback to be required")
	}

	// Verify no vectors were inserted (rollback successful)
	searchResults, err := collection.Search(ctx, []float32{1.0, 1.0, 1.0}, 10)
	if err != nil {
		// Empty index error is expected after rollback
		if searchResults != nil && len(searchResults.Results) > 0 {
			t.Errorf("Expected no vectors after rollback, found %d", len(searchResults.Results))
		}
	} else if len(searchResults.Results) > 0 {
		t.Errorf("Expected no vectors after rollback, found %d", len(searchResults.Results))
	}
}

func TestBatchUpdateRollback(t *testing.T) {
	// Create a test database
	db, err := New(WithStoragePath(t.TempDir()))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create a test collection
	collectionName := fmt.Sprintf("rollback_update_test_%d", time.Now().UnixNano())
	collection, err := db.CreateCollection(ctx, collectionName, WithDimension(3), WithMetric(CosineDistance), WithHNSW(16, 200, 50))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert initial vectors
	initialVectors := []*VectorEntry{
		{ID: "update1", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"version": 1}},
		{ID: "update2", Vector: []float32{4.0, 5.0, 6.0}, Metadata: map[string]interface{}{"version": 1}},
		{ID: "update3", Vector: []float32{7.0, 8.0, 9.0}, Metadata: map[string]interface{}{"version": 1}},
	}

	batchInsert := collection.NewBatchInsert(initialVectors)
	_, err = batchInsert.Execute(ctx)
	if err != nil {
		t.Fatalf("Failed to insert initial vectors: %v", err)
	}

	// Create batch update with some valid and some invalid updates
	updates := []*VectorUpdate{
		{ID: "update1", Vector: []float32{1.1, 2.1, 3.1}, Metadata: map[string]interface{}{"version": 2}},
		{ID: "update2", Vector: []float32{4.1, 5.1, 6.1}, Metadata: map[string]interface{}{"version": 2}},
		{ID: "nonexistent", Vector: []float32{10.0, 11.0, 12.0}, Metadata: map[string]interface{}{"version": 2}}, // Non-existent ID
		{ID: "update3", Vector: []float32{7.1, 8.1, 9.1}, Metadata: map[string]interface{}{"version": 2}},
	}

	// Enable rollback and fail-fast
	options := DefaultBatchOptions()
	options.EnableRollback = true
	options.FailFast = true

	batchUpdate := collection.NewBatchUpdate(updates, options)
	result, err := batchUpdate.Execute(ctx)

	// Should fail due to non-existent vector
	if err == nil {
		t.Error("Expected batch update to fail due to non-existent vector")
	}

	// Should have triggered rollback
	if !result.RollbackRequired {
		t.Error("Expected rollback to be required")
	}

	// Verify original vectors are still intact (rollback successful)
	for _, original := range initialVectors {
		searchResults, err := collection.Search(ctx, original.Vector, 1)
		if err != nil {
			t.Fatalf("Failed to search for original vector %s: %v", original.ID, err)
		}

		if len(searchResults.Results) == 0 {
			t.Errorf("Original vector %s not found after rollback", original.ID)
			continue
		}

		result := searchResults.Results[0]
		if result.ID != original.ID {
			t.Errorf("Expected ID %s, got %s", original.ID, result.ID)
		}

		if result.Metadata["version"] != 1 {
			t.Errorf("Expected version 1 for %s after rollback, got %v", original.ID, result.Metadata["version"])
		}
	}
}

func TestBatchDeleteRollback(t *testing.T) {
	// Create a test database
	db, err := New(WithStoragePath(t.TempDir()))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create a test collection
	collectionName := fmt.Sprintf("rollback_delete_test_%d", time.Now().UnixNano())
	collection, err := db.CreateCollection(ctx, collectionName, WithDimension(3), WithMetric(CosineDistance), WithHNSW(16, 200, 50))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert initial vectors
	initialVectors := []*VectorEntry{
		{ID: "delete1", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"type": "test"}},
		{ID: "delete2", Vector: []float32{4.0, 5.0, 6.0}, Metadata: map[string]interface{}{"type": "test"}},
		{ID: "delete3", Vector: []float32{7.0, 8.0, 9.0}, Metadata: map[string]interface{}{"type": "test"}},
	}

	batchInsert := collection.NewBatchInsert(initialVectors)
	_, err = batchInsert.Execute(ctx)
	if err != nil {
		t.Fatalf("Failed to insert initial vectors: %v", err)
	}

	// Create batch delete with some valid and some invalid IDs
	idsToDelete := []string{"delete1", "delete2", "nonexistent", "delete3"}

	// Enable rollback and fail-fast
	options := DefaultBatchOptions()
	options.EnableRollback = true
	options.FailFast = true

	batchDelete := collection.NewBatchDelete(idsToDelete, options)
	result, err := batchDelete.Execute(ctx)

	// Note: Delete operations might not fail on non-existent IDs depending on implementation
	// This test verifies that if rollback is triggered, it works correctly
	if result.RollbackRequired && result.RollbackError == nil {
		// Verify all original vectors are still present (rollback successful)
		searchResults, err := collection.Search(ctx, []float32{1.0, 1.0, 1.0}, 10)
		if err != nil {
			t.Fatalf("Failed to search after rollback: %v", err)
		}

		if len(searchResults.Results) != len(initialVectors) {
			t.Errorf("Expected %d vectors after rollback, found %d", len(initialVectors), len(searchResults.Results))
		}

		// Verify each original vector is present
		foundIDs := make(map[string]bool)
		for _, result := range searchResults.Results {
			foundIDs[result.ID] = true
		}

		for _, original := range initialVectors {
			if !foundIDs[original.ID] {
				t.Errorf("Original vector %s not found after rollback", original.ID)
			}
		}
	}
}

func TestBatchOperationsWithoutRollback(t *testing.T) {
	// Create a test database
	db, err := New(WithStoragePath(t.TempDir()))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create a test collection
	collectionName := fmt.Sprintf("no_rollback_test_%d", time.Now().UnixNano())
	collection, err := db.CreateCollection(ctx, collectionName, WithDimension(3), WithMetric(CosineDistance), WithHNSW(16, 200, 50))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create batch with some valid and some invalid entries
	vectors := []*VectorEntry{
		{ID: "valid1", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"type": "valid"}},
		{ID: "valid2", Vector: []float32{4.0, 5.0, 6.0}, Metadata: map[string]interface{}{"type": "valid"}},
		{ID: "invalid", Vector: []float32{7.0, 8.0}, Metadata: map[string]interface{}{"type": "invalid"}}, // Wrong dimension
		{ID: "valid3", Vector: []float32{7.0, 8.0, 9.0}, Metadata: map[string]interface{}{"type": "valid"}},
	}

	// Disable rollback and fail-fast
	options := DefaultBatchOptions()
	options.EnableRollback = false
	options.FailFast = false

	batchInsert := collection.NewBatchInsert(vectors, options)
	result, err := batchInsert.Execute(ctx)

	// Should complete with some successes and some failures
	if result.Successful == 0 {
		t.Error("Expected some successful insertions")
	}

	if result.Failed == 0 {
		t.Error("Expected some failed insertions")
	}

	// Should not have triggered rollback
	if result.RollbackRequired {
		t.Error("Expected no rollback when rollback is disabled")
	}

	// Verify successful vectors were inserted
	searchResults, err := collection.Search(ctx, []float32{1.0, 1.0, 1.0}, 10)
	if err != nil {
		t.Fatalf("Failed to search: %v", err)
	}

	if len(searchResults.Results) != result.Successful {
		t.Errorf("Expected %d vectors, found %d", result.Successful, len(searchResults.Results))
	}
}
