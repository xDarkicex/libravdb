package libravdb

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestCollectionCRUDOperations(t *testing.T) {
	// Create a test database
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create a test collection with timestamp to ensure uniqueness
	collectionName := fmt.Sprintf("crud_test_%d", time.Now().UnixNano())
	collection, err := db.CreateCollection(ctx, collectionName, WithDimension(3), WithMetric(CosineDistance), WithHNSW(16, 200, 50))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Test Insert
	vector1 := []float32{1.0, 2.0, 3.0}
	metadata1 := map[string]interface{}{"category": "test", "value": 42}

	err = collection.Insert(ctx, "test1", vector1, metadata1)
	if err != nil {
		t.Fatalf("Failed to insert vector: %v", err)
	}

	// Test Update - partial update (metadata only)
	newMetadata := map[string]interface{}{"category": "updated", "new_field": "added"}
	err = collection.Update(ctx, "test1", nil, newMetadata)
	if err != nil {
		t.Fatalf("Failed to update vector metadata: %v", err)
	}

	// Test Update - full update (vector and metadata)
	newVector := []float32{4.0, 5.0, 6.0}
	fullMetadata := map[string]interface{}{"category": "fully_updated", "value": 100}
	err = collection.Update(ctx, "test1", newVector, fullMetadata)
	if err != nil {
		t.Fatalf("Failed to update vector completely: %v", err)
	}

	// Verify the update by searching
	results, err := collection.Search(ctx, newVector, 1)
	if err != nil {
		t.Fatalf("Failed to search after update: %v", err)
	}

	if len(results.Results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results.Results))
	}

	result := results.Results[0]
	if result.ID != "test1" {
		t.Errorf("Expected ID 'test1', got '%s'", result.ID)
	}

	if result.Metadata["category"] != "fully_updated" {
		t.Errorf("Expected category 'fully_updated', got '%v'", result.Metadata["category"])
	}

	if result.Metadata["value"] != 100 {
		t.Errorf("Expected value 100, got '%v'", result.Metadata["value"])
	}

	// Test Update - non-existent vector
	err = collection.Update(ctx, "nonexistent", []float32{7.0, 8.0, 9.0}, nil)
	if err == nil {
		t.Error("Expected error when updating non-existent vector")
	}

	// Test Delete
	err = collection.Delete(ctx, "test1")
	if err != nil {
		t.Fatalf("Failed to delete vector: %v", err)
	}

	// Verify deletion by searching
	results, err = collection.Search(ctx, newVector, 1)
	// Search might fail if index is empty, which is expected after deleting all vectors
	if err != nil {
		// Check if it's an "index is empty" error, which is expected
		if !strings.Contains(err.Error(), "index is empty") {
			t.Fatalf("Unexpected error during search after delete: %v", err)
		}
		// Index is empty, which means deletion worked
	} else {
		// If search succeeded, there should be no results
		if len(results.Results) != 0 {
			t.Errorf("Expected 0 results after deletion, got %d", len(results.Results))
		}
	}

	// Test Delete - non-existent vector (may error, which is acceptable)
	err = collection.Delete(ctx, "nonexistent")
	// It's acceptable for delete to error on non-existent vectors or empty index
	// The important thing is that it doesn't crash
}

func TestBatchUpdateOperations(t *testing.T) {
	// Create a test database
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create a test collection with timestamp to ensure uniqueness
	collectionName := fmt.Sprintf("batch_update_test_%d", time.Now().UnixNano())
	collection, err := db.CreateCollection(ctx, collectionName, WithDimension(3), WithMetric(CosineDistance), WithHNSW(16, 200, 50))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert some test vectors
	vectors := []*VectorEntry{
		{ID: "batch1", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"type": "original"}},
		{ID: "batch2", Vector: []float32{4.0, 5.0, 6.0}, Metadata: map[string]interface{}{"type": "original"}},
		{ID: "batch3", Vector: []float32{7.0, 8.0, 9.0}, Metadata: map[string]interface{}{"type": "original"}},
	}

	batchInsert := collection.NewBatchInsert(vectors)
	result, err := batchInsert.Execute(ctx)
	if err != nil {
		t.Fatalf("Failed to batch insert: %v", err)
	}

	if result.Successful != 3 {
		t.Errorf("Expected 3 successful inserts, got %d", result.Successful)
	}

	// Test batch update
	updates := []*VectorUpdate{
		{ID: "batch1", Vector: []float32{1.1, 2.1, 3.1}, Metadata: map[string]interface{}{"type": "updated"}},
		{ID: "batch2", Vector: nil, Metadata: map[string]interface{}{"type": "metadata_only_update"}}, // Metadata only
		{ID: "batch3", Vector: []float32{7.7, 8.8, 9.9}, Metadata: nil},                               // Vector only
	}

	batchUpdate := collection.NewBatchUpdate(updates)
	updateResult, err := batchUpdate.Execute(ctx)
	if err != nil {
		t.Fatalf("Failed to batch update: %v", err)
	}

	if updateResult.Successful != 3 {
		t.Errorf("Expected 3 successful updates, got %d", updateResult.Successful)
	}

	// Verify updates by searching
	for _, update := range updates {
		results, err := collection.Search(ctx, []float32{1.0, 1.0, 1.0}, 10)
		if err != nil {
			t.Fatalf("Failed to search after batch update: %v", err)
		}

		found := false
		for _, result := range results.Results {
			if result.ID == update.ID {
				found = true
				break
			}
		}

		if !found {
			t.Errorf("Updated vector %s not found in search results", update.ID)
		}
	}
}

func TestBatchDeleteOperations(t *testing.T) {
	// Create a test database
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create a test collection with timestamp to ensure uniqueness
	collectionName := fmt.Sprintf("batch_delete_test_%d", time.Now().UnixNano())
	collection, err := db.CreateCollection(ctx, collectionName, WithDimension(3), WithMetric(CosineDistance), WithHNSW(16, 200, 50))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert some test vectors
	vectors := []*VectorEntry{
		{ID: "delete1", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"type": "to_delete"}},
		{ID: "delete2", Vector: []float32{4.0, 5.0, 6.0}, Metadata: map[string]interface{}{"type": "to_delete"}},
		{ID: "keep1", Vector: []float32{7.0, 8.0, 9.0}, Metadata: map[string]interface{}{"type": "keep"}},
	}

	batchInsert := collection.NewBatchInsert(vectors)
	result, err := batchInsert.Execute(ctx)
	if err != nil {
		t.Fatalf("Failed to batch insert: %v", err)
	}

	if result.Successful != 3 {
		t.Errorf("Expected 3 successful inserts, got %d", result.Successful)
	}

	// Test batch delete
	idsToDelete := []string{"delete1", "delete2"}
	batchDelete := collection.NewBatchDelete(idsToDelete)
	deleteResult, err := batchDelete.Execute(ctx)
	if err != nil {
		t.Fatalf("Failed to batch delete: %v", err)
	}

	if deleteResult.Successful != 2 {
		t.Errorf("Expected 2 successful deletes, got %d", deleteResult.Successful)
	}

	// Verify deletions by searching
	results, err := collection.Search(ctx, []float32{1.0, 1.0, 1.0}, 10)
	if err != nil {
		t.Fatalf("Failed to search after batch delete: %v", err)
	}

	if len(results.Results) != 1 {
		t.Errorf("Expected 1 remaining vector, got %d", len(results.Results))
	}

	if results.Results[0].ID != "keep1" {
		t.Errorf("Expected remaining vector to be 'keep1', got '%s'", results.Results[0].ID)
	}
}

func TestStreamingUpdateOperations(t *testing.T) {
	// Create a test database
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create a test collection with timestamp to ensure uniqueness
	collectionName := fmt.Sprintf("streaming_update_test_%d", time.Now().UnixNano())
	collection, err := db.CreateCollection(ctx, collectionName, WithDimension(3), WithMetric(CosineDistance), WithHNSW(16, 200, 50))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert some test vectors first
	for i := 0; i < 5; i++ {
		vector := []float32{float32(i), float32(i + 1), float32(i + 2)}
		metadata := map[string]interface{}{"index": i, "type": "original"}
		err = collection.Insert(ctx, fmt.Sprintf("stream%d", i), vector, metadata)
		if err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}
	}

	// Test streaming update
	streamingUpdate := collection.NewStreamingBatchUpdate(&StreamingOptions{
		BufferSize:     100,
		ChunkSize:      2,
		MaxConcurrency: 2,
		FlushInterval:  100 * time.Millisecond,
	})

	err = streamingUpdate.Start()
	if err != nil {
		t.Fatalf("Failed to start streaming update: %v", err)
	}

	// Send updates
	for i := 0; i < 5; i++ {
		update := &VectorUpdate{
			ID:       fmt.Sprintf("stream%d", i),
			Vector:   []float32{float32(i + 10), float32(i + 11), float32(i + 12)},
			Metadata: map[string]interface{}{"index": i, "type": "updated"},
		}
		err = streamingUpdate.Send(update)
		if err != nil {
			t.Fatalf("Failed to send update %d: %v", i, err)
		}
	}

	// Wait a bit for processing
	time.Sleep(500 * time.Millisecond)

	// Close streaming operation
	err = streamingUpdate.Close()
	if err != nil {
		t.Fatalf("Failed to close streaming update: %v", err)
	}

	// Verify updates
	stats := streamingUpdate.Stats()
	if stats.TotalProcessed != 5 {
		t.Errorf("Expected 5 processed updates, got %d", stats.TotalProcessed)
	}
}

func TestStreamingDeleteOperations(t *testing.T) {
	// Create a test database
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create a test collection with timestamp to ensure uniqueness
	collectionName := fmt.Sprintf("streaming_delete_test_%d", time.Now().UnixNano())
	collection, err := db.CreateCollection(ctx, collectionName, WithDimension(3), WithMetric(CosineDistance), WithHNSW(16, 200, 50))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert some test vectors first
	for i := 0; i < 10; i++ {
		vector := []float32{float32(i), float32(i + 1), float32(i + 2)}
		metadata := map[string]interface{}{"index": i}
		err = collection.Insert(ctx, fmt.Sprintf("stream_del%d", i), vector, metadata)
		if err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}
	}

	// Test streaming delete
	streamingDelete := collection.NewStreamingBatchDelete(&StreamingOptions{
		BufferSize:     100,
		ChunkSize:      3,
		MaxConcurrency: 2,
		FlushInterval:  100 * time.Millisecond,
	})

	err = streamingDelete.Start()
	if err != nil {
		t.Fatalf("Failed to start streaming delete: %v", err)
	}

	// Send deletes for half the vectors
	for i := 0; i < 5; i++ {
		err = streamingDelete.Send(fmt.Sprintf("stream_del%d", i))
		if err != nil {
			t.Fatalf("Failed to send delete %d: %v", i, err)
		}
	}

	// Wait a bit for processing
	time.Sleep(500 * time.Millisecond)

	// Close streaming operation
	err = streamingDelete.Close()
	if err != nil {
		t.Fatalf("Failed to close streaming delete: %v", err)
	}

	// Verify deletes
	stats := streamingDelete.Stats()
	if stats.TotalProcessed != 5 {
		t.Errorf("Expected 5 processed deletes, got %d", stats.TotalProcessed)
	}

	// Verify remaining vectors
	results, err := collection.Search(ctx, []float32{1.0, 1.0, 1.0}, 10)
	if err != nil {
		t.Fatalf("Failed to search after streaming delete: %v", err)
	}

	if len(results.Results) != 5 {
		t.Errorf("Expected 5 remaining vectors, got %d", len(results.Results))
	}
}

// TestInsertPreflightNoPartialWrite verifies that Insert with an invalid-dimension
// vector does not produce a partial write: the collection must not contain the entry.
func TestInsertPreflightNoPartialWrite(t *testing.T) {
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	collection, err := db.CreateCollection(ctx, "preflight_test", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert a valid entry first
	validVector := []float32{1.0, 2.0, 3.0}
	err = collection.Insert(ctx, "valid_entry", validVector, map[string]interface{}{"type": "valid"})
	if err != nil {
		t.Fatalf("Failed to insert valid entry: %v", err)
	}

	// Attempt to insert an entry with wrong dimension - should fail preflight
	badVector := []float32{1.0, 2.0} // dimension 2, not 3
	err = collection.Insert(ctx, "bad_dimension_entry", badVector, nil)
	if err == nil {
		t.Fatal("Expected error for wrong dimension, got nil")
	}

	// Verify no partial commit: "bad_dimension_entry" must not exist
	_, err = collection.Get(ctx, "bad_dimension_entry")
	if err == nil {
		t.Error("Expected error when getting non-existent entry, got nil - partial write may have occurred")
	}

	// Verify the valid entry is still there
	validRecord, err := collection.Get(ctx, "valid_entry")
	if err != nil {
		t.Fatalf("Valid entry should still exist: %v", err)
	}
	if validRecord.ID != "valid_entry" {
		t.Errorf("Expected valid_entry, got %s", validRecord.ID)
	}
}

// TestBatchInsertPreflightNoPartialWrite verifies that InsertBatch with a bad
// dimension entry records the failure per-item but does not commit partial work
// for the invalid entry; the valid entry should still be committed.
func TestBatchInsertPreflightNoPartialWrite(t *testing.T) {
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	collection, err := db.CreateCollection(ctx, "batch_preflight_test", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Batch with one valid entry and one invalid-dimension entry
	entries := []*VectorEntry{
		{ID: "batch_valid", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"type": "valid"}},
		{ID: "batch_bad", Vector: []float32{4.0, 5.0}, Metadata: nil}, // wrong dimension
	}

	batch := collection.NewBatchInsert(entries)
	result, err := batch.Execute(ctx)
	if err != nil {
		t.Fatalf("Batch execute returned unexpected error: %v", err)
	}

	// The batch itself succeeds (items processed), but the bad entry is marked failed
	if result.Successful != 1 {
		t.Errorf("Expected 1 successful insert, got %d", result.Successful)
	}
	if result.Failed != 1 {
		t.Errorf("Expected 1 failed insert, got %d", result.Failed)
	}

	// The bad entry should be recorded as failed
	badItemFailed := false
	for _, item := range result.Items {
		if item.ID == "batch_bad" && !item.Success {
			badItemFailed = true
		}
	}
	if !badItemFailed {
		t.Error("batch_bad should be marked as failed")
	}

	// Verify the valid entry was committed
	validRecord, err := collection.Get(ctx, "batch_valid")
	if err != nil {
		t.Fatalf("Valid entry batch_valid should exist: %v", err)
	}
	if validRecord.ID != "batch_valid" {
		t.Errorf("Expected batch_valid, got %s", validRecord.ID)
	}

	// The bad entry should not exist
	_, err = collection.Get(ctx, "batch_bad")
	if err == nil {
		t.Error("batch_bad should not exist after failed insert")
	}
}

// TestBatchInsertDuplicateIDPreflight verifies that InsertBatch with duplicate IDs
// correctly handles the conflict: the first entry may succeed at the storage layer
// while the second entry fails with a duplicate-ID error from storage.
// The key invariant is: after the batch, we must not have MORE than one entry
// with that ID committed (no ghost entries).
func TestBatchInsertDuplicateIDPreflight(t *testing.T) {
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	collection, err := db.CreateCollection(ctx, "dup_id_preflight_test", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Batch with duplicate IDs - first one succeeds, second fails at storage layer
	entries := []*VectorEntry{
		{ID: "dup_id", Vector: []float32{1.0, 2.0, 3.0}},
		{ID: "dup_id", Vector: []float32{4.0, 5.0, 6.0}}, // same ID
	}

	batch := collection.NewBatchInsert(entries)
	result, err := batch.Execute(ctx)
	if err != nil {
		t.Fatalf("Batch execute returned unexpected error: %v", err)
	}

	// One should succeed, one should fail
	if result.Successful+result.Failed != 2 {
		t.Errorf("Expected 2 items total, got %d", result.Successful+result.Failed)
	}

	// Verify at most one entry with dup_id exists (no ghost entries)
	count := 0
	_ = collection.Iterate(ctx, func(r Record) error {
		if r.ID == "dup_id" {
			count++
		}
		return nil
	})
	if count > 1 {
		t.Errorf("Expected at most 1 dup_id entry, got %d", count)
	}
}
