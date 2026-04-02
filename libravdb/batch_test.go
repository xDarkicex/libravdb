package libravdb

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

// createTestDB creates a unique database for testing
func createTestDB(t *testing.T) *Database {
	tmpDir, err := os.MkdirTemp("", "libravdb_test_*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}

	db, err := New(WithStoragePath(filepath.Join(tmpDir, "test.libravdb")))
	if err != nil {
		os.RemoveAll(tmpDir)
		t.Fatalf("Failed to create database: %v", err)
	}

	// Clean up when test completes
	t.Cleanup(func() {
		db.Close()
		os.RemoveAll(tmpDir)
	})

	return db
}

func TestBatchInsert_Basic(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_basic", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create test entries
	entries := []*VectorEntry{
		{ID: "1", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"type": "test"}},
		{ID: "2", Vector: []float32{4.0, 5.0, 6.0}, Metadata: map[string]interface{}{"type": "test"}},
		{ID: "3", Vector: []float32{7.0, 8.0, 9.0}, Metadata: map[string]interface{}{"type": "test"}},
	}

	// Create batch insert
	batch := collection.NewBatchInsert(entries)

	// Execute batch
	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch insert failed: %v", err)
	}

	// Verify results
	if result.Successful != 3 {
		t.Errorf("Expected 3 successful inserts, got %d", result.Successful)
	}
	if result.Failed != 0 {
		t.Errorf("Expected 0 failed inserts, got %d", result.Failed)
	}
	if len(result.Errors) != 0 {
		t.Errorf("Expected no errors, got %d", len(result.Errors))
	}
	if len(result.Items) != 3 {
		t.Errorf("Expected 3 item results, got %d", len(result.Items))
	}

	// Verify all items were successful
	for i, item := range result.Items {
		if !item.Success {
			t.Errorf("Item %d should be successful", i)
		}
		if item.Error != nil {
			t.Errorf("Item %d should not have error: %v", i, item.Error)
		}
	}
}

func TestBatchInsert_WithErrors(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_errors", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create test entries with some invalid ones
	entries := []*VectorEntry{
		{ID: "1", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"type": "test"}},
		{ID: "", Vector: []float32{4.0, 5.0, 6.0}, Metadata: map[string]interface{}{"type": "test"}}, // Invalid: empty ID
		{ID: "3", Vector: []float32{7.0, 8.0}, Metadata: map[string]interface{}{"type": "test"}},     // Invalid: wrong dimension
		{ID: "4", Vector: []float32{10.0, 11.0, 12.0}, Metadata: map[string]interface{}{"type": "test"}},
	}

	// Create batch insert
	batch := collection.NewBatchInsert(entries)

	// Execute batch
	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch insert failed: %v", err)
	}

	// Verify results
	if result.Successful != 2 {
		t.Errorf("Expected 2 successful inserts, got %d", result.Successful)
	}
	if result.Failed != 2 {
		t.Errorf("Expected 2 failed inserts, got %d", result.Failed)
	}
	if len(result.Errors) != 2 {
		t.Errorf("Expected 2 errors, got %d", len(result.Errors))
	}

	// Check specific errors
	if _, exists := result.Errors[1]; !exists {
		t.Error("Expected error for item 1 (empty ID)")
	}
	if _, exists := result.Errors[2]; !exists {
		t.Error("Expected error for item 2 (wrong dimension)")
	}
}

func TestBatchInsert_Chunking(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_chunking", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create many test entries
	entries := make([]*VectorEntry, 100)
	for i := 0; i < 100; i++ {
		entries[i] = &VectorEntry{
			ID:       fmt.Sprintf("item_%d", i),
			Vector:   []float32{float32(i), float32(i + 1), float32(i + 2)},
			Metadata: map[string]interface{}{"index": i},
		}
	}

	// Create batch insert with small chunk size
	options := &BatchOptions{
		ChunkSize:      10,
		MaxConcurrency: 2,
	}
	batch := collection.NewBatchInsert(entries, options)

	// Execute batch
	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch insert failed: %v", err)
	}

	// Verify results
	if result.Successful != 100 {
		t.Errorf("Expected 100 successful inserts, got %d", result.Successful)
	}
	if result.Failed != 0 {
		t.Errorf("Expected 0 failed inserts, got %d", result.Failed)
	}
}

func TestBatchInsert_ConcurrentChunkOrdering(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_concurrent_chunk_ordering", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	entries := make([]*VectorEntry, 24)
	for i := 0; i < len(entries); i++ {
		entries[i] = &VectorEntry{
			ID:       fmt.Sprintf("item_%02d", i),
			Vector:   []float32{float32(i), float32(i + 1), float32(i + 2)},
			Metadata: map[string]interface{}{"index": i},
		}
	}

	batch := collection.NewBatchInsert(entries, &BatchOptions{
		ChunkSize:      3,
		MaxConcurrency: 4,
	})

	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch insert failed: %v", err)
	}

	if result.Successful != len(entries) {
		t.Fatalf("Expected %d successful inserts, got %d", len(entries), result.Successful)
	}

	for i, item := range result.Items {
		if item.Index != i {
			t.Fatalf("Expected item result index %d, got %d", i, item.Index)
		}
	}
}

func TestBatchInsert_ProgressCallback(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_progress", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create test entries
	entries := make([]*VectorEntry, 50)
	for i := 0; i < 50; i++ {
		entries[i] = &VectorEntry{
			ID:       fmt.Sprintf("item_%d", i),
			Vector:   []float32{float32(i), float32(i + 1), float32(i + 2)},
			Metadata: map[string]interface{}{"index": i},
		}
	}

	// Track progress
	var progressCalls int32
	var lastCompleted, lastTotal int

	options := &BatchOptions{
		ChunkSize:      10,
		MaxConcurrency: 1, // Use single worker to ensure predictable progress
		ProgressCallback: func(completed, total int) {
			atomic.AddInt32(&progressCalls, 1)
			lastCompleted = completed
			lastTotal = total
		},
	}

	batch := collection.NewBatchInsert(entries, options)

	// Execute batch
	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch insert failed: %v", err)
	}

	// Verify progress was tracked
	if atomic.LoadInt32(&progressCalls) == 0 {
		t.Error("Progress callback was never called")
	}
	if lastTotal != 50 {
		t.Errorf("Expected total to be 50, got %d", lastTotal)
	}
	if lastCompleted != 50 {
		t.Errorf("Expected final completed to be 50, got %d", lastCompleted)
	}

	// Verify final result
	if result.Successful != 50 {
		t.Errorf("Expected 50 successful inserts, got %d", result.Successful)
	}
}

func TestBatchInsert_Timeout(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_timeout", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create test entries
	entries := []*VectorEntry{
		{ID: "1", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"type": "test"}},
	}

	// Create batch with reasonable timeout
	options := &BatchOptions{
		Timeout: 100 * time.Millisecond, // Short but reasonable timeout
	}
	batch := collection.NewBatchInsert(entries, options)

	// Execute batch - should complete successfully since it's a simple operation
	ctx := context.Background()
	result, err := batch.Execute(ctx)
	if err != nil {
		t.Fatalf("Batch insert failed: %v", err)
	}

	// Verify the batch completed successfully
	if result.Successful != 1 {
		t.Errorf("Expected 1 successful insert, got %d", result.Successful)
	}
}

func TestBatchInsert_MemoryEstimation(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_memory", WithDimension(100))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create test entries
	entries := []*VectorEntry{
		{ID: "1", Vector: make([]float32, 100), Metadata: map[string]interface{}{"type": "test"}},
		{ID: "2", Vector: make([]float32, 100), Metadata: map[string]interface{}{"type": "test"}},
	}

	batch := collection.NewBatchInsert(entries)

	// Test memory estimation
	memory := batch.EstimateMemory()
	if memory <= 0 {
		t.Error("Memory estimation should be positive")
	}

	// Should be roughly: 2 entries * (100 floats * 4 bytes + metadata + overhead)
	expectedMin := int64(2 * (100*4 + 100 + 50))
	if memory < expectedMin {
		t.Errorf("Memory estimation seems too low: got %d, expected at least %d", memory, expectedMin)
	}
}

func TestBatchUpdate_Basic(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_update_basic", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create test updates (upsert mode for now since update isn't implemented)
	updates := []*VectorUpdate{
		{ID: "1", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"type": "test"}, Upsert: true},
		{ID: "2", Vector: []float32{4.0, 5.0, 6.0}, Metadata: map[string]interface{}{"type": "test"}, Upsert: true},
	}

	batch := collection.NewBatchUpdate(updates)

	// Execute batch
	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch update failed: %v", err)
	}

	// Since upsert uses insert, these should succeed
	if result.Successful != 2 {
		t.Errorf("Expected 2 successful updates, got %d", result.Successful)
	}
}

func TestBatchUpdate_Validation(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_update_validation", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create invalid updates
	updates := []*VectorUpdate{
		{ID: "", Vector: []float32{1.0, 2.0, 3.0}, Upsert: true},                  // Invalid: empty ID
		{ID: "2", Vector: []float32{4.0, 5.0}, Upsert: true},                      // Invalid: wrong dimension
		{ID: "3", Vector: []float32{7.0, 8.0, 9.0}, Metadata: nil, Upsert: false}, // Invalid: not upsert and no update impl
	}

	batch := collection.NewBatchUpdate(updates)

	// Execute batch
	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch update failed: %v", err)
	}

	// All should fail due to validation or missing implementation
	if result.Failed != 3 {
		t.Errorf("Expected 3 failed updates, got %d", result.Failed)
	}
	if result.Successful != 0 {
		t.Errorf("Expected 0 successful updates, got %d", result.Successful)
	}
}

func TestBatchDelete_Basic(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_delete_basic", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create test deletes
	ids := []string{"1", "2", "3"}

	batch := collection.NewBatchDelete(ids)

	// Execute batch
	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch delete failed: %v", err)
	}

	// Since delete isn't implemented yet, all should fail
	if result.Failed != 3 {
		t.Errorf("Expected 3 failed deletes, got %d", result.Failed)
	}
	if result.Successful != 0 {
		t.Errorf("Expected 0 successful deletes, got %d", result.Successful)
	}
}

func TestBatchDelete_Validation(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_delete_validation", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create invalid deletes
	ids := []string{"1", "", "3"} // Empty ID in the middle

	batch := collection.NewBatchDelete(ids)

	// Execute batch
	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch delete failed: %v", err)
	}

	// All should fail - empty ID validation + missing delete implementation
	if result.Failed != 3 {
		t.Errorf("Expected 3 failed deletes, got %d", result.Failed)
	}

	// Check that empty ID error is properly reported
	if _, exists := result.Errors[1]; !exists {
		t.Error("Expected error for empty ID at index 1")
	}
}

func TestWorkerPool_Basic(t *testing.T) {
	pool := newWorkerPool(2)
	defer pool.close()

	// Submit some jobs
	var counter int32
	for i := 0; i < 10; i++ {
		pool.submit(func() error {
			atomic.AddInt32(&counter, 1)
			return nil
		})
	}

	// Wait for completion
	err := pool.wait(context.Background())
	if err != nil {
		t.Fatalf("Worker pool wait failed: %v", err)
	}

	// Verify all jobs were executed
	if atomic.LoadInt32(&counter) != 10 {
		t.Errorf("Expected 10 jobs executed, got %d", atomic.LoadInt32(&counter))
	}
}

func TestWorkerPool_WithErrors(t *testing.T) {
	pool := newWorkerPool(2)
	defer pool.close()

	// Submit jobs - errors are handled within the jobs themselves in this implementation
	var errorCount int32
	for i := 0; i < 5; i++ {
		i := i
		pool.submit(func() error {
			if i == 2 {
				atomic.AddInt32(&errorCount, 1)
			}
			return nil
		})
	}

	// Wait for completion
	err := pool.wait(context.Background())
	if err != nil {
		t.Errorf("Unexpected error from worker pool: %v", err)
	}

	// Verify that the error job was processed
	if atomic.LoadInt32(&errorCount) != 1 {
		t.Errorf("Expected 1 error job to be processed, got %d", atomic.LoadInt32(&errorCount))
	}
}

func TestWorkerPool_ContextCancellation(t *testing.T) {
	pool := newWorkerPool(1)
	defer pool.close()

	// Submit a job
	pool.submit(func() error {
		time.Sleep(100 * time.Millisecond)
		return nil
	})

	// Create context with short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	// Wait should timeout
	err := pool.wait(ctx)
	if err != context.DeadlineExceeded {
		t.Errorf("Expected context deadline exceeded, got %v", err)
	}
}

func TestDefaultBatchOptions(t *testing.T) {
	opts := DefaultBatchOptions()

	if opts.ChunkSize != 1000 {
		t.Errorf("Expected default chunk size 1000, got %d", opts.ChunkSize)
	}
	if opts.MaxConcurrency != 4 {
		t.Errorf("Expected default max concurrency 4, got %d", opts.MaxConcurrency)
	}
	if opts.FailFast != false {
		t.Errorf("Expected default fail fast false, got %t", opts.FailFast)
	}
	if opts.Timeout != 5*time.Minute {
		t.Errorf("Expected default timeout 5 minutes, got %v", opts.Timeout)
	}
	if opts.EnableRollback != false {
		t.Errorf("Expected default enable rollback false, got %t", opts.EnableRollback)
	}
	if opts.MaxRetries != 3 {
		t.Errorf("Expected default max retries 3, got %d", opts.MaxRetries)
	}
	if opts.RetryDelay != 100*time.Millisecond {
		t.Errorf("Expected default retry delay 100ms, got %v", opts.RetryDelay)
	}
}

// Test enhanced error handling and progress tracking
func TestBatchInsert_EnhancedErrorHandling(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_enhanced_errors", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create entries with various error conditions
	entries := []*VectorEntry{
		{ID: "1", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"type": "test"}},
		{ID: "", Vector: []float32{4.0, 5.0, 6.0}, Metadata: map[string]interface{}{"type": "test"}}, // Validation error
		{ID: "3", Vector: []float32{7.0, 8.0}, Metadata: map[string]interface{}{"type": "test"}},     // Dimension error
		{ID: "4", Vector: []float32{10.0, 11.0, 12.0}, Metadata: map[string]interface{}{"type": "test"}},
	}

	// Track error callbacks
	var errorCallbacks []string
	options := &BatchOptions{
		ChunkSize:  2,
		MaxRetries: 1,
		RetryDelay: 10 * time.Millisecond,
		ErrorCallback: func(item *BatchItemResult, err error) {
			errorCallbacks = append(errorCallbacks, fmt.Sprintf("Item %d: %s", item.Index, item.BatchErrorCode))
		},
	}

	batch := collection.NewBatchInsert(entries, options)
	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch insert failed: %v", err)
	}

	// Verify error categorization
	if result.Successful != 2 {
		t.Errorf("Expected 2 successful inserts, got %d", result.Successful)
	}
	if result.Failed != 2 {
		t.Errorf("Expected 2 failed inserts, got %d", result.Failed)
	}

	// Check error codes
	for _, item := range result.Items {
		if !item.Success {
			if item.BatchErrorCode == "" {
				t.Errorf("Item %d should have an error code", item.Index)
			}
			if item.Timestamp.IsZero() {
				t.Errorf("Item %d should have a timestamp", item.Index)
			}
		}
	}

	// Verify error callbacks were called
	if len(errorCallbacks) != 2 {
		t.Errorf("Expected 2 error callbacks, got %d", len(errorCallbacks))
	}
}

func TestBatchInsert_DetailedProgressTracking(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_detailed_progress", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create test entries
	entries := make([]*VectorEntry, 20)
	for i := 0; i < 20; i++ {
		entries[i] = &VectorEntry{
			ID:       fmt.Sprintf("item_%d", i),
			Vector:   []float32{float32(i), float32(i + 1), float32(i + 2)},
			Metadata: map[string]interface{}{"index": i},
		}
	}

	// Track detailed progress
	var progressUpdates []*BatchProgress
	options := &BatchOptions{
		ChunkSize: 5,
		DetailedProgress: func(progress *BatchProgress) {
			// Make a copy to avoid race conditions
			progressCopy := *progress
			progressUpdates = append(progressUpdates, &progressCopy)
		},
	}

	batch := collection.NewBatchInsert(entries, options)
	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch insert failed: %v", err)
	}

	// Verify progress tracking
	if len(progressUpdates) == 0 {
		t.Error("Expected progress updates")
	}

	lastProgress := progressUpdates[len(progressUpdates)-1]
	if lastProgress.Completed != 20 {
		t.Errorf("Expected final completed to be 20, got %d", lastProgress.Completed)
	}
	if lastProgress.Total != 20 {
		t.Errorf("Expected total to be 20, got %d", lastProgress.Total)
	}
	if lastProgress.TotalChunks != 4 {
		t.Errorf("Expected 4 total chunks, got %d", lastProgress.TotalChunks)
	}

	// Verify final result
	if result.Successful != 20 {
		t.Errorf("Expected 20 successful inserts, got %d", result.Successful)
	}
}

func TestBatchInsert_RetryMechanism(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_retry", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create entries with validation errors (non-retryable)
	entries := []*VectorEntry{
		{ID: "1", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"type": "test"}},
		{ID: "", Vector: []float32{4.0, 5.0, 6.0}, Metadata: map[string]interface{}{"type": "test"}}, // Non-retryable
	}

	options := &BatchOptions{
		MaxRetries: 2,
		RetryDelay: 10 * time.Millisecond,
	}

	batch := collection.NewBatchInsert(entries, options)
	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch insert failed: %v", err)
	}

	// Verify retry counts
	if result.Items[0].Retries != 0 {
		t.Errorf("Successful item should have 0 retries, got %d", result.Items[0].Retries)
	}
	if result.Items[1].Retries != 0 {
		t.Errorf("Non-retryable error should have 0 retries, got %d", result.Items[1].Retries)
	}
}

func TestBatchInsert_RollbackMechanism(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_rollback", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create entries with some failures
	entries := []*VectorEntry{
		{ID: "1", Vector: []float32{1.0, 2.0, 3.0}, Metadata: map[string]interface{}{"type": "test"}},
		{ID: "2", Vector: []float32{4.0, 5.0, 6.0}, Metadata: map[string]interface{}{"type": "test"}},
		{ID: "", Vector: []float32{7.0, 8.0, 9.0}, Metadata: map[string]interface{}{"type": "test"}}, // This will fail
	}

	options := &BatchOptions{
		EnableRollback: true,
		FailFast:       true,
	}

	batch := collection.NewBatchInsert(entries, options)
	result, err := batch.Execute(context.Background())
	if err == nil {
		t.Fatal("Expected strict rollback batch insert to return an error")
	}

	// Verify rollback was attempted (even though delete isn't implemented yet)
	if !result.RollbackRequired {
		t.Error("Expected rollback to be required")
	}

	// The rollback error should be nil since we don't have actual delete implementation
	if result.RollbackError != nil {
		t.Errorf("Unexpected rollback error: %v", result.RollbackError)
	}
}

func TestBatchUpdate_EnhancedErrorHandling(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_update_enhanced", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create updates with various error conditions
	updates := []*VectorUpdate{
		{ID: "1", Vector: []float32{1.0, 2.0, 3.0}, Upsert: true},
		{ID: "", Vector: []float32{4.0, 5.0, 6.0}, Upsert: true},                     // Validation error
		{ID: "3", Vector: []float32{7.0, 8.0}, Upsert: true},                         // Dimension error
		{ID: "4", Vector: []float32{10.0, 11.0, 12.0}, Metadata: nil, Upsert: false}, // Not implemented error
	}

	var errorCallbacks []string
	options := &BatchOptions{
		ErrorCallback: func(item *BatchItemResult, err error) {
			errorCallbacks = append(errorCallbacks, fmt.Sprintf("Update %d: %s", item.Index, item.BatchErrorCode))
		},
	}

	batch := collection.NewBatchUpdate(updates, options)
	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch update failed: %v", err)
	}

	// Verify error categorization
	if result.Successful != 1 {
		t.Errorf("Expected 1 successful update, got %d", result.Successful)
	}
	if result.Failed != 3 {
		t.Errorf("Expected 3 failed updates, got %d", result.Failed)
	}

	// Verify error callbacks
	if len(errorCallbacks) != 3 {
		t.Errorf("Expected 3 error callbacks, got %d", len(errorCallbacks))
	}
}

func TestBatchDelete_EnhancedErrorHandling(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_delete_enhanced", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create deletes with error conditions
	ids := []string{"1", "", "3"} // Empty ID will cause validation error

	var errorCallbacks []string
	options := &BatchOptions{
		ErrorCallback: func(item *BatchItemResult, err error) {
			errorCallbacks = append(errorCallbacks, fmt.Sprintf("Delete %d: %s", item.Index, item.BatchErrorCode))
		},
	}

	batch := collection.NewBatchDelete(ids, options)
	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch delete failed: %v", err)
	}

	// All should fail (validation + not implemented)
	if result.Failed != 3 {
		t.Errorf("Expected 3 failed deletes, got %d", result.Failed)
	}

	// Verify error callbacks
	if len(errorCallbacks) != 3 {
		t.Errorf("Expected 3 error callbacks, got %d", len(errorCallbacks))
	}
}

func TestProgressTracker(t *testing.T) {
	tracker := newProgressTracker(100, 10)

	// Initial state
	progress := tracker.getProgress()
	if progress.Total != 100 {
		t.Errorf("Expected total 100, got %d", progress.Total)
	}
	if progress.TotalChunks != 10 {
		t.Errorf("Expected 10 chunks, got %d", progress.TotalChunks)
	}

	// Add a small delay to ensure measurable elapsed time
	time.Sleep(10 * time.Millisecond)

	// Update progress
	tracker.update(50, 45, 5, 5, nil)
	progress = tracker.getProgress()

	if progress.Completed != 50 {
		t.Errorf("Expected completed 50, got %d", progress.Completed)
	}
	if progress.Successful != 45 {
		t.Errorf("Expected successful 45, got %d", progress.Successful)
	}
	if progress.Failed != 5 {
		t.Errorf("Expected failed 5, got %d", progress.Failed)
	}
	if progress.CurrentChunk != 5 {
		t.Errorf("Expected current chunk 5, got %d", progress.CurrentChunk)
	}

	// Verify ETA calculation
	if progress.ItemsPerSec <= 0 {
		t.Error("Items per second should be positive")
	}
	// ETA should be positive only if there are remaining items and we have a rate
	if progress.Completed < progress.Total && progress.ItemsPerSec > 0 && progress.EstimatedETA <= 0 {
		t.Errorf("ETA should be positive when there are remaining items and we have a processing rate. Completed: %d, Total: %d, ItemsPerSec: %f, ETA: %v",
			progress.Completed, progress.Total, progress.ItemsPerSec, progress.EstimatedETA)
	}
}

// TestBatchOperations_ComprehensiveIntegration tests all enhanced features together
func TestBatchOperations_ComprehensiveIntegration(t *testing.T) {
	db := createTestDB(t)

	collection, err := db.CreateCollection(context.Background(), "test_comprehensive", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create entries with mixed success/failure scenarios
	entries := make([]*VectorEntry, 25)
	for i := 0; i < 25; i++ {
		var entry *VectorEntry
		switch {
		case i%10 == 1: // Every 10th item starting at 1 has empty ID (validation error)
			entry = &VectorEntry{ID: "", Vector: []float32{float32(i), float32(i + 1), float32(i + 2)}}
		case i%10 == 2: // Every 10th item starting at 2 has wrong dimension
			entry = &VectorEntry{ID: fmt.Sprintf("item_%d", i), Vector: []float32{float32(i), float32(i + 1)}}
		default: // Valid entries
			entry = &VectorEntry{
				ID:       fmt.Sprintf("item_%d", i),
				Vector:   []float32{float32(i), float32(i + 1), float32(i + 2)},
				Metadata: map[string]interface{}{"index": i, "batch": "comprehensive"},
			}
		}
		entries[i] = entry
	}

	// Track all callbacks and progress
	var progressUpdates []*BatchProgress
	var errorCallbacks []string
	var simpleProgressCalls []int

	options := &BatchOptions{
		ChunkSize:      5,
		MaxRetries:     2,
		RetryDelay:     10 * time.Millisecond,
		EnableRollback: true,
		FailFast:       false, // Process all items
		ProgressCallback: func(completed, total int) {
			simpleProgressCalls = append(simpleProgressCalls, completed)
		},
		DetailedProgress: func(progress *BatchProgress) {
			// Make a copy to avoid race conditions
			progressCopy := *progress
			progressUpdates = append(progressUpdates, &progressCopy)
		},
		ErrorCallback: func(item *BatchItemResult, err error) {
			errorCallbacks = append(errorCallbacks, fmt.Sprintf("Item %d (%s): %s - %s",
				item.Index, item.ID, item.BatchErrorCode, err.Error()))
		},
	}

	batch := collection.NewBatchInsert(entries, options)

	// Execute the comprehensive batch
	result, err := batch.Execute(context.Background())
	if err != nil {
		t.Fatalf("Batch insert failed: %v", err)
	}

	// Verify comprehensive results
	expectedSuccessful := 19 // 25 total - 6 failures (items 1, 2, 11, 12, 21, 22)
	expectedFailed := 6

	if result.Successful != expectedSuccessful {
		t.Errorf("Expected %d successful inserts, got %d", expectedSuccessful, result.Successful)
	}
	if result.Failed != expectedFailed {
		t.Errorf("Expected %d failed inserts, got %d", expectedFailed, result.Failed)
	}

	// Verify error categorization
	validationErrors := 0
	for _, item := range result.Items {
		if !item.Success && item.BatchErrorCode == BatchErrorValidation {
			validationErrors++
		}
	}
	if validationErrors != expectedFailed {
		t.Errorf("Expected %d validation errors, got %d", expectedFailed, validationErrors)
	}

	// Verify progress tracking
	if len(progressUpdates) == 0 {
		t.Error("Expected detailed progress updates")
	}
	if len(simpleProgressCalls) == 0 {
		t.Error("Expected simple progress callbacks")
	}
	if len(errorCallbacks) != expectedFailed {
		t.Errorf("Expected %d error callbacks, got %d", expectedFailed, len(errorCallbacks))
	}

	// Verify final progress state
	lastProgress := progressUpdates[len(progressUpdates)-1]
	if lastProgress.Completed != 25 {
		t.Errorf("Expected final completed to be 25, got %d", lastProgress.Completed)
	}
	if lastProgress.Successful != expectedSuccessful {
		t.Errorf("Expected final successful to be %d, got %d", expectedSuccessful, lastProgress.Successful)
	}
	if lastProgress.Failed != expectedFailed {
		t.Errorf("Expected final failed to be %d, got %d", expectedFailed, lastProgress.Failed)
	}

	// Rollback is only required when FailFast is true and we have failures
	// Since FailFast is false, rollback should not be required
	if result.RollbackRequired {
		t.Error("Rollback should not be required when FailFast is false")
	}

	// Verify all items have timestamps and proper retry counts
	for _, item := range result.Items {
		if item.Timestamp.IsZero() {
			t.Errorf("Item %d should have a timestamp", item.Index)
		}
		if !item.Success && item.Retries < 0 {
			t.Errorf("Item %d should have non-negative retry count", item.Index)
		}
	}

	// Verify duration is reasonable
	if result.Duration <= 0 {
		t.Error("Batch duration should be positive")
	}

	t.Logf("Comprehensive batch test completed successfully:")
	t.Logf("  - Processed %d items in %d chunks", len(entries), len(progressUpdates))
	t.Logf("  - %d successful, %d failed", result.Successful, result.Failed)
	t.Logf("  - %d error callbacks received", len(errorCallbacks))
	t.Logf("  - Duration: %v", result.Duration)
	t.Logf("  - Rollback required: %t", result.RollbackRequired)
}
