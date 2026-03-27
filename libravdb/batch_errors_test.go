package libravdb

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestBatchError(t *testing.T) {
	t.Run("basic error creation", func(t *testing.T) {
		err := NewBatchError(
			ErrBatchSizeExceeded,
			"BatchInsert",
			"batch size exceeded maximum limit",
			1000,
		)

		if err.Code != ErrBatchSizeExceeded {
			t.Errorf("expected code %d, got %d", ErrBatchSizeExceeded, err.Code)
		}

		if err.Operation != "BatchInsert" {
			t.Errorf("expected operation 'BatchInsert', got '%s'", err.Operation)
		}

		if err.BatchSize != 1000 {
			t.Errorf("expected batch size 1000, got %d", err.BatchSize)
		}

		if err.Message != "batch size exceeded maximum limit" {
			t.Errorf("expected message 'batch size exceeded maximum limit', got '%s'", err.Message)
		}
	})

	t.Run("error with item failures", func(t *testing.T) {
		err := NewBatchError(
			ErrBatchPartialFailure,
			"BatchInsert",
			"some items failed to process",
			100,
		).WithItemError(5, errors.New("validation failed")).
			WithItemError(15, errors.New("duplicate key")).
			WithItemError(25, errors.New("invalid data")).
			WithRetryable(true).
			WithRecoverable(true).
			WithDuration(time.Second * 5)

		if err.Failed != 3 {
			t.Errorf("expected 3 failed items, got %d", err.Failed)
		}

		if len(err.ItemErrors) != 3 {
			t.Errorf("expected 3 item errors, got %d", len(err.ItemErrors))
		}

		if err.ItemErrors[5] == nil {
			t.Error("expected error for item 5")
		}

		if err.Duration != time.Second*5 {
			t.Errorf("expected duration 5s, got %v", err.Duration)
		}

		failedItems := err.GetFailedItems()
		if len(failedItems) != 3 {
			t.Errorf("expected 3 failed items, got %d", len(failedItems))
		}

		// Check that failed items are correct
		expectedFailed := map[int]bool{5: true, 15: true, 25: true}
		for _, item := range failedItems {
			if !expectedFailed[item] {
				t.Errorf("unexpected failed item: %d", item)
			}
		}
	})

	t.Run("successful items calculation", func(t *testing.T) {
		err := NewBatchError(
			ErrBatchPartialFailure,
			"BatchInsert",
			"partial failure",
			10,
		).WithItemError(2, errors.New("error")).
			WithItemError(7, errors.New("error"))

		err.Processed = 10 // All items were processed

		successfulItems := err.GetSuccessfulItems()
		expectedSuccessful := []int{0, 1, 3, 4, 5, 6, 8, 9}

		if len(successfulItems) != len(expectedSuccessful) {
			t.Errorf("expected %d successful items, got %d", len(expectedSuccessful), len(successfulItems))
		}

		for i, expected := range expectedSuccessful {
			if i >= len(successfulItems) || successfulItems[i] != expected {
				t.Errorf("expected successful item %d at position %d, got %d", expected, i, successfulItems[i])
			}
		}
	})
}

func TestBatchRecoveryManager(t *testing.T) {
	t.Run("recover from size exceeded", func(t *testing.T) {
		brm := NewBatchRecoveryManager()
		brm.maxChunkSize = 50

		err := NewBatchError(
			ErrBatchSizeExceeded,
			"BatchInsert",
			"batch too large",
			100,
		)

		processedChunks := [][]int{}
		retryFunc := func(ctx context.Context, items []int) error {
			processedChunks = append(processedChunks, items)
			return nil
		}

		ctx := context.Background()
		if recoveryErr := brm.RecoverFromBatchFailure(ctx, err, retryFunc); recoveryErr != nil {
			t.Errorf("expected recovery to succeed, got error: %v", recoveryErr)
		}

		// Should have processed 2 chunks of 50 items each
		if len(processedChunks) != 2 {
			t.Errorf("expected 2 chunks, got %d", len(processedChunks))
		}

		if len(processedChunks[0]) != 50 {
			t.Errorf("expected first chunk size 50, got %d", len(processedChunks[0]))
		}

		if len(processedChunks[1]) != 50 {
			t.Errorf("expected second chunk size 50, got %d", len(processedChunks[1]))
		}
	})

	t.Run("recover from partial failure", func(t *testing.T) {
		brm := NewBatchRecoveryManager()
		brm.allowPartialSuccess = true

		err := NewBatchError(
			ErrBatchPartialFailure,
			"BatchInsert",
			"some items failed",
			100,
		).WithItemError(10, errors.New("error")).
			WithItemError(20, errors.New("error")).
			WithItemError(30, errors.New("error"))

		var retriedItems []int
		retryFunc := func(ctx context.Context, items []int) error {
			retriedItems = items
			return nil
		}

		ctx := context.Background()
		if recoveryErr := brm.RecoverFromBatchFailure(ctx, err, retryFunc); recoveryErr != nil {
			t.Errorf("expected recovery to succeed, got error: %v", recoveryErr)
		}

		// Should only retry the failed items
		expectedRetried := []int{10, 20, 30}
		if len(retriedItems) != len(expectedRetried) {
			t.Errorf("expected %d retried items, got %d", len(expectedRetried), len(retriedItems))
		}

		for i, expected := range expectedRetried {
			if retriedItems[i] != expected {
				t.Errorf("expected retried item %d, got %d", expected, retriedItems[i])
			}
		}
	})

	t.Run("recover from memory exhaustion", func(t *testing.T) {
		brm := NewBatchRecoveryManager()

		err := NewBatchError(
			ErrBatchMemoryExhausted,
			"BatchInsert",
			"out of memory",
			100,
		)

		processedChunks := [][]int{}
		retryFunc := func(ctx context.Context, items []int) error {
			processedChunks = append(processedChunks, items)
			return nil
		}

		ctx := context.Background()
		if recoveryErr := brm.RecoverFromBatchFailure(ctx, err, retryFunc); recoveryErr != nil {
			t.Errorf("expected recovery to succeed, got error: %v", recoveryErr)
		}

		// Should process in very small chunks (batch_size / 4 = 25)
		if len(processedChunks) != 4 {
			t.Errorf("expected 4 chunks, got %d", len(processedChunks))
		}

		for i, chunk := range processedChunks {
			if len(chunk) != 25 {
				t.Errorf("expected chunk %d size 25, got %d", i, len(chunk))
			}
		}
	})

	t.Run("retry batch operation", func(t *testing.T) {
		brm := NewBatchRecoveryManager()
		brm.maxRetries = 2
		brm.retryBackoff = time.Millisecond * 10

		err := NewBatchError(
			ErrBatchTransactionFailed,
			"BatchInsert",
			"transaction failed",
			50,
		)

		callCount := 0
		retryFunc := func(ctx context.Context, items []int) error {
			callCount++
			if callCount == 1 {
				return errors.New("still failing")
			}
			return nil // Success on second try
		}

		ctx := context.Background()
		if recoveryErr := brm.RecoverFromBatchFailure(ctx, err, retryFunc); recoveryErr != nil {
			t.Errorf("expected recovery to succeed, got error: %v", recoveryErr)
		}

		if callCount != 2 {
			t.Errorf("expected 2 retry calls, got %d", callCount)
		}
	})

	t.Run("recovery failure after max retries", func(t *testing.T) {
		brm := NewBatchRecoveryManager()
		brm.maxRetries = 2
		brm.retryBackoff = time.Millisecond * 10

		err := NewBatchError(
			ErrBatchTransactionFailed,
			"BatchInsert",
			"transaction failed",
			50,
		)

		retryFunc := func(ctx context.Context, items []int) error {
			return errors.New("always failing")
		}

		ctx := context.Background()
		if recoveryErr := brm.RecoverFromBatchFailure(ctx, err, retryFunc); recoveryErr == nil {
			t.Error("expected recovery to fail after max retries")
		}
	})

	t.Run("context cancellation", func(t *testing.T) {
		brm := NewBatchRecoveryManager()
		brm.retryBackoff = time.Second

		err := NewBatchError(
			ErrBatchTransactionFailed,
			"BatchInsert",
			"transaction failed",
			100,
		)

		retryFunc := func(ctx context.Context, items []int) error {
			return errors.New("always failing")
		}

		ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond*100)
		defer cancel()

		recoveryErr := brm.RecoverFromBatchFailure(ctx, err, retryFunc)
		if recoveryErr == nil {
			t.Error("expected recovery to fail due to context cancellation")
		}

		if !errors.Is(recoveryErr, context.DeadlineExceeded) {
			t.Errorf("expected context deadline exceeded error, got: %v", recoveryErr)
		}
	})
}

func TestBatchOperationTracker(t *testing.T) {
	t.Run("track operation lifecycle", func(t *testing.T) {
		tracker := NewBatchOperationTracker()

		operationID := "op-123"

		// Start operation
		tracker.StartOperation(operationID, "BatchInsert", 1000)

		status, exists := tracker.GetOperationStatus(operationID)
		if !exists {
			t.Fatal("expected operation to exist")
		}

		if status.ID != operationID {
			t.Errorf("expected operation ID '%s', got '%s'", operationID, status.ID)
		}

		if status.Operation != "BatchInsert" {
			t.Errorf("expected operation 'BatchInsert', got '%s'", status.Operation)
		}

		if status.BatchSize != 1000 {
			t.Errorf("expected batch size 1000, got %d", status.BatchSize)
		}

		if status.Status != "running" {
			t.Errorf("expected status 'running', got '%s'", status.Status)
		}

		// Update progress
		tracker.UpdateProgress(operationID, 500, 10)

		status, _ = tracker.GetOperationStatus(operationID)
		if status.Processed != 500 {
			t.Errorf("expected processed 500, got %d", status.Processed)
		}

		if status.Failed != 10 {
			t.Errorf("expected failed 10, got %d", status.Failed)
		}

		// Complete with partial failure
		batchErr := NewBatchError(
			ErrBatchPartialFailure,
			"BatchInsert",
			"some items failed",
			1000,
		)
		batchErr.Failed = 10

		tracker.CompleteOperation(operationID, batchErr)

		status, _ = tracker.GetOperationStatus(operationID)
		if status.Status != "partial" {
			t.Errorf("expected status 'partial', got '%s'", status.Status)
		}

		if status.EndTime == nil {
			t.Error("expected end time to be set")
		}

		if len(status.Errors) != 1 {
			t.Errorf("expected 1 error, got %d", len(status.Errors))
		}
	})

	t.Run("complete operation successfully", func(t *testing.T) {
		tracker := NewBatchOperationTracker()

		operationID := "op-456"
		tracker.StartOperation(operationID, "BatchUpdate", 500)
		tracker.CompleteOperation(operationID, nil) // No error

		status, exists := tracker.GetOperationStatus(operationID)
		if !exists {
			t.Fatal("expected operation to exist")
		}

		if status.Status != "completed" {
			t.Errorf("expected status 'completed', got '%s'", status.Status)
		}

		if len(status.Errors) != 0 {
			t.Errorf("expected no errors, got %d", len(status.Errors))
		}
	})

	t.Run("complete operation with total failure", func(t *testing.T) {
		tracker := NewBatchOperationTracker()

		operationID := "op-789"
		tracker.StartOperation(operationID, "BatchDelete", 100)

		batchErr := NewBatchError(
			ErrBatchTransactionFailed,
			"BatchDelete",
			"transaction failed",
			100,
		)
		batchErr.Failed = 100 // All items failed

		tracker.CompleteOperation(operationID, batchErr)

		status, exists := tracker.GetOperationStatus(operationID)
		if !exists {
			t.Fatal("expected operation to exist")
		}

		if status.Status != "failed" {
			t.Errorf("expected status 'failed', got '%s'", status.Status)
		}
	})

	t.Run("get all operations", func(t *testing.T) {
		tracker := NewBatchOperationTracker()

		tracker.StartOperation("op-1", "BatchInsert", 100)
		tracker.StartOperation("op-2", "BatchUpdate", 200)
		tracker.StartOperation("op-3", "BatchDelete", 300)

		allOps := tracker.GetAllOperations()

		if len(allOps) != 3 {
			t.Errorf("expected 3 operations, got %d", len(allOps))
		}

		if allOps["op-1"] == nil {
			t.Error("expected op-1 to exist")
		}

		if allOps["op-2"] == nil {
			t.Error("expected op-2 to exist")
		}

		if allOps["op-3"] == nil {
			t.Error("expected op-3 to exist")
		}
	})

	t.Run("cleanup completed operations", func(t *testing.T) {
		tracker := NewBatchOperationTracker()

		// Start and complete operations
		tracker.StartOperation("op-old", "BatchInsert", 100)
		tracker.StartOperation("op-new", "BatchUpdate", 200)

		// Complete the old operation
		tracker.CompleteOperation("op-old", nil)

		// Manually set the end time to be old
		if status, exists := tracker.GetOperationStatus("op-old"); exists {
			oldTime := time.Now().Add(-time.Hour * 2)
			status.EndTime = &oldTime
			tracker.operations["op-old"] = status
		}

		// Cleanup operations older than 1 hour
		tracker.CleanupCompletedOperations(time.Hour)

		allOps := tracker.GetAllOperations()

		if len(allOps) != 1 {
			t.Errorf("expected 1 operation after cleanup, got %d", len(allOps))
		}

		if allOps["op-old"] != nil {
			t.Error("expected op-old to be cleaned up")
		}

		if allOps["op-new"] == nil {
			t.Error("expected op-new to remain")
		}
	})

	t.Run("nonexistent operation", func(t *testing.T) {
		tracker := NewBatchOperationTracker()

		status, exists := tracker.GetOperationStatus("nonexistent")
		if exists {
			t.Error("expected operation to not exist")
		}

		if status != nil {
			t.Error("expected nil status for nonexistent operation")
		}
	})
}
