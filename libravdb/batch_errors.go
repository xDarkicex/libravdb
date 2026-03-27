package libravdb

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"
)

// BatchErrorCode represents specific batch operation error types
type BatchErrorCode int

const (
	ErrBatchUnknown BatchErrorCode = iota
	ErrBatchSizeExceeded
	ErrBatchTimeout
	ErrBatchPartialFailure
	ErrBatchTransactionFailed
	ErrBatchMemoryExhausted
	ErrBatchConcurrencyLimit
	ErrBatchValidationFailed
	ErrBatchIndexCorrupted
	ErrBatchStorageFailure
	ErrBatchRecoveryFailed
)

// BatchError represents a batch operation specific error
type BatchError struct {
	Code        BatchErrorCode         `json:"code"`
	Message     string                 `json:"message"`
	Operation   string                 `json:"operation"`
	BatchSize   int                    `json:"batch_size"`
	Processed   int                    `json:"processed"`
	Failed      int                    `json:"failed"`
	Retryable   bool                   `json:"retryable"`
	Recoverable bool                   `json:"recoverable"`
	ItemErrors  map[int]error          `json:"item_errors,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	Cause       error                  `json:"cause,omitempty"`
	Timestamp   time.Time              `json:"timestamp"`
	Duration    time.Duration          `json:"duration"`
}

func (be *BatchError) Error() string {
	if be.Cause != nil {
		return fmt.Sprintf("batch %s error: %s (processed: %d/%d, failed: %d) - caused by: %v",
			be.Operation, be.Message, be.Processed, be.BatchSize, be.Failed, be.Cause)
	}
	return fmt.Sprintf("batch %s error: %s (processed: %d/%d, failed: %d)",
		be.Operation, be.Message, be.Processed, be.BatchSize, be.Failed)
}

// Unwrap returns the underlying cause error
func (be *BatchError) Unwrap() error {
	return be.Cause
}

// NewBatchError creates a new batch error
func NewBatchError(code BatchErrorCode, operation, message string, batchSize int) *BatchError {
	return &BatchError{
		Code:       code,
		Message:    message,
		Operation:  operation,
		BatchSize:  batchSize,
		ItemErrors: make(map[int]error),
		Metadata:   make(map[string]interface{}),
		Timestamp:  time.Now(),
	}
}

// WithCause adds a cause error
func (be *BatchError) WithCause(cause error) *BatchError {
	be.Cause = cause
	return be
}

// WithRetryable sets whether the error is retryable
func (be *BatchError) WithRetryable(retryable bool) *BatchError {
	be.Retryable = retryable
	return be
}

// WithRecoverable sets whether the error is recoverable
func (be *BatchError) WithRecoverable(recoverable bool) *BatchError {
	be.Recoverable = recoverable
	return be
}

// WithItemError adds an error for a specific item
func (be *BatchError) WithItemError(index int, err error) *BatchError {
	if be.ItemErrors == nil {
		be.ItemErrors = make(map[int]error)
	}
	be.ItemErrors[index] = err
	be.Failed++
	return be
}

// WithMetadata adds metadata to the error
func (be *BatchError) WithMetadata(key string, value interface{}) *BatchError {
	if be.Metadata == nil {
		be.Metadata = make(map[string]interface{})
	}
	be.Metadata[key] = value
	return be
}

// WithDuration sets the operation duration
func (be *BatchError) WithDuration(duration time.Duration) *BatchError {
	be.Duration = duration
	return be
}

// GetFailedItems returns the indices of failed items
func (be *BatchError) GetFailedItems() []int {
	indices := make([]int, 0, len(be.ItemErrors))
	for index := range be.ItemErrors {
		indices = append(indices, index)
	}
	sort.Ints(indices)
	return indices
}

// GetSuccessfulItems returns the indices of successful items
func (be *BatchError) GetSuccessfulItems() []int {
	failedSet := make(map[int]bool)
	for index := range be.ItemErrors {
		failedSet[index] = true
	}

	successful := make([]int, 0, be.BatchSize-len(be.ItemErrors))
	for i := 0; i < be.Processed; i++ {
		if !failedSet[i] {
			successful = append(successful, i)
		}
	}
	return successful
}

// BatchRecoveryManager handles batch operation error recovery
type BatchRecoveryManager struct {
	maxRetries          int
	retryBackoff        time.Duration
	allowPartialSuccess bool
	enableChunking      bool
	maxChunkSize        int
	enableConcurrency   bool
	maxConcurrency      int
}

// NewBatchRecoveryManager creates a new batch recovery manager
func NewBatchRecoveryManager() *BatchRecoveryManager {
	return &BatchRecoveryManager{
		maxRetries:          3,
		retryBackoff:        time.Second,
		allowPartialSuccess: true,
		enableChunking:      true,
		maxChunkSize:        1000,
		enableConcurrency:   true,
		maxConcurrency:      4,
	}
}

// RecoverFromBatchFailure attempts to recover from batch operation failures
func (brm *BatchRecoveryManager) RecoverFromBatchFailure(
	ctx context.Context,
	err *BatchError,
	retryFunc func(ctx context.Context, items []int) error,
) error {
	switch err.Code {
	case ErrBatchSizeExceeded:
		return brm.recoverFromSizeExceeded(ctx, err, retryFunc)
	case ErrBatchTimeout:
		return brm.recoverFromTimeout(ctx, err, retryFunc)
	case ErrBatchPartialFailure:
		return brm.recoverFromPartialFailure(ctx, err, retryFunc)
	case ErrBatchMemoryExhausted:
		return brm.recoverFromMemoryExhaustion(ctx, err, retryFunc)
	case ErrBatchConcurrencyLimit:
		return brm.recoverFromConcurrencyLimit(ctx, err, retryFunc)
	default:
		return brm.retryBatchOperation(ctx, err, retryFunc)
	}
}

// recoverFromSizeExceeded handles batch size exceeded errors
func (brm *BatchRecoveryManager) recoverFromSizeExceeded(
	ctx context.Context,
	err *BatchError,
	retryFunc func(ctx context.Context, items []int) error,
) error {
	if !brm.enableChunking {
		return fmt.Errorf("chunking disabled, cannot recover from size exceeded")
	}

	// Split the batch into smaller chunks
	chunkSize := brm.maxChunkSize
	if err.BatchSize < chunkSize {
		chunkSize = err.BatchSize / 2 // Use half the original size
	}

	if chunkSize <= 0 {
		return fmt.Errorf("cannot chunk batch further")
	}

	// Process in chunks
	for start := 0; start < err.BatchSize; start += chunkSize {
		end := start + chunkSize
		if end > err.BatchSize {
			end = err.BatchSize
		}

		chunk := make([]int, end-start)
		for i := range chunk {
			chunk[i] = start + i
		}

		if chunkErr := retryFunc(ctx, chunk); chunkErr != nil {
			return fmt.Errorf("chunk processing failed at %d-%d: %w", start, end-1, chunkErr)
		}
	}

	return nil
}

// recoverFromTimeout handles batch timeout errors
func (brm *BatchRecoveryManager) recoverFromTimeout(
	ctx context.Context,
	err *BatchError,
	retryFunc func(ctx context.Context, items []int) error,
) error {
	// For timeout errors, we can:
	// 1. Retry with smaller chunks
	// 2. Increase timeout (if possible)
	// 3. Process sequentially instead of concurrently

	if brm.enableChunking {
		// Try with smaller chunks
		return brm.recoverFromSizeExceeded(ctx, err, retryFunc)
	}

	// Retry with sequential processing
	items := make([]int, err.BatchSize)
	for i := range items {
		items[i] = i
	}

	return retryFunc(ctx, items)
}

// recoverFromPartialFailure handles partial failure errors
func (brm *BatchRecoveryManager) recoverFromPartialFailure(
	ctx context.Context,
	err *BatchError,
	retryFunc func(ctx context.Context, items []int) error,
) error {
	if brm.allowPartialSuccess && len(err.ItemErrors) < err.BatchSize/2 {
		// If less than half failed, just retry the failed items
		failedItems := err.GetFailedItems()
		if len(failedItems) == 0 {
			return nil // Nothing to retry
		}

		return retryFunc(ctx, failedItems)
	}

	// Too many failures, retry the entire batch
	items := make([]int, err.BatchSize)
	for i := range items {
		items[i] = i
	}

	return retryFunc(ctx, items)
}

// recoverFromMemoryExhaustion handles memory exhaustion errors
func (brm *BatchRecoveryManager) recoverFromMemoryExhaustion(
	ctx context.Context,
	err *BatchError,
	retryFunc func(ctx context.Context, items []int) error,
) error {
	// For memory exhaustion, reduce batch size and disable concurrency
	chunkSize := err.BatchSize / 4 // Use quarter of original size
	if chunkSize <= 0 {
		chunkSize = 1
	}

	// Process items one by one or in very small chunks
	for start := 0; start < err.BatchSize; start += chunkSize {
		end := start + chunkSize
		if end > err.BatchSize {
			end = err.BatchSize
		}

		chunk := make([]int, end-start)
		for i := range chunk {
			chunk[i] = start + i
		}

		// Force garbage collection before each chunk
		runtime.GC()

		if chunkErr := retryFunc(ctx, chunk); chunkErr != nil {
			return fmt.Errorf("memory-constrained chunk processing failed at %d-%d: %w",
				start, end-1, chunkErr)
		}
	}

	return nil
}

// recoverFromConcurrencyLimit handles concurrency limit errors
func (brm *BatchRecoveryManager) recoverFromConcurrencyLimit(
	ctx context.Context,
	err *BatchError,
	retryFunc func(ctx context.Context, items []int) error,
) error {
	// Reduce concurrency and retry
	items := make([]int, err.BatchSize)
	for i := range items {
		items[i] = i
	}

	// The retry function should handle reduced concurrency
	return retryFunc(ctx, items)
}

// retryBatchOperation performs simple retry with backoff
func (brm *BatchRecoveryManager) retryBatchOperation(
	ctx context.Context,
	err *BatchError,
	retryFunc func(ctx context.Context, items []int) error,
) error {
	items := make([]int, err.BatchSize)
	for i := range items {
		items[i] = i
	}

	for attempt := 1; attempt <= brm.maxRetries; attempt++ {
		if attempt > 1 {
			// Apply backoff
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(brm.retryBackoff * time.Duration(attempt)):
			}
		}

		if retryErr := retryFunc(ctx, items); retryErr == nil {
			return nil // Success
		}
	}

	return fmt.Errorf("batch operation failed after %d retries", brm.maxRetries)
}

// BatchOperationTracker tracks batch operations and their errors
type BatchOperationTracker struct {
	mu         sync.RWMutex
	operations map[string]*BatchOperationStatus
}

// BatchOperationStatus represents the status of a batch operation
type BatchOperationStatus struct {
	ID        string                 `json:"id"`
	Operation string                 `json:"operation"`
	BatchSize int                    `json:"batch_size"`
	Processed int                    `json:"processed"`
	Failed    int                    `json:"failed"`
	StartTime time.Time              `json:"start_time"`
	EndTime   *time.Time             `json:"end_time,omitempty"`
	Duration  time.Duration          `json:"duration"`
	Status    string                 `json:"status"` // "running", "completed", "failed", "partial"
	Errors    []*BatchError          `json:"errors,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// NewBatchOperationTracker creates a new batch operation tracker
func NewBatchOperationTracker() *BatchOperationTracker {
	return &BatchOperationTracker{
		operations: make(map[string]*BatchOperationStatus),
	}
}

// StartOperation starts tracking a batch operation
func (bot *BatchOperationTracker) StartOperation(id, operation string, batchSize int) {
	bot.mu.Lock()
	defer bot.mu.Unlock()

	bot.operations[id] = &BatchOperationStatus{
		ID:        id,
		Operation: operation,
		BatchSize: batchSize,
		StartTime: time.Now(),
		Status:    "running",
		Metadata:  make(map[string]interface{}),
	}
}

// UpdateProgress updates the progress of a batch operation
func (bot *BatchOperationTracker) UpdateProgress(id string, processed, failed int) {
	bot.mu.Lock()
	defer bot.mu.Unlock()

	if status, exists := bot.operations[id]; exists {
		status.Processed = processed
		status.Failed = failed
	}
}

// CompleteOperation marks a batch operation as completed
func (bot *BatchOperationTracker) CompleteOperation(id string, err *BatchError) {
	bot.mu.Lock()
	defer bot.mu.Unlock()

	status, exists := bot.operations[id]
	if !exists {
		return
	}

	now := time.Now()
	status.EndTime = &now
	status.Duration = now.Sub(status.StartTime)

	if err != nil {
		status.Errors = append(status.Errors, err)
		status.Failed += err.Failed
		status.Processed = err.Processed
		if err.Failed == err.BatchSize {
			status.Status = "failed"
		} else if err.Failed > 0 {
			status.Status = "partial"
		} else {
			status.Status = "completed"
		}
	} else {
		status.Status = "completed"
	}
}

// GetOperationStatus returns the status of a batch operation
func (bot *BatchOperationTracker) GetOperationStatus(id string) (*BatchOperationStatus, bool) {
	bot.mu.RLock()
	defer bot.mu.RUnlock()

	status, exists := bot.operations[id]
	if !exists {
		return nil, false
	}

	// Return a copy to avoid race conditions
	statusCopy := *status
	return &statusCopy, true
}

// GetAllOperations returns all tracked operations
func (bot *BatchOperationTracker) GetAllOperations() map[string]*BatchOperationStatus {
	bot.mu.RLock()
	defer bot.mu.RUnlock()

	result := make(map[string]*BatchOperationStatus)
	for id, status := range bot.operations {
		statusCopy := *status
		result[id] = &statusCopy
	}

	return result
}

// CleanupCompletedOperations removes completed operations older than the specified duration
func (bot *BatchOperationTracker) CleanupCompletedOperations(maxAge time.Duration) {
	bot.mu.Lock()
	defer bot.mu.Unlock()

	cutoff := time.Now().Add(-maxAge)

	for id, status := range bot.operations {
		if status.EndTime != nil && status.EndTime.Before(cutoff) {
			delete(bot.operations, id)
		}
	}
}
