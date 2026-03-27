package libravdb

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/index"
)

// BatchOperation represents a batch operation that can be executed
type BatchOperation interface {
	Execute(ctx context.Context) (*BatchResult, error)
	Size() int
	EstimateMemory() int64
}

// BatchResult contains the results of a batch operation
type BatchResult struct {
	Successful       int                `json:"successful"`
	Failed           int                `json:"failed"`
	Errors           map[int]error      `json:"errors"` // Index -> Error mapping
	Duration         time.Duration      `json:"duration"`
	Items            []*BatchItemResult `json:"items"`
	RollbackRequired bool               `json:"rollback_required"`
	RollbackError    error              `json:"rollback_error,omitempty"`
}

// BatchItemResult represents the result of a single item in a batch
type BatchItemResult struct {
	Index          int       `json:"index"`
	ID             string    `json:"id"`
	Success        bool      `json:"success"`
	Error          error     `json:"error,omitempty"`
	BatchErrorCode string    `json:"batch_error_code,omitempty"`
	Timestamp      time.Time `json:"timestamp"`
	Retries        int       `json:"retries"`
}

// Batch operation error codes
const (
	BatchErrorValidation   = "VALIDATION_ERROR"
	BatchErrorDuplicate    = "DUPLICATE_ERROR"
	BatchErrorNotFound     = "NOT_FOUND_ERROR"
	BatchErrorTimeout      = "TIMEOUT_ERROR"
	BatchErrorMemory       = "MEMORY_ERROR"
	BatchErrorInternal     = "INTERNAL_ERROR"
	BatchErrorCancellation = "CANCELLATION_ERROR"
)

// BatchOptions configures batch operation behavior
type BatchOptions struct {
	ChunkSize        int                                    `json:"chunk_size"`
	MaxConcurrency   int                                    `json:"max_concurrency"`
	FailFast         bool                                   `json:"fail_fast"`
	ProgressCallback func(completed, total int)             `json:"-"`
	Timeout          time.Duration                          `json:"timeout"`
	EnableRollback   bool                                   `json:"enable_rollback"`
	MaxRetries       int                                    `json:"max_retries"`
	RetryDelay       time.Duration                          `json:"retry_delay"`
	DetailedProgress func(progress *BatchProgress)          `json:"-"`
	ErrorCallback    func(item *BatchItemResult, err error) `json:"-"`
}

// BatchProgress provides detailed progress information
type BatchProgress struct {
	Completed    int           `json:"completed"`
	Total        int           `json:"total"`
	Successful   int           `json:"successful"`
	Failed       int           `json:"failed"`
	CurrentChunk int           `json:"current_chunk"`
	TotalChunks  int           `json:"total_chunks"`
	ElapsedTime  time.Duration `json:"elapsed_time"`
	EstimatedETA time.Duration `json:"estimated_eta"`
	ItemsPerSec  float64       `json:"items_per_sec"`
	LastError    error         `json:"last_error,omitempty"`
}

// DefaultBatchOptions returns sensible defaults for batch operations
func DefaultBatchOptions() *BatchOptions {
	return &BatchOptions{
		ChunkSize:      1000,
		MaxConcurrency: 4,
		FailFast:       false,
		Timeout:        5 * time.Minute,
		EnableRollback: false,
		MaxRetries:     3,
		RetryDelay:     100 * time.Millisecond,
	}
}

// VectorUpdate represents an update operation for a vector
type VectorUpdate struct {
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector,omitempty"`   // Optional: update vector
	Metadata map[string]interface{} `json:"metadata,omitempty"` // Optional: update metadata
	Upsert   bool                   `json:"upsert"`             // Create if not exists
}

// BatchInsert represents a batch insert operation
type BatchInsert struct {
	collection      *Collection
	entries         []*VectorEntry
	options         *BatchOptions
	insertedIDs     []string
	rollbackMutex   sync.Mutex
	progressTracker *progressTracker
}

// progressTracker tracks detailed progress for batch operations
type progressTracker struct {
	startTime    time.Time
	completed    int
	total        int
	successful   int
	failed       int
	currentChunk int
	totalChunks  int
	lastError    error
	mutex        sync.RWMutex
}

// newProgressTracker creates a new progress tracker
func newProgressTracker(total int, chunkSize int) *progressTracker {
	if chunkSize <= 0 {
		chunkSize = 1
	}
	totalChunks := (total + chunkSize - 1) / chunkSize
	return &progressTracker{
		startTime:   time.Now(),
		total:       total,
		totalChunks: totalChunks,
	}
}

// update updates the progress tracker
func (pt *progressTracker) update(completed, successful, failed int, currentChunk int, lastError error) {
	pt.mutex.Lock()
	defer pt.mutex.Unlock()

	pt.completed = completed
	pt.successful = successful
	pt.failed = failed
	pt.currentChunk = currentChunk
	pt.lastError = lastError
}

// getProgress returns current progress information
func (pt *progressTracker) getProgress() *BatchProgress {
	pt.mutex.RLock()
	defer pt.mutex.RUnlock()

	elapsed := time.Since(pt.startTime)
	var eta time.Duration
	var itemsPerSec float64

	if pt.completed > 0 && elapsed > 0 {
		itemsPerSec = float64(pt.completed) / elapsed.Seconds()
		if itemsPerSec > 0 && pt.completed < pt.total {
			remaining := pt.total - pt.completed
			etaSeconds := float64(remaining) / itemsPerSec
			eta = time.Duration(etaSeconds * float64(time.Second))
		}
	}

	return &BatchProgress{
		Completed:    pt.completed,
		Total:        pt.total,
		Successful:   pt.successful,
		Failed:       pt.failed,
		CurrentChunk: pt.currentChunk,
		TotalChunks:  pt.totalChunks,
		ElapsedTime:  elapsed,
		EstimatedETA: eta,
		ItemsPerSec:  itemsPerSec,
		LastError:    pt.lastError,
	}
}

// BatchUpdate represents a batch update operation
type BatchUpdate struct {
	collection      *Collection
	updates         []*VectorUpdate
	options         *BatchOptions
	modifiedIDs     []string
	originalEntries []*VectorEntry // Store original entries for rollback
	rollbackMutex   sync.Mutex
	progressTracker *progressTracker
}

// BatchDelete represents a batch delete operation
type BatchDelete struct {
	collection      *Collection
	ids             []string
	options         *BatchOptions
	deletedEntries  []*VectorEntry // Store for rollback
	rollbackMutex   sync.Mutex
	progressTracker *progressTracker
}

// NewBatchInsert creates a new batch insert operation
func (c *Collection) NewBatchInsert(entries []*VectorEntry, opts ...*BatchOptions) *BatchInsert {
	options := DefaultBatchOptions()
	if len(opts) > 0 && opts[0] != nil {
		options = opts[0]
	}

	return &BatchInsert{
		collection:      c,
		entries:         entries,
		options:         options,
		insertedIDs:     make([]string, 0, len(entries)),
		progressTracker: newProgressTracker(len(entries), options.ChunkSize),
	}
}

// NewBatchUpdate creates a new batch update operation
func (c *Collection) NewBatchUpdate(updates []*VectorUpdate, opts ...*BatchOptions) *BatchUpdate {
	options := DefaultBatchOptions()
	if len(opts) > 0 && opts[0] != nil {
		options = opts[0]
	}

	return &BatchUpdate{
		collection:      c,
		updates:         updates,
		options:         options,
		modifiedIDs:     make([]string, 0, len(updates)),
		originalEntries: make([]*VectorEntry, 0, len(updates)),
		progressTracker: newProgressTracker(len(updates), options.ChunkSize),
	}
}

// NewBatchDelete creates a new batch delete operation
func (c *Collection) NewBatchDelete(ids []string, opts ...*BatchOptions) *BatchDelete {
	options := DefaultBatchOptions()
	if len(opts) > 0 && opts[0] != nil {
		options = opts[0]
	}

	return &BatchDelete{
		collection:      c,
		ids:             ids,
		options:         options,
		deletedEntries:  make([]*VectorEntry, 0, len(ids)),
		progressTracker: newProgressTracker(len(ids), options.ChunkSize),
	}
}

// Size returns the number of items in the batch insert
func (b *BatchInsert) Size() int {
	return len(b.entries)
}

// EstimateMemory estimates the memory usage of the batch insert
func (b *BatchInsert) EstimateMemory() int64 {
	if len(b.entries) == 0 {
		return 0
	}

	// Estimate: vector size + metadata + overhead
	vectorSize := len(b.entries[0].Vector) * 4 // 4 bytes per float32
	metadataSize := 100                        // rough estimate for metadata
	overhead := 50                             // struct overhead

	return int64(len(b.entries) * (vectorSize + metadataSize + overhead))
}

// Execute performs the batch insert operation with enhanced error handling and rollback
func (b *BatchInsert) Execute(ctx context.Context) (*BatchResult, error) {
	startTime := time.Now()
	var operationErr error

	// Apply timeout if specified
	if b.options.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, b.options.Timeout)
		defer cancel()
	}

	result := &BatchResult{
		Errors: make(map[int]error),
		Items:  make([]*BatchItemResult, 0, len(b.entries)),
	}

	// Process in chunks for better memory management and progress tracking
	chunkSize := b.options.ChunkSize
	if chunkSize <= 0 {
		if len(b.entries) == 0 {
			chunkSize = 1 // Avoid division by zero
		} else {
			chunkSize = len(b.entries)
		}
	}

	// Handle empty batch case
	if len(b.entries) == 0 {
		result.Duration = time.Since(startTime)
		return result, nil
	}

	totalChunks := (len(b.entries) + chunkSize - 1) / chunkSize

	for chunkIdx := 0; chunkIdx < totalChunks; chunkIdx++ {
		startIdx := chunkIdx * chunkSize
		endIdx := startIdx + chunkSize
		if endIdx > len(b.entries) {
			endIdx = len(b.entries)
		}

		chunk := b.entries[startIdx:endIdx]

		// Process chunk
		chunkResult, err := b.processChunk(ctx, chunk, startIdx, chunkIdx)
		if err != nil {
			// Handle chunk processing error
			if b.options.EnableRollback {
				rollbackErr := b.rollback(ctx)
				result.RollbackRequired = true
				result.RollbackError = rollbackErr
			}
			result.Duration = time.Since(startTime)
			return result, err
		}

		// Merge chunk results
		b.mergeChunkResult(result, chunkResult)
		if chunkResult.lastError != nil {
			operationErr = chunkResult.lastError
		}

		// Update progress
		b.progressTracker.update(
			result.Successful+result.Failed,
			result.Successful,
			result.Failed,
			chunkIdx+1,
			chunkResult.lastError,
		)

		// Call progress callbacks
		if b.options.ProgressCallback != nil {
			b.options.ProgressCallback(result.Successful+result.Failed, len(b.entries))
		}
		if b.options.DetailedProgress != nil {
			b.options.DetailedProgress(b.progressTracker.getProgress())
		}

		// Check for early termination on fail-fast
		if b.options.FailFast && result.Failed > 0 {
			if b.options.EnableRollback {
				rollbackErr := b.rollback(ctx)
				result.RollbackRequired = true
				result.RollbackError = rollbackErr
			}
			result.Duration = time.Since(startTime)
			if operationErr == nil {
				operationErr = fmt.Errorf("batch insert failed")
			}
			return result, operationErr
		}

		// Check for context cancellation
		select {
		case <-ctx.Done():
			if b.options.EnableRollback {
				rollbackErr := b.rollback(ctx)
				result.RollbackRequired = true
				result.RollbackError = rollbackErr
			}
			result.Duration = time.Since(startTime)
			return result, ctx.Err()
		default:
		}
	}

	result.Duration = time.Since(startTime)
	return result, nil
}

// chunkResult represents the result of processing a single chunk
type chunkResult struct {
	items     []*BatchItemResult
	errors    map[int]error
	lastError error
}

// processChunk processes a single chunk of entries
func (b *BatchInsert) processChunk(ctx context.Context, chunk []*VectorEntry, startIndex int, chunkIdx int) (*chunkResult, error) {
	result := &chunkResult{
		items:  make([]*BatchItemResult, 0, len(chunk)),
		errors: make(map[int]error),
	}

	for i, entry := range chunk {
		globalIndex := startIndex + i
		itemResult := &BatchItemResult{
			Index:     globalIndex,
			ID:        entry.ID,
			Timestamp: time.Now(),
		}

		// Process with retries
		success, err := b.processItemWithRetries(ctx, entry, itemResult)
		itemResult.Success = success
		itemResult.Error = err

		if err != nil {
			result.errors[globalIndex] = err
			result.lastError = err
			itemResult.BatchErrorCode = b.categorizeError(err)

			// Call error callback if provided
			if b.options.ErrorCallback != nil {
				b.options.ErrorCallback(itemResult, err)
			}
		} else {
			// Track successful insert for rollback
			b.rollbackMutex.Lock()
			b.insertedIDs = append(b.insertedIDs, entry.ID)
			b.rollbackMutex.Unlock()
		}

		result.items = append(result.items, itemResult)

		// Check for context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}

	return result, nil
}

// processItemWithRetries processes a single item with retry logic
func (b *BatchInsert) processItemWithRetries(ctx context.Context, entry *VectorEntry, itemResult *BatchItemResult) (bool, error) {
	var lastErr error

	for attempt := 0; attempt <= b.options.MaxRetries; attempt++ {
		itemResult.Retries = attempt

		// Validate entry
		if err := b.validateEntry(entry); err != nil {
			return false, err
		}

		// Perform the insert
		err := b.collection.Insert(ctx, entry.ID, entry.Vector, entry.Metadata)
		if err == nil {
			return true, nil
		}

		lastErr = err

		// Don't retry validation errors or context cancellation
		if b.isNonRetryableError(err) || ctx.Err() != nil {
			break
		}

		// Wait before retry (except on last attempt)
		if attempt < b.options.MaxRetries && b.options.RetryDelay > 0 {
			select {
			case <-time.After(b.options.RetryDelay):
			case <-ctx.Done():
				return false, ctx.Err()
			}
		}
	}

	return false, lastErr
}

// mergeChunkResult merges chunk results into the main result
func (b *BatchInsert) mergeChunkResult(result *BatchResult, chunkResult *chunkResult) {
	for _, item := range chunkResult.items {
		result.Items = append(result.Items, item)
		if item.Success {
			result.Successful++
		} else {
			result.Failed++
		}
	}

	for idx, err := range chunkResult.errors {
		result.Errors[idx] = err
	}
}

// rollback removes all successfully inserted items
func (b *BatchInsert) rollback(ctx context.Context) error {
	b.rollbackMutex.Lock()
	defer b.rollbackMutex.Unlock()

	if len(b.insertedIDs) == 0 {
		return nil
	}

	var rollbackErrors []error

	for _, id := range b.insertedIDs {
		if err := b.collection.Delete(ctx, id); err != nil {
			rollbackErrors = append(rollbackErrors, fmt.Errorf("failed to rollback %s: %w", id, err))
		}
	}

	if len(rollbackErrors) > 0 {
		return fmt.Errorf("rollback failed for %d items", len(rollbackErrors))
	}

	b.insertedIDs = b.insertedIDs[:0] // Clear the slice
	return nil
}

// categorizeError categorizes errors into error codes
func (b *BatchInsert) categorizeError(err error) string {
	if err == nil {
		return ""
	}

	errStr := err.Error()
	switch {
	case err == context.DeadlineExceeded:
		return BatchErrorTimeout
	case err == context.Canceled:
		return BatchErrorCancellation
	case contains(errStr, "dimension"):
		return BatchErrorValidation
	case contains(errStr, "empty"):
		return BatchErrorValidation
	case contains(errStr, "duplicate"):
		return BatchErrorDuplicate
	case contains(errStr, "memory"):
		return BatchErrorMemory
	default:
		return BatchErrorInternal
	}
}

// isNonRetryableError determines if an error should not be retried
func (b *BatchInsert) isNonRetryableError(err error) bool {
	if err == nil {
		return false
	}

	errStr := err.Error()
	return contains(errStr, "dimension") ||
		contains(errStr, "empty") ||
		contains(errStr, "duplicate")
}

// validateEntry validates a vector entry before insertion
func (b *BatchInsert) validateEntry(entry *VectorEntry) error {
	if entry == nil {
		return fmt.Errorf("entry cannot be nil")
	}

	if entry.ID == "" {
		return fmt.Errorf("entry ID cannot be empty")
	}

	if len(entry.Vector) != b.collection.config.Dimension {
		return fmt.Errorf("vector dimension %d does not match collection dimension %d",
			len(entry.Vector), b.collection.config.Dimension)
	}

	return nil
}

// contains is a simple string contains helper
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || (len(s) > len(substr) &&
		(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr ||
			findSubstring(s, substr))))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// Size returns the number of items in the batch update
func (b *BatchUpdate) Size() int {
	return len(b.updates)
}

// EstimateMemory estimates the memory usage of the batch update
func (b *BatchUpdate) EstimateMemory() int64 {
	if len(b.updates) == 0 {
		return 0
	}

	// Estimate based on vector size and metadata
	avgVectorSize := 0
	for _, update := range b.updates {
		if update.Vector != nil {
			avgVectorSize += len(update.Vector) * 4 // 4 bytes per float32
		}
	}
	if len(b.updates) > 0 {
		avgVectorSize /= len(b.updates)
	}

	metadataSize := 100 // rough estimate
	overhead := 50

	return int64(len(b.updates) * (avgVectorSize + metadataSize + overhead))
}

// Execute performs the batch update operation with enhanced error handling and rollback
func (b *BatchUpdate) Execute(ctx context.Context) (*BatchResult, error) {
	startTime := time.Now()
	var operationErr error

	if b.options.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, b.options.Timeout)
		defer cancel()
	}

	result := &BatchResult{
		Errors: make(map[int]error),
		Items:  make([]*BatchItemResult, 0, len(b.updates)),
	}

	// Process in chunks
	chunkSize := b.options.ChunkSize
	if chunkSize <= 0 {
		chunkSize = len(b.updates)
	}

	totalChunks := (len(b.updates) + chunkSize - 1) / chunkSize

	for chunkIdx := 0; chunkIdx < totalChunks; chunkIdx++ {
		startIdx := chunkIdx * chunkSize
		endIdx := startIdx + chunkSize
		if endIdx > len(b.updates) {
			endIdx = len(b.updates)
		}

		chunk := b.updates[startIdx:endIdx]

		// Process chunk
		chunkResult, err := b.processUpdateChunk(ctx, chunk, startIdx, chunkIdx)
		if err != nil {
			if b.options.EnableRollback {
				rollbackErr := b.rollbackUpdates(ctx)
				result.RollbackRequired = true
				result.RollbackError = rollbackErr
			}
			result.Duration = time.Since(startTime)
			return result, err
		}

		// Merge chunk results
		b.mergeUpdateChunkResult(result, chunkResult)
		if chunkResult.lastError != nil {
			operationErr = chunkResult.lastError
		}

		// Update progress
		b.progressTracker.update(
			result.Successful+result.Failed,
			result.Successful,
			result.Failed,
			chunkIdx+1,
			chunkResult.lastError,
		)

		// Call progress callbacks
		if b.options.ProgressCallback != nil {
			b.options.ProgressCallback(result.Successful+result.Failed, len(b.updates))
		}
		if b.options.DetailedProgress != nil {
			b.options.DetailedProgress(b.progressTracker.getProgress())
		}

		// Check for early termination on fail-fast
		if b.options.FailFast && result.Failed > 0 {
			if b.options.EnableRollback {
				rollbackErr := b.rollbackUpdates(ctx)
				result.RollbackRequired = true
				result.RollbackError = rollbackErr
			}
			result.Duration = time.Since(startTime)
			if operationErr == nil {
				operationErr = fmt.Errorf("batch update failed")
			}
			return result, operationErr
		}

		// Check for context cancellation
		select {
		case <-ctx.Done():
			if b.options.EnableRollback {
				rollbackErr := b.rollbackUpdates(ctx)
				result.RollbackRequired = true
				result.RollbackError = rollbackErr
			}
			result.Duration = time.Since(startTime)
			return result, ctx.Err()
		default:
		}
	}

	result.Duration = time.Since(startTime)
	return result, nil
}

// processUpdateChunk processes a single chunk of updates
func (b *BatchUpdate) processUpdateChunk(ctx context.Context, chunk []*VectorUpdate, startIndex int, chunkIdx int) (*chunkResult, error) {
	result := &chunkResult{
		items:  make([]*BatchItemResult, 0, len(chunk)),
		errors: make(map[int]error),
	}

	for i, update := range chunk {
		globalIndex := startIndex + i
		itemResult := &BatchItemResult{
			Index:     globalIndex,
			ID:        update.ID,
			Timestamp: time.Now(),
		}

		// Process with retries
		success, err := b.processUpdateWithRetries(ctx, update, itemResult)
		itemResult.Success = success
		itemResult.Error = err

		if err != nil {
			result.errors[globalIndex] = err
			result.lastError = err
			itemResult.BatchErrorCode = b.categorizeUpdateError(err)

			if b.options.ErrorCallback != nil {
				b.options.ErrorCallback(itemResult, err)
			}
		} else {
			// Track successful update for rollback
			b.rollbackMutex.Lock()
			b.modifiedIDs = append(b.modifiedIDs, update.ID)
			b.rollbackMutex.Unlock()
		}

		result.items = append(result.items, itemResult)

		// Check for context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}

	return result, nil
}

// processUpdateWithRetries processes a single update with retry logic
func (b *BatchUpdate) processUpdateWithRetries(ctx context.Context, update *VectorUpdate, itemResult *BatchItemResult) (bool, error) {
	var lastErr error

	for attempt := 0; attempt <= b.options.MaxRetries; attempt++ {
		itemResult.Retries = attempt

		// Validate update
		if err := b.validateUpdate(update); err != nil {
			return false, err
		}

		// Store original entry for rollback (if not upsert)
		var originalEntry *VectorEntry
		if !update.Upsert && b.options.EnableRollback {
			// Retrieve original entry before updating
			_ = b.collection.storage.Iterate(ctx, func(entry *index.VectorEntry) error {
				if entry.ID == update.ID {
					originalEntry = &VectorEntry{
						ID:       entry.ID,
						Vector:   make([]float32, len(entry.Vector)),
						Metadata: make(map[string]interface{}),
					}
					copy(originalEntry.Vector, entry.Vector)
					for k, v := range entry.Metadata {
						originalEntry.Metadata[k] = v
					}
					return fmt.Errorf("found") // Use error to break iteration
				}
				return nil
			})
			// Ignore the "found" error - it's just to break iteration
		}

		// Perform the update/upsert
		var err error
		if update.Upsert && update.Vector != nil {
			err = b.collection.Insert(ctx, update.ID, update.Vector, update.Metadata)
		} else {
			// Use the new Update method
			err = b.collection.Update(ctx, update.ID, update.Vector, update.Metadata)
		}

		// Store original entry for rollback if update was successful
		if err == nil && originalEntry != nil {
			b.rollbackMutex.Lock()
			b.originalEntries = append(b.originalEntries, originalEntry)
			b.rollbackMutex.Unlock()
		}

		if err == nil {
			return true, nil
		}

		lastErr = err

		// Don't retry validation errors or context cancellation
		if b.isNonRetryableUpdateError(err) || ctx.Err() != nil {
			break
		}

		// Wait before retry
		if attempt < b.options.MaxRetries && b.options.RetryDelay > 0 {
			select {
			case <-time.After(b.options.RetryDelay):
			case <-ctx.Done():
				return false, ctx.Err()
			}
		}
	}

	return false, lastErr
}

// mergeUpdateChunkResult merges chunk results into the main result
func (b *BatchUpdate) mergeUpdateChunkResult(result *BatchResult, chunkResult *chunkResult) {
	for _, item := range chunkResult.items {
		result.Items = append(result.Items, item)
		if item.Success {
			result.Successful++
		} else {
			result.Failed++
		}
	}

	for idx, err := range chunkResult.errors {
		result.Errors[idx] = err
	}
}

// rollbackUpdates reverts all successful updates
func (b *BatchUpdate) rollbackUpdates(ctx context.Context) error {
	b.rollbackMutex.Lock()
	defer b.rollbackMutex.Unlock()

	if len(b.originalEntries) == 0 {
		return nil
	}

	var rollbackErrors []error

	// Restore original entries
	for _, originalEntry := range b.originalEntries {
		if err := b.collection.Update(ctx, originalEntry.ID, originalEntry.Vector, originalEntry.Metadata); err != nil {
			rollbackErrors = append(rollbackErrors, fmt.Errorf("failed to rollback update for %s: %w", originalEntry.ID, err))
		}
	}

	if len(rollbackErrors) > 0 {
		return fmt.Errorf("rollback failed for %d items", len(rollbackErrors))
	}

	b.modifiedIDs = b.modifiedIDs[:0]
	b.originalEntries = b.originalEntries[:0]
	return nil
}

// categorizeUpdateError categorizes update errors
func (b *BatchUpdate) categorizeUpdateError(err error) string {
	if err == nil {
		return ""
	}

	errStr := err.Error()
	switch {
	case err == context.DeadlineExceeded:
		return BatchErrorTimeout
	case err == context.Canceled:
		return BatchErrorCancellation
	case contains(errStr, "dimension"):
		return BatchErrorValidation
	case contains(errStr, "empty"):
		return BatchErrorValidation
	case contains(errStr, "not found"):
		return BatchErrorNotFound
	default:
		return BatchErrorInternal
	}
}

// isNonRetryableUpdateError determines if an update error should not be retried
func (b *BatchUpdate) isNonRetryableUpdateError(err error) bool {
	if err == nil {
		return false
	}

	errStr := err.Error()
	return contains(errStr, "dimension") ||
		contains(errStr, "empty")
}

// validateUpdate validates an update operation
func (b *BatchUpdate) validateUpdate(update *VectorUpdate) error {
	if update == nil {
		return fmt.Errorf("update cannot be nil")
	}

	if update.ID == "" {
		return fmt.Errorf("update ID cannot be empty")
	}

	if update.Vector != nil && len(update.Vector) != b.collection.config.Dimension {
		return fmt.Errorf("vector dimension %d does not match collection dimension %d",
			len(update.Vector), b.collection.config.Dimension)
	}

	return nil
}

// Size returns the number of items in the batch delete
func (b *BatchDelete) Size() int {
	return len(b.ids)
}

// EstimateMemory estimates the memory usage of the batch delete
func (b *BatchDelete) EstimateMemory() int64 {
	// Delete operations have minimal memory overhead
	return int64(len(b.ids) * 50) // rough estimate for ID storage
}

// Execute performs the batch delete operation with enhanced error handling and rollback
func (b *BatchDelete) Execute(ctx context.Context) (*BatchResult, error) {
	startTime := time.Now()
	var operationErr error

	if b.options.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, b.options.Timeout)
		defer cancel()
	}

	result := &BatchResult{
		Errors: make(map[int]error),
		Items:  make([]*BatchItemResult, 0, len(b.ids)),
	}

	// Process in chunks
	chunkSize := b.options.ChunkSize
	if chunkSize <= 0 {
		chunkSize = len(b.ids)
	}

	totalChunks := (len(b.ids) + chunkSize - 1) / chunkSize

	for chunkIdx := 0; chunkIdx < totalChunks; chunkIdx++ {
		startIdx := chunkIdx * chunkSize
		endIdx := startIdx + chunkSize
		if endIdx > len(b.ids) {
			endIdx = len(b.ids)
		}

		chunk := b.ids[startIdx:endIdx]

		// Process chunk
		chunkResult, err := b.processDeleteChunk(ctx, chunk, startIdx, chunkIdx)
		if err != nil {
			if b.options.EnableRollback {
				rollbackErr := b.rollbackDeletes(ctx)
				result.RollbackRequired = true
				result.RollbackError = rollbackErr
			}
			result.Duration = time.Since(startTime)
			return result, err
		}

		// Merge chunk results
		b.mergeDeleteChunkResult(result, chunkResult)
		if chunkResult.lastError != nil {
			operationErr = chunkResult.lastError
		}

		// Update progress
		b.progressTracker.update(
			result.Successful+result.Failed,
			result.Successful,
			result.Failed,
			chunkIdx+1,
			chunkResult.lastError,
		)

		// Call progress callbacks
		if b.options.ProgressCallback != nil {
			b.options.ProgressCallback(result.Successful+result.Failed, len(b.ids))
		}
		if b.options.DetailedProgress != nil {
			b.options.DetailedProgress(b.progressTracker.getProgress())
		}

		// Check for early termination on fail-fast
		if b.options.FailFast && result.Failed > 0 {
			if b.options.EnableRollback {
				rollbackErr := b.rollbackDeletes(ctx)
				result.RollbackRequired = true
				result.RollbackError = rollbackErr
			}
			result.Duration = time.Since(startTime)
			if operationErr == nil {
				operationErr = fmt.Errorf("batch delete failed")
			}
			return result, operationErr
		}

		// Check for context cancellation
		select {
		case <-ctx.Done():
			if b.options.EnableRollback {
				rollbackErr := b.rollbackDeletes(ctx)
				result.RollbackRequired = true
				result.RollbackError = rollbackErr
			}
			result.Duration = time.Since(startTime)
			return result, ctx.Err()
		default:
		}
	}

	result.Duration = time.Since(startTime)
	return result, nil
}

// processDeleteChunk processes a single chunk of deletes
func (b *BatchDelete) processDeleteChunk(ctx context.Context, chunk []string, startIndex int, chunkIdx int) (*chunkResult, error) {
	result := &chunkResult{
		items:  make([]*BatchItemResult, 0, len(chunk)),
		errors: make(map[int]error),
	}

	for i, id := range chunk {
		globalIndex := startIndex + i
		itemResult := &BatchItemResult{
			Index:     globalIndex,
			ID:        id,
			Timestamp: time.Now(),
		}

		// Process with retries
		success, err := b.processDeleteWithRetries(ctx, id, itemResult)
		itemResult.Success = success
		itemResult.Error = err

		if err != nil {
			result.errors[globalIndex] = err
			result.lastError = err
			itemResult.BatchErrorCode = b.categorizeDeleteError(err)

			if b.options.ErrorCallback != nil {
				b.options.ErrorCallback(itemResult, err)
			}
		} else {
			// Track successful delete for rollback (store the deleted entry)
			// TODO: Retrieve the entry before deletion for rollback
			b.rollbackMutex.Lock()
			// This would store the actual entry: b.deletedEntries = append(b.deletedEntries, entry)
			b.rollbackMutex.Unlock()
		}

		result.items = append(result.items, itemResult)

		// Check for context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}

	return result, nil
}

// processDeleteWithRetries processes a single delete with retry logic
func (b *BatchDelete) processDeleteWithRetries(ctx context.Context, id string, itemResult *BatchItemResult) (bool, error) {
	var lastErr error

	for attempt := 0; attempt <= b.options.MaxRetries; attempt++ {
		itemResult.Retries = attempt

		// Validate ID
		if id == "" {
			return false, fmt.Errorf("ID cannot be empty")
		}

		// Store original entry for rollback before deleting
		var originalEntry *VectorEntry
		if b.options.EnableRollback {
			_ = b.collection.storage.Iterate(ctx, func(entry *index.VectorEntry) error {
				if entry.ID == id {
					originalEntry = &VectorEntry{
						ID:       entry.ID,
						Vector:   make([]float32, len(entry.Vector)),
						Metadata: make(map[string]interface{}),
					}
					copy(originalEntry.Vector, entry.Vector)
					for k, v := range entry.Metadata {
						originalEntry.Metadata[k] = v
					}
					return fmt.Errorf("found") // Use error to break iteration
				}
				return nil
			})
			// Ignore the "found" error - it's just to break iteration
		}

		// Use the new Delete method
		err := b.collection.Delete(ctx, id)

		// Store original entry for rollback if delete was successful
		if err == nil && originalEntry != nil {
			b.rollbackMutex.Lock()
			b.deletedEntries = append(b.deletedEntries, originalEntry)
			b.rollbackMutex.Unlock()
		}

		if err == nil {
			return true, nil
		}

		lastErr = err

		// Don't retry validation errors or context cancellation
		if b.isNonRetryableDeleteError(err) || ctx.Err() != nil {
			break
		}

		// Wait before retry
		if attempt < b.options.MaxRetries && b.options.RetryDelay > 0 {
			select {
			case <-time.After(b.options.RetryDelay):
			case <-ctx.Done():
				return false, ctx.Err()
			}
		}
	}

	return false, lastErr
}

// mergeDeleteChunkResult merges chunk results into the main result
func (b *BatchDelete) mergeDeleteChunkResult(result *BatchResult, chunkResult *chunkResult) {
	for _, item := range chunkResult.items {
		result.Items = append(result.Items, item)
		if item.Success {
			result.Successful++
		} else {
			result.Failed++
		}
	}

	for idx, err := range chunkResult.errors {
		result.Errors[idx] = err
	}
}

// rollbackDeletes restores all successfully deleted items
func (b *BatchDelete) rollbackDeletes(ctx context.Context) error {
	b.rollbackMutex.Lock()
	defer b.rollbackMutex.Unlock()

	if len(b.deletedEntries) == 0 {
		return nil
	}

	var rollbackErrors []error

	// Re-insert deleted entries
	for _, entry := range b.deletedEntries {
		if err := b.collection.Insert(ctx, entry.ID, entry.Vector, entry.Metadata); err != nil {
			rollbackErrors = append(rollbackErrors, fmt.Errorf("failed to rollback delete for %s: %w", entry.ID, err))
		}
	}

	if len(rollbackErrors) > 0 {
		return fmt.Errorf("rollback failed for %d items", len(rollbackErrors))
	}

	b.deletedEntries = b.deletedEntries[:0]
	return nil
}

// categorizeDeleteError categorizes delete errors
func (b *BatchDelete) categorizeDeleteError(err error) string {
	if err == nil {
		return ""
	}

	errStr := err.Error()
	switch {
	case err == context.DeadlineExceeded:
		return BatchErrorTimeout
	case err == context.Canceled:
		return BatchErrorCancellation
	case contains(errStr, "empty"):
		return BatchErrorValidation
	case contains(errStr, "not found"):
		return BatchErrorNotFound
	case contains(errStr, "not yet implemented"):
		return BatchErrorInternal
	default:
		return BatchErrorInternal
	}
}

// isNonRetryableDeleteError determines if a delete error should not be retried
func (b *BatchDelete) isNonRetryableDeleteError(err error) bool {
	if err == nil {
		return false
	}

	errStr := err.Error()
	return contains(errStr, "empty") ||
		contains(errStr, "not yet implemented")
}
