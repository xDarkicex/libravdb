package libravdb

import (
	"context"
	"fmt"
	"runtime"
	"time"
)

// MemoryPressureRecoveryStrategy handles memory pressure recovery
type MemoryPressureRecoveryStrategy struct {
	memoryManager MemoryManager
}

// NewMemoryPressureRecoveryStrategy creates a new memory pressure recovery strategy
func NewMemoryPressureRecoveryStrategy(memoryManager MemoryManager) *MemoryPressureRecoveryStrategy {
	return &MemoryPressureRecoveryStrategy{
		memoryManager: memoryManager,
	}
}

// CanRecover checks if memory pressure can be recovered from
func (mprs *MemoryPressureRecoveryStrategy) CanRecover(err *VectorDBError) bool {
	return err.Code == ErrCodeMemoryExhausted || err.Code == ErrCodeMemoryPressure
}

// Recover attempts to recover from memory pressure
func (mprs *MemoryPressureRecoveryStrategy) Recover(ctx context.Context, err *VectorDBError) error {
	if mprs.memoryManager == nil {
		return fmt.Errorf("memory manager not available for recovery")
	}

	// Step 1: Force garbage collection
	runtime.GC()
	runtime.GC() // Run twice for better cleanup

	// Step 2: Get current memory usage
	usage := mprs.memoryManager.GetUsage()

	// Step 3: If still over limit, try to handle memory limit exceeded
	if usage.Limit > 0 && usage.Total > usage.Limit {
		if handleErr := mprs.memoryManager.HandleMemoryLimitExceeded(); handleErr != nil {
			return fmt.Errorf("failed to handle memory limit exceeded: %w", handleErr)
		}
	}

	// Step 4: Verify recovery was successful
	finalUsage := mprs.memoryManager.GetUsage()
	if finalUsage.Limit > 0 && finalUsage.Total > finalUsage.Limit {
		return fmt.Errorf("memory usage still exceeds limit after recovery: %d > %d",
			finalUsage.Total, finalUsage.Limit)
	}

	return nil
}

// GetRecoveryAction returns the recovery action type
func (mprs *MemoryPressureRecoveryStrategy) GetRecoveryAction() RecoveryAction {
	return RecoveryGracefulDegradation
}

// QuantizationRecoveryStrategy handles quantization failures
type QuantizationRecoveryStrategy struct {
	fallbackToUncompressed bool
}

// NewQuantizationRecoveryStrategy creates a new quantization recovery strategy
func NewQuantizationRecoveryStrategy(fallbackToUncompressed bool) *QuantizationRecoveryStrategy {
	return &QuantizationRecoveryStrategy{
		fallbackToUncompressed: fallbackToUncompressed,
	}
}

// CanRecover checks if quantization errors can be recovered from
func (qrs *QuantizationRecoveryStrategy) CanRecover(err *VectorDBError) bool {
	return err.Code == ErrCodeQuantizationFailure ||
		err.Code == ErrCodeQuantizationCorruption ||
		err.Code == ErrCodeQuantizationTraining
}

// Recover attempts to recover from quantization failures
func (qrs *QuantizationRecoveryStrategy) Recover(ctx context.Context, err *VectorDBError) error {
	if !qrs.fallbackToUncompressed {
		return fmt.Errorf("quantization recovery disabled")
	}

	// For quantization failures, we can fall back to uncompressed storage
	// This would need to be implemented in the specific quantizer or collection
	// For now, we'll mark it as recoverable and let the caller handle the fallback

	if err.Context != nil && err.Context.Metadata != nil {
		err.Context.Metadata["recovery_action"] = "fallback_to_uncompressed"
		err.Context.Metadata["recovery_timestamp"] = time.Now()
	}

	return nil
}

// GetRecoveryAction returns the recovery action type
func (qrs *QuantizationRecoveryStrategy) GetRecoveryAction() RecoveryAction {
	return RecoveryFallback
}

// IndexCorruptionRecoveryStrategy handles index corruption
type IndexCorruptionRecoveryStrategy struct {
	allowRebuild bool
}

// NewIndexCorruptionRecoveryStrategy creates a new index corruption recovery strategy
func NewIndexCorruptionRecoveryStrategy(allowRebuild bool) *IndexCorruptionRecoveryStrategy {
	return &IndexCorruptionRecoveryStrategy{
		allowRebuild: allowRebuild,
	}
}

// CanRecover checks if index corruption can be recovered from
func (icrs *IndexCorruptionRecoveryStrategy) CanRecover(err *VectorDBError) bool {
	return err.Code == ErrCodeIndexCorruption || err.Code == ErrCodeIndexFailure
}

// Recover attempts to recover from index corruption
func (icrs *IndexCorruptionRecoveryStrategy) Recover(ctx context.Context, err *VectorDBError) error {
	if !icrs.allowRebuild {
		return fmt.Errorf("index rebuild recovery disabled")
	}

	// Mark for rebuild - the actual rebuild would be handled by the collection
	if err.Context != nil && err.Context.Metadata != nil {
		err.Context.Metadata["recovery_action"] = "rebuild_index"
		err.Context.Metadata["recovery_timestamp"] = time.Now()
	}

	return nil
}

// GetRecoveryAction returns the recovery action type
func (icrs *IndexCorruptionRecoveryStrategy) GetRecoveryAction() RecoveryAction {
	return RecoveryRebuild
}

// BatchOperationRecoveryStrategy handles batch operation failures
type BatchOperationRecoveryStrategy struct {
	maxRetries   int
	retryBackoff time.Duration
	allowPartial bool
}

// NewBatchOperationRecoveryStrategy creates a new batch operation recovery strategy
func NewBatchOperationRecoveryStrategy(maxRetries int, retryBackoff time.Duration, allowPartial bool) *BatchOperationRecoveryStrategy {
	return &BatchOperationRecoveryStrategy{
		maxRetries:   maxRetries,
		retryBackoff: retryBackoff,
		allowPartial: allowPartial,
	}
}

// CanRecover checks if batch operation errors can be recovered from
func (bors *BatchOperationRecoveryStrategy) CanRecover(err *VectorDBError) bool {
	return err.Code == ErrCodeBatchFailure || err.Code == ErrCodeBatchTimeout
}

// Recover attempts to recover from batch operation failures
func (bors *BatchOperationRecoveryStrategy) Recover(ctx context.Context, err *VectorDBError) error {
	if err.RetryCount >= bors.maxRetries {
		if bors.allowPartial {
			// Allow partial success
			if err.Context != nil && err.Context.Metadata != nil {
				err.Context.Metadata["recovery_action"] = "allow_partial_success"
				err.Context.Metadata["recovery_timestamp"] = time.Now()
			}
			return nil
		}
		return fmt.Errorf("batch operation failed after %d retries", bors.maxRetries)
	}

	// Apply backoff before retry
	if err.RetryCount > 0 {
		backoff := bors.retryBackoff * time.Duration(err.RetryCount)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(backoff):
		}
	}

	// Mark for retry
	if err.Context != nil && err.Context.Metadata != nil {
		err.Context.Metadata["recovery_action"] = "retry_batch_operation"
		err.Context.Metadata["retry_attempt"] = err.RetryCount + 1
		err.Context.Metadata["recovery_timestamp"] = time.Now()
	}

	return nil
}

// GetRecoveryAction returns the recovery action type
func (bors *BatchOperationRecoveryStrategy) GetRecoveryAction() RecoveryAction {
	return RecoveryRetry
}

// Note: GracefulDegradationManager, DegradationConfig, and NewGracefulDegradationManager
// are now defined in degradation.go with enhanced functionality

// Note: HandleMemoryPressure, MemoryManager interface, and MemoryUsage struct
// are now defined in degradation.go with enhanced functionality
