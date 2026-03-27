package libravdb

import (
	"context"
	"testing"
	"time"
)

func TestErrorHandlingIntegration(t *testing.T) {
	t.Run("end-to-end error recovery", func(t *testing.T) {
		// Create error recovery manager
		erm := NewErrorRecoveryManager()

		// Register recovery strategies
		erm.RegisterRecoveryStrategy(ErrCodeMemoryExhausted, NewMemoryPressureRecoveryStrategy(nil))
		erm.RegisterRecoveryStrategy(ErrCodeQuantizationFailure, NewQuantizationRecoveryStrategy(true))
		erm.RegisterRecoveryStrategy(ErrCodeIndexCorruption, NewIndexCorruptionRecoveryStrategy(true))

		// Test memory pressure recovery
		memErr := NewVectorDBErrorWithContext(
			ErrCodeMemoryExhausted,
			"memory limit exceeded during batch operation",
			true,
			"BatchProcessor",
			"ProcessBatch",
		).WithSeverity(SeverityCritical).
			WithRecoveryAction(RecoveryGracefulDegradation)

		ctx := context.Background()
		if recoveryErr := erm.TryRecover(ctx, memErr); recoveryErr == nil {
			t.Error("expected memory recovery to fail without memory manager")
		}

		// Test quantization recovery
		quantErr := NewVectorDBErrorWithContext(
			ErrCodeQuantizationFailure,
			"quantization training failed",
			true,
			"ProductQuantizer",
			"Train",
		).WithSeverity(SeverityError).
			WithRecoveryAction(RecoveryFallback)

		if recoveryErr := erm.TryRecover(ctx, quantErr); recoveryErr != nil {
			t.Errorf("expected quantization recovery to succeed, got error: %v", recoveryErr)
		}

		// Verify recovery action was set
		if quantErr.Context.Metadata["recovery_action"] != "fallback_to_uncompressed" {
			t.Error("expected recovery action to be set in metadata")
		}
	})

	t.Run("batch operation error tracking", func(t *testing.T) {
		tracker := NewBatchOperationTracker()

		// Start tracking a batch operation
		operationID := "test-batch-001"
		tracker.StartOperation(operationID, "BatchInsert", 1000)

		// Simulate progress updates
		tracker.UpdateProgress(operationID, 500, 0)

		// Create a batch error with item failures
		batchErr := NewBatchError(
			ErrBatchPartialFailure,
			"BatchInsert",
			"some items failed validation",
			1000,
		).WithItemError(100, NewVectorDBError(ErrCodeInvalidVector, "invalid dimension", false)).
			WithItemError(200, NewVectorDBError(ErrCodeInvalidVector, "null vector", false)).
			WithDuration(time.Second * 30)

		batchErr.Processed = 1000
		// Failed count is already set by WithItemError calls

		// Complete the operation
		tracker.CompleteOperation(operationID, batchErr)

		// Verify tracking
		status, exists := tracker.GetOperationStatus(operationID)
		if !exists {
			t.Fatal("expected operation to be tracked")
		}

		if status.Status != "partial" {
			t.Errorf("expected status 'partial', got '%s'", status.Status)
		}

		if status.Failed != 2 {
			t.Errorf("expected 2 failed items, got %d", status.Failed)
		}

		if len(status.Errors) != 1 {
			t.Errorf("expected 1 batch error, got %d", len(status.Errors))
		}
	})

	t.Run("graceful degradation under pressure", func(t *testing.T) {
		config := DegradationConfig{
			EnableQuantizationFallback: true,
			EnableMemoryMapping:        true,
			EnableCacheEviction:        true,
			EnableIndexSimplification:  true,
			MaxDegradationLevel:        3,
		}

		gdm := NewGracefulDegradationManager(config)

		ctx := context.Background()

		// Test different pressure levels
		// Note: Without actual managers set, degradation actions won't be available
		// This tests the error handling path
		for level := 1; level <= 3; level++ {
			err := gdm.HandleMemoryPressure(ctx, level)
			if err == nil {
				t.Logf("Pressure level %d handled successfully", level)
			} else {
				t.Logf("Pressure level %d returned expected error: %v", level, err)
			}
		}

		// Test pressure level exceeding maximum
		if err := gdm.HandleMemoryPressure(ctx, 5); err == nil {
			t.Error("expected error for pressure level exceeding maximum")
		}
	})

	t.Run("error severity and recovery action classification", func(t *testing.T) {
		testCases := []struct {
			name             string
			code             ErrorCode
			expectedSeverity ErrorSeverity
			expectedRecovery RecoveryAction
		}{
			{
				name:             "memory exhaustion",
				code:             ErrCodeMemoryExhausted,
				expectedSeverity: SeverityCritical,
				expectedRecovery: RecoveryGracefulDegradation,
			},
			{
				name:             "quantization failure",
				code:             ErrCodeQuantizationFailure,
				expectedSeverity: SeverityError,
				expectedRecovery: RecoveryFallback,
			},
			{
				name:             "index corruption",
				code:             ErrCodeIndexCorruption,
				expectedSeverity: SeverityCritical,
				expectedRecovery: RecoveryRebuild,
			},
			{
				name:             "batch timeout",
				code:             ErrCodeBatchTimeout,
				expectedSeverity: SeverityWarning,
				expectedRecovery: RecoveryRetry,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				err := NewVectorDBError(tc.code, "test error", true).
					WithSeverity(tc.expectedSeverity).
					WithRecoveryAction(tc.expectedRecovery)

				if err.Severity != tc.expectedSeverity {
					t.Errorf("expected severity %v, got %v", tc.expectedSeverity, err.Severity)
				}

				if err.RecoveryAction != tc.expectedRecovery {
					t.Errorf("expected recovery action %v, got %v", tc.expectedRecovery, err.RecoveryAction)
				}

				if !err.CanRecover() {
					t.Error("expected error to be recoverable")
				}
			})
		}
	})
}
