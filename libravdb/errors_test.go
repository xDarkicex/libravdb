package libravdb

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestVectorDBError(t *testing.T) {
	t.Run("basic error creation", func(t *testing.T) {
		err := NewVectorDBError(ErrCodeMemoryExhausted, "memory limit exceeded", true)

		if err.Code != ErrCodeMemoryExhausted {
			t.Errorf("expected code %d, got %d", ErrCodeMemoryExhausted, err.Code)
		}

		if err.Message != "memory limit exceeded" {
			t.Errorf("expected message 'memory limit exceeded', got '%s'", err.Message)
		}

		if !err.Retryable {
			t.Error("expected error to be retryable")
		}

		if err.Severity != SeverityError {
			t.Errorf("expected severity %d, got %d", SeverityError, err.Severity)
		}
	})

	t.Run("error with context", func(t *testing.T) {
		err := NewVectorDBErrorWithContext(
			ErrCodeQuantizationFailure,
			"quantization training failed",
			true,
			"ProductQuantizer",
			"Train",
		)

		if err.Context == nil {
			t.Fatal("expected context to be set")
		}

		if err.Context.Component != "ProductQuantizer" {
			t.Errorf("expected component 'ProductQuantizer', got '%s'", err.Context.Component)
		}

		if err.Context.Operation != "Train" {
			t.Errorf("expected operation 'Train', got '%s'", err.Context.Operation)
		}

		if err.Context.StackTrace == "" {
			t.Error("expected stack trace to be captured")
		}
	})

	t.Run("error chaining", func(t *testing.T) {
		cause := errors.New("underlying error")
		err := NewVectorDBError(ErrCodeStorageFailure, "storage operation failed", false).
			WithCause(cause).
			WithSeverity(SeverityCritical).
			WithRecoveryAction(RecoveryRestart)

		if err.Cause != cause {
			t.Error("expected cause to be set")
		}

		if err.Severity != SeverityCritical {
			t.Errorf("expected severity %d, got %d", SeverityCritical, err.Severity)
		}

		if err.RecoveryAction != RecoveryRestart {
			t.Errorf("expected recovery action %d, got %d", RecoveryRestart, err.RecoveryAction)
		}

		if !errors.Is(err, cause) {
			t.Error("expected error to wrap cause")
		}
	})

	t.Run("metadata and request ID", func(t *testing.T) {
		err := NewVectorDBError(ErrCodeTimeout, "operation timed out", true).
			WithMetadata("timeout_duration", "30s").
			WithMetadata("operation_id", "op-123").
			WithRequestID("req-456")

		if err.Context == nil {
			t.Fatal("expected context to be set")
		}

		if err.Context.RequestID != "req-456" {
			t.Errorf("expected request ID 'req-456', got '%s'", err.Context.RequestID)
		}

		if err.Context.Metadata["timeout_duration"] != "30s" {
			t.Error("expected timeout_duration metadata to be set")
		}

		if err.Context.Metadata["operation_id"] != "op-123" {
			t.Error("expected operation_id metadata to be set")
		}
	})

	t.Run("retry logic", func(t *testing.T) {
		err := NewVectorDBError(ErrCodeRateLimited, "rate limit exceeded", true)
		err.MaxRetries = 3

		if !err.IsRetryable() {
			t.Error("expected error to be retryable initially")
		}

		// Simulate retries
		for i := 0; i < 3; i++ {
			err.IncrementRetry()
		}

		if err.IsRetryable() {
			t.Error("expected error to not be retryable after max retries")
		}

		if err.RetryCount != 3 {
			t.Errorf("expected retry count 3, got %d", err.RetryCount)
		}
	})
}

func TestErrorRecoveryManager(t *testing.T) {
	t.Run("register and recover", func(t *testing.T) {
		erm := NewErrorRecoveryManager()

		// Mock recovery strategy
		strategy := &mockRecoveryStrategy{
			canRecover: true,
			shouldFail: false,
		}

		erm.RegisterRecoveryStrategy(ErrCodeMemoryExhausted, strategy)

		err := NewVectorDBError(ErrCodeMemoryExhausted, "memory exhausted", true)

		ctx := context.Background()
		if recoveryErr := erm.TryRecover(ctx, err); recoveryErr != nil {
			t.Errorf("expected recovery to succeed, got error: %v", recoveryErr)
		}

		if !strategy.recoverCalled.Load() {
			t.Error("expected recovery strategy to be called")
		}
	})

	t.Run("recovery failure", func(t *testing.T) {
		erm := NewErrorRecoveryManager()

		strategy := &mockRecoveryStrategy{
			canRecover: true,
			shouldFail: true,
		}

		erm.RegisterRecoveryStrategy(ErrCodeMemoryExhausted, strategy)

		err := NewVectorDBError(ErrCodeMemoryExhausted, "memory exhausted", true)

		ctx := context.Background()
		if recoveryErr := erm.TryRecover(ctx, err); recoveryErr == nil {
			t.Error("expected recovery to fail")
		}
	})

	t.Run("no recovery strategy", func(t *testing.T) {
		erm := NewErrorRecoveryManager()

		err := NewVectorDBError(ErrCodeUnknown, "unknown error", false)

		ctx := context.Background()
		if recoveryErr := erm.TryRecover(ctx, err); recoveryErr == nil {
			t.Error("expected recovery to fail when no strategy is registered")
		}
	})

	t.Run("context cancellation", func(t *testing.T) {
		erm := NewErrorRecoveryManager()

		strategy := &mockRecoveryStrategy{
			canRecover: true,
			shouldFail: true,
			delay:      time.Second,
		}

		erm.RegisterRecoveryStrategy(ErrCodeMemoryExhausted, strategy)

		err := NewVectorDBError(ErrCodeMemoryExhausted, "memory exhausted", true)

		ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond*100)
		defer cancel()

		recoveryErr := erm.TryRecover(ctx, err)
		if recoveryErr == nil {
			t.Error("expected recovery to fail due to context cancellation")
		}

		if !errors.Is(recoveryErr, context.DeadlineExceeded) {
			t.Error("expected context deadline exceeded error")
		}
	})
}

func TestMemoryPressureRecoveryStrategy(t *testing.T) {
	t.Run("successful recovery", func(t *testing.T) {
		mockManager := &mockMemoryManager{
			usage: MemoryUsage{
				Total: 1200,
				Limit: 1000,
			},
		}

		strategy := NewMemoryPressureRecoveryStrategy(mockManager)

		err := NewVectorDBError(ErrCodeMemoryExhausted, "memory exhausted", true)

		ctx := context.Background()
		if recoveryErr := strategy.Recover(ctx, err); recoveryErr != nil {
			t.Errorf("expected recovery to succeed, got error: %v", recoveryErr)
		}

		if !mockManager.handleLimitCalled {
			t.Error("expected HandleMemoryLimitExceeded to be called")
		}
	})

	t.Run("recovery failure", func(t *testing.T) {
		mockManager := &mockMemoryManager{
			usage: MemoryUsage{
				Total: 1200,
				Limit: 1000,
			},
			handleLimitError: errors.New("handle limit failed"),
		}

		strategy := NewMemoryPressureRecoveryStrategy(mockManager)

		err := NewVectorDBError(ErrCodeMemoryExhausted, "memory exhausted", true)

		ctx := context.Background()
		if recoveryErr := strategy.Recover(ctx, err); recoveryErr == nil {
			t.Error("expected recovery to fail")
		}
	})

	t.Run("no memory manager", func(t *testing.T) {
		strategy := NewMemoryPressureRecoveryStrategy(nil)

		err := NewVectorDBError(ErrCodeMemoryExhausted, "memory exhausted", true)

		ctx := context.Background()
		if recoveryErr := strategy.Recover(ctx, err); recoveryErr == nil {
			t.Error("expected recovery to fail when no memory manager is available")
		}
	})
}

func TestQuantizationRecoveryStrategy(t *testing.T) {
	t.Run("fallback enabled", func(t *testing.T) {
		strategy := NewQuantizationRecoveryStrategy(true)

		err := NewVectorDBErrorWithContext(
			ErrCodeQuantizationFailure,
			"quantization failed",
			true,
			"ProductQuantizer",
			"Train",
		)

		ctx := context.Background()
		if recoveryErr := strategy.Recover(ctx, err); recoveryErr != nil {
			t.Errorf("expected recovery to succeed, got error: %v", recoveryErr)
		}

		if err.Context.Metadata["recovery_action"] != "fallback_to_uncompressed" {
			t.Error("expected recovery action to be set in metadata")
		}
	})

	t.Run("fallback disabled", func(t *testing.T) {
		strategy := NewQuantizationRecoveryStrategy(false)

		err := NewVectorDBError(ErrCodeQuantizationFailure, "quantization failed", true)

		ctx := context.Background()
		if recoveryErr := strategy.Recover(ctx, err); recoveryErr == nil {
			t.Error("expected recovery to fail when fallback is disabled")
		}
	})
}

func TestGracefulDegradationManager(t *testing.T) {
	t.Run("memory pressure handling", func(t *testing.T) {
		mockManager := &mockMemoryManager{
			usage: MemoryUsage{
				Total:  800,
				Caches: 200,
				Limit:  1000,
			},
		}

		config := DegradationConfig{
			EnableQuantizationFallback: true,
			EnableMemoryMapping:        true,
			EnableCacheEviction:        true,
			EnableIndexSimplification:  true,
			MaxDegradationLevel:        4,
		}

		gdm := NewGracefulDegradationManager(config)
		gdm.SetMemoryManager(mockManager)

		ctx := context.Background()

		// Test different pressure levels
		for level := 1; level <= 4; level++ {
			if err := gdm.HandleMemoryPressure(ctx, level); err != nil {
				t.Errorf("expected pressure level %d to be handled, got error: %v", level, err)
			}
		}
	})

	t.Run("pressure level too high", func(t *testing.T) {
		config := DegradationConfig{
			MaxDegradationLevel: 3,
		}

		gdm := NewGracefulDegradationManager(config)

		ctx := context.Background()
		if err := gdm.HandleMemoryPressure(ctx, 5); err == nil {
			t.Error("expected error for pressure level exceeding maximum")
		}
	})
}

// Mock implementations for testing

type mockRecoveryStrategy struct {
	canRecover    bool
	shouldFail    bool
	delay         time.Duration
	recoverCalled atomic.Bool
}

func (mrs *mockRecoveryStrategy) CanRecover(err *VectorDBError) bool {
	return mrs.canRecover
}

func (mrs *mockRecoveryStrategy) Recover(ctx context.Context, err *VectorDBError) error {
	mrs.recoverCalled.Store(true)

	if mrs.delay > 0 {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(mrs.delay):
		}
	}

	if mrs.shouldFail {
		return errors.New("recovery failed")
	}

	return nil
}

func (mrs *mockRecoveryStrategy) GetRecoveryAction() RecoveryAction {
	return RecoveryRetry
}

type mockMemoryManager struct {
	mu                sync.RWMutex
	usage             MemoryUsage
	handleLimitCalled bool
	handleLimitError  error
	gcTriggered       bool
	cachesEvicted     int64
	mmapEnabled       bool
	shouldFailGC      bool
	shouldFailEvict   bool
	shouldFailMmap    bool
}

func (mmm *mockMemoryManager) GetUsage() MemoryUsage {
	mmm.mu.RLock()
	defer mmm.mu.RUnlock()
	return mmm.usage
}

func (mmm *mockMemoryManager) HandleMemoryLimitExceeded() error {
	mmm.mu.Lock()
	defer mmm.mu.Unlock()
	mmm.handleLimitCalled = true
	if mmm.handleLimitError != nil {
		return mmm.handleLimitError
	}

	// Simulate successful memory reduction
	mmm.usage.Total = mmm.usage.Limit - 100
	return nil
}

func (mmm *mockMemoryManager) TriggerGC() error {
	mmm.mu.Lock()
	defer mmm.mu.Unlock()
	if mmm.shouldFailGC {
		return errors.New("GC failed")
	}
	mmm.gcTriggered = true
	return nil
}

func (mmm *mockMemoryManager) EvictCaches(bytes int64) (int64, error) {
	mmm.mu.Lock()
	defer mmm.mu.Unlock()
	if mmm.shouldFailEvict {
		return 0, errors.New("cache eviction failed")
	}
	mmm.cachesEvicted = bytes
	return bytes, nil
}

func (mmm *mockMemoryManager) EnableMemoryMapping() error {
	mmm.mu.Lock()
	defer mmm.mu.Unlock()
	if mmm.shouldFailMmap {
		return errors.New("memory mapping failed")
	}
	mmm.mmapEnabled = true
	return nil
}

func (mmm *mockMemoryManager) DisableMemoryMapping() error {
	mmm.mu.Lock()
	defer mmm.mu.Unlock()
	mmm.mmapEnabled = false
	return nil
}
