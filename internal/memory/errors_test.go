package memory

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestMemoryError(t *testing.T) {
	t.Run("basic error creation", func(t *testing.T) {
		err := NewMemoryError(
			ErrMemLimitExceeded,
			"MemoryManager",
			"SetLimit",
			"memory limit exceeded",
		)

		if err.Code != ErrMemLimitExceeded {
			t.Errorf("expected code %d, got %d", ErrMemLimitExceeded, err.Code)
		}

		if err.Component != "MemoryManager" {
			t.Errorf("expected component 'MemoryManager', got '%s'", err.Component)
		}

		if err.Operation != "SetLimit" {
			t.Errorf("expected operation 'SetLimit', got '%s'", err.Operation)
		}

		if err.Message != "memory limit exceeded" {
			t.Errorf("expected message 'memory limit exceeded', got '%s'", err.Message)
		}
	})

	t.Run("error with cause and usage", func(t *testing.T) {
		cause := errors.New("out of memory")
		usage := MemoryUsage{
			Total: 1200,
			Limit: 1000,
		}

		err := NewMemoryError(
			ErrMemPressureCritical,
			"MemoryManager",
			"checkMemoryUsage",
			"critical memory pressure detected",
		).WithCause(cause).
			WithRetryable(true).
			WithRecoverable(true).
			WithUsage(usage).
			WithMetadata("pressure_level", "critical")

		if err.Cause != cause {
			t.Error("expected cause to be set")
		}

		if !err.Retryable {
			t.Error("expected error to be retryable")
		}

		if !err.Recoverable {
			t.Error("expected error to be recoverable")
		}

		if err.Usage == nil {
			t.Fatal("expected usage to be set")
		}

		if err.Usage.Total != 1200 {
			t.Errorf("expected usage total 1200, got %d", err.Usage.Total)
		}

		if err.Metadata["pressure_level"] != "critical" {
			t.Error("expected pressure_level metadata to be set")
		}

		if !errors.Is(err, cause) {
			t.Error("expected error to wrap cause")
		}
	})
}

func TestMemoryRecoveryManager(t *testing.T) {
	t.Run("successful lightweight recovery", func(t *testing.T) {
		// Create a mock that simulates memory reduction after GC
		mockManager := &mockMemoryManager{
			usage: MemoryUsage{
				Total: 1100, // Start above limit
				Limit: 1000,
			},
		}

		mrm := NewMemoryRecoveryManager(mockManager)

		err := NewMemoryError(
			ErrMemLimitExceeded,
			"MemoryManager",
			"checkUsage",
			"memory limit exceeded",
		).WithRecoverable(true)

		ctx := context.Background()
		if recoveryErr := mrm.RecoverFromMemoryPressure(ctx, err); recoveryErr != nil {
			t.Errorf("expected recovery to succeed, got error: %v", recoveryErr)
		}
	})

	t.Run("moderate recovery with limit handling", func(t *testing.T) {
		mockManager := &mockMemoryManager{
			usage: MemoryUsage{
				Total: 1200,
				Limit: 1000,
			},
			handleLimitSuccess: true,
		}

		mrm := NewMemoryRecoveryManager(mockManager)

		err := NewMemoryError(
			ErrMemLimitExceeded,
			"MemoryManager",
			"checkUsage",
			"memory limit exceeded",
		).WithRecoverable(true)

		ctx := context.Background()
		if recoveryErr := mrm.RecoverFromMemoryPressure(ctx, err); recoveryErr != nil {
			t.Errorf("expected recovery to succeed, got error: %v", recoveryErr)
		}

		if !mockManager.handleLimitCalled {
			t.Error("expected HandleMemoryLimitExceeded to be called")
		}
	})

	t.Run("aggressive recovery", func(t *testing.T) {
		mockManager := &mockMemoryManager{
			usage: MemoryUsage{
				Total: 1500,
				Limit: 1000,
			},
			handleLimitSuccess: true,
		}

		mrm := NewMemoryRecoveryManager(mockManager)
		mrm.maxRecoveryAttempts = 3

		err := NewMemoryError(
			ErrMemPressureCritical,
			"MemoryManager",
			"checkUsage",
			"critical memory pressure",
		).WithRecoverable(true)

		ctx := context.Background()
		if recoveryErr := mrm.RecoverFromMemoryPressure(ctx, err); recoveryErr != nil {
			t.Errorf("expected recovery to succeed, got error: %v", recoveryErr)
		}
	})

	t.Run("recovery failure", func(t *testing.T) {
		mockManager := &mockMemoryManager{
			usage: MemoryUsage{
				Total: 1500,
				Limit: 1000,
			},
			handleLimitError: errors.New("handle limit failed"),
		}

		mrm := NewMemoryRecoveryManager(mockManager)
		mrm.maxRecoveryAttempts = 2
		mrm.recoveryBackoff = time.Millisecond * 10

		err := NewMemoryError(
			ErrMemPressureCritical,
			"MemoryManager",
			"checkUsage",
			"critical memory pressure",
		).WithRecoverable(true)

		ctx := context.Background()
		if recoveryErr := mrm.RecoverFromMemoryPressure(ctx, err); recoveryErr == nil {
			t.Error("expected recovery to fail")
		}
	})

	t.Run("no memory manager", func(t *testing.T) {
		mrm := NewMemoryRecoveryManager(nil)

		err := NewMemoryError(
			ErrMemLimitExceeded,
			"MemoryManager",
			"checkUsage",
			"memory limit exceeded",
		)

		ctx := context.Background()
		if recoveryErr := mrm.RecoverFromMemoryPressure(ctx, err); recoveryErr == nil {
			t.Error("expected recovery to fail when no memory manager is available")
		}
	})

	t.Run("context cancellation", func(t *testing.T) {
		mockManager := &mockMemoryManager{
			usage: MemoryUsage{
				Total: 1200,
				Limit: 1000,
			},
			handleLimitError: errors.New("always fail"),
		}

		mrm := NewMemoryRecoveryManager(mockManager)
		mrm.recoveryBackoff = time.Second

		err := NewMemoryError(
			ErrMemLimitExceeded,
			"MemoryManager",
			"checkUsage",
			"memory limit exceeded",
		)

		ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond*100)
		defer cancel()

		recoveryErr := mrm.RecoverFromMemoryPressure(ctx, err)
		if recoveryErr == nil {
			t.Error("expected recovery to fail due to context cancellation")
		}

		if !errors.Is(recoveryErr, context.DeadlineExceeded) {
			t.Error("expected context deadline exceeded error")
		}
	})
}

func TestMemoryHealthMonitor(t *testing.T) {
	t.Run("start and stop monitor", func(t *testing.T) {
		mockManager := &mockMemoryManager{
			usage: MemoryUsage{
				Total: 500,
				Limit: 1000,
			},
		}

		mrm := NewMemoryRecoveryManager(mockManager)
		monitor := NewMemoryHealthMonitor(mockManager, mrm)
		monitor.interval = time.Millisecond * 10

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		if err := monitor.Start(ctx); err != nil {
			t.Errorf("expected monitor to start successfully, got error: %v", err)
		}

		// Let it run for a short time
		time.Sleep(time.Millisecond * 50)

		if err := monitor.Stop(); err != nil {
			t.Errorf("expected monitor to stop successfully, got error: %v", err)
		}
	})

	t.Run("double start error", func(t *testing.T) {
		mockManager := &mockMemoryManager{}
		monitor := NewMemoryHealthMonitor(mockManager, nil)

		ctx := context.Background()

		if err := monitor.Start(ctx); err != nil {
			t.Errorf("expected first start to succeed, got error: %v", err)
		}

		if err := monitor.Start(ctx); err == nil {
			t.Error("expected second start to fail")
		}

		monitor.Stop()
	})

	t.Run("stop without start error", func(t *testing.T) {
		mockManager := &mockMemoryManager{}
		monitor := NewMemoryHealthMonitor(mockManager, nil)

		if err := monitor.Stop(); err == nil {
			t.Error("expected stop without start to fail")
		}
	})

	t.Run("memory pressure detection", func(t *testing.T) {
		mockManager := &mockMemoryManager{
			usage: MemoryUsage{
				Total: 950, // 95% of limit
				Limit: 1000,
			},
		}

		recoveryManager := NewMemoryRecoveryManager(mockManager)

		monitor := NewMemoryHealthMonitor(mockManager, recoveryManager)
		monitor.interval = time.Millisecond * 10
		monitor.thresholds.RecoveryThreshold = 0.9 // 90%

		ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond*100)
		defer cancel()

		monitor.Start(ctx)

		// Wait for monitoring to detect pressure
		time.Sleep(time.Millisecond * 50)

		monitor.Stop()

		// For this test, we just verify that the monitor runs without error
		// In a real implementation, we would check if recovery was triggered
	})

	t.Run("no limit set", func(t *testing.T) {
		mockManager := &mockMemoryManager{
			usage: MemoryUsage{
				Total: 1000,
				Limit: 0, // No limit
			},
		}

		recoveryManager := NewMemoryRecoveryManager(mockManager)

		monitor := NewMemoryHealthMonitor(mockManager, recoveryManager)
		monitor.interval = time.Millisecond * 10

		ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond*50)
		defer cancel()

		monitor.Start(ctx)

		// Wait for monitoring
		time.Sleep(time.Millisecond * 30)

		monitor.Stop()

		// For this test, we just verify that the monitor runs without error when no limit is set
	})
}

// Mock implementations for testing

type mockMemoryManager struct {
	usage              MemoryUsage
	handleLimitCalled  bool
	handleLimitError   error
	handleLimitSuccess bool
	gcCalled           bool
	getUsageCallCount  int
}

func (mmm *mockMemoryManager) SetLimit(bytes int64) error {
	mmm.usage.Limit = bytes
	return nil
}

func (mmm *mockMemoryManager) GetUsage() MemoryUsage {
	mmm.getUsageCallCount++

	// Simulate memory reduction after first call (simulating GC effect)
	if mmm.getUsageCallCount > 1 && mmm.usage.Total > 200 {
		mmm.usage.Total -= 200
	}

	// Return a copy to avoid race conditions
	return MemoryUsage{
		Total:        mmm.usage.Total,
		Indices:      mmm.usage.Indices,
		Caches:       mmm.usage.Caches,
		Quantized:    mmm.usage.Quantized,
		MemoryMapped: mmm.usage.MemoryMapped,
		Available:    mmm.usage.Available,
		Limit:        mmm.usage.Limit,
		Timestamp:    mmm.usage.Timestamp,
	}
}

func (mmm *mockMemoryManager) TriggerGC() error {
	mmm.gcCalled = true
	// Simulate significant memory reduction after GC
	if mmm.usage.Total > 200 {
		mmm.usage.Total -= 200
	}
	return nil
}

func (mmm *mockMemoryManager) RegisterCache(name string, cache Cache) error {
	return nil
}

func (mmm *mockMemoryManager) UnregisterCache(name string) error {
	return nil
}

func (mmm *mockMemoryManager) RegisterMemoryMappable(name string, mappable MemoryMappable) error {
	return nil
}

func (mmm *mockMemoryManager) UnregisterMemoryMappable(name string) error {
	return nil
}

func (mmm *mockMemoryManager) OnMemoryPressure(callback func(usage MemoryUsage)) {
	// No-op for testing
}

func (mmm *mockMemoryManager) OnMemoryRelease(callback func(freed int64)) {
	// No-op for testing
}

func (mmm *mockMemoryManager) HandleMemoryLimitExceeded() error {
	mmm.handleLimitCalled = true

	if mmm.handleLimitError != nil {
		return mmm.handleLimitError
	}

	if mmm.handleLimitSuccess {
		// Simulate successful memory reduction
		mmm.usage.Total = mmm.usage.Limit - 100
	}

	return nil
}

func (mmm *mockMemoryManager) Start(ctx context.Context) error {
	return nil
}

func (mmm *mockMemoryManager) Stop() error {
	return nil
}
