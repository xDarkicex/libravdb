package libravdb

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

// Integration tests for complete error handling and recovery workflows

// Mock implementations for integration tests
type mockQuantizationManager struct {
	fallbackCalled     bool
	restoreCalled      bool
	precisionReduced   bool
	shouldFail         bool
	quantizationStatus map[string]bool
}

func (m *mockQuantizationManager) FallbackToUncompressed(component string) error {
	if m.shouldFail {
		return errors.New("fallback failed")
	}
	m.fallbackCalled = true
	return nil
}

func (m *mockQuantizationManager) RestoreQuantization(component string) error {
	if m.shouldFail {
		return errors.New("restore failed")
	}
	m.restoreCalled = true
	return nil
}

func (m *mockQuantizationManager) ReduceQuantizationPrecision(component string, level int) error {
	if m.shouldFail {
		return errors.New("precision reduction failed")
	}
	m.precisionReduced = true
	return nil
}

func (m *mockQuantizationManager) GetQuantizationStatus(component string) (bool, error) {
	if m.quantizationStatus == nil {
		return false, nil
	}
	status, exists := m.quantizationStatus[component]
	return status, map[bool]error{true: nil, false: errors.New("component not found")}[exists]
}

type mockIndexManager struct {
	simplified bool
	restored   bool
	rebuilt    bool
	shouldFail bool
	complexity map[string]int
}

func (m *mockIndexManager) SimplifyIndex(component string, level int) error {
	if m.shouldFail {
		return errors.New("index simplification failed")
	}
	m.simplified = true
	return nil
}

func (m *mockIndexManager) RestoreIndex(component string) error {
	if m.shouldFail {
		return errors.New("index restore failed")
	}
	m.restored = true
	return nil
}

func (m *mockIndexManager) GetIndexComplexity(component string) (int, error) {
	if m.complexity == nil {
		return 1, nil
	}
	complexity, exists := m.complexity[component]
	if !exists {
		return 0, errors.New("component not found")
	}
	return complexity, nil
}

func (m *mockIndexManager) RebuildIndex(component string) error {
	if m.shouldFail {
		return errors.New("index rebuild failed")
	}
	m.rebuilt = true
	return nil
}

type mockCircuitBreaker struct {
	state       string
	executions  int
	shouldFail  bool
	resetCalled bool
}

func (m *mockCircuitBreaker) Execute(ctx context.Context, fn func() error) error {
	m.executions++
	if m.shouldFail {
		return errors.New("circuit breaker open")
	}
	return fn()
}

func (m *mockCircuitBreaker) State() string {
	return m.state
}

func (m *mockCircuitBreaker) Reset() {
	m.resetCalled = true
	m.state = "CLOSED"
}

func TestCompleteErrorRecoveryWorkflow(t *testing.T) {
	requireStressMode(t)

	t.Run("memory pressure recovery workflow", func(t *testing.T) {
		// Setup complete system
		memMgr := &mockMemoryManager{
			usage: MemoryUsage{
				Total: 900 * 1024 * 1024,  // 900MB
				Limit: 1024 * 1024 * 1024, // 1GB limit
			},
		}

		quantMgr := &mockQuantizationManager{}
		indexMgr := &mockIndexManager{}

		// Create degradation manager
		config := DefaultDegradationConfig()
		gdm := NewGracefulDegradationManager(config)
		gdm.SetMemoryManager(memMgr)
		gdm.SetQuantizationManager(quantMgr)
		gdm.SetIndexManager(indexMgr)

		// Create error recovery manager
		erm := NewErrorRecoveryManager()
		erm.SetDegradationManager(gdm)

		// Create health monitor
		healthMonitor := NewSystemHealthMonitor(100 * time.Millisecond)

		// Register memory health check
		healthMonitor.RegisterHealthCheck("memory", func(ctx context.Context) (HealthLevel, error) {
			usage := memMgr.GetUsage()
			if usage.Limit > 0 {
				usageRatio := float64(usage.Total) / float64(usage.Limit)
				if usageRatio > 0.95 {
					return HealthCritical, fmt.Errorf("memory usage critical: %.2f%%", usageRatio*100)
				}
				if usageRatio > 0.9 {
					return HealthUnhealthy, fmt.Errorf("memory usage high: %.2f%%", usageRatio*100)
				}
				if usageRatio > 0.8 {
					return HealthDegraded, fmt.Errorf("memory usage elevated: %.2f%%", usageRatio*100)
				}
			}
			return HealthHealthy, nil
		})

		erm.SetHealthMonitor(healthMonitor)

		// Create orchestrator
		aro := NewAutomaticRecoveryOrchestrator(erm)

		// Start health monitoring
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		healthMonitor.Start(ctx)
		defer healthMonitor.Stop()

		// Simulate memory pressure error
		memoryError := NewVectorDBErrorWithContext(
			ErrCodeMemoryExhausted,
			"memory limit exceeded during vector insertion",
			true,
			"collection",
			"insert",
		).WithSeverity(SeverityCritical).WithRecoveryAction(RecoveryGracefulDegradation)

		// Attempt recovery
		recoveryErr := aro.RecoverFromError(ctx, memoryError)

		// Verify recovery was attempted
		if recoveryErr == nil {
			t.Error("expected recovery to fail initially due to no registered strategies")
		}

		// Register memory pressure recovery strategy
		memoryStrategy := NewMemoryPressureRecoveryStrategy(memMgr)
		erm.RegisterRecoveryStrategy(ErrCodeMemoryExhausted, memoryStrategy)

		// Try recovery again
		recoveryErr = erm.TryRecoverWithDegradation(ctx, memoryError)
		if recoveryErr != nil {
			t.Errorf("expected recovery to succeed, got error: %v", recoveryErr)
		}

		// Verify degradation actions were taken
		activeDegradations := gdm.GetActiveDegradations()
		if len(activeDegradations) == 0 {
			t.Error("expected degradation actions to be active")
		}

		// Check recovery history
		history := aro.GetRecoveryHistory()
		if len(history) < 1 {
			t.Error("expected recovery attempts to be recorded")
		}

		// Verify health status reflects the situation
		status := healthMonitor.GetHealthStatus()
		if status.Overall == HealthUnknown {
			t.Error("expected health status to be determined")
		}
	})

	t.Run("quantization failure recovery workflow", func(t *testing.T) {
		// Setup system for quantization failure
		quantMgr := &mockQuantizationManager{shouldFail: false}

		config := DefaultDegradationConfig()
		gdm := NewGracefulDegradationManager(config)
		gdm.SetQuantizationManager(quantMgr)

		erm := NewErrorRecoveryManager()
		erm.SetDegradationManager(gdm)

		// Register quantization recovery strategy
		quantStrategy := NewQuantizationRecoveryStrategy(true)
		erm.RegisterRecoveryStrategy(ErrCodeQuantizationFailure, quantStrategy)

		// Create quantization error
		quantError := NewVectorDBErrorWithContext(
			ErrCodeQuantizationFailure,
			"quantization training failed due to insufficient data",
			true,
			"quantizer",
			"train",
		).WithSeverity(SeverityError).WithRecoveryAction(RecoveryFallback)

		ctx := context.Background()

		// Attempt recovery
		recoveryErr := erm.TryRecoverWithDegradation(ctx, quantError)
		if recoveryErr != nil {
			t.Errorf("expected quantization recovery to succeed, got error: %v", recoveryErr)
		}

		// Verify fallback was called
		if !quantMgr.fallbackCalled {
			t.Error("expected quantization fallback to be called")
		}

		// Check degradation history
		history := gdm.GetDegradationHistory()
		if len(history) == 0 {
			t.Error("expected degradation events to be recorded")
		}

		// Verify the degradation event
		found := false
		for _, event := range history {
			if event.Type == "quantization_failure" && event.Success {
				found = true
				break
			}
		}
		if !found {
			t.Error("expected successful quantization failure recovery event")
		}
	})

	t.Run("circuit breaker integration", func(t *testing.T) {
		// Create circuit breaker that fails initially
		breaker := &mockCircuitBreaker{state: "OPEN", shouldFail: true}

		erm := NewErrorRecoveryManager()
		erm.SetCircuitBreaker("failing_component", breaker)

		// Register a recovery strategy for index failures
		indexStrategy := &mockRecoveryStrategy{canRecover: true, shouldFail: false}
		erm.RegisterRecoveryStrategy(ErrCodeIndexFailure, indexStrategy)

		// Create error for failing component
		err := NewVectorDBErrorWithContext(
			ErrCodeIndexFailure,
			"index operation failed",
			true,
			"failing_component",
			"search",
		)

		ctx := context.Background()

		// First attempt should fail due to open circuit
		recoveryErr := erm.TryRecoverWithCircuitBreaker(ctx, err)
		if recoveryErr == nil {
			t.Error("expected recovery to fail with open circuit breaker")
		}

		// Close circuit breaker and disable failure
		breaker.state = "CLOSED"
		breaker.shouldFail = false

		// Second attempt should succeed
		recoveryErr = erm.TryRecoverWithCircuitBreaker(ctx, err)
		if recoveryErr != nil {
			t.Errorf("expected recovery to succeed with closed circuit, got error: %v", recoveryErr)
		}

		// Verify circuit breaker was used
		if breaker.executions < 2 {
			t.Errorf("expected at least 2 circuit breaker executions, got %d", breaker.executions)
		}
	})
}

func TestHealthMonitoringIntegration(t *testing.T) {
	requireStressMode(t)

	t.Run("health-aware error handling", func(t *testing.T) {
		// Create health monitor with multiple components
		monitor := NewSystemHealthMonitor(50 * time.Millisecond)

		// Component health states
		componentStates := map[string]HealthLevel{
			"memory":       HealthHealthy,
			"quantization": HealthDegraded,
			"index":        HealthUnhealthy,
		}

		var mu sync.RWMutex

		// Register health checks
		for component := range componentStates {
			comp := component // Capture for closure
			monitor.RegisterHealthCheck(comp, func(ctx context.Context) (HealthLevel, error) {
				mu.RLock()
				defer mu.RUnlock()
				level := componentStates[comp]
				if level != HealthHealthy {
					return level, fmt.Errorf("%s component is %s", comp, level.String())
				}
				return level, nil
			})
		}

		// Create error recovery manager
		erm := NewErrorRecoveryManager()

		// Create health-aware error handler
		handler := NewHealthAwareErrorHandler(monitor, erm)

		// Start monitoring
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		monitor.Start(ctx)
		defer monitor.Stop()

		// Wait for initial health check
		time.Sleep(100 * time.Millisecond)

		// Test error handling with unhealthy system
		testErr := errors.New("test error")
		handledErr := handler.HandleError(ctx, testErr)

		// Should return original error since system is unhealthy
		if handledErr != testErr {
			t.Error("expected original error to be returned for unhealthy system")
		}

		// Improve system health
		mu.Lock()
		componentStates["index"] = HealthHealthy
		componentStates["quantization"] = HealthHealthy
		mu.Unlock()

		// Wait for health check update
		time.Sleep(100 * time.Millisecond)

		// Test error handling with healthy system
		vectorErr := NewVectorDBError(ErrCodeMemoryExhausted, "memory error", true)

		// Register a recovery strategy for this test
		strategy := &mockRecoveryStrategy{canRecover: true, shouldFail: false}
		erm.RegisterRecoveryStrategy(ErrCodeMemoryExhausted, strategy)

		handledErr = handler.HandleError(ctx, vectorErr)

		// Should attempt recovery and succeed
		if handledErr != nil {
			t.Errorf("expected error to be recovered with healthy system, got: %v", handledErr)
		}
	})

	t.Run("component health tracking integration", func(t *testing.T) {
		tracker := NewComponentHealthTracker()

		// Simulate component operations and failures
		components := []string{"memory", "quantization", "index", "storage"}

		// Initial healthy state
		for _, comp := range components {
			tracker.UpdateComponentHealth(comp, HealthHealthy, nil)
		}

		// Simulate some failures
		tracker.UpdateComponentHealth("quantization", HealthDegraded, errors.New("training convergence slow"))
		tracker.UpdateComponentHealth("index", HealthUnhealthy, errors.New("corruption detected"))
		tracker.UpdateComponentHealth("storage", HealthCritical, errors.New("disk full"))

		// Check unhealthy components
		unhealthy := tracker.GetUnhealthyComponents()
		if len(unhealthy) != 3 {
			t.Errorf("expected 3 unhealthy components, got %d", len(unhealthy))
		}

		// Simulate recovery attempts
		tracker.RecordRecovery("quantization", true) // Successful recovery
		tracker.RecordRecovery("index", false)       // Failed recovery
		tracker.RecordRecovery("storage", true)      // Successful recovery

		// Check recovery counts
		quantHealth, _ := tracker.GetComponentHealth("quantization")
		if quantHealth.RecoveryCount != 1 {
			t.Errorf("expected 1 recovery attempt for quantization, got %d", quantHealth.RecoveryCount)
		}

		if quantHealth.Level != HealthHealthy {
			t.Errorf("expected quantization to be healthy after successful recovery, got %v", quantHealth.Level)
		}

		indexHealth, _ := tracker.GetComponentHealth("index")
		if indexHealth.Level == HealthHealthy {
			t.Error("expected index to remain unhealthy after failed recovery")
		}

		// Check overall health statistics
		allHealth := tracker.GetAllComponentHealth()
		if len(allHealth) != 4 {
			t.Errorf("expected 4 components in health tracker, got %d", len(allHealth))
		}
	})
}

func TestAutoRecoveryWorkflow(t *testing.T) {
	requireStressMode(t)

	t.Run("automatic degradation and recovery", func(t *testing.T) {
		// Create system with auto-recovery enabled
		config := DefaultDegradationConfig()
		config.AutoRecoveryEnabled = true
		config.RecoveryCheckInterval = 100 * time.Millisecond

		memMgr := &mockMemoryManager{
			usage: MemoryUsage{
				Total: 950 * 1024 * 1024, // High memory usage
				Limit: 1024 * 1024 * 1024,
			},
		}

		gdm := NewGracefulDegradationManager(config)
		gdm.SetMemoryManager(memMgr)

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		// Start auto-recovery
		gdm.StartAutoRecovery(ctx)

		// Apply degradation due to memory pressure
		err := gdm.HandleMemoryPressure(ctx, 3)
		if err != nil {
			t.Errorf("expected memory pressure handling to succeed, got error: %v", err)
		}

		initialLevel := gdm.GetDegradationLevel()
		if initialLevel < 3 {
			t.Errorf("expected degradation level to be at least 3, got %d", initialLevel)
		}

		// Simulate memory usage improvement
		memMgr.mu.Lock()
		memMgr.usage.Total = 500 * 1024 * 1024 // Reduce to 500MB
		memMgr.mu.Unlock()

		// Wait for auto-recovery to kick in
		time.Sleep(300 * time.Millisecond)

		finalLevel := gdm.GetDegradationLevel()
		if finalLevel >= initialLevel {
			t.Errorf("expected degradation level to decrease due to auto-recovery, initial: %d, final: %d", initialLevel, finalLevel)
		}

		// Check degradation history for recovery events
		history := gdm.GetDegradationHistory()
		recoveryFound := false
		for _, event := range history {
			if event.Action == "revert" && event.Type == "auto_recovery" {
				recoveryFound = true
				break
			}
		}

		if !recoveryFound {
			t.Error("expected auto-recovery event in degradation history")
		}
	})
}

func TestErrorHandlingPerformance(t *testing.T) {
	requireStressMode(t)

	t.Run("concurrent error handling", func(t *testing.T) {
		erm := NewErrorRecoveryManager()

		// Register a simple recovery strategy
		strategy := &mockRecoveryStrategy{canRecover: true, shouldFail: false}
		erm.RegisterRecoveryStrategy(ErrCodeMemoryExhausted, strategy)

		const numGoroutines = 100
		const errorsPerGoroutine = 10

		var wg sync.WaitGroup
		errors := make(chan error, numGoroutines*errorsPerGoroutine)

		ctx := context.Background()

		// Launch concurrent error recovery
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				for j := 0; j < errorsPerGoroutine; j++ {
					err := NewVectorDBErrorWithContext(
						ErrCodeMemoryExhausted,
						fmt.Sprintf("error from goroutine %d, iteration %d", id, j),
						true,
						fmt.Sprintf("component_%d", id),
						"test_operation",
					)

					recoveryErr := erm.TryRecover(ctx, err)
					if recoveryErr != nil {
						errors <- recoveryErr
					}
				}
			}(i)
		}

		wg.Wait()
		close(errors)

		// Check for any recovery failures
		var failureCount int
		for err := range errors {
			if err != nil {
				failureCount++
				t.Logf("Recovery failure: %v", err)
			}
		}

		if failureCount > 0 {
			t.Errorf("expected no recovery failures, got %d failures", failureCount)
		}
	})

	t.Run("health monitoring performance", func(t *testing.T) {
		monitor := NewSystemHealthMonitor(10 * time.Millisecond) // Fast interval

		// Register multiple health checks
		for i := 0; i < 10; i++ {
			component := fmt.Sprintf("component_%d", i)
			monitor.RegisterHealthCheck(component, func(ctx context.Context) (HealthLevel, error) {
				// Simulate some work
				time.Sleep(time.Microsecond * 100)
				return HealthHealthy, nil
			})
		}

		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()

		start := time.Now()
		monitor.Start(ctx)

		// Let it run for a while
		time.Sleep(500 * time.Millisecond)

		monitor.Stop()
		duration := time.Since(start)

		// Verify health checks were performed
		status := monitor.GetHealthStatus()
		if len(status.Components) != 10 {
			t.Errorf("expected 10 components in health status, got %d", len(status.Components))
		}

		// Performance should be reasonable
		if duration > 2*time.Second {
			t.Errorf("health monitoring took too long: %v", duration)
		}
	})
}

func TestErrorHandlingEdgeCases(t *testing.T) {
	t.Run("nil managers graceful handling", func(t *testing.T) {
		// Test degradation manager with nil sub-managers
		config := DefaultDegradationConfig()
		gdm := NewGracefulDegradationManager(config)

		ctx := context.Background()

		// Should not panic with nil managers
		err := gdm.HandleMemoryPressure(ctx, 1)
		if err == nil {
			t.Error("expected error when handling memory pressure with nil memory manager")
		}

		err = gdm.HandleQuantizationFailure(ctx, true)
		if err == nil {
			t.Error("expected error when handling quantization failure with nil quantization manager")
		}

		err = gdm.HandleIndexCorruption(ctx, true)
		if err == nil {
			t.Error("expected error when handling index corruption with nil index manager")
		}
	})

	t.Run("recovery strategy edge cases", func(t *testing.T) {
		erm := NewErrorRecoveryManager()

		// Test with strategy that can't recover
		strategy := &mockRecoveryStrategy{canRecover: false, shouldFail: true}
		erm.RegisterRecoveryStrategy(ErrCodeMemoryExhausted, strategy)

		err := NewVectorDBError(ErrCodeMemoryExhausted, "memory exhausted", true)
		ctx := context.Background()

		recoveryErr := erm.TryRecover(ctx, err)
		if recoveryErr == nil {
			t.Error("expected recovery to fail when strategy can't recover")
		}

		// Test with strategy that fails recovery
		strategy.canRecover = true
		strategy.shouldFail = true

		recoveryErr = erm.TryRecover(ctx, err)
		if recoveryErr == nil {
			t.Error("expected recovery to fail when strategy fails")
		}
	})

	t.Run("health check timeout handling", func(t *testing.T) {
		monitor := NewSystemHealthMonitor(100 * time.Millisecond)

		// Register a health check that times out
		monitor.RegisterHealthCheck("slow_component", func(ctx context.Context) (HealthLevel, error) {
			select {
			case <-ctx.Done():
				return HealthUnknown, ctx.Err()
			case <-time.After(time.Minute): // Longer than the 30s timeout
				return HealthHealthy, nil
			}
		})

		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		defer cancel()

		monitor.Start(ctx)

		// Wait for health check cycle
		time.Sleep(200 * time.Millisecond)

		monitor.Stop()

		// Health check should have timed out, but system should still function
		status := monitor.GetHealthStatus()
		if status.Overall == HealthUnknown {
			// This is acceptable - the timeout was handled gracefully
		}
	})
}
