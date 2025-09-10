package memory

import (
	"context"
	"fmt"
	"runtime"
	"time"
)

// MemoryErrorCode represents specific memory management error types
type MemoryErrorCode int

const (
	ErrMemUnknown MemoryErrorCode = iota
	ErrMemLimitExceeded
	ErrMemPressureCritical
	ErrMemMappingFailed
	ErrMemCacheEvictionFailed
	ErrMemMonitoringFailed
	ErrMemConfigInvalid
	ErrMemManagerNotStarted
	ErrMemManagerAlreadyStarted
	ErrMemCacheNotFound
	ErrMemMappableNotFound
	ErrMemRecoveryFailed
)

// MemoryError represents a memory management specific error
type MemoryError struct {
	Code        MemoryErrorCode        `json:"code"`
	Message     string                 `json:"message"`
	Component   string                 `json:"component"`
	Operation   string                 `json:"operation"`
	Retryable   bool                   `json:"retryable"`
	Recoverable bool                   `json:"recoverable"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	Cause       error                  `json:"cause,omitempty"`
	Timestamp   time.Time              `json:"timestamp"`
	Usage       *MemoryUsage           `json:"usage,omitempty"`
}

func (me *MemoryError) Error() string {
	if me.Cause != nil {
		return fmt.Sprintf("memory error in %s.%s: %s (caused by: %v)",
			me.Component, me.Operation, me.Message, me.Cause)
	}
	return fmt.Sprintf("memory error in %s.%s: %s",
		me.Component, me.Operation, me.Message)
}

// Unwrap returns the underlying cause error
func (me *MemoryError) Unwrap() error {
	return me.Cause
}

// NewMemoryError creates a new memory error
func NewMemoryError(code MemoryErrorCode, component, operation, message string) *MemoryError {
	return &MemoryError{
		Code:      code,
		Message:   message,
		Component: component,
		Operation: operation,
		Metadata:  make(map[string]interface{}),
		Timestamp: time.Now(),
	}
}

// WithCause adds a cause error
func (me *MemoryError) WithCause(cause error) *MemoryError {
	me.Cause = cause
	return me
}

// WithRetryable sets whether the error is retryable
func (me *MemoryError) WithRetryable(retryable bool) *MemoryError {
	me.Retryable = retryable
	return me
}

// WithRecoverable sets whether the error is recoverable
func (me *MemoryError) WithRecoverable(recoverable bool) *MemoryError {
	me.Recoverable = recoverable
	return me
}

// WithMetadata adds metadata to the error
func (me *MemoryError) WithMetadata(key string, value interface{}) *MemoryError {
	if me.Metadata == nil {
		me.Metadata = make(map[string]interface{})
	}
	me.Metadata[key] = value
	return me
}

// WithUsage adds memory usage information
func (me *MemoryError) WithUsage(usage MemoryUsage) *MemoryError {
	me.Usage = &usage
	return me
}

// MemoryRecoveryManager handles memory-related error recovery
type MemoryRecoveryManager struct {
	manager             MemoryManager
	maxRecoveryAttempts int
	recoveryBackoff     time.Duration
	enableAggressiveGC  bool
	enableMemoryMapping bool
	enableCacheEviction bool
}

// NewMemoryRecoveryManager creates a new memory recovery manager
func NewMemoryRecoveryManager(manager MemoryManager) *MemoryRecoveryManager {
	return &MemoryRecoveryManager{
		manager:             manager,
		maxRecoveryAttempts: 3,
		recoveryBackoff:     time.Second,
		enableAggressiveGC:  true,
		enableMemoryMapping: true,
		enableCacheEviction: true,
	}
}

// RecoverFromMemoryPressure attempts to recover from memory pressure
func (mrm *MemoryRecoveryManager) RecoverFromMemoryPressure(ctx context.Context, err *MemoryError) error {
	if mrm.manager == nil {
		return NewMemoryError(
			ErrMemRecoveryFailed,
			"MemoryRecoveryManager",
			"RecoverFromMemoryPressure",
			"memory manager not available",
		).WithCause(err)
	}

	initialUsage := mrm.manager.GetUsage()

	for attempt := 1; attempt <= mrm.maxRecoveryAttempts; attempt++ {
		if attempt > 1 {
			// Apply backoff
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(mrm.recoveryBackoff * time.Duration(attempt)):
			}
		}

		// Try different recovery strategies
		if recoveryErr := mrm.attemptRecovery(ctx, err, attempt); recoveryErr == nil {
			// Verify recovery was successful
			finalUsage := mrm.manager.GetUsage()
			if mrm.isRecoverySuccessful(initialUsage, finalUsage) {
				return nil
			}
		}
	}

	return NewMemoryError(
		ErrMemRecoveryFailed,
		"MemoryRecoveryManager",
		"RecoverFromMemoryPressure",
		fmt.Sprintf("recovery failed after %d attempts", mrm.maxRecoveryAttempts),
	).WithCause(err).WithUsage(mrm.manager.GetUsage())
}

// attemptRecovery tries different recovery strategies based on attempt number
func (mrm *MemoryRecoveryManager) attemptRecovery(ctx context.Context, err *MemoryError, attempt int) error {
	switch attempt {
	case 1:
		return mrm.lightweightRecovery(ctx)
	case 2:
		return mrm.moderateRecovery(ctx)
	case 3:
		return mrm.aggressiveRecovery(ctx)
	default:
		return fmt.Errorf("no recovery strategy for attempt %d", attempt)
	}
}

// lightweightRecovery performs lightweight memory recovery
func (mrm *MemoryRecoveryManager) lightweightRecovery(ctx context.Context) error {
	// Step 1: Trigger garbage collection
	if mrm.enableAggressiveGC {
		runtime.GC()
	}

	// Step 2: Check if recovery is sufficient
	usage := mrm.manager.GetUsage()
	if usage.Limit > 0 && usage.Total <= usage.Limit {
		return nil
	}

	return fmt.Errorf("lightweight recovery insufficient")
}

// moderateRecovery performs moderate memory recovery
func (mrm *MemoryRecoveryManager) moderateRecovery(ctx context.Context) error {
	// Step 1: Aggressive garbage collection
	if mrm.enableAggressiveGC {
		runtime.GC()
		runtime.GC() // Run twice
	}

	// Step 2: Try to handle memory limit exceeded
	if handleErr := mrm.manager.HandleMemoryLimitExceeded(); handleErr != nil {
		return fmt.Errorf("failed to handle memory limit: %w", handleErr)
	}

	return nil
}

// aggressiveRecovery performs aggressive memory recovery
func (mrm *MemoryRecoveryManager) aggressiveRecovery(ctx context.Context) error {
	// Step 1: Multiple garbage collection cycles
	if mrm.enableAggressiveGC {
		for i := 0; i < 3; i++ {
			runtime.GC()
			time.Sleep(time.Millisecond * 100) // Allow GC to complete
		}
	}

	// Step 2: Force memory limit handling
	if handleErr := mrm.manager.HandleMemoryLimitExceeded(); handleErr != nil {
		return fmt.Errorf("aggressive memory limit handling failed: %w", handleErr)
	}

	// Step 3: Final verification
	usage := mrm.manager.GetUsage()
	if usage.Limit > 0 && usage.Total > usage.Limit {
		return fmt.Errorf("aggressive recovery failed: usage %d > limit %d", usage.Total, usage.Limit)
	}

	return nil
}

// isRecoverySuccessful checks if recovery was successful
func (mrm *MemoryRecoveryManager) isRecoverySuccessful(before, after MemoryUsage) bool {
	// Recovery is successful if:
	// 1. Memory usage is below limit (if limit is set)
	// 2. Memory usage has decreased

	if after.Limit > 0 && after.Total > after.Limit {
		return false // Still over limit
	}

	if after.Total >= before.Total {
		return false // Memory usage didn't decrease
	}

	return true
}

// RecoverFromMappingFailure attempts to recover from memory mapping failures
func (mrm *MemoryRecoveryManager) RecoverFromMappingFailure(ctx context.Context, err *MemoryError) error {
	// For memory mapping failures, we can try:
	// 1. Retry with different parameters
	// 2. Fall back to regular memory allocation
	// 3. Free up space and retry

	if !mrm.enableMemoryMapping {
		return fmt.Errorf("memory mapping recovery disabled")
	}

	// Try to free up memory first
	if recoveryErr := mrm.lightweightRecovery(ctx); recoveryErr != nil {
		return fmt.Errorf("failed to free memory for mapping retry: %w", recoveryErr)
	}

	// The actual retry would be handled by the caller
	// We just ensure memory is available
	return nil
}

// RecoverFromCacheFailure attempts to recover from cache operation failures
func (mrm *MemoryRecoveryManager) RecoverFromCacheFailure(ctx context.Context, err *MemoryError) error {
	if !mrm.enableCacheEviction {
		return fmt.Errorf("cache recovery disabled")
	}

	// For cache failures, we can:
	// 1. Clear problematic caches
	// 2. Reduce cache sizes
	// 3. Disable caching temporarily

	// The specific recovery would depend on the cache implementation
	// For now, we'll just trigger memory cleanup
	return mrm.lightweightRecovery(ctx)
}

// MonitorMemoryHealth continuously monitors memory health and triggers recovery
type MemoryHealthMonitor struct {
	manager         MemoryManager
	recoveryManager *MemoryRecoveryManager
	ctx             context.Context
	cancel          context.CancelFunc
	interval        time.Duration
	thresholds      MemoryHealthThresholds
}

// MemoryHealthThresholds defines thresholds for memory health monitoring
type MemoryHealthThresholds struct {
	WarningThreshold  float64 // Percentage of limit
	CriticalThreshold float64 // Percentage of limit
	RecoveryThreshold float64 // Percentage of limit for triggering recovery
}

// NewMemoryHealthMonitor creates a new memory health monitor
func NewMemoryHealthMonitor(manager MemoryManager, recoveryManager *MemoryRecoveryManager) *MemoryHealthMonitor {
	return &MemoryHealthMonitor{
		manager:         manager,
		recoveryManager: recoveryManager,
		interval:        time.Second * 5,
		thresholds: MemoryHealthThresholds{
			WarningThreshold:  0.8,  // 80%
			CriticalThreshold: 0.9,  // 90%
			RecoveryThreshold: 0.95, // 95%
		},
	}
}

// Start begins memory health monitoring
func (mhm *MemoryHealthMonitor) Start(ctx context.Context) error {
	if mhm.ctx != nil {
		return NewMemoryError(
			ErrMemManagerAlreadyStarted,
			"MemoryHealthMonitor",
			"Start",
			"monitor already started",
		)
	}

	mhm.ctx, mhm.cancel = context.WithCancel(ctx)
	go mhm.monitorLoop()
	return nil
}

// Stop stops memory health monitoring
func (mhm *MemoryHealthMonitor) Stop() error {
	if mhm.cancel == nil {
		return NewMemoryError(
			ErrMemManagerNotStarted,
			"MemoryHealthMonitor",
			"Stop",
			"monitor not started",
		)
	}

	mhm.cancel()
	mhm.ctx = nil
	mhm.cancel = nil
	return nil
}

// monitorLoop runs the monitoring loop
func (mhm *MemoryHealthMonitor) monitorLoop() {
	if mhm.interval <= 0 {
		mhm.interval = time.Second * 5 // Default interval
	}

	if mhm.ctx == nil {
		return // No context available
	}

	ticker := time.NewTicker(mhm.interval)
	defer ticker.Stop()

	for {
		select {
		case <-mhm.ctx.Done():
			return
		case <-ticker.C:
			mhm.checkMemoryHealth()
		}
	}
}

// checkMemoryHealth checks current memory health and triggers recovery if needed
func (mhm *MemoryHealthMonitor) checkMemoryHealth() {
	if mhm.manager == nil {
		return // No manager available
	}

	usage := mhm.manager.GetUsage()

	if usage.Limit <= 0 {
		return // No limit set, nothing to monitor
	}

	usageRatio := float64(usage.Total) / float64(usage.Limit)

	// Check if recovery is needed
	if usageRatio >= mhm.thresholds.RecoveryThreshold {
		err := NewMemoryError(
			ErrMemPressureCritical,
			"MemoryHealthMonitor",
			"checkMemoryHealth",
			fmt.Sprintf("memory usage %.2f%% exceeds recovery threshold %.2f%%",
				usageRatio*100, mhm.thresholds.RecoveryThreshold*100),
		).WithUsage(usage).WithRecoverable(true)

		if mhm.recoveryManager != nil {
			if recoveryErr := mhm.recoveryManager.RecoverFromMemoryPressure(mhm.ctx, err); recoveryErr != nil {
				// Log recovery failure (in a real implementation, use proper logging)
				fmt.Printf("Memory recovery failed: %v\n", recoveryErr)
			}
		}
	} else if usageRatio >= mhm.thresholds.CriticalThreshold {
		// Log critical memory usage (in a real implementation, use proper logging)
		fmt.Printf("Critical memory usage: %.2f%% of limit\n", usageRatio*100)
	} else if usageRatio >= mhm.thresholds.WarningThreshold {
		// Log warning (in a real implementation, use proper logging)
		fmt.Printf("High memory usage: %.2f%% of limit\n", usageRatio*100)
	}
}
