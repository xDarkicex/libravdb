package libravdb

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Core errors
var (
	ErrDatabaseClosed     = errors.New("database is closed")
	ErrCollectionClosed   = errors.New("collection is closed")
	ErrTooManyCollections = errors.New("maximum number of collections exceeded")
	ErrCollectionNotFound = errors.New("collection not found")
	ErrInvalidDimension   = errors.New("invalid vector dimension")
	ErrInvalidK           = errors.New("k must be positive")
	ErrEmptyIndex         = errors.New("index is empty")
)

// Streaming errors
var (
	ErrBackpressureActive = errors.New("backpressure is active, cannot send more data")
	ErrStreamingStopped   = errors.New("streaming operation has been stopped")
	ErrStreamingTimeout   = errors.New("streaming operation timed out")
	ErrBufferFull         = errors.New("streaming buffer is full")
)

// Memory management errors
var (
	ErrMemoryLimitExceeded    = errors.New("memory limit exceeded")
	ErrMemoryPressureCritical = errors.New("critical memory pressure detected")
	ErrMemoryMappingFailed    = errors.New("memory mapping operation failed")
	ErrCacheEvictionFailed    = errors.New("cache eviction failed")
	ErrMemoryRecoveryFailed   = errors.New("automatic memory recovery failed")
)

// Quantization errors
var (
	ErrQuantizationNotTrained     = errors.New("quantizer not trained")
	ErrQuantizationTrainingFailed = errors.New("quantization training failed")
	ErrQuantizationCorrupted      = errors.New("quantization data corrupted")
	ErrQuantizationIncompatible   = errors.New("quantization configuration incompatible")
	ErrQuantizationRecoveryFailed = errors.New("quantization recovery failed")
)

// Index errors
var (
	ErrIndexCorrupted          = errors.New("index data corrupted")
	ErrIndexTypeMismatch       = errors.New("index type mismatch")
	ErrIndexRebuildRequired    = errors.New("index rebuild required")
	ErrIndexOptimizationFailed = errors.New("index optimization failed")
)

// Filter errors
var (
	ErrFilterInvalid         = errors.New("invalid filter expression")
	ErrFilterTypeUnsupported = errors.New("unsupported filter type")
	ErrFilterExecutionFailed = errors.New("filter execution failed")
)

// ErrorCode represents structured error codes
type ErrorCode int

const (
	ErrCodeUnknown ErrorCode = iota
	ErrCodeInvalidVector
	ErrCodeIndexCorrupted
	ErrCodeStorageFailure
	ErrCodeMemoryExhausted
	ErrCodeTimeout
	ErrCodeRateLimited
	// Memory management error codes
	ErrCodeMemoryPressure
	ErrCodeMemoryMappingFailure
	ErrCodeCacheFailure
	// Quantization error codes
	ErrCodeQuantizationFailure
	ErrCodeQuantizationCorruption
	ErrCodeQuantizationTraining
	// Batch operation error codes
	ErrCodeBatchFailure
	ErrCodeBatchTimeout
	ErrCodeBatchSizeLimit
	// Index error codes
	ErrCodeIndexFailure
	ErrCodeIndexCorruption
	ErrCodeIndexRebuild
	// Filter error codes
	ErrCodeFilterFailure
	ErrCodeFilterInvalid
)

// ErrorSeverity represents the severity level of an error
type ErrorSeverity int

const (
	SeverityInfo ErrorSeverity = iota
	SeverityWarning
	SeverityError
	SeverityCritical
	SeverityFatal
)

// String returns the string representation of error severity
func (s ErrorSeverity) String() string {
	switch s {
	case SeverityInfo:
		return "INFO"
	case SeverityWarning:
		return "WARNING"
	case SeverityError:
		return "ERROR"
	case SeverityCritical:
		return "CRITICAL"
	case SeverityFatal:
		return "FATAL"
	default:
		return "UNKNOWN"
	}
}

// RecoveryAction represents possible recovery actions
type RecoveryAction int

const (
	RecoveryNone RecoveryAction = iota
	RecoveryRetry
	RecoveryFallback
	RecoveryGracefulDegradation
	RecoveryRestart
	RecoveryRebuild
)

// String returns the string representation of recovery action
func (r RecoveryAction) String() string {
	switch r {
	case RecoveryNone:
		return "NONE"
	case RecoveryRetry:
		return "RETRY"
	case RecoveryFallback:
		return "FALLBACK"
	case RecoveryGracefulDegradation:
		return "GRACEFUL_DEGRADATION"
	case RecoveryRestart:
		return "RESTART"
	case RecoveryRebuild:
		return "REBUILD"
	default:
		return "UNKNOWN"
	}
}

// ErrorContext provides additional context about where and when an error occurred
type ErrorContext struct {
	Component  string                 `json:"component"`
	Operation  string                 `json:"operation"`
	StackTrace string                 `json:"stack_trace,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	Timestamp  time.Time              `json:"timestamp"`
	RequestID  string                 `json:"request_id,omitempty"`
	UserID     string                 `json:"user_id,omitempty"`
}

// VectorDBError represents a structured error with additional context
type VectorDBError struct {
	Code           ErrorCode      `json:"code"`
	Message        string         `json:"message"`
	Details        any            `json:"details,omitempty"`
	Retryable      bool           `json:"retryable"`
	Severity       ErrorSeverity  `json:"severity"`
	RecoveryAction RecoveryAction `json:"recovery_action"`
	Context        *ErrorContext  `json:"context,omitempty"`
	Cause          error          `json:"cause,omitempty"`
	Timestamp      time.Time      `json:"timestamp"`
	RetryCount     int            `json:"retry_count"`
	MaxRetries     int            `json:"max_retries"`
}

func (e *VectorDBError) Error() string {
	var parts []string
	parts = append(parts, fmt.Sprintf("[%s] VectorDB Error %d: %s", e.Severity.String(), e.Code, e.Message))

	if e.Context != nil && e.Context.Component != "" {
		parts = append(parts, fmt.Sprintf("Component: %s", e.Context.Component))
	}

	if e.Context != nil && e.Context.Operation != "" {
		parts = append(parts, fmt.Sprintf("Operation: %s", e.Context.Operation))
	}

	if e.Cause != nil {
		parts = append(parts, fmt.Sprintf("Cause: %v", e.Cause))
	}

	return strings.Join(parts, " | ")
}

// Unwrap returns the underlying cause error
func (e *VectorDBError) Unwrap() error {
	return e.Cause
}

// IsRetryable returns true if the error can be retried
func (e *VectorDBError) IsRetryable() bool {
	return e.Retryable && e.RetryCount < e.MaxRetries
}

// CanRecover returns true if automatic recovery is possible
func (e *VectorDBError) CanRecover() bool {
	return e.RecoveryAction != RecoveryNone
}

// NewVectorDBError creates a new structured error
func NewVectorDBError(code ErrorCode, message string, retryable bool) *VectorDBError {
	return &VectorDBError{
		Code:           code,
		Message:        message,
		Retryable:      retryable,
		Severity:       SeverityError,
		RecoveryAction: RecoveryNone,
		Timestamp:      time.Now(),
		MaxRetries:     3,
	}
}

// NewVectorDBErrorWithContext creates a new structured error with context
func NewVectorDBErrorWithContext(code ErrorCode, message string, retryable bool, component, operation string) *VectorDBError {
	err := NewVectorDBError(code, message, retryable)
	err.Context = &ErrorContext{
		Component: component,
		Operation: operation,
		Timestamp: time.Now(),
		Metadata:  make(map[string]interface{}),
	}

	// Capture stack trace for debugging
	if code >= ErrCodeMemoryExhausted {
		err.Context.StackTrace = captureStackTrace()
	}

	return err
}

// WithCause adds a cause error to the VectorDBError
func (e *VectorDBError) WithCause(cause error) *VectorDBError {
	e.Cause = cause
	return e
}

// WithSeverity sets the severity level
func (e *VectorDBError) WithSeverity(severity ErrorSeverity) *VectorDBError {
	e.Severity = severity
	return e
}

// WithRecoveryAction sets the recovery action
func (e *VectorDBError) WithRecoveryAction(action RecoveryAction) *VectorDBError {
	e.RecoveryAction = action
	return e
}

// WithMetadata adds metadata to the error context
func (e *VectorDBError) WithMetadata(key string, value interface{}) *VectorDBError {
	if e.Context == nil {
		e.Context = &ErrorContext{
			Timestamp: time.Now(),
			Metadata:  make(map[string]interface{}),
		}
	}
	if e.Context.Metadata == nil {
		e.Context.Metadata = make(map[string]interface{})
	}
	e.Context.Metadata[key] = value
	return e
}

// WithRequestID sets the request ID for tracing
func (e *VectorDBError) WithRequestID(requestID string) *VectorDBError {
	if e.Context == nil {
		e.Context = &ErrorContext{
			Timestamp: time.Now(),
			Metadata:  make(map[string]interface{}),
		}
	}
	e.Context.RequestID = requestID
	return e
}

// IncrementRetry increments the retry count
func (e *VectorDBError) IncrementRetry() {
	e.RetryCount++
}

// captureStackTrace captures the current stack trace
func captureStackTrace() string {
	buf := make([]byte, 4096)
	n := runtime.Stack(buf, false)
	return string(buf[:n])
}

// ErrorRecoveryManager handles automatic error recovery
type ErrorRecoveryManager struct {
	recoveryStrategies map[ErrorCode]RecoveryStrategy
	maxRetryAttempts   int
	retryBackoff       time.Duration
	circuitBreakers    map[string]CircuitBreaker
	degradationManager GracefulDegradationManager
	healthMonitor      SystemHealthMonitor
	mu                 sync.RWMutex
}

// RecoveryStrategy defines how to recover from specific error types
type RecoveryStrategy interface {
	CanRecover(err *VectorDBError) bool
	Recover(ctx context.Context, err *VectorDBError) error
	GetRecoveryAction() RecoveryAction
}

// NewErrorRecoveryManager creates a new error recovery manager
func NewErrorRecoveryManager() *ErrorRecoveryManager {
	return &ErrorRecoveryManager{
		recoveryStrategies: make(map[ErrorCode]RecoveryStrategy),
		maxRetryAttempts:   3,
		retryBackoff:       time.Second,
		circuitBreakers:    make(map[string]CircuitBreaker),
	}
}

// RegisterRecoveryStrategy registers a recovery strategy for an error code
func (erm *ErrorRecoveryManager) RegisterRecoveryStrategy(code ErrorCode, strategy RecoveryStrategy) {
	erm.recoveryStrategies[code] = strategy
}

// TryRecover attempts to recover from an error
func (erm *ErrorRecoveryManager) TryRecover(ctx context.Context, err *VectorDBError) error {
	strategy, exists := erm.recoveryStrategies[err.Code]
	if !exists || !strategy.CanRecover(err) {
		return err
	}

	// Set recovery action
	err.RecoveryAction = strategy.GetRecoveryAction()

	// Attempt recovery with retries
	for attempt := 0; attempt < erm.maxRetryAttempts; attempt++ {
		if attempt > 0 {
			// Apply backoff
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(erm.retryBackoff * time.Duration(attempt)):
			}
		}

		if recoveryErr := strategy.Recover(ctx, err); recoveryErr == nil {
			return nil // Recovery successful
		}

		err.IncrementRetry()
	}

	// Recovery failed after all attempts
	return NewVectorDBErrorWithContext(
		ErrCodeUnknown,
		fmt.Sprintf("recovery failed after %d attempts", erm.maxRetryAttempts),
		false,
		"ErrorRecoveryManager",
		"TryRecover",
	).WithCause(err)
}

// CircuitBreaker represents a circuit breaker for fault tolerance
type CircuitBreaker interface {
	Execute(ctx context.Context, fn func() error) error
	State() string
	Reset()
}

// GracefulDegradationManager handles graceful degradation under failure conditions
type GracefulDegradationManager interface {
	HandleMemoryPressure(ctx context.Context, pressureLevel int) error
	HandleQuantizationFailure(ctx context.Context, fallbackEnabled bool) error
	HandleIndexCorruption(ctx context.Context, rebuildEnabled bool) error
	GetDegradationLevel() int
	SetDegradationLevel(level int) error
}

// SystemHealthMonitor monitors overall system health
type SystemHealthMonitor interface {
	GetHealthStatus() HealthStatus
	RegisterHealthCheck(name string, check HealthCheck) error
	UnregisterHealthCheck(name string) error
	Start(ctx context.Context) error
	Stop() error
}

// HealthStatus represents the overall health of the system
type HealthStatus struct {
	Overall    HealthLevel            `json:"overall"`
	Components map[string]HealthLevel `json:"components"`
	Timestamp  time.Time              `json:"timestamp"`
	Details    map[string]any         `json:"details,omitempty"`
}

// HealthLevel represents the health level of a component
type HealthLevel int

const (
	HealthUnknown HealthLevel = iota
	HealthHealthy
	HealthDegraded
	HealthUnhealthy
	HealthCritical
)

// String returns the string representation of health level
func (hl HealthLevel) String() string {
	switch hl {
	case HealthHealthy:
		return "HEALTHY"
	case HealthDegraded:
		return "DEGRADED"
	case HealthUnhealthy:
		return "UNHEALTHY"
	case HealthCritical:
		return "CRITICAL"
	default:
		return "UNKNOWN"
	}
}

// HealthCheck represents a health check function
type HealthCheck func(ctx context.Context) (HealthLevel, error)

// SetCircuitBreaker sets a circuit breaker for a specific component
func (erm *ErrorRecoveryManager) SetCircuitBreaker(component string, breaker CircuitBreaker) {
	erm.mu.Lock()
	defer erm.mu.Unlock()
	erm.circuitBreakers[component] = breaker
}

// GetCircuitBreaker gets a circuit breaker for a specific component
func (erm *ErrorRecoveryManager) GetCircuitBreaker(component string) (CircuitBreaker, bool) {
	erm.mu.RLock()
	defer erm.mu.RUnlock()
	breaker, exists := erm.circuitBreakers[component]
	return breaker, exists
}

// SetDegradationManager sets the graceful degradation manager
func (erm *ErrorRecoveryManager) SetDegradationManager(manager GracefulDegradationManager) {
	erm.mu.Lock()
	defer erm.mu.Unlock()
	erm.degradationManager = manager
}

// SetHealthMonitor sets the system health monitor
func (erm *ErrorRecoveryManager) SetHealthMonitor(monitor SystemHealthMonitor) {
	erm.mu.Lock()
	defer erm.mu.Unlock()
	erm.healthMonitor = monitor
}

// TryRecoverWithCircuitBreaker attempts recovery with circuit breaker protection
func (erm *ErrorRecoveryManager) TryRecoverWithCircuitBreaker(ctx context.Context, err *VectorDBError) error {
	// Check if we have a circuit breaker for this component
	if err.Context != nil && err.Context.Component != "" {
		if breaker, exists := erm.GetCircuitBreaker(err.Context.Component); exists {
			return breaker.Execute(ctx, func() error {
				return erm.TryRecover(ctx, err)
			})
		}
	}

	// Fallback to regular recovery
	return erm.TryRecover(ctx, err)
}

// TryRecoverWithDegradation attempts recovery with graceful degradation
func (erm *ErrorRecoveryManager) TryRecoverWithDegradation(ctx context.Context, err *VectorDBError) error {
	erm.mu.RLock()
	degradationManager := erm.degradationManager
	erm.mu.RUnlock()

	if degradationManager == nil {
		return erm.TryRecoverWithCircuitBreaker(ctx, err)
	}

	// Apply graceful degradation based on error type
	switch err.Code {
	case ErrCodeMemoryExhausted, ErrCodeMemoryPressure:
		if degradationErr := degradationManager.HandleMemoryPressure(ctx, 1); degradationErr != nil {
			return fmt.Errorf("graceful degradation failed: %w", degradationErr)
		}
	case ErrCodeQuantizationFailure, ErrCodeQuantizationCorruption:
		if degradationErr := degradationManager.HandleQuantizationFailure(ctx, true); degradationErr != nil {
			return fmt.Errorf("quantization degradation failed: %w", degradationErr)
		}
	case ErrCodeIndexCorruption, ErrCodeIndexFailure:
		if degradationErr := degradationManager.HandleIndexCorruption(ctx, true); degradationErr != nil {
			return fmt.Errorf("index degradation failed: %w", degradationErr)
		}
	}

	// Try normal recovery after degradation
	return erm.TryRecoverWithCircuitBreaker(ctx, err)
}

// AutomaticRecoveryOrchestrator coordinates automatic recovery across all systems
type AutomaticRecoveryOrchestrator struct {
	errorRecoveryManager  *ErrorRecoveryManager
	memoryRecoveryManager interface {
		RecoverFromMemoryPressure(ctx context.Context, err any) error
	}
	quantizationRecoveryManager interface {
		RecoverFromTrainingFailure(ctx context.Context, quantizer any, vectors [][]float32, err any) error
	}
	batchRecoveryManager interface {
		RecoverFromBatchFailure(ctx context.Context, err any, retryFunc func(ctx context.Context, items []int) error) error
	}
	recoveryHistory []RecoveryAttempt
	mu              sync.RWMutex
}

// RecoveryAttempt represents a recovery attempt
type RecoveryAttempt struct {
	Timestamp     time.Time     `json:"timestamp"`
	ErrorCode     ErrorCode     `json:"error_code"`
	Component     string        `json:"component"`
	Operation     string        `json:"operation"`
	RecoveryType  string        `json:"recovery_type"`
	Success       bool          `json:"success"`
	Duration      time.Duration `json:"duration"`
	ErrorMessage  string        `json:"error_message,omitempty"`
	RecoverySteps []string      `json:"recovery_steps,omitempty"`
}

// NewAutomaticRecoveryOrchestrator creates a new recovery orchestrator
func NewAutomaticRecoveryOrchestrator(errorRecoveryManager *ErrorRecoveryManager) *AutomaticRecoveryOrchestrator {
	return &AutomaticRecoveryOrchestrator{
		errorRecoveryManager: errorRecoveryManager,
		recoveryHistory:      make([]RecoveryAttempt, 0),
	}
}

// SetMemoryRecoveryManager sets the memory recovery manager
func (aro *AutomaticRecoveryOrchestrator) SetMemoryRecoveryManager(manager any) {
	aro.mu.Lock()
	defer aro.mu.Unlock()
	if mgr, ok := manager.(interface {
		RecoverFromMemoryPressure(ctx context.Context, err any) error
	}); ok {
		aro.memoryRecoveryManager = mgr
	}
}

// SetQuantizationRecoveryManager sets the quantization recovery manager
func (aro *AutomaticRecoveryOrchestrator) SetQuantizationRecoveryManager(manager any) {
	aro.mu.Lock()
	defer aro.mu.Unlock()
	if mgr, ok := manager.(interface {
		RecoverFromTrainingFailure(ctx context.Context, quantizer any, vectors [][]float32, err any) error
	}); ok {
		aro.quantizationRecoveryManager = mgr
	}
}

// SetBatchRecoveryManager sets the batch recovery manager
func (aro *AutomaticRecoveryOrchestrator) SetBatchRecoveryManager(manager any) {
	aro.mu.Lock()
	defer aro.mu.Unlock()
	if mgr, ok := manager.(interface {
		RecoverFromBatchFailure(ctx context.Context, err any, retryFunc func(ctx context.Context, items []int) error) error
	}); ok {
		aro.batchRecoveryManager = mgr
	}
}

// RecoverFromError attempts comprehensive recovery from any error
func (aro *AutomaticRecoveryOrchestrator) RecoverFromError(ctx context.Context, err error) error {
	startTime := time.Now()

	attempt := RecoveryAttempt{
		Timestamp:     startTime,
		RecoverySteps: make([]string, 0),
	}

	// Determine error type and recovery strategy
	switch e := err.(type) {
	case *VectorDBError:
		attempt.ErrorCode = e.Code
		if e.Context != nil {
			attempt.Component = e.Context.Component
			attempt.Operation = e.Context.Operation
		}

		// Try VectorDB-specific recovery
		attempt.RecoveryType = "vectordb_recovery"
		attempt.RecoverySteps = append(attempt.RecoverySteps, "attempting_vectordb_recovery")

		if recoveryErr := aro.errorRecoveryManager.TryRecoverWithDegradation(ctx, e); recoveryErr == nil {
			attempt.Success = true
			attempt.RecoverySteps = append(attempt.RecoverySteps, "vectordb_recovery_successful")
		} else {
			attempt.ErrorMessage = recoveryErr.Error()
			attempt.RecoverySteps = append(attempt.RecoverySteps, "vectordb_recovery_failed")
		}

	default:
		// Try generic recovery strategies
		attempt.RecoveryType = "generic_recovery"
		attempt.RecoverySteps = append(attempt.RecoverySteps, "attempting_generic_recovery")

		// For now, mark as failed since we don't have specific recovery
		attempt.Success = false
		attempt.ErrorMessage = err.Error()
		attempt.RecoverySteps = append(attempt.RecoverySteps, "no_specific_recovery_available")
	}

	attempt.Duration = time.Since(startTime)

	// Record recovery attempt
	aro.mu.Lock()
	aro.recoveryHistory = append(aro.recoveryHistory, attempt)
	// Keep only last 100 attempts
	if len(aro.recoveryHistory) > 100 {
		aro.recoveryHistory = aro.recoveryHistory[1:]
	}
	aro.mu.Unlock()

	if !attempt.Success {
		return fmt.Errorf("automatic recovery failed: %s", attempt.ErrorMessage)
	}

	return nil
}

// GetRecoveryHistory returns the recovery attempt history
func (aro *AutomaticRecoveryOrchestrator) GetRecoveryHistory() []RecoveryAttempt {
	aro.mu.RLock()
	defer aro.mu.RUnlock()

	// Return a copy to avoid race conditions
	history := make([]RecoveryAttempt, len(aro.recoveryHistory))
	copy(history, aro.recoveryHistory)
	return history
}

// GetRecoveryStats returns statistics about recovery attempts
func (aro *AutomaticRecoveryOrchestrator) GetRecoveryStats() RecoveryStats {
	aro.mu.RLock()
	defer aro.mu.RUnlock()

	stats := RecoveryStats{
		TotalAttempts:  len(aro.recoveryHistory),
		ByErrorCode:    make(map[ErrorCode]int),
		ByComponent:    make(map[string]int),
		ByRecoveryType: make(map[string]int),
	}

	for _, attempt := range aro.recoveryHistory {
		if attempt.Success {
			stats.SuccessfulAttempts++
		}
		stats.ByErrorCode[attempt.ErrorCode]++
		stats.ByComponent[attempt.Component]++
		stats.ByRecoveryType[attempt.RecoveryType]++

		stats.TotalDuration += attempt.Duration
	}

	if stats.TotalAttempts > 0 {
		stats.SuccessRate = float64(stats.SuccessfulAttempts) / float64(stats.TotalAttempts)
		stats.AverageDuration = stats.TotalDuration / time.Duration(stats.TotalAttempts)
	}

	return stats
}

// RecoveryStats represents statistics about recovery attempts
type RecoveryStats struct {
	TotalAttempts      int               `json:"total_attempts"`
	SuccessfulAttempts int               `json:"successful_attempts"`
	SuccessRate        float64           `json:"success_rate"`
	TotalDuration      time.Duration     `json:"total_duration"`
	AverageDuration    time.Duration     `json:"average_duration"`
	ByErrorCode        map[ErrorCode]int `json:"by_error_code"`
	ByComponent        map[string]int    `json:"by_component"`
	ByRecoveryType     map[string]int    `json:"by_recovery_type"`
}
