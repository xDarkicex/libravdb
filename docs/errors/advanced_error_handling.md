# Advanced Error Handling and Recovery System

This document describes the comprehensive error handling and recovery system implemented in LibraVDB as part of the competitive features phase 3.

## Overview

The advanced error handling system provides:

1. **Structured Error Types** - Detailed error classification with context and metadata
2. **Automatic Recovery Mechanisms** - Intelligent recovery strategies for different failure modes
3. **Graceful Degradation** - System-wide degradation under resource pressure
4. **Circuit Breaker Pattern** - Fault tolerance for failing components
5. **Health Monitoring** - Continuous system health assessment
6. **Recovery Orchestration** - Coordinated recovery across all system components

## Core Components

### 1. VectorDBError

Enhanced error type with rich context and recovery information:

```go
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
```

**Features:**
- Structured error codes for different failure types
- Severity levels (Info, Warning, Error, Critical, Fatal)
- Recovery action hints (None, Retry, Fallback, Graceful Degradation, Restart, Rebuild)
- Rich context with component, operation, stack trace, and metadata
- Automatic retry logic with backoff

**Usage:**
```go
// Create a structured error
err := NewVectorDBErrorWithContext(
    ErrCodeMemoryExhausted,
    "memory limit exceeded during vector insertion",
    true, // retryable
    "collection",
    "insert",
).WithSeverity(SeverityCritical).
  WithRecoveryAction(RecoveryGracefulDegradation).
  WithMetadata("memory_usage", "1.2GB").
  WithRequestID("req-123")
```

### 2. Error Recovery Manager

Coordinates automatic error recovery using pluggable strategies:

```go
type ErrorRecoveryManager struct {
    recoveryStrategies map[ErrorCode]RecoveryStrategy
    circuitBreakers    map[string]CircuitBreaker
    degradationManager GracefulDegradationManager
    healthMonitor      SystemHealthMonitor
}
```

**Recovery Strategies:**
- **Memory Pressure Recovery** - GC, cache eviction, memory mapping
- **Quantization Recovery** - Fallback to uncompressed storage
- **Index Corruption Recovery** - Index rebuild or simplification
- **Batch Operation Recovery** - Chunking, retry with backoff

**Usage:**
```go
erm := NewErrorRecoveryManager()

// Register recovery strategies
memoryStrategy := NewMemoryPressureRecoveryStrategy(memoryManager)
erm.RegisterRecoveryStrategy(ErrCodeMemoryExhausted, memoryStrategy)

// Attempt recovery
err := NewVectorDBError(ErrCodeMemoryExhausted, "memory exhausted", true)
if recoveryErr := erm.TryRecover(ctx, err); recoveryErr != nil {
    // Recovery failed
}
```

### 3. Graceful Degradation Manager

Implements system-wide graceful degradation under various failure conditions:

```go
type GracefulDegradationManager interface {
    HandleMemoryPressure(ctx context.Context, pressureLevel int) error
    HandleQuantizationFailure(ctx context.Context, fallbackEnabled bool) error
    HandleIndexCorruption(ctx context.Context, rebuildEnabled bool) error
    GetDegradationLevel() int
    SetDegradationLevel(level int) error
}
```

**Degradation Levels:**
1. **Level 1** - Garbage collection, lightweight cleanup
2. **Level 2** - Cache eviction, memory optimization
3. **Level 3** - Memory mapping activation
4. **Level 4** - Quantization fallback to uncompressed storage
5. **Level 5** - Index simplification (aggressive)

**Configuration:**
```go
config := DegradationConfig{
    EnableQuantizationFallback: true,
    EnableMemoryMapping:        true,
    EnableCacheEviction:        true,
    EnableIndexSimplification:  false, // More aggressive
    MaxDegradationLevel:        5,
    AutoRecoveryEnabled:        true,
    RecoveryCheckInterval:      30 * time.Second,
}

gdm := NewGracefulDegradationManager(config)
```

### 4. Circuit Breaker

Implements the circuit breaker pattern for fault tolerance:

```go
type CircuitBreaker interface {
    Execute(ctx context.Context, fn func() error) error
    State() string // "CLOSED", "OPEN", "HALF_OPEN"
    Reset()
}
```

**States:**
- **CLOSED** - Normal operation, requests allowed
- **OPEN** - Circuit open, requests rejected immediately
- **HALF_OPEN** - Testing recovery, limited requests allowed

**Configuration:**
```go
config := CircuitBreakerConfig{
    Name:             "quantization",
    MaxFailures:      5,
    Timeout:          30 * time.Second,
    MaxRequests:      3,
    FailureThreshold: 0.6, // 60% failure rate
    MinRequests:      10,
}

breaker := NewCircuitBreaker(config)
```

### 5. System Health Monitor

Continuously monitors system health and triggers recovery:

```go
type SystemHealthMonitor interface {
    RegisterHealthCheck(name string, check HealthCheck) error
    GetHealthStatus() HealthStatus
    Start(ctx context.Context) error
    Stop() error
}
```

**Health Levels:**
- **HEALTHY** - Component operating normally
- **DEGRADED** - Component experiencing minor issues
- **UNHEALTHY** - Component experiencing significant issues
- **CRITICAL** - Component in critical state
- **UNKNOWN** - Health status unknown

**Usage:**
```go
monitor := NewSystemHealthMonitor(5 * time.Second)

// Register health checks
monitor.RegisterHealthCheck("memory", func(ctx context.Context) (HealthLevel, error) {
    usage := memoryManager.GetUsage()
    if usage.Limit > 0 && float64(usage.Total)/float64(usage.Limit) > 0.9 {
        return HealthUnhealthy, fmt.Errorf("high memory usage")
    }
    return HealthHealthy, nil
})

monitor.Start(ctx)
```

### 6. Automatic Recovery Orchestrator

Coordinates recovery across all system components:

```go
type AutomaticRecoveryOrchestrator struct {
    errorRecoveryManager        *ErrorRecoveryManager
    memoryRecoveryManager       MemoryRecoveryManager
    quantizationRecoveryManager QuantizationRecoveryManager
    batchRecoveryManager        BatchRecoveryManager
}
```

**Features:**
- Cross-component recovery coordination
- Recovery attempt tracking and statistics
- Failure pattern analysis
- Recovery success rate monitoring

## Error Types and Recovery Actions

### Memory-Related Errors

| Error Code | Description | Recovery Actions |
|------------|-------------|------------------|
| `ErrCodeMemoryExhausted` | Memory limit exceeded | GC → Cache eviction → Memory mapping → Quantization fallback |
| `ErrCodeMemoryPressure` | High memory usage | Proactive cleanup and optimization |
| `ErrCodeMemoryMappingFailure` | Memory mapping failed | Retry with different parameters, fallback to RAM |
| `ErrCodeCacheFailure` | Cache operation failed | Cache cleanup, size reduction |

### Quantization-Related Errors

| Error Code | Description | Recovery Actions |
|------------|-------------|------------------|
| `ErrCodeQuantizationFailure` | Quantization operation failed | Retry with reduced complexity → Fallback to uncompressed |
| `ErrCodeQuantizationCorruption` | Quantization data corrupted | Retrain quantizer → Fallback to uncompressed |
| `ErrCodeQuantizationTraining` | Training failed | Adjust parameters → Reduce complexity → Fallback |

### Index-Related Errors

| Error Code | Description | Recovery Actions |
|------------|-------------|------------------|
| `ErrCodeIndexCorruption` | Index data corrupted | Rebuild index → Simplify index → Fallback to flat search |
| `ErrCodeIndexFailure` | Index operation failed | Retry → Rebuild → Simplify |
| `ErrCodeIndexRebuild` | Index rebuild required | Automatic rebuild with progress tracking |

### Batch Operation Errors

| Error Code | Description | Recovery Actions |
|------------|-------------|------------------|
| `ErrCodeBatchFailure` | Batch operation failed | Retry failed items → Reduce batch size → Sequential processing |
| `ErrCodeBatchTimeout` | Batch operation timed out | Reduce batch size → Increase timeout → Chunking |
| `ErrCodeBatchSizeLimit` | Batch too large | Automatic chunking with optimal size |

## Integration Examples

### Basic Error Handling

```go
// Create collection with error handling
collection, err := db.CreateCollection("vectors", CollectionConfig{
    Dimension: 128,
    Metric:    DistanceEuclidean,
})
if err != nil {
    if vectorErr, ok := err.(*VectorDBError); ok {
        log.Printf("Error: %s (Code: %d, Severity: %s)", 
            vectorErr.Message, vectorErr.Code, vectorErr.Severity)
        
        if vectorErr.IsRetryable() {
            // Retry logic
        }
    }
    return err
}
```

### Automatic Recovery Setup

```go
// Setup comprehensive error recovery
erm := NewErrorRecoveryManager()

// Memory recovery
memoryManager := NewMemoryManager(MemoryConfig{MaxMemory: 2 * 1024 * 1024 * 1024}) // 2GB
memoryStrategy := NewMemoryPressureRecoveryStrategy(memoryManager)
erm.RegisterRecoveryStrategy(ErrCodeMemoryExhausted, memoryStrategy)

// Quantization recovery
quantStrategy := NewQuantizationRecoveryStrategy(true) // Enable fallback
erm.RegisterRecoveryStrategy(ErrCodeQuantizationFailure, quantStrategy)

// Graceful degradation
config := DefaultDegradationConfig()
gdm := NewGracefulDegradationManager(config)
gdm.SetMemoryManager(memoryManager)
erm.SetDegradationManager(gdm)

// Health monitoring
healthMonitor := NewSystemHealthMonitor(5 * time.Second)
healthMonitor.RegisterHealthCheck("memory", memoryHealthCheck)
healthMonitor.RegisterHealthCheck("quantization", quantizationHealthCheck)
erm.SetHealthMonitor(healthMonitor)

// Start monitoring
healthMonitor.Start(ctx)
gdm.StartAutoRecovery(ctx)
```

### Circuit Breaker Integration

```go
// Setup circuit breakers for critical components
cbManager := NewCircuitBreakerManager()

quantizationBreaker := cbManager.GetOrCreate("quantization", 
    DefaultCircuitBreakerConfig("quantization"))

// Use circuit breaker for quantization operations
err := quantizationBreaker.Execute(ctx, func() error {
    return quantizer.Train(ctx, vectors)
})

if err != nil {
    log.Printf("Quantization failed: %v (Circuit state: %s)", 
        err, quantizationBreaker.State())
}
```

### Health-Aware Operations

```go
// Health-aware error handler
handler := NewHealthAwareErrorHandler(healthMonitor, erm)

// Operations automatically adapt to system health
err := someOperation()
if err != nil {
    recoveredErr := handler.HandleError(ctx, err)
    if recoveredErr != nil {
        // Recovery failed, handle appropriately
        return recoveredErr
    }
    // Recovery succeeded, continue
}
```

## Monitoring and Observability

### Recovery Statistics

```go
// Get recovery statistics
orchestrator := NewAutomaticRecoveryOrchestrator(erm)
stats := orchestrator.GetRecoveryStats()

fmt.Printf("Recovery Success Rate: %.2f%%\n", stats.SuccessRate*100)
fmt.Printf("Total Attempts: %d\n", stats.TotalAttempts)
fmt.Printf("Average Duration: %v\n", stats.AverageDuration)

// By error type
for errorCode, count := range stats.ByErrorCode {
    fmt.Printf("Error %d: %d attempts\n", errorCode, count)
}
```

### Health Status Monitoring

```go
// Monitor health status changes
healthMonitor.RegisterCallback(func(status HealthStatus) {
    log.Printf("System health: %s", status.Overall)
    
    for component, level := range status.Components {
        if level != HealthHealthy {
            log.Printf("Component %s: %s", component, level)
        }
    }
})
```

### Degradation Event Tracking

```go
// Track degradation events
gdm.RegisterCallback(func(event DegradationEvent) {
    log.Printf("Degradation %s: %s (Level %d, Success: %t)", 
        event.Action, event.Type, event.Level, event.Success)
})

// Get degradation history
history := gdm.GetDegradationHistory()
for _, event := range history {
    fmt.Printf("%s: %s - %s\n", 
        event.Timestamp.Format(time.RFC3339), 
        event.Type, 
        event.Reason)
}
```

## Best Practices

### 1. Error Classification

- Use appropriate error codes for different failure types
- Set correct severity levels (Critical for data loss, Warning for performance issues)
- Include relevant context and metadata
- Use structured errors consistently across the codebase

### 2. Recovery Strategy Design

- Implement recovery strategies that are idempotent
- Use progressive recovery (try lightweight solutions first)
- Include circuit breakers for external dependencies
- Log recovery attempts and outcomes for debugging

### 3. Graceful Degradation

- Design degradation levels that maintain core functionality
- Make degradation actions reversible when possible
- Monitor system health continuously
- Implement automatic recovery when conditions improve

### 4. Health Monitoring

- Register health checks for all critical components
- Use appropriate health check intervals (not too frequent)
- Include meaningful error messages in health check failures
- Monitor health trends, not just current status

### 5. Testing

- Test error conditions and recovery paths
- Simulate various failure scenarios
- Verify graceful degradation behavior
- Test circuit breaker state transitions
- Validate health monitoring accuracy

## Performance Considerations

### Memory Usage

The error handling system is designed to be lightweight:
- Error objects are created only when needed
- Recovery strategies are stateless and reusable
- Health monitoring uses efficient polling
- Circuit breakers have minimal overhead

### Latency Impact

- Error creation: ~1-5μs overhead
- Recovery attempts: Variable (depends on strategy)
- Health checks: Configurable interval (default 5s)
- Circuit breaker checks: ~100ns overhead

### Concurrency

All components are thread-safe and designed for concurrent use:
- Error recovery manager uses read-write locks
- Health monitor supports concurrent health checks
- Circuit breakers handle concurrent requests safely
- Degradation manager coordinates actions atomically

## Troubleshooting

### Common Issues

1. **Recovery Loops** - Ensure recovery strategies don't cause the same error
2. **Circuit Breaker Stuck Open** - Check failure thresholds and timeout settings
3. **Health Check Timeouts** - Verify health check implementation efficiency
4. **Memory Leaks in Error Tracking** - Recovery history is automatically pruned

### Debugging

Enable detailed error logging:
```go
// Log all error recovery attempts
erm.RegisterRecoveryStrategy(errorCode, &LoggingRecoveryStrategy{
    underlying: actualStrategy,
    logger:     log.New(os.Stdout, "RECOVERY: ", log.LstdFlags),
})
```

Monitor system metrics:
```go
// Export metrics for monitoring
metrics := map[string]interface{}{
    "recovery_success_rate": stats.SuccessRate,
    "circuit_breaker_states": cbManager.GetStates(),
    "degradation_level": gdm.GetDegradationLevel(),
    "health_status": healthMonitor.GetHealthStatus(),
}
```

This advanced error handling system provides comprehensive fault tolerance and automatic recovery capabilities, making LibraVDB more robust and suitable for production deployments.