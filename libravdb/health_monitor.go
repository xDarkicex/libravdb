package libravdb

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"
)

// SystemHealthMonitorImpl implements SystemHealthMonitor
type SystemHealthMonitorImpl struct {
	mu           sync.RWMutex
	healthChecks map[string]HealthCheck
	lastStatus   HealthStatus
	running      bool
	ctx          context.Context
	cancel       context.CancelFunc
	interval     time.Duration
	callbacks    []HealthStatusCallback
}

// HealthStatusCallback is called when health status changes
type HealthStatusCallback func(status HealthStatus)

// NewSystemHealthMonitor creates a new system health monitor
func NewSystemHealthMonitor(interval time.Duration) *SystemHealthMonitorImpl {
	return &SystemHealthMonitorImpl{
		healthChecks: make(map[string]HealthCheck),
		interval:     interval,
		callbacks:    make([]HealthStatusCallback, 0),
		lastStatus: HealthStatus{
			Overall:    HealthUnknown,
			Components: make(map[string]HealthLevel),
			Timestamp:  time.Now(),
			Details:    make(map[string]any),
		},
	}
}

// RegisterHealthCheck registers a health check for a component
func (shm *SystemHealthMonitorImpl) RegisterHealthCheck(name string, check HealthCheck) error {
	shm.mu.Lock()
	defer shm.mu.Unlock()

	shm.healthChecks[name] = check
	return nil
}

// UnregisterHealthCheck removes a health check
func (shm *SystemHealthMonitorImpl) UnregisterHealthCheck(name string) error {
	shm.mu.Lock()
	defer shm.mu.Unlock()

	delete(shm.healthChecks, name)
	return nil
}

// RegisterCallback registers a callback for health status changes
func (shm *SystemHealthMonitorImpl) RegisterCallback(callback HealthStatusCallback) {
	shm.mu.Lock()
	defer shm.mu.Unlock()

	shm.callbacks = append(shm.callbacks, callback)
}

// Start begins health monitoring
func (shm *SystemHealthMonitorImpl) Start(ctx context.Context) error {
	shm.mu.Lock()
	defer shm.mu.Unlock()

	if shm.running {
		return fmt.Errorf("health monitor already running")
	}

	shm.ctx, shm.cancel = context.WithCancel(ctx)
	shm.running = true

	go shm.monitorLoop()
	return nil
}

// Stop stops health monitoring
func (shm *SystemHealthMonitorImpl) Stop() error {
	shm.mu.Lock()
	defer shm.mu.Unlock()

	if !shm.running {
		return fmt.Errorf("health monitor not running")
	}

	shm.cancel()
	shm.running = false
	return nil
}

// GetHealthStatus returns the current health status
func (shm *SystemHealthMonitorImpl) GetHealthStatus() HealthStatus {
	shm.mu.RLock()
	defer shm.mu.RUnlock()

	// Return a copy to avoid race conditions
	status := HealthStatus{
		Overall:    shm.lastStatus.Overall,
		Components: make(map[string]HealthLevel),
		Timestamp:  shm.lastStatus.Timestamp,
		Details:    make(map[string]any),
	}

	for k, v := range shm.lastStatus.Components {
		status.Components[k] = v
	}

	for k, v := range shm.lastStatus.Details {
		status.Details[k] = v
	}

	return status
}

// monitorLoop runs the health monitoring loop
func (shm *SystemHealthMonitorImpl) monitorLoop() {
	ticker := time.NewTicker(shm.interval)
	defer ticker.Stop()

	// Perform initial health check
	shm.performHealthCheck()

	for {
		select {
		case <-shm.ctx.Done():
			return
		case <-ticker.C:
			shm.performHealthCheck()
		}
	}
}

// performHealthCheck executes all registered health checks
func (shm *SystemHealthMonitorImpl) performHealthCheck() {
	shm.mu.RLock()
	checks := make(map[string]HealthCheck)
	for name, check := range shm.healthChecks {
		checks[name] = check
	}
	callbacks := make([]HealthStatusCallback, len(shm.callbacks))
	copy(callbacks, shm.callbacks)
	shm.mu.RUnlock()

	newStatus := HealthStatus{
		Components: make(map[string]HealthLevel),
		Timestamp:  time.Now(),
		Details:    make(map[string]any),
	}

	// Execute health checks with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	overallHealth := HealthHealthy

	for name, check := range checks {
		level, err := check(ctx)
		newStatus.Components[name] = level

		if err != nil {
			newStatus.Details[name+"_error"] = err.Error()
		}

		// Determine overall health (worst component health)
		if level > overallHealth {
			overallHealth = level
		}
	}

	newStatus.Overall = overallHealth

	// Check if status changed
	shm.mu.Lock()
	statusChanged := shm.hasStatusChanged(shm.lastStatus, newStatus)
	shm.lastStatus = newStatus
	shm.mu.Unlock()

	// Notify callbacks if status changed
	if statusChanged {
		for _, callback := range callbacks {
			go callback(newStatus)
		}
	}
}

// hasStatusChanged checks if the health status has changed significantly
func (shm *SystemHealthMonitorImpl) hasStatusChanged(old, new HealthStatus) bool {
	if old.Overall != new.Overall {
		return true
	}

	// Check if any component status changed
	for name, newLevel := range new.Components {
		if oldLevel, exists := old.Components[name]; !exists || oldLevel != newLevel {
			return true
		}
	}

	// Check if any component was removed
	for name := range old.Components {
		if _, exists := new.Components[name]; !exists {
			return true
		}
	}

	return false
}

// DefaultHealthChecks returns a set of default health checks
func DefaultHealthChecks() map[string]HealthCheck {
	return map[string]HealthCheck{
		"memory": func(ctx context.Context) (HealthLevel, error) {
			// Simple memory health check
			var m runtime.MemStats
			runtime.ReadMemStats(&m)

			// Consider unhealthy if using more than 1GB
			if m.Alloc > 1024*1024*1024 {
				return HealthUnhealthy, fmt.Errorf("high memory usage: %d bytes", m.Alloc)
			}

			// Consider degraded if using more than 512MB
			if m.Alloc > 512*1024*1024 {
				return HealthDegraded, fmt.Errorf("elevated memory usage: %d bytes", m.Alloc)
			}

			return HealthHealthy, nil
		},

		"goroutines": func(ctx context.Context) (HealthLevel, error) {
			numGoroutines := runtime.NumGoroutine()

			// Consider critical if more than 10000 goroutines
			if numGoroutines > 10000 {
				return HealthCritical, fmt.Errorf("excessive goroutines: %d", numGoroutines)
			}

			// Consider unhealthy if more than 1000 goroutines
			if numGoroutines > 1000 {
				return HealthUnhealthy, fmt.Errorf("high goroutine count: %d", numGoroutines)
			}

			// Consider degraded if more than 100 goroutines
			if numGoroutines > 100 {
				return HealthDegraded, fmt.Errorf("elevated goroutine count: %d", numGoroutines)
			}

			return HealthHealthy, nil
		},
	}
}

// HealthAwareErrorHandler wraps error handling with health monitoring
type HealthAwareErrorHandler struct {
	healthMonitor SystemHealthMonitor
	errorRecovery *ErrorRecoveryManager
}

// NewHealthAwareErrorHandler creates a new health-aware error handler
func NewHealthAwareErrorHandler(healthMonitor SystemHealthMonitor, errorRecovery *ErrorRecoveryManager) *HealthAwareErrorHandler {
	return &HealthAwareErrorHandler{
		healthMonitor: healthMonitor,
		errorRecovery: errorRecovery,
	}
}

// HandleError handles an error with health awareness
func (haeh *HealthAwareErrorHandler) HandleError(ctx context.Context, err error) error {
	// Check system health before attempting recovery
	if haeh.healthMonitor != nil {
		status := haeh.healthMonitor.GetHealthStatus()

		// If system is in critical state, avoid aggressive recovery
		if status.Overall == HealthCritical {
			return fmt.Errorf("system in critical state, avoiding recovery: %w", err)
		}

		// If system is unhealthy, use conservative recovery
		if status.Overall == HealthUnhealthy {
			if vectorErr, ok := err.(*VectorDBError); ok {
				vectorErr.WithSeverity(SeverityCritical)
				vectorErr.WithRecoveryAction(RecoveryGracefulDegradation)
			}
		}
	}

	// Attempt recovery if error recovery manager is available
	if haeh.errorRecovery != nil {
		if vectorErr, ok := err.(*VectorDBError); ok {
			return haeh.errorRecovery.TryRecoverWithDegradation(ctx, vectorErr)
		}
	}

	return err
}

// ComponentHealthTracker tracks health of individual components
type ComponentHealthTracker struct {
	mu              sync.RWMutex
	componentHealth map[string]ComponentHealth
}

// ComponentHealth represents the health of a single component
type ComponentHealth struct {
	Name          string         `json:"name"`
	Level         HealthLevel    `json:"level"`
	LastCheck     time.Time      `json:"last_check"`
	LastError     string         `json:"last_error,omitempty"`
	ErrorCount    int            `json:"error_count"`
	SuccessCount  int            `json:"success_count"`
	Metadata      map[string]any `json:"metadata,omitempty"`
	RecoveryCount int            `json:"recovery_count"`
	LastRecovery  *time.Time     `json:"last_recovery,omitempty"`
}

// NewComponentHealthTracker creates a new component health tracker
func NewComponentHealthTracker() *ComponentHealthTracker {
	return &ComponentHealthTracker{
		componentHealth: make(map[string]ComponentHealth),
	}
}

// UpdateComponentHealth updates the health of a component
func (cht *ComponentHealthTracker) UpdateComponentHealth(name string, level HealthLevel, err error) {
	cht.mu.Lock()
	defer cht.mu.Unlock()

	health, exists := cht.componentHealth[name]
	if !exists {
		health = ComponentHealth{
			Name:     name,
			Metadata: make(map[string]any),
		}
	}

	health.Level = level
	health.LastCheck = time.Now()

	if err != nil {
		health.LastError = err.Error()
		health.ErrorCount++
	} else {
		health.SuccessCount++
		health.LastError = ""
	}

	cht.componentHealth[name] = health
}

// RecordRecovery records a recovery attempt for a component
func (cht *ComponentHealthTracker) RecordRecovery(name string, success bool) {
	cht.mu.Lock()
	defer cht.mu.Unlock()

	health, exists := cht.componentHealth[name]
	if !exists {
		health = ComponentHealth{
			Name:     name,
			Metadata: make(map[string]any),
		}
	}

	health.RecoveryCount++
	now := time.Now()
	health.LastRecovery = &now

	if success {
		health.Level = HealthHealthy
		health.LastError = ""
	}

	cht.componentHealth[name] = health
}

// GetComponentHealth returns the health of a specific component
func (cht *ComponentHealthTracker) GetComponentHealth(name string) (ComponentHealth, bool) {
	cht.mu.RLock()
	defer cht.mu.RUnlock()

	health, exists := cht.componentHealth[name]
	return health, exists
}

// GetAllComponentHealth returns the health of all components
func (cht *ComponentHealthTracker) GetAllComponentHealth() map[string]ComponentHealth {
	cht.mu.RLock()
	defer cht.mu.RUnlock()

	result := make(map[string]ComponentHealth)
	for name, health := range cht.componentHealth {
		result[name] = health
	}
	return result
}

// GetUnhealthyComponents returns components that are not healthy
func (cht *ComponentHealthTracker) GetUnhealthyComponents() []ComponentHealth {
	cht.mu.RLock()
	defer cht.mu.RUnlock()

	var unhealthy []ComponentHealth
	for _, health := range cht.componentHealth {
		if health.Level != HealthHealthy {
			unhealthy = append(unhealthy, health)
		}
	}
	return unhealthy
}
