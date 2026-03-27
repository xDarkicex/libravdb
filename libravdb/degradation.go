package libravdb

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// GracefulDegradationManagerImpl implements GracefulDegradationManager
type GracefulDegradationManagerImpl struct {
	mu                  sync.RWMutex
	degradationLevel    int
	maxDegradationLevel int
	memoryManager       MemoryManager
	quantizationManager QuantizationManager
	indexManager        IndexManager
	config              DegradationConfig
	activeDegradations  map[string]DegradationAction
	degradationHistory  []DegradationEvent
	callbacks           []DegradationCallback
}

// DegradationConfig configures graceful degradation behavior
type DegradationConfig struct {
	EnableQuantizationFallback bool          `json:"enable_quantization_fallback"`
	EnableMemoryMapping        bool          `json:"enable_memory_mapping"`
	EnableCacheEviction        bool          `json:"enable_cache_eviction"`
	EnableIndexSimplification  bool          `json:"enable_index_simplification"`
	MaxDegradationLevel        int           `json:"max_degradation_level"`
	RecoveryTimeout            time.Duration `json:"recovery_timeout"`
	AutoRecoveryEnabled        bool          `json:"auto_recovery_enabled"`
	RecoveryCheckInterval      time.Duration `json:"recovery_check_interval"`
}

// DefaultDegradationConfig returns sensible defaults
func DefaultDegradationConfig() DegradationConfig {
	return DegradationConfig{
		EnableQuantizationFallback: true,
		EnableMemoryMapping:        true,
		EnableCacheEviction:        true,
		EnableIndexSimplification:  false, // More aggressive
		MaxDegradationLevel:        5,
		RecoveryTimeout:            5 * time.Minute,
		AutoRecoveryEnabled:        true,
		RecoveryCheckInterval:      30 * time.Second,
	}
}

// DegradationAction represents an active degradation
type DegradationAction struct {
	Type        string                 `json:"type"`
	Level       int                    `json:"level"`
	StartTime   time.Time              `json:"start_time"`
	Description string                 `json:"description"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	Reversible  bool                   `json:"reversible"`
}

// DegradationEvent represents a degradation event in history
type DegradationEvent struct {
	Timestamp time.Time              `json:"timestamp"`
	Action    string                 `json:"action"` // "apply" or "revert"
	Type      string                 `json:"type"`
	Level     int                    `json:"level"`
	Reason    string                 `json:"reason"`
	Success   bool                   `json:"success"`
	Duration  time.Duration          `json:"duration"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// DegradationCallback is called when degradation actions are applied or reverted
type DegradationCallback func(event DegradationEvent)

// MemoryManager interface for degradation
type MemoryManager interface {
	GetUsage() MemoryUsage
	HandleMemoryLimitExceeded() error
	TriggerGC() error
	EvictCaches(bytes int64) (int64, error)
	EnableMemoryMapping() error
	DisableMemoryMapping() error
}

// MemoryUsage represents memory usage statistics
type MemoryUsage struct {
	Total        int64     `json:"total"`
	Indices      int64     `json:"indices"`
	Caches       int64     `json:"caches"`
	Quantized    int64     `json:"quantized"`
	MemoryMapped int64     `json:"memory_mapped"`
	Available    int64     `json:"available"`
	Limit        int64     `json:"limit"`
	Timestamp    time.Time `json:"timestamp"`
}

// QuantizationManager interface for degradation
type QuantizationManager interface {
	FallbackToUncompressed(component string) error
	RestoreQuantization(component string) error
	ReduceQuantizationPrecision(component string, level int) error
	GetQuantizationStatus(component string) (bool, error)
}

// IndexManager interface for degradation
type IndexManager interface {
	SimplifyIndex(component string, level int) error
	RestoreIndex(component string) error
	GetIndexComplexity(component string) (int, error)
	RebuildIndex(component string) error
}

// NewGracefulDegradationManager creates a new graceful degradation manager
func NewGracefulDegradationManager(config DegradationConfig) *GracefulDegradationManagerImpl {
	return &GracefulDegradationManagerImpl{
		config:              config,
		maxDegradationLevel: config.MaxDegradationLevel,
		activeDegradations:  make(map[string]DegradationAction),
		degradationHistory:  make([]DegradationEvent, 0),
		callbacks:           make([]DegradationCallback, 0),
	}
}

// SetMemoryManager sets the memory manager
func (gdm *GracefulDegradationManagerImpl) SetMemoryManager(manager MemoryManager) {
	gdm.mu.Lock()
	defer gdm.mu.Unlock()
	gdm.memoryManager = manager
}

// SetQuantizationManager sets the quantization manager
func (gdm *GracefulDegradationManagerImpl) SetQuantizationManager(manager QuantizationManager) {
	gdm.mu.Lock()
	defer gdm.mu.Unlock()
	gdm.quantizationManager = manager
}

// SetIndexManager sets the index manager
func (gdm *GracefulDegradationManagerImpl) SetIndexManager(manager IndexManager) {
	gdm.mu.Lock()
	defer gdm.mu.Unlock()
	gdm.indexManager = manager
}

// RegisterCallback registers a callback for degradation events
func (gdm *GracefulDegradationManagerImpl) RegisterCallback(callback DegradationCallback) {
	gdm.mu.Lock()
	defer gdm.mu.Unlock()
	gdm.callbacks = append(gdm.callbacks, callback)
}

// GetDegradationLevel returns the current degradation level
func (gdm *GracefulDegradationManagerImpl) GetDegradationLevel() int {
	gdm.mu.RLock()
	defer gdm.mu.RUnlock()
	return gdm.degradationLevel
}

// SetDegradationLevel sets the degradation level
func (gdm *GracefulDegradationManagerImpl) SetDegradationLevel(level int) error {
	gdm.mu.Lock()
	defer gdm.mu.Unlock()

	if level < 0 || level > gdm.maxDegradationLevel {
		return fmt.Errorf("degradation level %d out of range [0, %d]", level, gdm.maxDegradationLevel)
	}

	oldLevel := gdm.degradationLevel
	gdm.degradationLevel = level

	// Apply or revert degradations as needed
	if level > oldLevel {
		return gdm.applyDegradationLevel(level, "manual_set")
	} else if level < oldLevel {
		return gdm.revertDegradationLevel(oldLevel, level, "manual_set")
	}

	return nil
}

// HandleMemoryPressure implements graceful degradation under memory pressure
func (gdm *GracefulDegradationManagerImpl) HandleMemoryPressure(ctx context.Context, pressureLevel int) error {
	startTime := time.Now()

	if pressureLevel > gdm.config.MaxDegradationLevel {
		return fmt.Errorf("memory pressure level %d exceeds maximum degradation level %d",
			pressureLevel, gdm.config.MaxDegradationLevel)
	}

	gdm.mu.Lock()
	defer gdm.mu.Unlock()

	var appliedActions []string

	// Level 1: Trigger garbage collection
	if pressureLevel >= 1 {
		if gdm.memoryManager != nil {
			if err := gdm.memoryManager.TriggerGC(); err == nil {
				appliedActions = append(appliedActions, "garbage_collection")
				gdm.recordDegradationAction("memory_gc", 1, "Triggered garbage collection", true)
			}
		}
	}

	// Level 2: Enable cache eviction
	if pressureLevel >= 2 && gdm.config.EnableCacheEviction {
		if gdm.memoryManager != nil {
			if freed, err := gdm.memoryManager.EvictCaches(0); err == nil && freed > 0 {
				appliedActions = append(appliedActions, "cache_eviction")
				gdm.recordDegradationAction("cache_eviction", 2, fmt.Sprintf("Evicted %d bytes from caches", freed), true)
			}
		}
	}

	// Level 3: Enable memory mapping
	if pressureLevel >= 3 && gdm.config.EnableMemoryMapping {
		if gdm.memoryManager != nil {
			if err := gdm.memoryManager.EnableMemoryMapping(); err == nil {
				appliedActions = append(appliedActions, "memory_mapping")
				gdm.recordDegradationAction("memory_mapping", 3, "Enabled memory mapping", true)
			}
		}
	}

	// Level 4: Fallback from quantization
	if pressureLevel >= 4 && gdm.config.EnableQuantizationFallback {
		if gdm.quantizationManager != nil {
			if err := gdm.quantizationManager.FallbackToUncompressed("all"); err == nil {
				appliedActions = append(appliedActions, "quantization_fallback")
				gdm.recordDegradationAction("quantization_fallback", 4, "Fallback to uncompressed storage", true)
			}
		}
	}

	// Level 5: Simplify index structures
	if pressureLevel >= 5 && gdm.config.EnableIndexSimplification {
		if gdm.indexManager != nil {
			if err := gdm.indexManager.SimplifyIndex("all", pressureLevel); err == nil {
				appliedActions = append(appliedActions, "index_simplification")
				gdm.recordDegradationAction("index_simplification", 5, "Simplified index structures", false)
			}
		}
	}

	// Update degradation level
	if pressureLevel > gdm.degradationLevel {
		gdm.degradationLevel = pressureLevel
	}

	// Record event
	event := DegradationEvent{
		Timestamp: startTime,
		Action:    "apply",
		Type:      "memory_pressure",
		Level:     pressureLevel,
		Reason:    fmt.Sprintf("Memory pressure level %d", pressureLevel),
		Success:   len(appliedActions) > 0,
		Duration:  time.Since(startTime),
		Metadata: map[string]interface{}{
			"applied_actions": appliedActions,
		},
	}

	gdm.degradationHistory = append(gdm.degradationHistory, event)
	gdm.notifyCallbacks(event)

	if len(appliedActions) == 0 {
		return fmt.Errorf("no degradation actions available for memory pressure level %d", pressureLevel)
	}

	return nil
}

// HandleQuantizationFailure implements graceful degradation for quantization failures
func (gdm *GracefulDegradationManagerImpl) HandleQuantizationFailure(ctx context.Context, fallbackEnabled bool) error {
	startTime := time.Now()

	gdm.mu.Lock()
	defer gdm.mu.Unlock()

	if !fallbackEnabled || !gdm.config.EnableQuantizationFallback {
		return fmt.Errorf("quantization fallback disabled")
	}

	var success bool
	var errorMsg string

	if gdm.quantizationManager != nil {
		if err := gdm.quantizationManager.FallbackToUncompressed("failed_component"); err == nil {
			success = true
			gdm.recordDegradationAction("quantization_failure_fallback", gdm.degradationLevel+1, "Fallback due to quantization failure", true)
		} else {
			errorMsg = err.Error()
		}
	} else {
		errorMsg = "quantization manager not available"
	}

	// Record event
	event := DegradationEvent{
		Timestamp: startTime,
		Action:    "apply",
		Type:      "quantization_failure",
		Level:     gdm.degradationLevel + 1,
		Reason:    "Quantization failure recovery",
		Success:   success,
		Duration:  time.Since(startTime),
		Metadata: map[string]interface{}{
			"fallback_enabled": fallbackEnabled,
			"error":            errorMsg,
		},
	}

	gdm.degradationHistory = append(gdm.degradationHistory, event)
	gdm.notifyCallbacks(event)

	if !success {
		return fmt.Errorf("quantization failure recovery failed: %s", errorMsg)
	}

	return nil
}

// HandleIndexCorruption implements graceful degradation for index corruption
func (gdm *GracefulDegradationManagerImpl) HandleIndexCorruption(ctx context.Context, rebuildEnabled bool) error {
	startTime := time.Now()

	gdm.mu.Lock()
	defer gdm.mu.Unlock()

	var success bool
	var errorMsg string
	var action string

	if gdm.indexManager != nil {
		if rebuildEnabled {
			// Try to rebuild the index
			if err := gdm.indexManager.RebuildIndex("corrupted_component"); err == nil {
				success = true
				action = "index_rebuild"
				gdm.recordDegradationAction("index_corruption_rebuild", gdm.degradationLevel, "Rebuilt corrupted index", false)
			} else if gdm.config.EnableIndexSimplification {
				// Fallback to index simplification
				if err := gdm.indexManager.SimplifyIndex("corrupted_component", 1); err == nil {
					success = true
					action = "index_simplification"
					gdm.recordDegradationAction("index_corruption_simplify", gdm.degradationLevel+1, "Simplified corrupted index", false)
				} else {
					errorMsg = err.Error()
				}
			} else {
				errorMsg = err.Error()
			}
		} else {
			errorMsg = "index rebuild disabled"
		}
	} else {
		errorMsg = "index manager not available"
	}

	// Record event
	event := DegradationEvent{
		Timestamp: startTime,
		Action:    "apply",
		Type:      "index_corruption",
		Level:     gdm.degradationLevel,
		Reason:    "Index corruption recovery",
		Success:   success,
		Duration:  time.Since(startTime),
		Metadata: map[string]interface{}{
			"rebuild_enabled": rebuildEnabled,
			"action_taken":    action,
			"error":           errorMsg,
		},
	}

	gdm.degradationHistory = append(gdm.degradationHistory, event)
	gdm.notifyCallbacks(event)

	if !success {
		return fmt.Errorf("index corruption recovery failed: %s", errorMsg)
	}

	return nil
}

// applyDegradationLevel applies degradation up to the specified level
func (gdm *GracefulDegradationManagerImpl) applyDegradationLevel(level int, reason string) error {
	// This would contain the logic to apply degradations incrementally
	// For now, we'll just record the action
	gdm.recordDegradationAction("level_increase", level, fmt.Sprintf("Increased degradation level to %d: %s", level, reason), true)
	return nil
}

// revertDegradationLevel reverts degradation from oldLevel to newLevel
func (gdm *GracefulDegradationManagerImpl) revertDegradationLevel(oldLevel, newLevel int, reason string) error {
	// This would contain the logic to revert degradations
	// For now, we'll just record the action
	gdm.recordDegradationAction("level_decrease", newLevel, fmt.Sprintf("Decreased degradation level from %d to %d: %s", oldLevel, newLevel, reason), true)
	return nil
}

// recordDegradationAction records an active degradation action
func (gdm *GracefulDegradationManagerImpl) recordDegradationAction(actionType string, level int, description string, reversible bool) {
	action := DegradationAction{
		Type:        actionType,
		Level:       level,
		StartTime:   time.Now(),
		Description: description,
		Metadata:    make(map[string]interface{}),
		Reversible:  reversible,
	}

	gdm.activeDegradations[actionType] = action
}

// notifyCallbacks notifies all registered callbacks
func (gdm *GracefulDegradationManagerImpl) notifyCallbacks(event DegradationEvent) {
	for _, callback := range gdm.callbacks {
		go callback(event)
	}
}

// GetActiveDegradations returns currently active degradations
func (gdm *GracefulDegradationManagerImpl) GetActiveDegradations() map[string]DegradationAction {
	gdm.mu.RLock()
	defer gdm.mu.RUnlock()

	result := make(map[string]DegradationAction)
	for k, v := range gdm.activeDegradations {
		result[k] = v
	}
	return result
}

// GetDegradationHistory returns the degradation event history
func (gdm *GracefulDegradationManagerImpl) GetDegradationHistory() []DegradationEvent {
	gdm.mu.RLock()
	defer gdm.mu.RUnlock()

	history := make([]DegradationEvent, len(gdm.degradationHistory))
	copy(history, gdm.degradationHistory)
	return history
}

// StartAutoRecovery starts automatic recovery monitoring
func (gdm *GracefulDegradationManagerImpl) StartAutoRecovery(ctx context.Context) error {
	if !gdm.config.AutoRecoveryEnabled {
		return fmt.Errorf("auto recovery disabled")
	}

	go gdm.autoRecoveryLoop(ctx)
	return nil
}

// autoRecoveryLoop monitors system health and attempts recovery
func (gdm *GracefulDegradationManagerImpl) autoRecoveryLoop(ctx context.Context) {
	ticker := time.NewTicker(gdm.config.RecoveryCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			gdm.checkAndAttemptRecovery()
		}
	}
}

// checkAndAttemptRecovery checks if recovery is possible and attempts it
func (gdm *GracefulDegradationManagerImpl) checkAndAttemptRecovery() {
	gdm.mu.Lock()
	defer gdm.mu.Unlock()

	if gdm.degradationLevel == 0 {
		return // No degradation to recover from
	}

	// Check if system conditions have improved
	canRecover := gdm.assessRecoveryConditions()
	if !canRecover {
		return
	}

	// Attempt to reduce degradation level
	newLevel := gdm.degradationLevel - 1
	if newLevel < 0 {
		newLevel = 0
	}

	if err := gdm.revertDegradationLevel(gdm.degradationLevel, newLevel, "auto_recovery"); err == nil {
		gdm.degradationLevel = newLevel

		// Record recovery event
		event := DegradationEvent{
			Timestamp: time.Now(),
			Action:    "revert",
			Type:      "auto_recovery",
			Level:     newLevel,
			Reason:    "Automatic recovery attempt",
			Success:   true,
			Duration:  0,
		}

		gdm.degradationHistory = append(gdm.degradationHistory, event)
		gdm.notifyCallbacks(event)
	}
}

// assessRecoveryConditions assesses whether recovery conditions are met
func (gdm *GracefulDegradationManagerImpl) assessRecoveryConditions() bool {
	// Simple heuristic: check memory usage
	if gdm.memoryManager != nil {
		usage := gdm.memoryManager.GetUsage()
		if usage.Limit > 0 {
			usageRatio := float64(usage.Total) / float64(usage.Limit)
			// Allow recovery if memory usage is below 70%
			return usageRatio < 0.7
		}
	}

	// If no memory manager or no limit, assume recovery is possible
	return true
}
