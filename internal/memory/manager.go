package memory

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"
)

// manager implements the MemoryManager interface
type manager struct {
	config MemoryConfig

	// Memory tracking
	mu        sync.RWMutex
	limit     int64
	caches    map[string]Cache
	mappables map[string]MemoryMappable
	lastUsage MemoryUsage

	// Memory mapping
	mmapManager *MemoryMapManager

	// Callbacks
	pressureCallbacks []func(usage MemoryUsage)
	releaseCallbacks  []func(freed int64)

	// Monitoring
	ctx    context.Context
	cancel context.CancelFunc
	done   chan struct{}

	// Pressure tracking
	lastPressureLevel MemoryPressureLevel
}

// NewManager creates a new memory manager with the given configuration
func NewManager(config MemoryConfig) MemoryManager {
	// Ensure config has default values if not set
	if config.PressureThresholds == nil {
		config.PressureThresholds = DefaultMemoryConfig().PressureThresholds
	}
	if config.MonitorInterval == 0 {
		config.MonitorInterval = DefaultMemoryConfig().MonitorInterval
	}
	if config.MMapPath == "" {
		config.MMapPath = DefaultMemoryConfig().MMapPath
	}
	if config.MMapThreshold == 0 {
		config.MMapThreshold = DefaultMemoryConfig().MMapThreshold
	}

	var mmapManager *MemoryMapManager
	if config.EnableMMap {
		mmapManager = NewMemoryMapManager(config.MMapPath)
	}

	return &manager{
		config:            config,
		limit:             config.MaxMemory,
		caches:            make(map[string]Cache),
		mappables:         make(map[string]MemoryMappable),
		mmapManager:       mmapManager,
		pressureCallbacks: make([]func(usage MemoryUsage), 0),
		releaseCallbacks:  make([]func(freed int64), 0),
		done:              make(chan struct{}),
		lastPressureLevel: NoPressure,
	}
}

// SetLimit configures the maximum memory usage in bytes
func (m *manager) SetLimit(bytes int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if bytes < 0 {
		return fmt.Errorf("memory limit cannot be negative: %d", bytes)
	}

	m.limit = bytes
	m.config.MaxMemory = bytes

	// Check if we're already over the new limit
	if bytes > 0 {
		usage := m.getCurrentUsage()
		if usage.Total > bytes {
			// Trigger immediate pressure response
			go m.handleMemoryPressure(usage)
		}
	}

	return nil
}

// GetUsage returns current memory usage statistics
func (m *manager) GetUsage() MemoryUsage {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.getCurrentUsage()
}

// getCurrentUsage calculates current memory usage (must be called with lock held)
func (m *manager) getCurrentUsage() MemoryUsage {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	// Calculate cache usage
	var cacheUsage int64
	for _, cache := range m.caches {
		cacheUsage += cache.Size()
	}

	// Calculate memory-mapped usage
	var mmapUsage int64
	for _, mappable := range m.mappables {
		if mappable.IsMemoryMapped() {
			mmapUsage += mappable.MemoryMappedSize()
		}
	}

	// For now, we'll use heap usage as our primary metric
	// Memory-mapped data doesn't count against heap usage
	heapUsage := int64(memStats.HeapInuse)
	totalUsage := heapUsage + mmapUsage

	usage := MemoryUsage{
		Total:        totalUsage,
		Indices:      heapUsage - cacheUsage, // Approximate
		Caches:       cacheUsage,
		Quantized:    0, // TODO: Track quantized memory separately
		MemoryMapped: mmapUsage,
		Limit:        m.limit,
		Timestamp:    time.Now(),
	}

	if m.limit > 0 {
		// Only count heap usage against the limit, not memory-mapped data
		usage.Available = m.limit - heapUsage
		if usage.Available < 0 {
			usage.Available = 0
		}
	} else {
		usage.Available = -1 // Unlimited
	}

	m.lastUsage = usage
	return usage
}

// TriggerGC forces garbage collection
func (m *manager) TriggerGC() error {
	runtime.GC()
	runtime.GC() // Run twice for better cleanup
	return nil
}

// RegisterCache registers a cache for memory management
func (m *manager) RegisterCache(name string, cache Cache) error {
	if cache == nil {
		return fmt.Errorf("cache cannot be nil")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.caches[name]; exists {
		return fmt.Errorf("cache with name %s already registered", name)
	}

	m.caches[name] = cache
	return nil
}

// UnregisterCache removes a cache from management
func (m *manager) UnregisterCache(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.caches[name]; !exists {
		return fmt.Errorf("cache with name %s not found", name)
	}

	delete(m.caches, name)
	return nil
}

// RegisterMemoryMappable registers a memory-mappable component
func (m *manager) RegisterMemoryMappable(name string, mappable MemoryMappable) error {
	if mappable == nil {
		return fmt.Errorf("mappable cannot be nil")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.mappables[name]; exists {
		return fmt.Errorf("mappable with name %s already registered", name)
	}

	m.mappables[name] = mappable

	// Check if we should automatically enable memory mapping
	if m.config.EnableMMap && mappable.CanMemoryMap() {
		size := mappable.EstimateSize()
		if size >= m.config.MMapThreshold {
			// Enable memory mapping in a separate goroutine to avoid blocking
			go func() {
				if err := mappable.EnableMemoryMapping(m.config.MMapPath); err != nil {
					// Log error but don't fail registration
					// In a real implementation, we'd use a proper logger
					fmt.Printf("Failed to enable memory mapping for %s: %v\n", name, err)
				}
			}()
		}
	}

	return nil
}

// UnregisterMemoryMappable removes a memory-mappable component
func (m *manager) UnregisterMemoryMappable(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	mappable, exists := m.mappables[name]
	if !exists {
		return fmt.Errorf("mappable with name %s not found", name)
	}

	// Disable memory mapping if enabled
	if mappable.IsMemoryMapped() {
		if err := mappable.DisableMemoryMapping(); err != nil {
			return fmt.Errorf("failed to disable memory mapping: %w", err)
		}
	}

	delete(m.mappables, name)
	return nil
}

// OnMemoryPressure registers a callback for memory pressure events
func (m *manager) OnMemoryPressure(callback func(usage MemoryUsage)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pressureCallbacks = append(m.pressureCallbacks, callback)
}

// OnMemoryRelease registers a callback for memory release events
func (m *manager) OnMemoryRelease(callback func(freed int64)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.releaseCallbacks = append(m.releaseCallbacks, callback)
}

// Start begins memory monitoring
func (m *manager) Start(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.ctx != nil {
		return fmt.Errorf("memory manager already started")
	}

	m.ctx, m.cancel = context.WithCancel(ctx)

	go m.monitorLoop()

	return nil
}

// Stop ends memory monitoring
func (m *manager) Stop() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.cancel == nil {
		return fmt.Errorf("memory manager not started")
	}

	m.cancel()

	// Wait for monitor loop to finish
	select {
	case <-m.done:
	case <-time.After(5 * time.Second):
		// Timeout waiting for graceful shutdown
	}

	// Clean up memory mappings
	if m.mmapManager != nil {
		if err := m.mmapManager.Close(); err != nil {
			fmt.Printf("Failed to close memory map manager: %v\n", err)
		}
	}

	m.ctx = nil
	m.cancel = nil

	return nil
}

// monitorLoop runs the memory monitoring in a separate goroutine
func (m *manager) monitorLoop() {
	defer close(m.done)

	// Safety check for context
	m.mu.RLock()
	ctx := m.ctx
	interval := m.config.MonitorInterval
	m.mu.RUnlock()

	if ctx == nil {
		return
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.checkMemoryUsage()
		}
	}
}

// checkMemoryUsage monitors memory and triggers appropriate responses
func (m *manager) checkMemoryUsage() {
	m.mu.RLock()
	usage := m.getCurrentUsage()
	limit := m.limit
	config := m.config
	ctx := m.ctx
	m.mu.RUnlock()

	// Always check for automatic memory mapping opportunities
	m.checkAndEnableAutomaticMemoryMapping()

	// Skip pressure monitoring if no limit is set or context is nil
	if limit <= 0 || ctx == nil {
		return
	}

	usageRatio := float64(usage.Total) / float64(limit)

	// Determine pressure level
	pressureLevel := m.calculatePressureLevel(usageRatio)

	// Trigger pressure callbacks if level changed
	if pressureLevel != m.lastPressureLevel && pressureLevel != NoPressure {
		m.handleMemoryPressure(usage)
		m.lastPressureLevel = pressureLevel
	}

	// Trigger automatic GC if enabled and threshold exceeded
	if config.EnableGC && usageRatio >= config.GCThreshold {
		beforeGC := usage.Total
		m.TriggerGC()

		// Measure freed memory
		afterUsage := m.GetUsage()
		freed := beforeGC - afterUsage.Total
		if freed > 0 {
			m.notifyMemoryRelease(freed)
		}
	}
}

// calculatePressureLevel determines the current memory pressure level
func (m *manager) calculatePressureLevel(usageRatio float64) MemoryPressureLevel {
	thresholds := m.config.PressureThresholds

	// Use default thresholds if not configured
	if thresholds == nil {
		thresholds = DefaultMemoryConfig().PressureThresholds
	}

	if usageRatio >= thresholds[CriticalPressure] {
		return CriticalPressure
	} else if usageRatio >= thresholds[HighPressure] {
		return HighPressure
	} else if usageRatio >= thresholds[ModeratePressure] {
		return ModeratePressure
	} else if usageRatio >= thresholds[LowPressure] {
		return LowPressure
	}

	return NoPressure
}

// handleMemoryPressure responds to memory pressure by evicting caches and enabling memory mapping
func (m *manager) handleMemoryPressure(usage MemoryUsage) {
	// Calculate how much memory we need to free
	if usage.Limit <= 0 {
		return
	}

	// Only consider heap usage for pressure calculation (not memory-mapped data)
	heapUsage := usage.Total - usage.MemoryMapped
	targetUsage := int64(float64(usage.Limit) * 0.8) // Target 80% usage
	needToFree := heapUsage - targetUsage

	if needToFree <= 0 {
		return
	}

	var totalFreed int64

	// Step 1: Try to free memory from caches
	totalFreed += m.evictFromCaches(needToFree)

	// Step 2: Enable memory mapping for eligible components
	if m.config.EnableMMap && totalFreed < needToFree {
		totalFreed += m.enableMemoryMappingForPressure(needToFree - totalFreed)
	}

	// Notify callbacks
	m.notifyPressureCallbacks(usage)

	if totalFreed > 0 {
		m.notifyMemoryRelease(totalFreed)
	}
}

// evictFromCaches attempts to free memory from registered caches
func (m *manager) evictFromCaches(targetBytes int64) int64 {
	m.mu.RLock()
	caches := make([]Cache, 0, len(m.caches))
	for _, cache := range m.caches {
		caches = append(caches, cache)
	}
	m.mu.RUnlock()

	var totalFreed int64
	remainingToFree := targetBytes

	// Evict from each cache proportionally
	for _, cache := range caches {
		if remainingToFree <= 0 {
			break
		}

		cacheSize := cache.Size()
		if cacheSize == 0 {
			continue
		}

		// Calculate how much to evict from this cache
		toEvict := remainingToFree
		if cacheSize < toEvict {
			toEvict = cacheSize
		}

		freed := cache.Evict(toEvict)
		totalFreed += freed
		remainingToFree -= freed
	}

	return totalFreed
}

// notifyPressureCallbacks calls all registered pressure callbacks
func (m *manager) notifyPressureCallbacks(usage MemoryUsage) {
	m.mu.RLock()
	callbacks := make([]func(usage MemoryUsage), len(m.pressureCallbacks))
	copy(callbacks, m.pressureCallbacks)
	m.mu.RUnlock()

	for _, callback := range callbacks {
		go callback(usage)
	}
}

// enableMemoryMappingForPressure enables memory mapping for components to free memory
func (m *manager) enableMemoryMappingForPressure(targetBytes int64) int64 {
	var totalFreed int64
	remainingToFree := targetBytes

	// Get list of mappables that can be memory mapped
	candidates := make([]struct {
		name     string
		mappable MemoryMappable
		size     int64
	}, 0)

	m.mu.RLock()
	for name, mappable := range m.mappables {
		if !mappable.IsMemoryMapped() && mappable.CanMemoryMap() {
			size := mappable.EstimateSize()
			if size > 0 {
				candidates = append(candidates, struct {
					name     string
					mappable MemoryMappable
					size     int64
				}{name, mappable, size})
			}
		}
	}
	m.mu.RUnlock()

	// Sort candidates by size (largest first) to maximize memory savings
	for i := 0; i < len(candidates)-1; i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[i].size < candidates[j].size {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	// Enable memory mapping for candidates until we've freed enough memory
	for _, candidate := range candidates {
		if remainingToFree <= 0 {
			break
		}

		if err := candidate.mappable.EnableMemoryMapping(m.config.MMapPath); err != nil {
			// Log error but continue with other candidates
			fmt.Printf("Failed to enable memory mapping for %s: %v\n", candidate.name, err)
			continue
		}

		// Assume we freed approximately the estimated size
		freed := candidate.size
		totalFreed += freed
		remainingToFree -= freed
	}

	return totalFreed
}

// checkAndEnableAutomaticMemoryMapping checks if any registered mappables should be automatically memory mapped
func (m *manager) checkAndEnableAutomaticMemoryMapping() {
	if !m.config.EnableMMap {
		return
	}

	m.mu.RLock()
	candidates := make([]struct {
		name     string
		mappable MemoryMappable
	}, 0)

	for name, mappable := range m.mappables {
		if !mappable.IsMemoryMapped() && mappable.CanMemoryMap() {
			size := mappable.EstimateSize()
			// Enable memory mapping if size exceeds threshold
			if size >= m.config.MMapThreshold {
				candidates = append(candidates, struct {
					name     string
					mappable MemoryMappable
				}{name, mappable})
			}
		}
	}
	m.mu.RUnlock()

	// Enable memory mapping for eligible candidates
	for _, candidate := range candidates {
		if err := candidate.mappable.EnableMemoryMapping(m.config.MMapPath); err != nil {
			fmt.Printf("Failed to auto-enable memory mapping for %s: %v\n", candidate.name, err)
		}
	}
}

// notifyMemoryRelease calls all registered release callbacks
func (m *manager) notifyMemoryRelease(freed int64) {
	m.mu.RLock()
	callbacks := make([]func(freed int64), len(m.releaseCallbacks))
	copy(callbacks, m.releaseCallbacks)
	m.mu.RUnlock()

	for _, callback := range callbacks {
		go callback(freed)
	}
}

// HandleMemoryLimitExceeded provides graceful handling when memory limits are exceeded
func (m *manager) HandleMemoryLimitExceeded() error {
	usage := m.GetUsage()

	if usage.Limit <= 0 {
		return nil // No limit set
	}

	if usage.Total <= usage.Limit {
		return nil // Not exceeded
	}

	// Calculate how much we need to free
	excessMemory := usage.Total - usage.Limit
	targetFree := excessMemory + (usage.Limit / 10) // Free 10% extra as buffer

	var totalFreed int64

	// Step 1: Force garbage collection
	beforeGC := usage.Total
	m.TriggerGC()
	afterGC := m.GetUsage().Total
	gcFreed := beforeGC - afterGC
	totalFreed += gcFreed

	if totalFreed >= targetFree {
		m.notifyMemoryRelease(totalFreed)
		return nil
	}

	// Step 2: Evict from caches aggressively
	cacheFreed := m.evictFromCaches(targetFree - totalFreed)
	totalFreed += cacheFreed

	if totalFreed >= targetFree {
		m.notifyMemoryRelease(totalFreed)
		return nil
	}

	// Step 3: Enable memory mapping for all eligible components
	mmapFreed := m.enableMemoryMappingForPressure(targetFree - totalFreed)
	totalFreed += mmapFreed

	if totalFreed >= targetFree {
		m.notifyMemoryRelease(totalFreed)
		return nil
	}

	// Step 4: If still over limit, return error but don't crash
	finalUsage := m.GetUsage()
	if finalUsage.Total > finalUsage.Limit {
		return fmt.Errorf("unable to reduce memory usage below limit: current=%d, limit=%d, freed=%d",
			finalUsage.Total, finalUsage.Limit, totalFreed)
	}

	m.notifyMemoryRelease(totalFreed)
	return nil
}
