package memory

import (
	"context"
	"time"
)

// MemoryManager provides memory usage monitoring and limit enforcement
type MemoryManager interface {
	// SetLimit configures the maximum memory usage in bytes
	SetLimit(bytes int64) error

	// GetUsage returns current memory usage statistics
	GetUsage() MemoryUsage

	// TriggerGC forces garbage collection
	TriggerGC() error

	// RegisterCache registers a cache for memory management
	RegisterCache(name string, cache Cache) error

	// UnregisterCache removes a cache from management
	UnregisterCache(name string) error

	// RegisterMemoryMappable registers a memory-mappable component
	RegisterMemoryMappable(name string, mappable MemoryMappable) error

	// UnregisterMemoryMappable removes a memory-mappable component
	UnregisterMemoryMappable(name string) error

	// OnMemoryPressure registers a callback for memory pressure events
	OnMemoryPressure(callback func(usage MemoryUsage))

	// OnMemoryRelease registers a callback for memory release events
	OnMemoryRelease(callback func(freed int64))

	// HandleMemoryLimitExceeded provides graceful handling when memory limits are exceeded
	HandleMemoryLimitExceeded() error

	// Start begins memory monitoring
	Start(ctx context.Context) error

	// Stop ends memory monitoring
	Stop() error
}

// MemoryUsage represents current memory usage statistics
type MemoryUsage struct {
	Timestamp    time.Time
	Total        int64
	Indices      int64
	Caches       int64
	Quantized    int64
	MemoryMapped int64
	Available    int64
	Limit        int64
}

// Cache interface for memory-managed caches
type Cache interface {
	// Evict removes items to free the specified number of bytes
	// Returns the actual number of bytes freed
	Evict(bytes int64) int64

	// Size returns current cache size in bytes
	Size() int64

	// Clear removes all items from the cache
	Clear()

	// Name returns the cache identifier
	Name() string
}

// MemoryMappable interface for components that support memory mapping
type MemoryMappable interface {
	// CanMemoryMap returns true if the component can be memory mapped
	CanMemoryMap() bool

	// EstimateSize returns the estimated size in bytes if memory mapped
	EstimateSize() int64

	// EnableMemoryMapping enables memory mapping for the component
	EnableMemoryMapping(path string) error

	// DisableMemoryMapping disables memory mapping and loads data back to RAM
	DisableMemoryMapping() error

	// IsMemoryMapped returns true if currently using memory mapping
	IsMemoryMapped() bool

	// MemoryMappedSize returns the size of memory-mapped data
	MemoryMappedSize() int64
}

// MemoryPressureLevel indicates severity of memory pressure
type MemoryPressureLevel int

const (
	NoPressure MemoryPressureLevel = iota
	LowPressure
	ModeratePressure
	HighPressure
	CriticalPressure
)

// String returns string representation of pressure level
func (l MemoryPressureLevel) String() string {
	switch l {
	case NoPressure:
		return "none"
	case LowPressure:
		return "low"
	case ModeratePressure:
		return "moderate"
	case HighPressure:
		return "high"
	case CriticalPressure:
		return "critical"
	default:
		return "unknown"
	}
}

// MemoryConfig configures memory management behavior
type MemoryConfig struct {
	PressureThresholds map[MemoryPressureLevel]float64
	MMapPath           string
	MaxMemory          int64
	MonitorInterval    time.Duration
	GCThreshold        float64
	MMapThreshold      int64
	EnableGC           bool
	EnableMMap         bool
}

// DefaultMemoryConfig returns sensible default configuration
func DefaultMemoryConfig() MemoryConfig {
	return MemoryConfig{
		MaxMemory: 0, // No limit by default
		PressureThresholds: map[MemoryPressureLevel]float64{
			LowPressure:      0.7,  // 70% of limit
			ModeratePressure: 0.8,  // 80% of limit
			HighPressure:     0.9,  // 90% of limit
			CriticalPressure: 0.95, // 95% of limit
		},
		MonitorInterval: 5 * time.Second,
		EnableGC:        true,
		GCThreshold:     0.85, // Trigger GC at 85% usage
		EnableMMap:      true,
		MMapThreshold:   100 * 1024 * 1024, // 100MB threshold for mmap activation
		MMapPath:        "/tmp/libravdb",   // Default mmap directory
	}
}
