package memory

import (
	"runtime"
	"sync"
	"time"
)

// Monitor provides utilities for memory monitoring and statistics
type Monitor struct {
	mu    sync.RWMutex
	stats []MemorySnapshot

	// Configuration
	maxSnapshots int
	interval     time.Duration
}

// MemorySnapshot represents a point-in-time memory measurement
type MemorySnapshot struct {
	Timestamp time.Time

	// Go runtime memory stats
	HeapAlloc    uint64
	HeapSys      uint64
	HeapInuse    uint64
	HeapReleased uint64

	// GC stats
	NumGC        uint32
	PauseTotalNs uint64

	// Custom tracking
	CacheUsage   int64
	IndexUsage   int64
	TotalManaged int64
}

// NewMonitor creates a new memory monitor
func NewMonitor(maxSnapshots int, interval time.Duration) *Monitor {
	return &Monitor{
		maxSnapshots: maxSnapshots,
		interval:     interval,
		stats:        make([]MemorySnapshot, 0, maxSnapshots),
	}
}

// TakeSnapshot captures current memory statistics
func (m *Monitor) TakeSnapshot() MemorySnapshot {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	snapshot := MemorySnapshot{
		Timestamp:    time.Now(),
		HeapAlloc:    memStats.HeapAlloc,
		HeapSys:      memStats.HeapSys,
		HeapInuse:    memStats.HeapInuse,
		HeapReleased: memStats.HeapReleased,
		NumGC:        memStats.NumGC,
		PauseTotalNs: memStats.PauseTotalNs,
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Add snapshot to history
	m.stats = append(m.stats, snapshot)

	// Keep only the most recent snapshots
	if len(m.stats) > m.maxSnapshots {
		m.stats = m.stats[1:]
	}

	return snapshot
}

// GetSnapshots returns all stored memory snapshots
func (m *Monitor) GetSnapshots() []MemorySnapshot {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Return a copy to avoid race conditions
	snapshots := make([]MemorySnapshot, len(m.stats))
	copy(snapshots, m.stats)
	return snapshots
}

// GetLatestSnapshot returns the most recent memory snapshot
func (m *Monitor) GetLatestSnapshot() (MemorySnapshot, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.stats) == 0 {
		return MemorySnapshot{}, false
	}

	return m.stats[len(m.stats)-1], true
}

// CalculateMemoryTrend analyzes memory usage trend over time
func (m *Monitor) CalculateMemoryTrend() MemoryTrend {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.stats) < 2 {
		return MemoryTrend{
			Direction: TrendUnknown,
			Rate:      0,
		}
	}

	// Calculate trend using first and last snapshots
	first := m.stats[0]
	last := m.stats[len(m.stats)-1]

	timeDiff := last.Timestamp.Sub(first.Timestamp).Seconds()
	if timeDiff <= 0 {
		return MemoryTrend{
			Direction: TrendStable,
			Rate:      0,
		}
	}

	memoryDiff := int64(last.HeapInuse) - int64(first.HeapInuse)
	rate := float64(memoryDiff) / timeDiff // bytes per second

	var direction TrendDirection
	if rate > 1024*1024 { // Growing more than 1MB/sec
		direction = TrendIncreasing
	} else if rate < -1024*1024 { // Decreasing more than 1MB/sec
		direction = TrendDecreasing
	} else {
		direction = TrendStable
	}

	return MemoryTrend{
		Direction: direction,
		Rate:      rate,
		Duration:  last.Timestamp.Sub(first.Timestamp),
	}
}

// MemoryTrend represents memory usage trend analysis
type MemoryTrend struct {
	Direction TrendDirection
	Rate      float64 // bytes per second
	Duration  time.Duration
}

// TrendDirection indicates the direction of memory usage change
type TrendDirection int

const (
	TrendUnknown TrendDirection = iota
	TrendIncreasing
	TrendDecreasing
	TrendStable
)

// String returns string representation of trend direction
func (d TrendDirection) String() string {
	switch d {
	case TrendIncreasing:
		return "increasing"
	case TrendDecreasing:
		return "decreasing"
	case TrendStable:
		return "stable"
	default:
		return "unknown"
	}
}

// GetMemoryPressure calculates current memory pressure based on available system memory
func GetMemoryPressure() (float64, error) {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	// This is a simplified calculation
	// In production, you might want to use system-specific APIs
	heapUsage := float64(memStats.HeapInuse)
	heapSys := float64(memStats.HeapSys)

	if heapSys == 0 {
		return 0, nil
	}

	pressure := heapUsage / heapSys
	return pressure, nil
}

// ForceGC triggers garbage collection and returns memory freed
func ForceGC() int64 {
	var before, after runtime.MemStats

	runtime.ReadMemStats(&before)
	runtime.GC()
	runtime.GC() // Run twice for better cleanup
	runtime.ReadMemStats(&after)

	// Calculate freed memory (approximate)
	freed := int64(before.HeapInuse) - int64(after.HeapInuse)
	if freed < 0 {
		freed = 0
	}

	return freed
}
