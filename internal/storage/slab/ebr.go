package slab

import (
	"math"
	"math/bits"
	"sync"
	"sync/atomic"
	"time"
)

// RegistryShards defines the striped array size.
// 64 is typically sufficient to avoid false sharing across standard core counts.
const RegistryShards = 64

// EpochCell is a cache-line padded struct (64 bytes).
// It isolates each reader's snapshot epoch, eliminating cache coherence traffic
// between unrelated concurrent readers.
type EpochCell struct {
	Epoch   uint64
	Padding [56]byte // 64 - 8
}

// EpochRegistry implements the Sharded Epoch Registry for Go.
type EpochRegistry struct {
	cells    [RegistryShards]EpochCell
	freelist uint64 // Atomic bitmask for 64 cells: 1 = in-use, 0 = free
}

// NewEpochRegistry initializes a registry with all cells marked inactive.
func NewEpochRegistry() *EpochRegistry {
	r := &EpochRegistry{}
	for i := range r.cells {
		r.cells[i].Epoch = math.MaxUint64
	}
	return r
}

// Acquire registers a reader's snapshot and returns an exclusively claimed slot ID (0-63).
// It uses a lock-free bitmask (freelist) to ensure no collisions between goroutines.
func (r *EpochRegistry) Acquire(snapshotXid uint64) uint32 {
	for {
		free := atomic.LoadUint64(&r.freelist)
		if free == math.MaxUint64 {
			// Extremely rare: all 64 slots are taken. Fallback or yield.
			time.Sleep(1 * time.Microsecond)
			continue
		}
		
		// Find the lowest 0-bit
		slot := uint32(bits.TrailingZeros64(^free))
		mask := uint64(1) << slot
		
		if atomic.CompareAndSwapUint64(&r.freelist, free, free|mask) {
			atomic.StoreUint64(&r.cells[slot].Epoch, snapshotXid)
			return slot
		}
	}
}

// Release unregisters a reader when its transaction completes and frees the slot.
func (r *EpochRegistry) Release(slot uint32) {
	atomic.StoreUint64(&r.cells[slot].Epoch, math.MaxUint64) // math.MaxUint64 = Inactive
	
	// Free the slot in the bitmask
	mask := ^(uint64(1) << slot)
	for {
		curr := atomic.LoadUint64(&r.freelist)
		if atomic.CompareAndSwapUint64(&r.freelist, curr, curr&mask) {
			break
		}
	}
}

// MinimumActiveEpoch scans the registry to find the oldest active reader.
// It returns currentXid if no readers are active.
func (r *EpochRegistry) MinimumActiveEpoch(currentXid uint64) uint64 {
	minEpoch := currentXid
	for i := 0; i < RegistryShards; i++ {
		e := atomic.LoadUint64(&r.cells[i].Epoch)
		if e < minEpoch {
			minEpoch = e
		}
	}
	return minEpoch
}

// LimboEntry represents a discarded slab waiting for reclamation.
type LimboEntry struct {
	DeletionXid uint64
	Offset      uint64
	Size        uint64
}

// LimboManager manages the lock-free reclamation lifecycle.
type LimboManager struct {
	registry   *EpochRegistry
	budget     uint64 // Max allowed bytes in limbo before stalling writers
	
	mu         sync.Mutex
	entries    []LimboEntry
	totalBytes uint64
}

func NewLimboManager(registry *EpochRegistry, maxBudgetBytes uint64) *LimboManager {
	return &LimboManager{
		registry: registry,
		budget:   maxBudgetBytes,
		entries:  make([]LimboEntry, 0, 1024),
	}
}

// Retire marks a slab for deferred reclamation.
// This is the writer's safety valve. If the limbo budget is exhausted, 
// the writer will gently spin/yield until the background reclaimer catches up,
// preventing unbounded memory growth.
func (m *LimboManager) Retire(deletionXid uint64, offset uint64, size uint64) {
	// Writer Backpressure: Soft stall if we exceed the budget.
	// This forces writers to pay the cost of reclamation delay, rather than aborting long reads.
	for {
		m.mu.Lock()
		if m.totalBytes+size <= m.budget {
			m.entries = append(m.entries, LimboEntry{
				DeletionXid: deletionXid,
				Offset:      offset,
				Size:        size,
			})
			m.totalBytes += size
			m.mu.Unlock()
			return
		}
		m.mu.Unlock()
		
		// Gently yield back to the Go scheduler.
		// The CAS-retry loop already makes writers resilient to delays.
		time.Sleep(1 * time.Millisecond)
	}
}

// Reclaim is called periodically by a background worker to drain the limbo list.
func (m *LimboManager) Reclaim(currentXid uint64) []LimboEntry {
	watermark := m.registry.MinimumActiveEpoch(currentXid)
	
	m.mu.Lock()
	defer m.mu.Unlock()
	
	var reclaimed []LimboEntry
	var remaining []LimboEntry
	
	for _, entry := range m.entries {
		if entry.DeletionXid < watermark {
			// The slab was deleted before the oldest active reader started.
			// It is mathematically invisible to all current and future readers.
			reclaimed = append(reclaimed, entry)
			m.totalBytes -= entry.Size
		} else {
			remaining = append(remaining, entry)
		}
	}
	
	m.entries = remaining
	return reclaimed
}
