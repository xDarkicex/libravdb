package singlefile

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/memory"
)

// ---------------------------------------------------------------------------
// Off-heap reverse directory: GraphNodeID → (collection index, ordinal)
//
// The reverse directory uses a lock-free HashMap indexing GraphNodeID to an
// off-heap pointer of reverseEntry. This bounds the directory to live
// cardinality rather than the maximum assigned ID, preventing massive memory
// consumption when IDs are sparse or mostly deleted.
//
// Go-heap cost: O(1) — a single HashMap struct with atomic pointers.
// Off-heap cost: O(V) — 16 bytes per entry + HashMap overhead.
// ---------------------------------------------------------------------------

// reverseEntry is the off-heap fixed-width struct.
type reverseEntry struct {
	collectionID uint64 // ID of the collection
	ordinal      uint32 // local ordinal within that collection
	tombstone    bool   // true if the record has been deleted
	_            [3]byte
}

// reverseDirectory provides off-heap GraphNodeID → physical location resolution.
// All mutating methods (put, tombstone, reserve, commit, close) acquire the
// exclusive lock. Reads (get) acquire the shared lock.
type reverseDirectory struct {
	mu   sync.RWMutex
	pool *memory.Pool
	hash *memory.HashMap
}

// newReverseDirectory creates an empty reverse directory.
func newReverseDirectory(poolSize int) (*reverseDirectory, error) {
	if poolSize <= 0 {
		poolSize = 64 * 1024 * 1024
	}
	pool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize: uint64(poolSize),
	}, 8)
	if err != nil {
		return nil, fmt.Errorf("reverseDirectory: pool alloc failed: %w", err)
	}

	hash, err := memory.NewHashMap(memory.HashMapConfig{
		Capacity: 64,
	})
	if err != nil {
		pool.Free()
		return nil, fmt.Errorf("reverseDirectory: hashmap alloc failed: %w", err)
	}

	return &reverseDirectory{
		pool: pool,
		hash: hash,
	}, nil
}

// put stores a reverse mapping. Acquires exclusive lock because it may mutate
// off-heap entries and the HashMap.
func (rd *reverseDirectory) put(id uint64, entry reverseEntry) error {
	if id == 0 {
		return nil // Cannot map zero ID
	}

	rd.mu.Lock()
	defer rd.mu.Unlock()

	if rd.hash == nil {
		return fmt.Errorf("reverseDirectory: closed")
	}

	existingPtr, ok := rd.hash.Get(id)
	if ok && existingPtr != nil {
		// Update existing off-heap entry in place
		existing := (*reverseEntry)(existingPtr)
		existing.collectionID = entry.collectionID
		existing.ordinal = entry.ordinal
		existing.tombstone = entry.tombstone
		return nil
	}

	// Allocate new entry (reverseEntry is 16 bytes)
	b, err := rd.pool.Allocate(16)
	if err != nil {
		return storage.ErrMemoryLimitExceeded
	}
	ptr := (*reverseEntry)(unsafe.Pointer(&b[0]))
	*ptr = entry

	// Insert into hash map
	rd.hash.Put(id, unsafe.Pointer(ptr))
	return nil
}

// get returns the reverse entry for a GraphNodeID, and whether it was found and not tombstoned.
func (rd *reverseDirectory) get(id uint64) (reverseEntry, bool) {
	if id == 0 {
		return reverseEntry{}, false
	}
	rd.mu.RLock()
	defer rd.mu.RUnlock()
	if rd.hash == nil {
		return reverseEntry{}, false
	}
	ptr, ok := rd.hash.Get(id)
	if !ok || ptr == nil {
		return reverseEntry{}, false
	}
	entry := *(*reverseEntry)(ptr)
	if entry.collectionID == 0 && entry.ordinal == 0 && !entry.tombstone {
		return reverseEntry{}, false
	}
	return entry, true
}

// tombstone marks a GraphNodeID as deleted. If no entry exists yet
// (e.g. during WAL replay where the live entry was never published),
// a tombstone-only entry is created. Acquires exclusive lock.
func (rd *reverseDirectory) tombstone(id uint64) error {
	if id == 0 {
		return fmt.Errorf("reverseDirectory: cannot tombstone zero ID")
	}
	rd.mu.Lock()
	defer rd.mu.Unlock()
	if rd.hash == nil {
		return fmt.Errorf("reverseDirectory: closed")
	}
	ptr, ok := rd.hash.Get(id)
	if ok && ptr != nil {
		entry := (*reverseEntry)(ptr)
		entry.tombstone = true
		return nil
	}
	// Create a tombstone-only entry – during WAL replay the
	// live mapping may not yet exist.
	b, err := rd.pool.Allocate(16)
	if err != nil {
		return storage.ErrMemoryLimitExceeded
	}
	e := (*reverseEntry)(unsafe.Pointer(&b[0]))
	*e = reverseEntry{tombstone: true}
	rd.hash.Put(id, unsafe.Pointer(e))
	return nil
}

// reserve pre-allocates n off-heap entries WITHOUT inserting them into the
// HashMap. It is the pre-admission gate: if the pool cannot allocate, the
// caller MUST NOT proceed with WAL append. Returns pointers that must be
// handed to commit (after durability) or silently dropped (on abort — pool
// freed at Close). Caller must hold the directory lock if concurrent with
// close, or ensure single-writer access during the reserve→commit window.
func (rd *reverseDirectory) reserve(n int) ([]unsafe.Pointer, error) {
	if n == 0 {
		return nil, nil
	}
	if n < 0 {
		return nil, fmt.Errorf("reverseDirectory: negative reserve count %d", n)
	}
	rd.mu.Lock()
	defer rd.mu.Unlock()
	if rd.hash == nil {
		return nil, fmt.Errorf("reverseDirectory: closed")
	}
	ptrs := make([]unsafe.Pointer, n)
	for i := 0; i < n; i++ {
		b, err := rd.pool.Allocate(16)
		if err != nil {
			// Leak the already-allocated entries; they'll be freed
			// when the pool is Freed at Close.
			return nil, storage.ErrMemoryLimitExceeded
		}
		ptrs[i] = unsafe.Pointer(&b[0])
	}
	return ptrs, nil
}

// commitEntry publishes a previously reserved entry into the HashMap. Must be
// called with a pointer returned by reserve. id must be nonzero. The caller
// must ensure that between reserve and commit, no other goroutine committed
// the same id.
func (rd *reverseDirectory) commitEntry(id uint64, ptr unsafe.Pointer, entry reverseEntry) {
	rd.mu.Lock()
	defer rd.mu.Unlock()
	if rd.hash == nil {
		return
	}
	existingPtr, ok := rd.hash.Get(id)
	if ok && existingPtr != nil {
		// Another entry already exists for this ID; update in place
		// and the reserved pointer is abandoned (pool will free on close).
		existing := (*reverseEntry)(existingPtr)
		existing.collectionID = entry.collectionID
		existing.ordinal = entry.ordinal
		existing.tombstone = entry.tombstone
		return
	}
	e := (*reverseEntry)(ptr)
	*e = entry
	rd.hash.Put(id, ptr)
}

// close releases all off-heap memory.
func (rd *reverseDirectory) close() {
	rd.mu.Lock()
	defer rd.mu.Unlock()
	if rd.hash != nil {
		rd.hash.Free()
		rd.hash = nil
	}
	if rd.pool != nil {
		rd.pool.Free()
		rd.pool = nil
	}
}
