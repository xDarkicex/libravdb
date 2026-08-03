package btree

import (
	"sync"
	"unsafe"
)

// pageRegistry provides sharded, lock-free-read page lookup.
// Same 64-shard RWMutex pattern as internal/graph/registry.go.
type pageRegistry struct {
	nextID uint32
	shards [64]*registryShard
}

type registryShard struct {
	sync.RWMutex
	pages map[uint32]uintptr
}

func newPageRegistry() *pageRegistry {
	r := &pageRegistry{nextID: 1}
	for i := 0; i < 64; i++ {
		r.shards[i] = &registryShard{
			pages: make(map[uint32]uintptr),
		}
	}
	return r
}

func (r *pageRegistry) register(page *BTreePage) uint32 {
	shard := r.shards[r.nextID%64]
	shard.Lock()
	id := r.nextID
	r.nextID++
	page.Header.PageSlot = id
	shard.pages[id] = uintptr(unsafe.Pointer(page))
	shard.Unlock()
	return id
}

func (r *pageRegistry) get(id uint32) *BTreePage {
	if id == 0 {
		return nil
	}
	shard := r.shards[id%64]
	shard.RLock()
	ptr := shard.pages[id]
	shard.RUnlock()
	if ptr == 0 {
		return nil
	}
	return (*BTreePage)(unsafe.Pointer(ptr))
}

func (r *pageRegistry) unregister(id uint32) {
	if id == 0 {
		return
	}
	shard := r.shards[id%64]
	shard.Lock()
	delete(shard.pages, id)
	shard.Unlock()
}

// replace atomically replaces the page at slotID with a new page.
// Keeps the same slot ID so child pointers remain valid.
func (r *pageRegistry) replace(slotID uint32, newPage *BTreePage) {
	if slotID == 0 {
		return
	}
	shard := r.shards[slotID%64]
	shard.Lock()
	shard.pages[slotID] = uintptr(unsafe.Pointer(newPage))
	shard.Unlock()
	newPage.Header.PageSlot = slotID
}

// snapshotIDs returns all registered page IDs (for persistence).
func (r *pageRegistry) snapshotIDs() []uint32 {
	ids := make([]uint32, 0, int(r.nextID))
	for i := 0; i < 64; i++ {
		shard := r.shards[i]
		shard.RLock()
		for id := range shard.pages {
			ids = append(ids, id)
		}
		shard.RUnlock()
	}
	return ids
}
