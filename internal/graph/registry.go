package graph

import (
	"sync"
	"sync/atomic"
	"unsafe"
)

// PageRegistry provides a lock-free read map from uint32 slot index to *EdgeTablePage.
// It is used because EdgeTable structures use 32-bit uint32 references to pages
// rather than 64-bit pointers to save space.
type PageRegistry struct {
	nextID atomic.Uint32

	// Sharded maps for fast concurrent insertions/removals
	shards [64]*registryShard
}

type registryShard struct {
	sync.RWMutex
	pages map[uint32]uintptr
}

func NewPageRegistry() *PageRegistry {
	r := &PageRegistry{}
	// Start at 1 so 0 is a clear null/empty value
	r.nextID.Store(1)

	for i := 0; i < 64; i++ {
		r.shards[i] = &registryShard{
			pages: make(map[uint32]uintptr),
		}
	}
	return r
}

func (r *PageRegistry) Register(page *EdgeTablePage) uint32 {
	id := r.nextID.Add(1)
	shardIdx := id % 64

	shard := r.shards[shardIdx]
	shard.Lock()
	shard.pages[id] = uintptr(unsafe.Pointer(page))
	shard.Unlock()

	return id
}

//go:nocheckptr
func (r *PageRegistry) Get(id uint32) *EdgeTablePage {
	if id == 0 {
		return nil
	}

	shardIdx := id % 64
	shard := r.shards[shardIdx]

	shard.RLock()
	ptr := shard.pages[id]
	shard.RUnlock()

	if ptr == 0 {
		return nil
	}

	return (*EdgeTablePage)(unsafe.Pointer(ptr))
}

func (r *PageRegistry) Unregister(id uint32) {
	if id == 0 {
		return
	}

	shardIdx := id % 64
	shard := r.shards[shardIdx]

	shard.Lock()
	delete(shard.pages, id)
	shard.Unlock()
}

// PropertyPageRegistry is the page-chain registry for node-owned edge
// property bytes. It is separate from PageRegistry because the same allocator
// slot can contain either an EdgeTablePage or an EdgePropertyPage.
type PropertyPageRegistry struct {
	nextID atomic.Uint32
	shards [64]*propertyRegistryShard
}

type propertyRegistryShard struct {
	sync.RWMutex
	pages map[uint32]uintptr
}

func NewPropertyPageRegistry() *PropertyPageRegistry {
	r := &PropertyPageRegistry{}
	r.nextID.Store(1)
	for i := 0; i < 64; i++ {
		r.shards[i] = &propertyRegistryShard{pages: make(map[uint32]uintptr)}
	}
	return r
}

func (r *PropertyPageRegistry) Register(page *EdgePropertyPage) uint32 {
	id := r.nextID.Add(1)
	shard := r.shards[id%64]
	shard.Lock()
	shard.pages[id] = uintptr(unsafe.Pointer(page))
	shard.Unlock()
	return id
}

//go:nocheckptr
func (r *PropertyPageRegistry) Get(id uint32) *EdgePropertyPage {
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
	return (*EdgePropertyPage)(unsafe.Pointer(ptr))
}

func (r *PropertyPageRegistry) Unregister(id uint32) {
	if id == 0 {
		return
	}
	shard := r.shards[id%64]
	shard.Lock()
	delete(shard.pages, id)
	shard.Unlock()
}
