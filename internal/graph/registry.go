package graph

import (
	"sync"
	"sync/atomic"
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
	pages map[uint32]*EdgeTablePage
}

func NewPageRegistry() *PageRegistry {
	r := &PageRegistry{}
	// Start at 1 so 0 is a clear null/empty value
	r.nextID.Store(1)

	for i := 0; i < 64; i++ {
		r.shards[i] = &registryShard{
			pages: make(map[uint32]*EdgeTablePage),
		}
	}
	return r
}

func (r *PageRegistry) Register(page *EdgeTablePage) uint32 {
	id := r.nextID.Add(1)
	shardIdx := id % 64

	shard := r.shards[shardIdx]
	shard.Lock()
	shard.pages[id] = page
	shard.Unlock()

	return id
}

func (r *PageRegistry) Get(id uint32) *EdgeTablePage {
	if id == 0 {
		return nil
	}

	shardIdx := id % 64
	shard := r.shards[shardIdx]

	shard.RLock()
	ptr := shard.pages[id]
	shard.RUnlock()

	if ptr == nil {
		return nil
	}

	return ptr
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
