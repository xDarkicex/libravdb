package graph

import "github.com/xDarkicex/memory"

// ReverseIndex maintains inbound edges symmetrically to the forward index.
// It is used to quickly locate all inbound edges for a node when performing DropNodeEdges.
type ReverseIndex struct {
	locator *EdgeTableIndex
	pool    *memory.ShardedFreeList
}

func newReverseIndex(cfg GraphConfig) (*ReverseIndex, error) {
	pool, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  uint64(cfg.PageSlots * 4096),
		SlotSize:  4096,
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 32,
		Prealloc:  false,
	}, cfg.PageShards)
	if err != nil {
		return nil, err
	}

	return &ReverseIndex{
		locator: NewEdgeTableIndex(1024),
		pool:    pool,
	}, nil
}

func (r *ReverseIndex) Close() error {
	return r.pool.Free()
}
