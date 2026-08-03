package graph

// GraphConfig configures the off-heap memory budgets for the graph layer.
// Pools use segmented ShardedFreeLists ([]*memory.ShardedFreeList) that grow
// by appending new segments on exhaustion — same pattern as the B-tree page pool.
// EdgeSlots / PageSlots are the per-segment capacity; total capacity is unbounded.
type GraphConfig struct {
	EdgeSlots        int // per-segment ShardedFreeList capacity (default: 100K)
	EdgeSlotSize     int // Edge slot size (default: 80 bytes)
	EdgeShards       int // Edge shards (default: 64)
	PageSlots        int // per-segment ShardedFreeList capacity for pages (default: 1M)
	PageShards       int // ShardedFreeList shards for pages (default: 64)
	BitsetPoolSize   int // Slot pool size for bitsets (default: 8)
	FrontierPoolSize int // Slot pool size for frontiers (default: 8)
	ArenaPages       int // Arena page count (default: 256)
}

// DefaultGraphConfig returns a GraphConfig with sensible defaults.
// Each segment is modest (~8 MB edges, ~256 MB pages); the segmented
// pool grows on demand when a segment is exhausted.
func DefaultGraphConfig() GraphConfig {
	return GraphConfig{
		EdgeSlots:        100_000, // 100K edges per segment, ~8 MB
		EdgeSlotSize:     80,
		EdgeShards:       64,
		PageSlots:        1_000_000, // 1M pages per segment, ~4 GB
		PageShards:       64,
		BitsetPoolSize:   8,
		FrontierPoolSize: 8,
		ArenaPages:       256,
	}
}
