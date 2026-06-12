package graph

// GraphConfig configures the off-heap memory budgets for the graph layer.
type GraphConfig struct {
	EdgeSlots        int // ShardedFreeList capacity (default: 333K for 1M edges)
	EdgeSlotSize     int // Edge slot size (default: 80 bytes)
	EdgeShards       int // Edge shards (default: 64)
	PageSlots        int // ShardedFreeList capacity for pages (default: 128K)
	PageShards       int // ShardedFreeList shards for pages (default: 64)
	BitsetPoolSize   int // Slot pool size for bitsets (default: 8)
	FrontierPoolSize int // Slot pool size for frontiers (default: 8)
	ArenaPages       int // Arena page count (default: 256)
}

// DefaultGraphConfig returns a GraphConfig with sensible defaults
// targeting approximately 576MB of off-heap memory.
func DefaultGraphConfig() GraphConfig {
	return GraphConfig{
		EdgeSlots:        333333,
		EdgeSlotSize:     80,
		EdgeShards:       64,
		PageSlots:        131072, // 128K
		PageShards:       64,
		BitsetPoolSize:   8,
		FrontierPoolSize: 8,
		ArenaPages:       256,
	}
}
