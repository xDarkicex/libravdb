package libravdb

import (
	"github.com/xDarkicex/libravdb/internal/graph"
)

// Graph defines the public consumer API for the graph layer.
type Graph interface {
	Stats() graph.GraphStats
	InboundNeighbors(nodeID uint64) ([]Edge, error)
	InboundDegree(nodeID uint64) (int, error)
	ForEachEdge(fn func(src, tgt uint64, edge Edge) bool)
	Close() error
}

// Edge represents a directed edge in the graph.
type Edge = graph.Edge

// KindSet represents a set of allowed edge kinds.
type KindSet = graph.KindSet

// GraphConfig represents configuration for the graph layer.
type GraphConfig struct {
	EdgeSlots        int
	EdgeSlotSize     int
	EdgeShards       int
	PageSlots        int
	PageShards       int
	BitsetPoolSize   int
	FrontierPoolSize int
	ArenaPages       int
}

// GraphFilter is an interface used to filter search candidates based on a graph bitset.
type GraphFilter interface {
	Test(idx uint64) bool
}

// NewGraph creates a new Graph instance.
func NewGraph(config GraphConfig) (Graph, error) {
	// Use defaults for zero values
	internalConfig := graph.DefaultGraphConfig()
	if config.EdgeSlots > 0 {
		internalConfig.EdgeSlots = config.EdgeSlots
	}
	if config.EdgeSlotSize > 0 {
		internalConfig.EdgeSlotSize = config.EdgeSlotSize
	}
	if config.EdgeShards > 0 {
		internalConfig.EdgeShards = config.EdgeShards
	}
	if config.PageSlots > 0 {
		internalConfig.PageSlots = config.PageSlots
	}
	if config.PageShards > 0 {
		internalConfig.PageShards = config.PageShards
	}
	if config.BitsetPoolSize > 0 {
		internalConfig.BitsetPoolSize = config.BitsetPoolSize
	}
	if config.FrontierPoolSize > 0 {
		internalConfig.FrontierPoolSize = config.FrontierPoolSize
	}
	if config.ArenaPages > 0 {
		internalConfig.ArenaPages = config.ArenaPages
	}
	
	return graph.NewGraph(internalConfig)
}
