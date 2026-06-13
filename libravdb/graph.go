package libravdb

import (
	"github.com/xDarkicex/libravdb/internal/graph"
)

// Graph defines the public consumer API for the graph layer.
type Graph interface {
	// Transaction lifecycle.
	BeginTxn() *graph.Txn

	// Edge mutations (must be called within a transaction).
	AddEdge(txn *graph.Txn, src, tgt uint64, weight float32, kind uint8) error
	RemoveEdge(txn *graph.Txn, src, tgt uint64, kind uint8) error
	DropNodeEdges(txn *graph.Txn, nodeID uint64) error

	// Edge queries.
	Neighbors(nodeID uint64) ([]Edge, error)
	Degree(nodeID uint64) (int, error)
	InboundNeighbors(nodeID uint64) ([]Edge, error)
	InboundDegree(nodeID uint64) (int, error)
	NeighborsAny(nodeID uint64, kindSet KindSet) ([]Edge, error)
	ForEachEdge(fn func(src, tgt uint64, edge Edge) bool)

	// Traversal.
	BFS(start uint64, maxDepth int, visit graph.VisitAction, bitset *graph.Bitset, frontier *graph.FrontierBuf) error

	// Pool management (caller-managed zero-alloc BFS).
	GetBitset() (*graph.Bitset, error)
	PutBitset(b *graph.Bitset)
	GetFrontierBuf() (*graph.FrontierBuf, error)
	PutFrontierBuf(f *graph.FrontierBuf)

	// Lifecycle.
	Stats() graph.GraphStats
	Close() error
}

// Txn wraps a graph transaction. After calling BeginTxn, use AddEdge/RemoveEdge/DropNodeEdges,
// then call Commit or let the transaction be discarded.
type Txn = graph.Txn

// Edge represents a directed edge in the graph.
type Edge = graph.Edge

// KindSet represents a set of allowed edge kinds.
type KindSet = graph.KindSet

// Bitset is a reusable off-heap bitset for BFS deduplication.
type Bitset = graph.Bitset

// FrontierBuf is a reusable off-heap ring buffer for BFS queueing.
type FrontierBuf = graph.FrontierBuf

// VisitAction is invoked for each node during BFS traversal.
type VisitAction = graph.VisitAction

// GraphStats contains metrics from the graph layer.
type GraphStats = graph.GraphStats

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
