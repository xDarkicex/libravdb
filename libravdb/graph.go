package libravdb

import (
	"time"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// Graph defines the public consumer API for the graph layer.
type Graph interface {
	// Transaction lifecycle.
	BeginTxn() *graph.Txn

	// Edge mutations (must be called within a transaction).
	AddEdge(txn *graph.Txn, src, tgt uint64, weight float32, kind uint8) error
	AddEdgeWithProperties(txn *graph.Txn, src, tgt uint64, weight float32, kind uint8, properties map[string]interface{}) error
	RemoveEdge(txn *graph.Txn, src, tgt uint64, kind uint8) error
	DropNodeEdges(txn *graph.Txn, nodeID uint64) error
	SetEdgeKindDirection(kind uint8, undirected bool)
	IsEdgeKindUndirected(kind uint8) bool

	// Edge queries.
	Neighbors(nodeID uint64) ([]Edge, error)
	NeighborsWithProperties(nodeID uint64) ([]graph.EdgeView, error)
	NeighborsAtLSN(nodeID uint64, snapshotLSN uint64) ([]Edge, error)
	NeighborsAtLSNWithProperties(nodeID uint64, snapshotLSN uint64) ([]graph.EdgeView, error)
	Degree(nodeID uint64) (int, error)
	InboundNeighbors(nodeID uint64) ([]Edge, error)
	InboundNeighborsWithProperties(nodeID uint64) ([]graph.EdgeView, error)
	InboundDegree(nodeID uint64) (int, error)
	NeighborsAny(nodeID uint64, kindSet KindSet) ([]Edge, error)
	ForEachEdge(fn func(src, tgt uint64, edge Edge) bool)

	// Traversal.
	BFS(start uint64, maxDepth int, visit graph.VisitAction, bitset *graph.Bitset, frontier *graph.FrontierBuf) error
	BFSPattern(start uint64, edges []EdgePlan, maxDepth int, visit graph.VisitAction, bitset *graph.Bitset, frontier *graph.FrontierBuf) error

	// Pool management (caller-managed zero-alloc BFS).
	GetBitset() (*graph.Bitset, error)
	PutBitset(b *graph.Bitset)
	GetFrontierBuf() (*graph.FrontierBuf, error)
	PutFrontierBuf(f *graph.FrontierBuf)

	// Vertex label registry. Labels are WAL-backed when the graph is attached
	// to a collection, and replayed when that collection is reopened.
	RegisterVertexLabel(nodeID uint64, label string)
	GetLabelNodes(label string) []uint64

	// Lifecycle.
	Stats() graph.GraphStats
	GraphCentrality(nodeID uint64) float64
	CentralityAtLSN(nodeID uint64, snapshotLSN uint64) float64
	// RecordPageRankPublication publishes maintenance metadata for a derived
	// PageRank vector. It does not compute PageRank; callers should invoke this
	// after atomically publishing their own vector.
	RecordPageRankPublication(snapshotLSN uint64, duration time.Duration)
	Close() error
}

// Txn wraps a graph transaction. After calling BeginTxn, use AddEdge/RemoveEdge/DropNodeEdges,
// then call Commit or let the transaction be discarded.
type Txn = graph.Txn

// Edge represents one stored graph edge. Its traversal direction is defined by
// the registered edge kind; directed kinds are the default.
type Edge = graph.Edge

// EdgeView is an edge with its canonical, versioned JSON property envelope.
type EdgeView = graph.EdgeView

// KindSet represents a set of allowed edge kinds.
type KindSet = graph.KindSet

// Bitset is a reusable off-heap bitset for BFS deduplication.
type Bitset = graph.Bitset

// FrontierBuf is a reusable off-heap ring buffer for BFS queueing.
type FrontierBuf = graph.FrontierBuf

// EdgePlan describes a single edge band in a BFSPattern traversal.
type EdgePlan = graph.EdgePlan

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

// RegisterEdgeKind makes a named edge kind available to SQL GRAPH_EDGES,
// MATCH, JOIN MATCH, and other graph-aware query paths. Kind 0 is reserved for
// untyped edges. Registration is process-wide and idempotent for the same
// name/kind pair.
func RegisterEdgeKind(name string, kind uint8) bool {
	return graph.RegisterEdgeKind(name, kind)
}

// RegisterEdgeKindWithDirection registers a named edge kind and explicitly
// declares whether it is bidirectional. Undirected kinds still use one
// canonical stored edge; traversal exposes the reverse direction without
// duplicating physical edges or WAL records.
func RegisterEdgeKindWithDirection(name string, kind uint8, undirected bool) bool {
	return graph.RegisterEdgeKindWithDirection(name, kind, undirected)
}

// RegisterUndirectedEdgeKind registers a bidirectional edge kind.
func RegisterUndirectedEdgeKind(name string, kind uint8) bool {
	return graph.RegisterUndirectedEdgeKind(name, kind)
}

// ResolveEdgeKind returns the numeric kind for a registered edge name, or 0
// when the name is unknown.
func ResolveEdgeKind(name string) uint8 {
	return graph.ResolveEdgeKind(name)
}
