package storage

import (
	"context"

	"errors"
	"time"

	"github.com/xDarkicex/libravdb/internal/index"
)

var ErrV1FormatMigrationRequired = errors.New("v1 format migration required")

// CollectionConfig is the engine-level persisted collection configuration.
type CollectionConfig struct {
	RawVectorStore string
	Dimension      int
	Metric         int
	IndexType      int
	M              int
	EfConstruction int
	EfSearch       int
	NClusters      int
	NProbes        int
	ML             float64
	Version        int
	RawStoreCap    int
	IDMapCapacity  int
	// CostModelStats is an opaque, versioned collection-statistics payload.
	// The storage engine persists it but intentionally does not interpret it.
	// DataLSN is populated on reads only and is never serialized as config.
	CostModelStats []byte
	// MetadataSchema contains application metadata field type codes. The
	// storage layer treats the codes as opaque; the owning package interprets
	// them when it rebuilds the public collection configuration.
	MetadataSchema map[string]uint8
	// IndexedFields contains the metadata fields whose derived posting lists
	// should be used for equality lookups.
	IndexedFields []string
	// SQLIndexes contains durable ordinary SQL index declarations. The
	// physical posting lists remain derived from records; these declarations
	// are the source of truth for DDL replay and DROP INDEX.
	SQLIndexes []SQLIndexDefinition
	// SQLIndexedFields records which IndexedFields were introduced by SQL DDL,
	// so DROP INDEX can remove only its own declarations and preserve fields
	// configured through the native API.
	SQLIndexedFields []string
	// GraphEnabled persists that the collection owns a graph layer. The graph
	// object itself is runtime state and is recreated by libravdb on reopen.
	GraphEnabled bool
	DataLSN      uint64
}

// SQLIndexDefinition is the storage-neutral form of a named SQL index.
type SQLIndexDefinition struct {
	Name    string
	Columns []string
	Unique  bool
}

// EdgeKindStore is the optional database-level durable registry used by the
// SQL CREATE EDGE TYPE surface. It is separate from Engine so alternate
// storage implementations can opt in without breaking the core interface.
type EdgeKindStore interface {
	ListEdgeKinds() (map[string]uint8, error)
	CreateEdgeKind(name string, kind uint8) error
}

// EdgeKindDefinition is the durable SQL graph edge-kind contract. The
// numeric kind remains the compact value stored in every edge; Undirected is
// metadata about how that kind is traversed and does not duplicate physical
// edges or WAL records.
type EdgeKindDefinition struct {
	Kind       uint8
	Undirected bool
}

// EdgeKindDefinitionStore is the direction-aware extension of EdgeKindStore.
// EdgeKindStore remains available for older/custom storage engines, whose
// definitions are interpreted as directed.
type EdgeKindDefinitionStore interface {
	ListEdgeKindDefinitions() (map[string]EdgeKindDefinition, error)
	CreateEdgeKindDefinition(name string, kind uint8, undirected bool) error
}

// CostModelStatisticsStore is an optional persistence seam for optimizer
// statistics.  Keeping this separate from Engine avoids forcing alternate
// storage backends to implement the feature before they can serve queries.
type CostModelStatisticsStore interface {
	SetCollectionCostModelStatsIfDataLSN(ctx context.Context, name string, expectedDataLSN uint64, stats []byte) (bool, error)
	CollectionDataLSN(name string) (uint64, error)
}

// CollectionConfigStore provides an atomic WAL-backed update for durable
// logical collection declarations such as metadata/index fields. It is
// optional so alternate engines can continue implementing the base Engine
// interface without adopting SQL DDL immediately.
type CollectionConfigStore interface {
	UpdateCollectionConfig(ctx context.Context, name string, config *CollectionConfig) error
}

// Engine defines the storage engine interface
type Engine interface {
	CreateCollection(name string, config interface{}) (Collection, error)
	GetCollection(name string) (Collection, error)
	ListCollections() ([]string, error)
	DeleteCollection(name string) error
	Close() error
	Vacuum(ctx context.Context) error
	Backup(ctx context.Context, destPath string) error
	Drop(ctx context.Context) error

	// Graph Identity API
	GetNodeID(ctx context.Context, collection, id string) (uint64, error)
	ResolveNodeID(ctx context.Context, graphNodeID uint64) (string, string, error)

	// Graph recovery wiring — called before WAL replay so committed graph edge
	// records can be routed back into the in-memory graph.
	SetGraphRecoveryTarget(collection string, target GraphRecoveryTarget)
}

// WriteStats captures coarse write-path instrumentation for benchmarking.
type WriteStats struct {
	WALTransactions       uint64
	WALBytes              uint64
	BatchFlushes          uint64
	BufferedVectorEntries uint64
	Checkpoints           uint64
}

// EngineStatus represents the engine's recovery and operational lifecycle.
type EngineStatus int32

const (
	StatusStarting           EngineStatus = iota // New() called, not yet opened
	StatusRecoveringSnapshot                     // loading snapshot from disk
	StatusRecoveringIndexes                      // loading or rebuilding indexes
	StatusReplayingWAL                           // replaying WAL from last checkpoint LSN
	StatusReady                                  // fully operational, queries accepted
	StatusFailed                                 // fatal recovery error, engine unusable
)

var (
	ErrMemoryLimitExceeded   = errors.New("memory limit exceeded")
	ErrUnknownGraphNodeID    = errors.New("unknown graph node ID")
	ErrTombstonedGraphNodeID = errors.New("tombstoned graph node ID")
)

// WriteStatsProvider is an optional interface for engines that expose write-path counters.
type WriteStatsProvider interface {
	WriteStats() WriteStats
}

// TxOperationType describes a transactional row mutation.
type TxOperationType uint8

const (
	TxOperationPut             TxOperationType = iota // record insert/update
	TxOperationDelete                                 // record delete
	TxOperationGraphEdgeAdd                           // graph edge add
	TxOperationGraphEdgeRemove                        // graph edge remove
	TxOperationGraphNodeDrop                          // graph node drop (all edges)
)

// TxOperation represents one row-level or graph mutation in a transactional batch.
type TxOperation struct {
	Metadata           map[string]interface{}
	Collection         string
	ID                 string
	Vector             []float32
	ExpectedVersion    uint64
	Ordinal            uint32
	GraphNodeID        uint64
	Type               TxOperationType
	HasExpectedVersion bool

	// Graph edge fields (used when Type is TxOperationGraphEdge*).
	EdgeSrc    uint64
	EdgeTgt    uint64
	EdgeWeight float32
	EdgeKind   uint8
	// EdgeProperties is the versioned JSON property envelope attached to the
	// node-owned edge record. Empty means no arbitrary properties.
	EdgeProperties []byte
}

// TransactionalEngine extends Engine with atomic multi-collection commit support.
type TransactionalEngine interface {
	PrepareTx(ctx context.Context, ops []TxOperation) ([]TxOperation, error)
	CommitTx(ctx context.Context, ops []TxOperation) error
	// ReserveGraphNodeIDs reserves n sequential graph node IDs and returns
	// the first ID. Used by epoch transactions that need to pre-assign
	// node IDs for remapping provisional graph edge references before
	// the combined record+graph WAL commit.
	ReserveGraphNodeIDs(ctx context.Context, n int) (uint64, error)
}

// CommitReceipt identifies the exact durable transaction boundary produced by
// a successful WAL commit. It is intentionally separate from
// TransactionalEngine so storage implementations that only support the
// legacy error-only CommitTx method remain source-compatible.
type CommitReceipt struct {
	CommitLSN uint64
}

// DurableTransactionalEngine is the optional exact-receipt extension to
// TransactionalEngine. LatestCommitLSN reads the persisted commit catalog
// rather than the next allocated WAL sequence number.
type DurableTransactionalEngine interface {
	TransactionalEngine
	CommitTxDurable(ctx context.Context, ops []TxOperation) (CommitReceipt, error)
	LatestCommitLSN() (uint64, error)
}

// Collection defines the collection storage interface
type Collection interface {
	AssignOrdinals(ctx context.Context, entries []*index.VectorEntry) error
	Insert(ctx context.Context, entry *index.VectorEntry) error
	InsertBatch(ctx context.Context, entries []*index.VectorEntry) error
	Exists(ctx context.Context, id string) (bool, error)
	Get(ctx context.Context, id string) (*index.VectorEntry, error)
	GetIDByOrdinal(ctx context.Context, ordinal uint32) (string, error)
	MemoryUsage(ctx context.Context) (int64, error)
	Delete(ctx context.Context, id string) error
	Iterate(ctx context.Context, fn func(*index.VectorEntry) error) error
	Count(ctx context.Context) (int, error)
	NextOrdinal(ctx context.Context) (uint32, error)
	Close() error
}

// DurableCollection exposes the WAL commit LSN associated with a successful
// storage mutation. Callers use this to track derived-index lag without making
// the base Collection interface storage-engine-specific.
type DurableCollection interface {
	Collection
	InsertDurable(ctx context.Context, entry *index.VectorEntry) (uint64, error)
	InsertBatchDurable(ctx context.Context, entries []*index.VectorEntry) (uint64, error)
}

// DurableRange identifies the operation and commit boundaries of one durable
// WAL transaction. Derived indexes use FirstLSN-1 as the safe frontier while
// any operation from the transaction remains unapplied.
type DurableRange struct {
	FirstLSN  uint64
	CommitLSN uint64
}

// DurableRangeCollection exposes precise transaction boundaries for bounded
// asynchronous derived-index tracking.
type DurableRangeCollection interface {
	DurableCollection
	InsertDurableRange(ctx context.Context, entry *index.VectorEntry) (DurableRange, error)
	InsertBatchDurableRange(ctx context.Context, entries []*index.VectorEntry) (DurableRange, error)
}

// OrdinalAssigner assigns stable internal ordinals to entries before indexing.
type OrdinalAssigner interface {
	AssignOrdinals(ctx context.Context, entries []*index.VectorEntry) error
}

// GraphEdgeOp is a single edge mutation queued for WAL recording.
type GraphEdgeOp struct {
	Collection string
	Src        uint64
	Tgt        uint64
	Weight     float32
	Kind       uint8
	Properties []byte
}

// GraphNodeDropOp is a collection-aware node-drop mutation. It carries the
// owning collection so that WAL frames, deferred recovery, and live graph
// publication can route the operation to the correct collection's graph.
type GraphNodeDropOp struct {
	Collection string
	NodeID     uint64
}

// GraphWALWriter is implemented by the storage engine to durably record graph
// edge mutations. Graph ops submitted through this interface share a commit
// LSN with any concurrent record writes in the same batch flush. The onCommit
// callback is invoked after WAL sync with the shared commit LSN.
type GraphWALWriter interface {
	AppendGraphEdges(ctx context.Context, adds, removes []GraphEdgeOp, nodeDrops []GraphNodeDropOp, onCommit func(lsn uint64)) (commitLSN uint64, err error)
}

// GraphLabelWALWriter durably records a vertex-label assignment.
type GraphLabelWALWriter interface {
	AppendGraphLabel(ctx context.Context, nodeID uint64, label string, onCommit func(lsn uint64)) error
}

// GraphRecoveryTarget is implemented by the graph store to replay edge
// operations during WAL recovery. These methods mutate the in-memory edge
// table directly — the WAL frames are already committed.
type GraphRecoveryTarget interface {
	ReplayEdgeAdd(src, tgt uint64, weight float32, kind uint8, properties []byte, commitLSN uint64) error
	ReplayEdgeRemove(src, tgt uint64, kind uint8, commitLSN uint64) error
	ReplayNodeEdgeDrop(nodeID uint64, commitLSN uint64) error
	ReplayVertexLabel(nodeID uint64, label string, commitLSN uint64) error
}

// TemporalRecord is a resolved historical record returned by temporal read
// APIs. It bridges the storage engine's MVCC layer to the public libravdb API.
type TemporalRecord struct {
	Metadata map[string]interface{}
	ID       string
	Vector   []float32
	Ordinal  uint32
	Version  uint64
}

// TemporalVersion is one retained record version and its validity interval.
// BeginTime is inclusive; EndTime is exclusive and zero for a still-live
// version. The interval is derived from the durable commit catalog, not wall
// clock observation at query time.
type TemporalVersion struct {
	Metadata  map[string]interface{}
	ID        string
	Vector    []float32
	Ordinal   uint32
	Version   uint64
	BeginLSN  uint64
	EndLSN    uint64
	BeginTime time.Time
	EndTime   time.Time
}

// TemporalReader is optionally implemented by storage engines that support
// MVCC record visibility at historical snapshot LSNs.
type TemporalReader interface {
	GetRecordAtLSN(collectionName, id string, snapshotLSN uint64) (*TemporalRecord, error)
	ListVisibleAtLSN(collectionName string, snapshotLSN uint64, fn func(*TemporalRecord) bool) error
}

// TemporalRangeReader enumerates retained versions whose validity intervals
// overlap an inclusive LSN range. It is optional so alternate storage engines
// can continue implementing point-in-time reads independently.
type TemporalRangeReader interface {
	ListVersionsBetween(collectionName string, startLSN, endLSN uint64, fn func(*TemporalVersion) bool) error
}
