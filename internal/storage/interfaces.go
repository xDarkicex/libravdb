package storage

import (
	"context"

	"github.com/xDarkicex/libravdb/internal/index"
)

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

// WriteStatsProvider is an optional interface for engines that expose write-path counters.
type WriteStatsProvider interface {
	WriteStats() WriteStats
}

// TxOperationType describes a transactional row mutation.
type TxOperationType uint8

const (
	TxOperationPut TxOperationType = iota
	TxOperationDelete
)

// TxOperation represents one row-level mutation in a transactional batch.
type TxOperation struct {
	Metadata           map[string]interface{}
	Collection         string
	ID                 string
	Vector             []float32
	ExpectedVersion    uint64
	Ordinal            uint32
	Type               TxOperationType
	HasExpectedVersion bool
}

// TransactionalEngine extends Engine with atomic multi-collection commit support.
type TransactionalEngine interface {
	PrepareTx(ctx context.Context, ops []TxOperation) ([]TxOperation, error)
	CommitTx(ctx context.Context, ops []TxOperation) error
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

// OrdinalAssigner assigns stable internal ordinals to entries before indexing.
type OrdinalAssigner interface {
	AssignOrdinals(ctx context.Context, entries []*index.VectorEntry) error
}
