package storage

import (
	"context"

	"github.com/xDarkicex/libravdb/internal/index"
)

// CollectionConfig is the engine-level persisted collection configuration.
type CollectionConfig struct {
	Dimension      int
	Metric         int
	IndexType      int
	M              int
	EfConstruction int
	EfSearch       int
	ML             float64
	Version        int
	RawVectorStore string
	RawStoreCap    int
}

// Engine defines the storage engine interface
type Engine interface {
	CreateCollection(name string, config interface{}) (Collection, error)
	GetCollection(name string) (Collection, error)
	ListCollections() ([]string, error)
	DeleteCollection(name string) error
	Close() error
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
	Close() error
}

// OrdinalAssigner assigns stable internal ordinals to entries before indexing.
type OrdinalAssigner interface {
	AssignOrdinals(ctx context.Context, entries []*index.VectorEntry) error
}
