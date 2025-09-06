package storage

import (
	"context"

	"github.com/xDarkicex/libravdb/internal/index"
)

// Engine defines the storage engine interface
type Engine interface {
	CreateCollection(name string, config interface{}) (Collection, error)
	GetCollection(name string) (Collection, error)
	Close() error
}

// Collection defines the collection storage interface
type Collection interface {
	Insert(ctx context.Context, entry *index.VectorEntry) error
	Get(ctx context.Context, id string) (*index.VectorEntry, error)
	Delete(ctx context.Context, id string) error
	Iterate(ctx context.Context, fn func(*index.VectorEntry) error) error
	Close() error
}
