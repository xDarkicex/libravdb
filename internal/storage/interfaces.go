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
	Close() error
}

// NewLSM creates a new LSM storage engine (temporary mock for Phase 1)
func NewLSM(path string) (Engine, error) {
	return &mockEngine{}, nil
}

// Temporary implementations for Phase 1
type mockEngine struct{}

func (m *mockEngine) CreateCollection(name string, config interface{}) (Collection, error) {
	return &mockCollection{}, nil
}

func (m *mockEngine) GetCollection(name string) (Collection, error) {
	return &mockCollection{}, nil
}

func (m *mockEngine) Close() error { return nil }

type mockCollection struct{}

func (m *mockCollection) Insert(ctx context.Context, entry *index.VectorEntry) error { return nil }
func (m *mockCollection) Get(ctx context.Context, id string) (*index.VectorEntry, error) {
	return nil, nil
}
func (m *mockCollection) Delete(ctx context.Context, id string) error { return nil }
func (m *mockCollection) Close() error                                { return nil }
