package index

import (
	"context"
)

// Index defines the interface for vector indexes
type Index interface {
	Insert(ctx context.Context, entry *VectorEntry) error
	Search(ctx context.Context, query []float32, k int) ([]*SearchResult, error)
	Delete(ctx context.Context, id string) error
	Size() int
	MemoryUsage() int64
	Close() error
}

// VectorEntry represents a vector entry (avoid circular imports)
type VectorEntry struct {
	ID       string
	Vector   []float32
	Metadata map[string]interface{}
}

// SearchResult represents a search result (avoid circular imports)
type SearchResult struct {
	ID       string
	Score    float32
	Vector   []float32
	Metadata map[string]interface{}
}
