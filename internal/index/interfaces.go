package index

import (
	"context"

	"github.com/xDarkicex/libravdb/internal/index/hnsw"
	"github.com/xDarkicex/libravdb/internal/util"
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

// HNSWConfig holds configuration for HNSW index
type HNSWConfig struct {
	Dimension      int
	M              int
	EfConstruction int
	EfSearch       int
	ML             float64
	Metric         util.DistanceMetric
}

// hnswWrapper wraps the HNSW index to adapt between interface types
type hnswWrapper struct {
	index *hnsw.Index
}

// Insert adapts the interface VectorEntry to HNSW VectorEntry
func (w *hnswWrapper) Insert(ctx context.Context, entry *VectorEntry) error {
	hnswEntry := &hnsw.VectorEntry{
		ID:       entry.ID,
		Vector:   entry.Vector,
		Metadata: entry.Metadata,
	}
	return w.index.Insert(ctx, hnswEntry)
}

// Search adapts the search results from HNSW to interface types
func (w *hnswWrapper) Search(ctx context.Context, query []float32, k int) ([]*SearchResult, error) {
	hnswResults, err := w.index.Search(ctx, query, k)
	if err != nil {
		return nil, err
	}

	results := make([]*SearchResult, len(hnswResults))
	for i, r := range hnswResults {
		results[i] = &SearchResult{
			ID:       r.ID,
			Score:    r.Score,
			Vector:   r.Vector,
			Metadata: r.Metadata,
		}
	}
	return results, nil
}

// Delete delegates to the wrapped index
func (w *hnswWrapper) Delete(ctx context.Context, id string) error {
	return w.index.Delete(ctx, id)
}

// Size delegates to the wrapped index
func (w *hnswWrapper) Size() int {
	return w.index.Size()
}

// MemoryUsage delegates to the wrapped index
func (w *hnswWrapper) MemoryUsage() int64 {
	return w.index.MemoryUsage()
}

// Close delegates to the wrapped index
func (w *hnswWrapper) Close() error {
	return w.index.Close()
}

// NewHNSW creates a new HNSW index
func NewHNSW(config *HNSWConfig) (Index, error) {
	// Convert to internal HNSW config
	hnswConfig := &hnsw.Config{
		Dimension:      config.Dimension,
		M:              config.M,
		EfConstruction: config.EfConstruction,
		EfSearch:       config.EfSearch,
		ML:             config.ML,
		Metric:         config.Metric,
		RandomSeed:     0, // Default seed for Phase 1
	}

	hnswIndex, err := hnsw.NewHNSW(hnswConfig)
	if err != nil {
		return nil, err
	}

	return &hnswWrapper{index: hnswIndex}, nil
}
