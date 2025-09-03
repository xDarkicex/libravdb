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

	return hnsw.NewHNSW(hnswConfig)
}
