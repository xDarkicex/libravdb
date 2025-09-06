package libravdb

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/obs"
	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/libravdb/internal/storage/lsm"
	"github.com/xDarkicex/libravdb/internal/util"
)

// Collection represents a named collection of vectors with a specific schema
type Collection struct {
	mu      sync.RWMutex
	name    string
	config  *CollectionConfig
	index   index.Index
	storage storage.Collection
	metrics *obs.Metrics
	closed  bool
}

// CollectionConfig holds collection-specific configuration
type CollectionConfig struct {
	Dimension int            `json:"dimension"`
	Metric    DistanceMetric `json:"metric"`
	IndexType IndexType      `json:"index_type"`
	// HNSW specific parameters
	M              int     `json:"m"`               // Max connections per node
	EfConstruction int     `json:"ef_construction"` // Size of dynamic candidate list during construction
	EfSearch       int     `json:"ef_search"`       // Size of dynamic candidate list during search
	ML             float64 `json:"ml"`              // Level generation factor
	Version        int     `json:"version"`         // Config version for future compatibility
}

// DistanceMetric defines the distance function to use
type DistanceMetric int

const (
	L2Distance DistanceMetric = iota
	InnerProduct
	CosineDistance
)

// IndexType defines the index algorithm to use
type IndexType int

const (
	HNSW IndexType = iota
	IVFPQ
)

// newCollection creates a new collection instance
func newCollection(name string, storageEngine storage.Engine, metrics *obs.Metrics, opts ...CollectionOption) (*Collection, error) {
	config := &CollectionConfig{
		Dimension:      768, // Default for common embeddings
		Metric:         CosineDistance,
		IndexType:      HNSW,
		M:              32,
		EfConstruction: 200,
		EfSearch:       50,
		ML:             1.0 / math.Log(2.0),
	}

	// Apply options
	for _, opt := range opts {
		if err := opt(config); err != nil {
			return nil, fmt.Errorf("failed to apply collection option: %w", err)
		}
	}

	// Validate configuration
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid collection config: %w", err)
	}

	// Convert to LSM config format
	lsmConfig := &lsm.CollectionConfig{
		Dimension:      config.Dimension,
		Metric:         int(config.Metric),
		IndexType:      int(config.IndexType),
		M:              config.M,
		EfConstruction: config.EfConstruction,
		EfSearch:       config.EfSearch,
		ML:             config.ML,
		Version:        1,
	}

	// Create storage for this collection - PASS THE CONFIG
	collectionStorage, err := storageEngine.CreateCollection(name, lsmConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection storage: %w", err)
	}

	// Create index
	var idx index.Index
	switch config.IndexType {
	case HNSW:
		idx, err = index.NewHNSW(&index.HNSWConfig{
			Dimension:      config.Dimension,
			M:              config.M,
			EfConstruction: config.EfConstruction,
			EfSearch:       config.EfSearch,
			ML:             config.ML,
			Metric:         util.DistanceMetric(config.Metric),
		})
	default:
		return nil, fmt.Errorf("unsupported index type: %v", config.IndexType)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create index: %w", err)
	}

	return &Collection{
		name:    name,
		config:  config,
		index:   idx,
		storage: collectionStorage,
		metrics: metrics,
	}, nil
}

// newCollectionFromStorage creates a collection instance from existing storage
func newCollectionFromStorage(name string, storageCollection storage.Collection, metrics *obs.Metrics, lsmConfig *lsm.CollectionConfig) (*Collection, error) {
	// Convert LSM config to libravdb config
	config := &CollectionConfig{
		Dimension:      lsmConfig.Dimension,
		Metric:         DistanceMetric(lsmConfig.Metric),
		IndexType:      IndexType(lsmConfig.IndexType),
		M:              lsmConfig.M,
		EfConstruction: lsmConfig.EfConstruction,
		EfSearch:       lsmConfig.EfSearch,
		ML:             lsmConfig.ML,
		Version:        lsmConfig.Version,
	}

	// Create index with stored config
	idx, err := index.NewHNSW(&index.HNSWConfig{
		Dimension:      config.Dimension,
		M:              config.M,
		EfConstruction: config.EfConstruction,
		EfSearch:       config.EfSearch,
		ML:             config.ML,
		Metric:         util.DistanceMetric(config.Metric),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create index: %w", err)
	}

	collection := &Collection{
		name:    name,
		config:  config,
		index:   idx,
		storage: storageCollection,
		metrics: metrics,
	}

	// Rebuild index from storage data
	if err := collection.rebuildIndex(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to rebuild index: %w", err)
	}

	return collection, nil
}

// rebuildIndex rebuilds the index from storage data
func (c *Collection) rebuildIndex(ctx context.Context) error {
	return c.storage.Iterate(ctx, func(entry *index.VectorEntry) error {
		return c.index.Insert(ctx, entry)
	})
}

// Insert adds or updates a vector in the collection
func (c *Collection) Insert(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrCollectionClosed
	}

	// Validate input
	if len(vector) != c.config.Dimension {
		return fmt.Errorf("vector dimension %d does not match collection dimension %d",
			len(vector), c.config.Dimension)
	}

	// Create vector entry for storage (avoiding circular imports)
	storageEntry := &index.VectorEntry{
		ID:       id,
		Vector:   vector,
		Metadata: metadata,
	}

	// Insert into index (convert to index VectorEntry if needed)
	if err := c.index.Insert(ctx, storageEntry); err != nil {
		return fmt.Errorf("failed to insert into index: %w", err)
	}

	// Write to storage (WAL)
	if err := c.storage.Insert(ctx, storageEntry); err != nil {
		// TODO: Rollback index insertion
		return fmt.Errorf("failed to write to storage: %w", err)
	}

	// Update metrics
	if c.metrics != nil {
		c.metrics.VectorInserts.Inc()
	}

	return nil
}

// Search performs a vector similarity search
func (c *Collection) Search(ctx context.Context, vector []float32, k int) (*SearchResults, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, ErrCollectionClosed
	}

	// Validate input
	if len(vector) != c.config.Dimension {
		return nil, fmt.Errorf("query vector dimension %d does not match collection dimension %d",
			len(vector), c.config.Dimension)
	}

	if k <= 0 {
		return nil, fmt.Errorf("k must be positive, got %d", k)
	}

	// Start timing
	start := time.Now()
	defer func() {
		if c.metrics != nil {
			c.metrics.SearchLatency.Observe(time.Since(start).Seconds())
		}
	}()

	// Search index
	indexResults, err := c.index.Search(ctx, vector, k)
	if err != nil {
		if c.metrics != nil {
			c.metrics.SearchErrors.Inc()
		}
		return nil, fmt.Errorf("index search failed: %w", err)
	}

	// Convert from index.SearchResult to libravdb.SearchResult
	results := make([]*SearchResult, len(indexResults))
	for i, r := range indexResults {
		results[i] = &SearchResult{
			ID:       r.ID,
			Score:    r.Score,
			Vector:   r.Vector,
			Metadata: r.Metadata,
		}
	}

	// Update metrics
	if c.metrics != nil {
		c.metrics.SearchQueries.Inc()
	}

	return &SearchResults{
		Results: results,
		Took:    time.Since(start),
		Total:   len(results),
	}, nil
}

// Query returns a new query builder for this collection
func (c *Collection) Query(ctx context.Context) *QueryBuilder {
	return &QueryBuilder{
		ctx:        ctx,
		collection: c,
		limit:      10, // default
	}
}

// Stats returns collection statistics
func (c *Collection) Stats() *CollectionStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return &CollectionStats{
		Name:        c.name,
		VectorCount: c.index.Size(),
		Dimension:   c.config.Dimension,
		IndexType:   c.config.IndexType.String(),
		MemoryUsage: c.index.MemoryUsage(),
	}
}

// Close shuts down the collection
func (c *Collection) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}

	var errors []error

	if err := c.index.Close(); err != nil {
		errors = append(errors, err)
	}

	if err := c.storage.Close(); err != nil {
		errors = append(errors, err)
	}

	c.closed = true

	if len(errors) > 0 {
		return fmt.Errorf("errors during collection shutdown: %v", errors)
	}

	return nil
}

// validate checks if the collection configuration is valid
func (config *CollectionConfig) validate() error {
	if config.Dimension <= 0 {
		return fmt.Errorf("dimension must be positive, got %d", config.Dimension)
	}

	if config.M <= 0 {
		return fmt.Errorf("M must be positive, got %d", config.M)
	}

	if config.EfConstruction <= 0 {
		return fmt.Errorf("EfConstruction must be positive, got %d", config.EfConstruction)
	}

	if config.EfSearch <= 0 {
		return fmt.Errorf("EfSearch must be positive, got %d", config.EfSearch)
	}

	return nil
}
