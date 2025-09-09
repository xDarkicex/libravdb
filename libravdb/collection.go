package libravdb

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/memory"
	"github.com/xDarkicex/libravdb/internal/obs"
	"github.com/xDarkicex/libravdb/internal/quant"
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
	// Persistence configuration
	AutoSave     bool          `json:"auto_save"`     // Enable automatic index saving
	SaveInterval time.Duration `json:"save_interval"` // Interval between automatic saves
	SavePath     string        `json:"save_path"`     // Path for automatic saves
	// Quantization configuration (optional)
	Quantization *quant.QuantizationConfig `json:"quantization,omitempty"`
	// Automatic index selection based on collection size
	AutoIndexSelection bool `json:"auto_index_selection,omitempty"`

	// NEW: Memory management configuration
	MemoryLimit    int64                `json:"memory_limit,omitempty"`    // Maximum memory usage in bytes (0 = no limit)
	CachePolicy    CachePolicy          `json:"cache_policy,omitempty"`    // Cache eviction policy
	EnableMMapping bool                 `json:"enable_mmapping,omitempty"` // Enable memory mapping for large indices
	MemoryConfig   *memory.MemoryConfig `json:"memory_config,omitempty"`   // Advanced memory management settings

	// NEW: Metadata schema and filtering configuration
	MetadataSchema MetadataSchema `json:"metadata_schema,omitempty"` // Schema definition for metadata fields
	IndexedFields  []string       `json:"indexed_fields,omitempty"`  // Fields to create indices for (for faster filtering)

	// NEW: Batch processing configuration
	BatchConfig BatchConfig `json:"batch_config,omitempty"` // Batch operation settings
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
	Flat
)

// selectOptimalIndexType chooses the best index type based on collection size
func selectOptimalIndexType(vectorCount int) IndexType {
	if vectorCount < 10000 {
		// Small collections: use Flat for exact search and simplicity
		return Flat
	} else if vectorCount < 1000000 {
		// Medium collections: use HNSW for good balance of speed and accuracy
		return HNSW
	} else {
		// Large collections: use IVF-PQ for memory efficiency
		return IVFPQ
	}
}

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
		// Default memory management settings
		MemoryLimit:    0, // No limit by default
		CachePolicy:    LRUCache,
		EnableMMapping: false, // Disabled by default
		// Default batch configuration
		BatchConfig: DefaultBatchConfig(),
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

	// Apply automatic index selection if enabled
	if config.AutoIndexSelection {
		config.IndexType = selectOptimalIndexType(0) // Start with 0 vectors
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
			Quantization:   config.Quantization,
		})
	case IVFPQ:
		idx, err = index.NewIVFPQ(&index.IVFPQConfig{
			Dimension:     config.Dimension,
			NClusters:     100, // Default cluster count
			NProbes:       10,  // Default probe count
			Metric:        util.DistanceMetric(config.Metric),
			Quantization:  config.Quantization,
			MaxIterations: 100,
			Tolerance:     1e-4,
			RandomSeed:    42,
		})
	case Flat:
		idx, err = index.NewFlat(&index.FlatConfig{
			Dimension:    config.Dimension,
			Metric:       util.DistanceMetric(config.Metric),
			Quantization: config.Quantization,
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
		Quantization:   config.Quantization,
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

	// Check if we should switch index type (automatic index selection)
	if c.config.AutoIndexSelection {
		if err := c.checkAndSwitchIndexType(ctx); err != nil {
			// Log the error but don't fail the insertion
			// TODO: Add proper logging
		}
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

// SaveIndex persists the collection's index to disk
func (c *Collection) SaveIndex(ctx context.Context, path string) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return fmt.Errorf("collection is closed")
	}

	return c.index.SaveToDisk(ctx, path)
}

// LoadIndex loads the collection's index from disk
func (c *Collection) LoadIndex(ctx context.Context, path string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return fmt.Errorf("collection is closed")
	}

	return c.index.LoadFromDisk(ctx, path)
}

// GetIndexMetadata returns metadata about the collection's index
func (c *Collection) GetIndexMetadata() *index.PersistenceMetadata {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil
	}

	// Get HNSW-specific metadata if available
	if hnswIndex, ok := c.index.(interface {
		GetPersistenceMetadata() *index.PersistenceMetadata
	}); ok {
		return hnswIndex.GetPersistenceMetadata()
	}

	return nil
}

// checkAndSwitchIndexType checks if the index type should be changed based on collection size
func (c *Collection) checkAndSwitchIndexType(ctx context.Context) error {
	currentSize := c.index.Size()
	optimalType := selectOptimalIndexType(currentSize)

	// If the optimal type is different from current, switch
	if optimalType != c.config.IndexType {
		return c.switchIndexType(ctx, optimalType)
	}

	return nil
}

// switchIndexType rebuilds the index with a new type
func (c *Collection) switchIndexType(ctx context.Context, newType IndexType) error {
	// Get all vectors from current index
	vectors, err := c.getAllVectors(ctx)
	if err != nil {
		return fmt.Errorf("failed to get vectors for index switch: %w", err)
	}

	// Create new index with the new type
	var newIndex index.Index
	switch newType {
	case HNSW:
		newIndex, err = index.NewHNSW(&index.HNSWConfig{
			Dimension:      c.config.Dimension,
			M:              c.config.M,
			EfConstruction: c.config.EfConstruction,
			EfSearch:       c.config.EfSearch,
			ML:             c.config.ML,
			Metric:         util.DistanceMetric(c.config.Metric),
			Quantization:   c.config.Quantization,
		})
	case IVFPQ:
		newIndex, err = index.NewIVFPQ(&index.IVFPQConfig{
			Dimension:     c.config.Dimension,
			NClusters:     100,
			NProbes:       10,
			Metric:        util.DistanceMetric(c.config.Metric),
			Quantization:  c.config.Quantization,
			MaxIterations: 100,
			Tolerance:     1e-4,
			RandomSeed:    42,
		})
	case Flat:
		newIndex, err = index.NewFlat(&index.FlatConfig{
			Dimension:    c.config.Dimension,
			Metric:       util.DistanceMetric(c.config.Metric),
			Quantization: c.config.Quantization,
		})
	default:
		return fmt.Errorf("unsupported index type: %v", newType)
	}

	if err != nil {
		return fmt.Errorf("failed to create new index: %w", err)
	}

	// Insert all vectors into new index
	for _, vector := range vectors {
		if err := newIndex.Insert(ctx, vector); err != nil {
			newIndex.Close() // Clean up on failure
			return fmt.Errorf("failed to insert vector during index switch: %w", err)
		}
	}

	// Close old index and switch
	c.index.Close()
	c.index = newIndex
	c.config.IndexType = newType

	return nil
}

// getAllVectors retrieves all vectors from the storage layer
func (c *Collection) getAllVectors(ctx context.Context) ([]*index.VectorEntry, error) {
	var vectors []*index.VectorEntry

	err := c.storage.Iterate(ctx, func(entry *index.VectorEntry) error {
		// Create a copy to avoid reference issues
		vectorCopy := &index.VectorEntry{
			ID:       entry.ID,
			Vector:   make([]float32, len(entry.Vector)),
			Metadata: make(map[string]interface{}),
		}
		copy(vectorCopy.Vector, entry.Vector)
		for k, v := range entry.Metadata {
			vectorCopy.Metadata[k] = v
		}
		vectors = append(vectors, vectorCopy)
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to iterate storage: %w", err)
	}

	return vectors, nil
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

	// Validate quantization configuration if provided
	if config.Quantization != nil {
		if err := config.Quantization.Validate(); err != nil {
			return fmt.Errorf("invalid quantization config: %w", err)
		}
	}

	// Validate memory configuration
	if config.MemoryLimit < 0 {
		return fmt.Errorf("memory limit must be non-negative, got %d", config.MemoryLimit)
	}

	if config.MemoryConfig != nil {
		if config.MemoryConfig.MaxMemory < 0 {
			return fmt.Errorf("max memory must be non-negative, got %d", config.MemoryConfig.MaxMemory)
		}
		if config.MemoryConfig.MonitorInterval <= 0 {
			return fmt.Errorf("monitor interval must be positive, got %v", config.MemoryConfig.MonitorInterval)
		}
		if config.MemoryConfig.GCThreshold < 0 || config.MemoryConfig.GCThreshold > 1 {
			return fmt.Errorf("GC threshold must be between 0 and 1, got %f", config.MemoryConfig.GCThreshold)
		}
		if config.MemoryConfig.MMapThreshold < 0 {
			return fmt.Errorf("mmap threshold must be non-negative, got %d", config.MemoryConfig.MMapThreshold)
		}
	}

	// Validate metadata schema if provided
	if config.MetadataSchema != nil {
		if err := config.MetadataSchema.Validate(); err != nil {
			return fmt.Errorf("invalid metadata schema: %w", err)
		}
	}

	// Validate indexed fields
	if len(config.IndexedFields) > 0 && config.MetadataSchema != nil {
		for _, field := range config.IndexedFields {
			if _, exists := config.MetadataSchema[field]; !exists {
				return fmt.Errorf("indexed field '%s' not found in metadata schema", field)
			}
		}
	}

	// Apply default batch configuration if not set (for backward compatibility)
	if config.BatchConfig.ChunkSize == 0 {
		config.BatchConfig = DefaultBatchConfig()
	}

	// Validate batch configuration
	if config.BatchConfig.ChunkSize <= 0 {
		return fmt.Errorf("batch chunk size must be positive, got %d", config.BatchConfig.ChunkSize)
	}
	if config.BatchConfig.MaxConcurrency <= 0 {
		return fmt.Errorf("batch max concurrency must be positive, got %d", config.BatchConfig.MaxConcurrency)
	}
	if config.BatchConfig.TimeoutPerChunk <= 0 {
		return fmt.Errorf("batch timeout per chunk must be positive, got %v", config.BatchConfig.TimeoutPerChunk)
	}

	return nil
}
