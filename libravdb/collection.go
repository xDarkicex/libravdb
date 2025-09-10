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
	mu            sync.RWMutex
	name          string
	config        *CollectionConfig
	index         index.Index
	storage       storage.Collection
	metrics       *obs.Metrics
	memoryManager memory.MemoryManager
	closed        bool

	// Runtime optimization state
	optimizationInProgress bool
	lastOptimization       time.Time
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

	// Initialize memory manager if memory management is configured
	var memManager memory.MemoryManager
	if config.MemoryLimit > 0 || config.MemoryConfig != nil {
		memConfig := memory.DefaultMemoryConfig()
		if config.MemoryConfig != nil {
			memConfig = *config.MemoryConfig
		}
		if config.MemoryLimit > 0 {
			memConfig.MaxMemory = config.MemoryLimit
		}
		memConfig.EnableMMap = config.EnableMMapping

		memManager = memory.NewManager(memConfig)

		// Register the index as a memory-mappable component if supported
		if mappable, ok := idx.(memory.MemoryMappable); ok {
			if err := memManager.RegisterMemoryMappable(fmt.Sprintf("index_%s", name), mappable); err != nil {
				return nil, fmt.Errorf("failed to register index for memory management: %w", err)
			}
		}

		// Start memory monitoring
		if err := memManager.Start(context.Background()); err != nil {
			return nil, fmt.Errorf("failed to start memory manager: %w", err)
		}
	}

	return &Collection{
		name:          name,
		config:        config,
		index:         idx,
		storage:       collectionStorage,
		metrics:       metrics,
		memoryManager: memManager,
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

	// Initialize memory manager if memory management is configured
	var memManager memory.MemoryManager
	if config.MemoryLimit > 0 || config.MemoryConfig != nil {
		memConfig := memory.DefaultMemoryConfig()
		if config.MemoryConfig != nil {
			memConfig = *config.MemoryConfig
		}
		if config.MemoryLimit > 0 {
			memConfig.MaxMemory = config.MemoryLimit
		}
		memConfig.EnableMMap = config.EnableMMapping

		memManager = memory.NewManager(memConfig)

		// Register the index as a memory-mappable component if supported
		if mappable, ok := idx.(memory.MemoryMappable); ok {
			if err := memManager.RegisterMemoryMappable(fmt.Sprintf("index_%s", name), mappable); err != nil {
				return nil, fmt.Errorf("failed to register index for memory management: %w", err)
			}
		}

		// Start memory monitoring
		if err := memManager.Start(context.Background()); err != nil {
			return nil, fmt.Errorf("failed to start memory manager: %w", err)
		}
	}

	collection := &Collection{
		name:          name,
		config:        config,
		index:         idx,
		storage:       storageCollection,
		metrics:       metrics,
		memoryManager: memManager,
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

// Update modifies an existing vector in the collection
func (c *Collection) Update(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrCollectionClosed
	}

	// Validate input
	if id == "" {
		return fmt.Errorf("vector ID cannot be empty")
	}

	if vector != nil && len(vector) != c.config.Dimension {
		return fmt.Errorf("vector dimension %d does not match collection dimension %d",
			len(vector), c.config.Dimension)
	}

	// For now, implement update as delete + insert
	// This ensures consistency across index and storage layers
	// TODO: Optimize with native update operations when available

	// First, try to get the existing entry for partial updates
	var existingEntry *index.VectorEntry
	if vector == nil || metadata == nil {
		// Need to retrieve existing data for partial update
		_ = c.storage.Iterate(ctx, func(entry *index.VectorEntry) error {
			if entry.ID == id {
				existingEntry = &index.VectorEntry{
					ID:       entry.ID,
					Vector:   make([]float32, len(entry.Vector)),
					Metadata: make(map[string]interface{}),
				}
				copy(existingEntry.Vector, entry.Vector)
				for k, v := range entry.Metadata {
					existingEntry.Metadata[k] = v
				}
				return fmt.Errorf("found") // Use error to break iteration
			}
			return nil
		})

		if existingEntry == nil {
			return fmt.Errorf("vector with ID %s not found", id)
		}
	}

	// Prepare the updated entry
	updatedEntry := &index.VectorEntry{
		ID:       id,
		Vector:   vector,
		Metadata: metadata,
	}

	// Use existing data for partial updates
	if existingEntry != nil {
		if vector == nil {
			updatedEntry.Vector = existingEntry.Vector
		}
		if metadata == nil {
			updatedEntry.Metadata = existingEntry.Metadata
		} else if existingEntry.Metadata != nil {
			// Merge metadata (new values override existing ones)
			mergedMetadata := make(map[string]interface{})
			for k, v := range existingEntry.Metadata {
				mergedMetadata[k] = v
			}
			for k, v := range metadata {
				mergedMetadata[k] = v
			}
			updatedEntry.Metadata = mergedMetadata
		}
	}

	// Delete the existing entry from index
	if err := c.index.Delete(ctx, id); err != nil {
		return fmt.Errorf("failed to delete existing vector from index: %w", err)
	}

	// Insert the updated entry into index
	if err := c.index.Insert(ctx, updatedEntry); err != nil {
		// TODO: Rollback the delete operation
		return fmt.Errorf("failed to insert updated vector into index: %w", err)
	}

	// Update storage (this will append to WAL)
	if err := c.storage.Insert(ctx, updatedEntry); err != nil {
		// TODO: Rollback index operations
		return fmt.Errorf("failed to write update to storage: %w", err)
	}

	// Update metrics
	if c.metrics != nil {
		c.metrics.VectorUpdates.Inc()
	}

	return nil
}

// Delete removes a vector from the collection
func (c *Collection) Delete(ctx context.Context, id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrCollectionClosed
	}

	// Validate input
	if id == "" {
		return fmt.Errorf("vector ID cannot be empty")
	}

	// Delete from index
	if err := c.index.Delete(ctx, id); err != nil {
		return fmt.Errorf("failed to delete vector from index: %w", err)
	}

	// Mark as deleted in storage (tombstone record)
	// Create a tombstone entry to mark the vector as deleted
	tombstone := &index.VectorEntry{
		ID:       id,
		Vector:   nil, // Nil vector indicates deletion
		Metadata: map[string]interface{}{"_deleted": true, "_deleted_at": time.Now()},
	}

	if err := c.storage.Insert(ctx, tombstone); err != nil {
		// TODO: Rollback index deletion
		return fmt.Errorf("failed to write deletion to storage: %w", err)
	}

	// Update metrics
	if c.metrics != nil {
		c.metrics.VectorDeletes.Inc()
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

	stats := &CollectionStats{
		Name:                 c.name,
		VectorCount:          c.index.Size(),
		Dimension:            c.config.Dimension,
		IndexType:            c.config.IndexType.String(),
		MemoryUsage:          c.index.MemoryUsage(),
		HasQuantization:      c.config.Quantization != nil,
		HasMemoryLimit:       c.config.MemoryLimit > 0,
		MemoryMappingEnabled: c.config.EnableMMapping,
	}

	// Add enhanced memory statistics if memory manager is available
	if c.memoryManager != nil {
		usage := c.memoryManager.GetUsage()
		stats.MemoryStats = &CollectionMemoryStats{
			Total:         usage.Total,
			Index:         usage.Indices,
			Cache:         usage.Caches,
			Quantized:     usage.Quantized,
			MemoryMapped:  usage.MemoryMapped,
			Limit:         usage.Limit,
			Available:     usage.Available,
			PressureLevel: "normal", // TODO: Calculate actual pressure level
			Timestamp:     usage.Timestamp,
		}
	}

	// Add optimization status
	stats.OptimizationStatus = &OptimizationStatus{
		InProgress:       c.optimizationInProgress,
		LastOptimization: c.lastOptimization,
		CanOptimize:      !c.closed && !c.optimizationInProgress,
	}

	return stats
}

// GetMemoryUsage returns current memory usage statistics for the collection
func (c *Collection) GetMemoryUsage() (*memory.MemoryUsage, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, ErrCollectionClosed
	}

	if c.memoryManager == nil {
		// Return basic memory usage from index if no memory manager
		usage := &memory.MemoryUsage{
			Total:     c.index.MemoryUsage(),
			Indices:   c.index.MemoryUsage(),
			Timestamp: time.Now(),
		}
		return usage, nil
	}

	usage := c.memoryManager.GetUsage()
	return &usage, nil
}

// SetMemoryLimit updates the memory limit for the collection
func (c *Collection) SetMemoryLimit(bytes int64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrCollectionClosed
	}

	// Update config
	c.config.MemoryLimit = bytes

	// Update memory manager if it exists
	if c.memoryManager != nil {
		return c.memoryManager.SetLimit(bytes)
	}

	// If no memory manager exists and limit is set, create one
	if bytes > 0 {
		memConfig := memory.DefaultMemoryConfig()
		if c.config.MemoryConfig != nil {
			memConfig = *c.config.MemoryConfig
		}
		memConfig.MaxMemory = bytes
		memConfig.EnableMMap = c.config.EnableMMapping

		memManager := memory.NewManager(memConfig)

		// Register the index as a memory-mappable component if supported
		if mappable, ok := c.index.(memory.MemoryMappable); ok {
			if err := memManager.RegisterMemoryMappable(fmt.Sprintf("index_%s", c.name), mappable); err != nil {
				return fmt.Errorf("failed to register index for memory management: %w", err)
			}
		}

		// Start memory monitoring
		if err := memManager.Start(context.Background()); err != nil {
			return fmt.Errorf("failed to start memory manager: %w", err)
		}

		c.memoryManager = memManager
	}

	return nil
}

// TriggerGC forces garbage collection for the collection
func (c *Collection) TriggerGC() error {
	c.mu.RLock()
	closed := c.closed
	memManager := c.memoryManager
	c.mu.RUnlock()

	if closed {
		return ErrCollectionClosed
	}

	if memManager != nil {
		return memManager.TriggerGC()
	}

	// Fallback to runtime GC if no memory manager
	memory.ForceGC()
	return nil
}

// OptimizeCollection performs collection optimization including index rebuilding and memory optimization
func (c *Collection) OptimizeCollection(ctx context.Context, options *OptimizationOptions) error {
	// Check initial state and set optimization in progress
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return ErrCollectionClosed
	}

	if c.optimizationInProgress {
		c.mu.Unlock()
		return fmt.Errorf("optimization already in progress")
	}

	// Set default options if not provided
	if options == nil {
		options = &OptimizationOptions{
			RebuildIndex:       true,
			OptimizeMemory:     true,
			CompactStorage:     true,
			UpdateQuantization: false,
		}
	}

	c.optimizationInProgress = true
	memManager := c.memoryManager
	hasQuantization := c.config.Quantization != nil
	c.mu.Unlock()

	// Ensure we reset optimization status on exit
	defer func() {
		c.mu.Lock()
		c.optimizationInProgress = false
		c.lastOptimization = time.Now()
		c.mu.Unlock()
	}()

	// Step 1: Optimize memory if requested
	if options.OptimizeMemory && memManager != nil {
		if err := memManager.HandleMemoryLimitExceeded(); err != nil {
			return fmt.Errorf("memory optimization failed: %w", err)
		}
	}

	// Step 2: Rebuild index if requested
	if options.RebuildIndex {
		if err := c.rebuildIndexOptimized(ctx, options); err != nil {
			return fmt.Errorf("index rebuild failed: %w", err)
		}
	}

	// Step 3: Update quantization if requested
	if options.UpdateQuantization && hasQuantization {
		if err := c.updateQuantization(ctx); err != nil {
			return fmt.Errorf("quantization update failed: %w", err)
		}
	}

	// Step 4: Compact storage if requested
	if options.CompactStorage {
		// Note: This would require storage layer support for compaction
		// For now, we'll just trigger GC
		if err := c.TriggerGC(); err != nil {
			return fmt.Errorf("storage compaction failed: %w", err)
		}
	}

	return nil
}

// rebuildIndexOptimized rebuilds the index with optimization considerations
func (c *Collection) rebuildIndexOptimized(ctx context.Context, options *OptimizationOptions) error {
	c.mu.Lock()
	currentSize := c.index.Size()
	autoIndexSelection := c.config.AutoIndexSelection
	currentType := c.config.IndexType
	c.mu.Unlock()

	if autoIndexSelection {
		optimalType := selectOptimalIndexType(currentSize)
		if optimalType != currentType {
			return c.switchIndexType(ctx, optimalType)
		}
	}

	// For optimization, we don't need to rebuild if the index is already populated
	// and we're not switching types. This avoids the duplicate insertion issue.
	if currentSize > 0 {
		// Index is already built, no need to rebuild unless switching types
		return nil
	}

	// Only rebuild if index is empty (e.g., after loading from storage)
	return c.rebuildIndex(ctx)
}

// updateQuantization retrains quantization parameters with current data
func (c *Collection) updateQuantization(ctx context.Context) error {
	if c.config.Quantization == nil {
		return fmt.Errorf("no quantization configured")
	}

	// Get all vectors for retraining
	vectors, err := c.getAllVectors(ctx)
	if err != nil {
		return fmt.Errorf("failed to get vectors for quantization update: %w", err)
	}

	if len(vectors) == 0 {
		return nil // Nothing to retrain
	}

	// Extract vector data for training
	trainingVectors := make([][]float32, len(vectors))
	for i, entry := range vectors {
		trainingVectors[i] = entry.Vector
	}

	// Create new quantizer and train it
	quantizer, err := quant.Create(c.config.Quantization)
	if err != nil {
		return fmt.Errorf("failed to create quantizer: %w", err)
	}

	if err := quantizer.Train(ctx, trainingVectors); err != nil {
		return fmt.Errorf("failed to train quantizer: %w", err)
	}

	// Update the index with the new quantizer
	// Note: This would require index-specific support for quantizer updates
	// For now, we'll rebuild the index with the new quantizer
	return c.rebuildIndex(ctx)
}

// GetOptimizationStatus returns the current optimization status
func (c *Collection) GetOptimizationStatus() *OptimizationStatus {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return &OptimizationStatus{
		InProgress:       c.optimizationInProgress,
		LastOptimization: c.lastOptimization,
		CanOptimize:      !c.closed && !c.optimizationInProgress,
	}
}

// EnableMemoryMapping enables memory mapping for the collection's index
func (c *Collection) EnableMemoryMapping(path string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrCollectionClosed
	}

	// Update config
	c.config.EnableMMapping = true

	// Enable memory mapping on the index if supported
	if mappable, ok := c.index.(memory.MemoryMappable); ok {
		if mappable.CanMemoryMap() {
			return mappable.EnableMemoryMapping(path)
		}
	}

	return fmt.Errorf("index does not support memory mapping")
}

// DisableMemoryMapping disables memory mapping for the collection's index
func (c *Collection) DisableMemoryMapping() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrCollectionClosed
	}

	// Update config
	c.config.EnableMMapping = false

	// Disable memory mapping on the index if supported
	if mappable, ok := c.index.(memory.MemoryMappable); ok {
		if mappable.IsMemoryMapped() {
			return mappable.DisableMemoryMapping()
		}
	}

	return nil
}

// Close shuts down the collection
func (c *Collection) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}

	var errors []error

	// Stop memory manager if it exists
	if c.memoryManager != nil {
		if err := c.memoryManager.Stop(); err != nil {
			errors = append(errors, err)
		}
	}

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
