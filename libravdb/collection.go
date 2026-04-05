package libravdb

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/filter"
	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/memory"
	"github.com/xDarkicex/libravdb/internal/obs"
	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/libravdb/internal/util"
)

// Collection represents a named collection of vectors with a specific schema
type Collection struct {
	mu            sync.RWMutex
	db            *Database
	name          string
	config        *CollectionConfig
	index         index.Index
	storage       storage.Collection
	writes        *writeController
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
	NClusters      int     `json:"n_clusters,omitempty"`
	NProbes        int     `json:"n_probes,omitempty"`
	ML             float64 `json:"ml"`      // Level generation factor
	Version        int     `json:"version"` // Config version for future compatibility
	RawVectorStore string  `json:"raw_vector_store,omitempty"`
	RawStoreCap    int     `json:"raw_store_cap,omitempty"`
	// Persistence configuration
	AutoSave     bool          `json:"auto_save"`     // Enable automatic index saving
	SaveInterval time.Duration `json:"save_interval"` // Interval between automatic saves
	SavePath     string        `json:"save_path"`     // Path for automatic saves
	// Quantization configuration (optional)
	Quantization *quant.QuantizationConfig `json:"quantization,omitempty"`
	// Automatic index selection based on collection size
	AutoIndexSelection bool `json:"auto_index_selection,omitempty"`
	// AutoIndexThresholds overrides the default thresholds for auto-index selection.
	// If set to zero values, DefaultHNSWThreshold and DefaultIVFPQThreshold are used.
	AutoIndexThresholds struct {
		HNSWThreshold  int `json:"hnsw_threshold,omitempty"`
		IVFPQThreshold int `json:"ivfpq_threshold,omitempty"`
	} `json:"auto_index_thresholds,omitempty"`

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

type trainableIndex interface {
	Train(ctx context.Context, vectors [][]float32) error
	IsTrained() bool
}

func trainingIndexState(idx index.Index) (trainableIndex, bool) {
	trainable, ok := idx.(trainableIndex)
	if !ok || trainable.IsTrained() {
		return nil, false
	}
	return trainable, true
}

func (c *Collection) ivfpqConfig() *index.IVFPQConfig {
	nClusters := c.config.NClusters
	if nClusters <= 0 {
		nClusters = 100
	}

	nProbes := c.config.NProbes
	if nProbes <= 0 {
		nProbes = 10
	}
	if nProbes > nClusters {
		nProbes = nClusters
	}

	return &index.IVFPQConfig{
		Dimension:     c.config.Dimension,
		NClusters:     nClusters,
		NProbes:       nProbes,
		Metric:        util.DistanceMetric(c.config.Metric),
		Quantization:  c.config.Quantization,
		MaxIterations: 100,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}
}

func prepareIndexForEntries(ctx context.Context, idx index.Index, entries []*index.VectorEntry) error {
	trainable, ok := trainingIndexState(idx)
	if !ok {
		return nil
	}
	if len(entries) == 0 {
		return nil
	}

	vectors := make([][]float32, len(entries))
	for i, entry := range entries {
		vectors[i] = entry.Vector
	}

	if err := trainable.Train(ctx, vectors); err != nil {
		return fmt.Errorf("failed to train index: %w", err)
	}
	return nil
}

func insertEntriesIntoIndex(ctx context.Context, idx index.Index, entries []*index.VectorEntry) error {
	if len(entries) == 0 {
		return nil
	}

	if _, ok := idx.(trainableIndex); ok {
		return idx.BatchInsert(ctx, entries)
	}

	for _, entry := range entries {
		if err := idx.Insert(ctx, entry); err != nil {
			return err
		}
	}
	return nil
}

func createIndexForCollection(config *CollectionConfig, provider interface {
	GetByOrdinal(uint32) ([]float32, error)
	Distance([]float32, uint32) (float32, error)
}) (index.Index, error) {
	switch config.IndexType {
	case HNSW:
		return index.NewHNSW(&index.HNSWConfig{
			Dimension:      config.Dimension,
			M:              config.M,
			EfConstruction: config.EfConstruction,
			EfSearch:       config.EfSearch,
			ML:             config.ML,
			Metric:         util.DistanceMetric(config.Metric),
			Provider:       provider,
			RawVectorStore: config.RawVectorStore,
			RawStoreCap:    config.RawStoreCap,
			Quantization:   config.Quantization,
		})
	case IVFPQ:
		temp := &Collection{config: config}
		return index.NewIVFPQ(temp.ivfpqConfig())
	case Flat:
		return index.NewFlat(&index.FlatConfig{
			Dimension:    config.Dimension,
			Metric:       util.DistanceMetric(config.Metric),
			Quantization: config.Quantization,
		})
	default:
		return nil, fmt.Errorf("unsupported index type: %v", config.IndexType)
	}
}

func buildIndexForEntries(ctx context.Context, config *CollectionConfig, provider interface {
	GetByOrdinal(uint32) ([]float32, error)
	Distance([]float32, uint32) (float32, error)
}, entries []*index.VectorEntry) (index.Index, error) {
	idx, err := createIndexForCollection(config, provider)
	if err != nil {
		return nil, err
	}
	if err := prepareIndexForEntries(ctx, idx, entries); err != nil {
		idx.Close()
		return nil, err
	}
	if err := insertEntriesIntoIndex(ctx, idx, entries); err != nil {
		idx.Close()
		return nil, fmt.Errorf("failed to insert vectors into index: %w", err)
	}
	return idx, nil
}

// IndexType defines the index algorithm to use
type IndexType int

const (
	HNSW IndexType = iota
	IVFPQ
	Flat
)

// DefaultAutoIndexThresholds defines the default thresholds for auto-index selection.
// These can be overridden via CollectionOption when creating a collection.
const (
	// DefaultHNSWThreshold is the default vector count at which HNSW is selected over Flat.
	// Collections with fewer vectors use Flat (exact search).
	// Collections at or above this count use HNSW (approximate search with better asymptotic performance).
	// The value 2000 balances query latency savings against HNSW build/update overhead.
	DefaultHNSWThreshold = 2000

	// DefaultIVFPQThreshold is the default vector count at which IVF-PQ is selected over HNSW.
	// Collections below this use HNSW for accuracy/speed balance.
	// Collections at or above this use IVF-PQ for memory efficiency at scale.
	DefaultIVFPQThreshold = 1000000
)

// selectOptimalIndexType chooses the best index type based on collection size.
// Uses the provided thresholds to determine the switching points.
func selectOptimalIndexType(vectorCount int, hnswThreshold, ivfpqThreshold int) IndexType {
	if vectorCount < hnswThreshold {
		// Small collections: use Flat for exact search and simplicity
		return Flat
	} else if vectorCount < ivfpqThreshold {
		// Medium collections: use HNSW for good balance of speed and accuracy
		return HNSW
	} else {
		// Large collections: use IVF-PQ for memory efficiency
		return IVFPQ
	}
}

// newCollection creates a new collection instance
func newCollection(name string, storageEngine storage.Engine, metrics *obs.Metrics, writes *writeController, opts ...CollectionOption) (*Collection, error) {
	config := &CollectionConfig{
		Dimension:      768, // Default for common embeddings
		Metric:         CosineDistance,
		IndexType:      HNSW,
		M:              32,
		EfConstruction: 200,
		EfSearch:       50,
		NClusters:      100,
		NProbes:        10,
		ML:             1.0 / math.Log(2.0),
		RawVectorStore: "slabby",
		RawStoreCap:    4096,
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
		hnswThreshold := config.AutoIndexThresholds.HNSWThreshold
		if hnswThreshold == 0 {
			hnswThreshold = DefaultHNSWThreshold
		}
		ivfpqThreshold := config.AutoIndexThresholds.IVFPQThreshold
		if ivfpqThreshold == 0 {
			ivfpqThreshold = DefaultIVFPQThreshold
		}
		config.IndexType = selectOptimalIndexType(0, hnswThreshold, ivfpqThreshold) // Start with 0 vectors
	}

	// Convert to LSM config format
	engineConfig := &storage.CollectionConfig{
		Dimension:      config.Dimension,
		Metric:         int(config.Metric),
		IndexType:      int(config.IndexType),
		M:              config.M,
		EfConstruction: config.EfConstruction,
		EfSearch:       config.EfSearch,
		NClusters:      config.NClusters,
		NProbes:        config.NProbes,
		ML:             config.ML,
		Version:        1,
		RawVectorStore: config.RawVectorStore,
		RawStoreCap:    config.RawStoreCap,
	}

	// Create storage for this collection - PASS THE CONFIG
	collectionStorage, err := storageEngine.CreateCollection(name, engineConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection storage: %w", err)
	}

	// Create index
	provider, _ := collectionStorage.(interface {
		GetByOrdinal(uint32) ([]float32, error)
		Distance([]float32, uint32) (float32, error)
	})
	idx, err := createIndexForCollection(config, provider)

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
		writes:        writes,
		metrics:       metrics,
		memoryManager: memManager,
	}, nil
}

// newCollectionFromStorage creates a collection instance from existing storage
func newCollectionFromStorage(name string, storageCollection storage.Collection, metrics *obs.Metrics, engineConfig *storage.CollectionConfig, writes *writeController) (*Collection, error) {
	// Convert LSM config to libravdb config
	config := &CollectionConfig{
		Dimension:      engineConfig.Dimension,
		Metric:         DistanceMetric(engineConfig.Metric),
		IndexType:      IndexType(engineConfig.IndexType),
		M:              engineConfig.M,
		EfConstruction: engineConfig.EfConstruction,
		EfSearch:       engineConfig.EfSearch,
		NClusters:      engineConfig.NClusters,
		NProbes:        engineConfig.NProbes,
		ML:             engineConfig.ML,
		Version:        engineConfig.Version,
		RawVectorStore: engineConfig.RawVectorStore,
		RawStoreCap:    engineConfig.RawStoreCap,
	}
	if config.NClusters <= 0 {
		config.NClusters = 100
	}
	if config.NProbes <= 0 {
		config.NProbes = min(config.NClusters, 10)
	}

	// Create index with stored config.
	provider, _ := storageCollection.(interface {
		GetByOrdinal(uint32) ([]float32, error)
		Distance([]float32, uint32) (float32, error)
	})
	idx, err := createIndexForCollection(config, provider)
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
		writes:        writes,
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
	vectors, err := c.getAllVectors(ctx)
	if err != nil {
		return err
	}
	if err := prepareIndexForEntries(ctx, c.index, vectors); err != nil {
		return err
	}
	return insertEntriesIntoIndex(ctx, c.index, vectors)
}

// Insert adds or updates a vector in the collection
func (c *Collection) Insert(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error {
	release, err := c.acquireWrite(ctx)
	if err != nil {
		return err
	}
	defer release()

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

	if exists, err := c.storage.Exists(ctx, id); err != nil {
		return fmt.Errorf("failed to check existing vector: %w", err)
	} else if exists {
		return fmt.Errorf("failed to insert into index: node with ID '%s' already exists", id)
	}

	if err := c.storage.AssignOrdinals(ctx, []*index.VectorEntry{storageEntry}); err != nil {
		return fmt.Errorf("failed to assign ordinal: %w", err)
	}

	if _, needsTraining := trainingIndexState(c.index); needsTraining {
		existingVectors, err := c.getAllVectors(ctx)
		if err != nil {
			return fmt.Errorf("failed to prepare index training set: %w", err)
		}
		if err := prepareIndexForEntries(ctx, c.index, append(existingVectors, storageEntry)); err != nil {
			return err
		}
	}

	if err := c.storage.Insert(ctx, storageEntry); err != nil {
		return fmt.Errorf("failed to write to storage: %w", err)
	}
	if err := c.index.Insert(ctx, storageEntry); err != nil {
		_ = c.storage.Delete(ctx, id)
		return fmt.Errorf("failed to insert into index: %w", err)
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

func (c *Collection) insertBatch(ctx context.Context, entries []*index.VectorEntry) error {
	release, err := c.acquireWrite(ctx)
	if err != nil {
		return err
	}
	defer release()

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrCollectionClosed
	}

	if len(entries) == 0 {
		return nil
	}

	seen := make(map[string]struct{}, len(entries))
	for _, entry := range entries {
		if _, ok := seen[entry.ID]; ok {
			return fmt.Errorf("failed to insert into index: node with ID '%s' already exists", entry.ID)
		}
		seen[entry.ID] = struct{}{}
		exists, err := c.storage.Exists(ctx, entry.ID)
		if err != nil {
			return fmt.Errorf("failed to check existing vector: %w", err)
		}
		if exists {
			return fmt.Errorf("failed to insert into index: node with ID '%s' already exists", entry.ID)
		}
	}

	if err := c.storage.AssignOrdinals(ctx, entries); err != nil {
		return fmt.Errorf("failed to assign ordinals: %w", err)
	}
	if _, needsTraining := trainingIndexState(c.index); needsTraining {
		existingVectors, err := c.getAllVectors(ctx)
		if err != nil {
			return fmt.Errorf("failed to prepare index training set: %w", err)
		}
		trainingVectors := append(existingVectors, entries...)
		if err := prepareIndexForEntries(ctx, c.index, trainingVectors); err != nil {
			return err
		}
	}
	insertedIDs := make([]string, 0, len(entries))
	for _, entry := range entries {
		if len(entry.Vector) != c.config.Dimension {
			return fmt.Errorf("vector dimension %d does not match collection dimension %d",
				len(entry.Vector), c.config.Dimension)
		}
	}

	if err := c.storage.InsertBatch(ctx, entries); err != nil {
		return fmt.Errorf("failed to write batch to storage: %w", err)
	}
	for _, entry := range entries {
		if err := c.index.Insert(ctx, entry); err != nil {
			for _, storedEntry := range entries {
				_ = c.storage.Delete(ctx, storedEntry.ID)
			}
			return fmt.Errorf("failed to insert into index: %w", err)
		}
		insertedIDs = append(insertedIDs, entry.ID)
	}

	if c.metrics != nil {
		c.metrics.VectorInserts.Add(float64(len(entries)))
	}

	if c.config.AutoIndexSelection {
		if err := c.checkAndSwitchIndexType(ctx); err != nil {
			// TODO: Add proper logging
		}
	}

	return nil
}

func (c *Collection) rollbackBatchIndex(ctx context.Context, ids []string) {
	for i := len(ids) - 1; i >= 0; i-- {
		_ = c.index.Delete(ctx, ids[i])
	}
}

// Update modifies an existing vector in the collection
func (c *Collection) Update(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error {
	release, err := c.acquireWrite(ctx)
	if err != nil {
		return err
	}
	defer release()

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

	if err := c.storage.AssignOrdinals(ctx, []*index.VectorEntry{updatedEntry}); err != nil {
		return fmt.Errorf("failed to assign ordinal: %w", err)
	}
	if deleter, ok := c.index.(interface {
		DeleteByOrdinal(context.Context, uint32) error
	}); ok {
		if err := deleter.DeleteByOrdinal(ctx, updatedEntry.Ordinal); err != nil {
			return fmt.Errorf("failed to delete existing vector from index: %w", err)
		}
	} else if err := c.index.Delete(ctx, id); err != nil {
		return fmt.Errorf("failed to delete existing vector from index: %w", err)
	}

	if err := c.storage.Insert(ctx, updatedEntry); err != nil {
		return fmt.Errorf("failed to write update to storage: %w", err)
	}
	if err := c.index.Insert(ctx, updatedEntry); err != nil {
		return fmt.Errorf("failed to insert updated vector into index: %w", err)
	}

	// Update metrics
	if c.metrics != nil {
		c.metrics.VectorUpdates.Inc()
	}

	return nil
}

// Delete removes a vector from the collection
func (c *Collection) Delete(ctx context.Context, id string) error {
	release, err := c.acquireWrite(ctx)
	if err != nil {
		return err
	}
	defer release()

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrCollectionClosed
	}

	// Validate input
	if id == "" {
		return fmt.Errorf("vector ID cannot be empty")
	}

	if entry, err := c.storage.Get(ctx, id); err == nil {
		if deleter, ok := c.index.(interface {
			DeleteByOrdinal(context.Context, uint32) error
		}); ok {
			if err := deleter.DeleteByOrdinal(ctx, entry.Ordinal); err != nil {
				return fmt.Errorf("failed to delete vector from index: %w", err)
			}
		} else if err := c.index.Delete(ctx, id); err != nil {
			return fmt.Errorf("failed to delete vector from index: %w", err)
		}
	} else if err := c.index.Delete(ctx, id); err != nil {
		return fmt.Errorf("failed to delete vector from index: %w", err)
	}

	if err := c.storage.Delete(ctx, id); err != nil {
		// TODO: Rollback index deletion
		return fmt.Errorf("failed to write deletion to storage: %w", err)
	}

	// Update metrics
	if c.metrics != nil {
		c.metrics.VectorDeletes.Inc()
	}

	return nil
}

// InsertBatch inserts multiple vectors using the public collection API.
func (c *Collection) InsertBatch(ctx context.Context, entries []VectorEntry) error {
	indexEntries := make([]*index.VectorEntry, 0, len(entries))
	for _, entry := range entries {
		indexEntries = append(indexEntries, &index.VectorEntry{
			ID:       entry.ID,
			Vector:   cloneVector(entry.Vector),
			Metadata: cloneMetadata(entry.Metadata),
		})
	}
	return c.insertBatch(ctx, indexEntries)
}

// DeleteBatch deletes multiple vectors by ID.
func (c *Collection) DeleteBatch(ctx context.Context, ids []string) error {
	for _, id := range ids {
		if err := c.Delete(ctx, id); err != nil {
			return err
		}
	}
	return nil
}

// Get returns a persisted record by ID.
func (c *Collection) Get(ctx context.Context, id string) (Record, error) {
	c.mu.RLock()
	if c.closed {
		c.mu.RUnlock()
		return Record{}, ErrCollectionClosed
	}
	c.mu.RUnlock()

	entry, err := c.storage.Get(ctx, id)
	if err != nil {
		return Record{}, fmt.Errorf("%w: %s", ErrRecordNotFound, id)
	}
	return recordFromIndexEntry(entry), nil
}

// UpdateIfVersion updates a record only if its current committed version matches expectedVersion.
func (c *Collection) UpdateIfVersion(ctx context.Context, id string, vector []float32, metadata map[string]interface{}, expectedVersion uint64) error {
	return c.withCAS(ctx, func(tx Tx) error {
		return tx.UpdateIfVersion(ctx, c.name, id, vector, metadata, expectedVersion)
	})
}

// DeleteIfVersion deletes a record only if its current committed version matches expectedVersion.
func (c *Collection) DeleteIfVersion(ctx context.Context, id string, expectedVersion uint64) error {
	return c.withCAS(ctx, func(tx Tx) error {
		return tx.DeleteIfVersion(ctx, c.name, id, expectedVersion)
	})
}

func (c *Collection) withCAS(ctx context.Context, fn func(tx Tx) error) error {
	if c == nil {
		return ErrCollectionClosed
	}
	if c.db == nil {
		return ErrTxEngineUnsupported
	}
	return c.db.WithTx(ctx, fn)
}

// Iterate walks all persisted records in the collection.
func (c *Collection) Iterate(ctx context.Context, fn func(Record) error) error {
	c.mu.RLock()
	if c.closed {
		c.mu.RUnlock()
		return ErrCollectionClosed
	}
	c.mu.RUnlock()

	return c.storage.Iterate(ctx, func(entry *index.VectorEntry) error {
		return fn(recordFromIndexEntry(entry))
	})
}

// ListAll returns all persisted records in the collection.
func (c *Collection) ListAll(ctx context.Context) ([]Record, error) {
	records := make([]Record, 0)
	if err := c.Iterate(ctx, func(record Record) error {
		records = append(records, record)
		return nil
	}); err != nil {
		return nil, err
	}
	return records, nil
}

// ListByMetadata returns records where the given metadata field equals the provided value.
func (c *Collection) ListByMetadata(ctx context.Context, field string, value interface{}) ([]Record, error) {
	records, err := c.ListAll(ctx)
	if err != nil {
		return nil, err
	}

	filtered, err := filter.NewEqualityFilter(field, value).Apply(ctx, filterEntriesFromRecords(records))
	if err != nil {
		return nil, err
	}

	result := make([]Record, 0, len(filtered))
	for _, entry := range filtered {
		result = append(result, Record{
			ID:       entry.ID,
			Vector:   cloneVector(entry.Vector),
			Metadata: cloneMetadata(entry.Metadata),
		})
	}
	return result, nil
}

// Count returns the exact number of live records in the collection.
func (c *Collection) Count(ctx context.Context) (int, error) {
	c.mu.RLock()
	if c.closed {
		c.mu.RUnlock()
		return 0, ErrCollectionClosed
	}
	c.mu.RUnlock()

	return c.storage.Count(ctx)
}

func (c *Collection) acquireWrite(ctx context.Context) (func(), error) {
	if c == nil {
		return func() {}, nil
	}
	if c.writes == nil {
		return func() {}, nil
	}
	return c.writes.acquire(ctx)
}

func (c *Collection) effectiveWriteConcurrency(requested int) int {
	if requested <= 0 {
		requested = 1
	}
	if c == nil || c.writes == nil {
		return requested
	}
	if limit := c.writes.maxParallelism(); limit > 0 && requested > limit {
		return limit
	}
	return requested
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

	// Convert from index.SearchResult to libravdb.SearchResult.
	// Public API scores are normalized to consumer-facing relevance semantics.
	results := make([]*SearchResult, len(indexResults))
	for i, r := range indexResults {
		result := &SearchResult{
			ID:      r.ID,
			Score:   r.Score,
			Version: r.Version,
		}
		if len(r.Vector) > 0 {
			result.Vector = cloneVector(r.Vector)
		}
		if r.Metadata != nil {
			result.Metadata = cloneMetadata(r.Metadata)
		}
		if result.ID == "" {
			if idGetter, ok := c.storage.(interface {
				GetIDByOrdinal(context.Context, uint32) (string, error)
			}); ok {
				if resolvedID, idErr := idGetter.GetIDByOrdinal(ctx, r.Ordinal); idErr == nil {
					result.ID = resolvedID
				}
			}
		}
		if result.Vector == nil || result.Metadata == nil || result.Version == 0 {
			entry, getErr := c.storage.Get(ctx, result.ID)
			if getErr == nil {
				result.ID = entry.ID
				result.Version = entry.Version
				if result.Vector == nil {
					result.Vector = cloneVector(entry.Vector)
				}
				if result.Metadata == nil {
					result.Metadata = cloneMetadata(entry.Metadata)
				}
			} else {
				if result.Metadata == nil {
					result.Metadata = map[string]interface{}{}
				}
			}
		}
		results[i] = result
	}

	normalizePublicSearchResults(c.config.Metric, results)

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

	storageUsage := c.storageMemoryUsageLocked()
	indexUsage := c.index.MemoryUsage()
	stats := &CollectionStats{
		Name:                 c.name,
		VectorCount:          c.index.Size(),
		Dimension:            c.config.Dimension,
		IndexType:            c.config.IndexType.String(),
		MemoryUsage:          storageUsage + indexUsage,
		HasQuantization:      c.config.Quantization != nil,
		HasMemoryLimit:       c.config.MemoryLimit > 0,
		MemoryMappingEnabled: c.config.EnableMMapping,
	}

	// Add enhanced memory statistics if memory manager is available
	if c.memoryManager != nil {
		usage := c.memoryManager.GetUsage()
		stats.MemoryStats = &CollectionMemoryStats{
			Total:         storageUsage + usage.Total,
			Storage:       storageUsage,
			Index:         usage.Indices,
			Cache:         usage.Caches,
			Quantized:     usage.Quantized,
			MemoryMapped:  usage.MemoryMapped,
			Limit:         usage.Limit,
			Available:     usage.Available,
			PressureLevel: "normal", // TODO: Calculate actual pressure level
			Timestamp:     usage.Timestamp,
		}
	} else {
		stats.MemoryStats = &CollectionMemoryStats{
			Total:         storageUsage + indexUsage,
			Storage:       storageUsage,
			Index:         indexUsage,
			PressureLevel: "normal",
			Timestamp:     time.Now(),
		}
	}
	if rawProfile := c.DebugRawVectorStoreProfile(); rawProfile != nil {
		stats.RawVectorStoreStats = &RawVectorStoreStats{
			Backend:             rawProfile["backend"].(string),
			VectorCount:         rawProfile["vector_count"].(int),
			Dimension:           rawProfile["dimension"].(int),
			BytesPerVector:      rawProfile["bytes_per_vector"].(int),
			MemoryUsage:         rawProfile["memory_usage"].(int64),
			ReservedBytes:       rawProfile["reserved_bytes"].(int64),
			ReservedDataBytes:   rawProfile["reserved_data_bytes"].(int64),
			ReservedMetaBytes:   rawProfile["reserved_meta_bytes"].(int64),
			ReservedGuardBytes:  rawProfile["reserved_guard_bytes"].(int64),
			LiveBytes:           rawProfile["live_bytes"].(int64),
			FreeBytes:           rawProfile["free_bytes"].(int64),
			CapacityUtilization: rawProfile["capacity_utilization"].(float64),
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

	storageUsage := c.storageMemoryUsageLocked()
	if c.memoryManager == nil {
		usage := &memory.MemoryUsage{
			Total:     storageUsage + c.index.MemoryUsage(),
			Indices:   c.index.MemoryUsage(),
			Caches:    storageUsage,
			Timestamp: time.Now(),
		}
		return usage, nil
	}

	usage := c.memoryManager.GetUsage()
	usage.Total += storageUsage
	usage.Caches += storageUsage
	return &usage, nil
}

func (c *Collection) storageMemoryUsageLocked() int64 {
	if c.storage == nil {
		return 0
	}
	usage, err := c.storage.MemoryUsage(context.Background())
	if err != nil {
		return 0
	}
	return usage
}

// DebugRawVectorStoreProfile exposes backend-specific raw vector storage stats
// for profiling and benchmarking.
func (c *Collection) DebugRawVectorStoreProfile() map[string]any {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if profiler, ok := c.index.(interface{ RawVectorStoreProfile() map[string]any }); ok {
		return profiler.RawVectorStoreProfile()
	}
	return nil
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
	hnswThreshold := c.config.AutoIndexThresholds.HNSWThreshold
	if hnswThreshold == 0 {
		hnswThreshold = DefaultHNSWThreshold
	}
	ivfpqThreshold := c.config.AutoIndexThresholds.IVFPQThreshold
	if ivfpqThreshold == 0 {
		ivfpqThreshold = DefaultIVFPQThreshold
	}
	c.mu.Unlock()

	if autoIndexSelection {
		optimalType := selectOptimalIndexType(currentSize, hnswThreshold, ivfpqThreshold)
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
	hnswThreshold := c.config.AutoIndexThresholds.HNSWThreshold
	if hnswThreshold == 0 {
		hnswThreshold = DefaultHNSWThreshold
	}
	ivfpqThreshold := c.config.AutoIndexThresholds.IVFPQThreshold
	if ivfpqThreshold == 0 {
		ivfpqThreshold = DefaultIVFPQThreshold
	}
	optimalType := selectOptimalIndexType(currentSize, hnswThreshold, ivfpqThreshold)

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

	provider, _ := c.storage.(interface {
		GetByOrdinal(uint32) ([]float32, error)
		Distance([]float32, uint32) (float32, error)
	})
	updatedConfig := *c.config
	updatedConfig.IndexType = newType
	newIndex, err := buildIndexForEntries(ctx, &updatedConfig, provider, vectors)
	if err != nil {
		return fmt.Errorf("failed to build new index: %w", err)
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
			Ordinal:  entry.Ordinal,
			Vector:   make([]float32, len(entry.Vector)),
			Metadata: make(map[string]interface{}),
			Version:  entry.Version,
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

func recordFromIndexEntry(entry *index.VectorEntry) Record {
	if entry == nil {
		return Record{}
	}

	return Record{
		ID:       entry.ID,
		Vector:   cloneVector(entry.Vector),
		Metadata: cloneMetadata(entry.Metadata),
		Version:  entry.Version,
	}
}

func filterEntriesFromRecords(records []Record) []*filter.VectorEntry {
	entries := make([]*filter.VectorEntry, 0, len(records))
	for _, record := range records {
		entries = append(entries, &filter.VectorEntry{
			ID:       record.ID,
			Vector:   cloneVector(record.Vector),
			Metadata: cloneMetadata(record.Metadata),
		})
	}
	return entries
}

func cloneVector(vector []float32) []float32 {
	if vector == nil {
		return nil
	}
	return append([]float32(nil), vector...)
}

func cloneMetadata(metadata map[string]interface{}) map[string]interface{} {
	if metadata == nil {
		return nil
	}

	cloned := make(map[string]interface{}, len(metadata))
	for k, v := range metadata {
		cloned[k] = v
	}
	return cloned
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

	switch config.RawVectorStore {
	case "", "memory", "slabby":
	default:
		return fmt.Errorf("unsupported raw vector store backend: %s", config.RawVectorStore)
	}
	if config.RawVectorStore == "slabby" && config.RawStoreCap <= 0 {
		return fmt.Errorf("slabby raw store capacity must be positive, got %d", config.RawStoreCap)
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
