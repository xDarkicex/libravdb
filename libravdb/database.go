// Package libravdb provides a high-performance vector database library
// optimized for Go applications with HNSW indexing and LSM storage.
package libravdb

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/obs"
	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/libravdb/internal/storage/singlefile"
	"github.com/xDarkicex/memory"
)

// Logger is the logging interface accepted by the database.
// It is compatible with the standard library's log.Printf signature.
type Logger interface {
	Printf(format string, v ...interface{})
}

// Database represents the main vector database instance
type Database struct {
	storage       storage.Engine
	logger        Logger
	collections   map[string]*Collection
	bridge        *indexPersistenceBridge
	metrics       *obs.Metrics
	health        *obs.HealthChecker
	healthMonitor *SystemHealthMonitorImpl
	config        *Config
	scratchPool   *sync.Pool
	mu            sync.RWMutex
	closed        bool
}

// Config holds database-wide configuration
type Config struct {
	Logger               Logger
	StoragePath          string
	MaxCollections       int
	MaxConcurrentWrites  int
	MaxWriteQueueDepth   int
	AsyncIndexQueueDepth int
	AsyncIndexWorkers    int
	MetricsEnabled       bool
	TracingEnabled       bool
	Durability           DurabilityMode
	maxWritesExplicit    bool
	writeQueueExplicit   bool
}

// DurabilityMode controls when a successful write may be acknowledged.
type DurabilityMode uint8

const (
	// DurabilitySynchronous acknowledges writes only after the WAL group reaches
	// stable storage through file.Sync.
	DurabilitySynchronous DurabilityMode = iota
	// DurabilityUnsafeNoSync is an explicit benchmark-only mode. A successful
	// write may still be lost after power failure or kernel crash.
	DurabilityUnsafeNoSync
)

// Open opens a Database at the configured path, creating it if necessary.
func Open(opts ...Option) (*Database, error) {
	config := &Config{
		StoragePath:          "./data",
		MetricsEnabled:       true,
		TracingEnabled:       false,
		MaxCollections:       100,
		MaxConcurrentWrites:  defaultMaxConcurrentWrites(),
		MaxWriteQueueDepth:   32,
		AsyncIndexQueueDepth: 0,
		AsyncIndexWorkers:    min(4, runtime.GOMAXPROCS(0)),
		Durability:           DurabilitySynchronous,
	}

	// Apply options
	for _, opt := range opts {
		if err := opt(config); err != nil {
			return nil, fmt.Errorf("failed to apply option: %w", err)
		}
	}
	if config.AsyncIndexQueueDepth > 0 {
		if !config.maxWritesExplicit {
			config.MaxConcurrentWrites = 32
		}
		if !config.writeQueueExplicit {
			config.MaxWriteQueueDepth = config.AsyncIndexQueueDepth
		}
	}

	// Create the index persistence bridge so persisted indexes can be
	// deserialized during recovery (avoiding full rebuild from records).
	bridge := &indexPersistenceBridge{cache: make(map[string]index.Index)}
	storageOptions := []singlefile.Option{
		singlefile.WithIndexSnapshotProvider(bridge),
		singlefile.WithWALSync(config.Durability == DurabilitySynchronous),
	}
	if config.AsyncIndexQueueDepth > 0 {
		storageOptions = append(storageOptions, singlefile.WithWALGroupCommitTarget(min(28, config.MaxConcurrentWrites), 5*time.Millisecond))
	}
	storageEngine, err := singlefile.New(config.StoragePath, storageOptions...)

	if err != nil {
		if errors.Is(err, storage.ErrV1FormatMigrationRequired) {
			if err := Migrate(context.Background(), config.StoragePath); err != nil {
				return nil, fmt.Errorf("auto-migration failed: %w", err)
			}
			// Retry opening the newly migrated database
			storageEngine, err = singlefile.New(config.StoragePath, storageOptions...)
			if err != nil {
				return nil, fmt.Errorf("failed to open database after migration: %w", err)
			}
		} else {
			return nil, fmt.Errorf("failed to initialize storage engine: %w", err)
		}
	}

	// Initialize observability
	var metrics *obs.Metrics
	if config.MetricsEnabled {
		metrics = obs.NewMetrics()
	}

	db := &Database{
		collections: make(map[string]*Collection),
		storage:     storageEngine,
		bridge:      bridge,
		metrics:     metrics,
		config:      config,
		logger:      config.Logger,
		scratchPool: &sync.Pool{
			New: func() interface{} {
				arena, err := memory.NewArena(1024*1024, 64)
				if err != nil {
					panic(fmt.Sprintf("failed to allocate scratch arena: %v", err))
				}
				return arena
			},
		},
	}

	// Wire the bridge back to the database so SerializeIndex can access
	// collection indexes during checkpoint.
	bridge.mu.Lock()
	bridge.db = db
	bridge.mu.Unlock()

	// Initialize health checker
	db.health = obs.NewHealthChecker(db)

	// Start the background health monitor. Registers a storage engine
	// liveness check and begins periodic monitoring with callbacks.
	db.healthMonitor = NewSystemHealthMonitor(30 * time.Second)
	db.healthMonitor.RegisterHealthCheck("storage", func(ctx context.Context) (HealthLevel, error) {
		_, err := storageEngine.ListCollections()
		if err != nil {
			return HealthCritical, fmt.Errorf("storage list: %w", err)
		}
		return HealthHealthy, nil
	})
	if err := db.healthMonitor.Start(context.Background()); err != nil {
		storageEngine.Close()
		return nil, fmt.Errorf("failed to start health monitor: %w", err)
	}

	// Load existing collections from storage, preferring cached indexes
	// that were deserialized or rebuilt during recovery.
	if err := db.loadExistingCollections(context.Background(), bridge); err != nil {
		db.healthMonitor.Stop()
		storageEngine.Close()
		return nil, fmt.Errorf("failed to load existing collections: %w", err)
	}

	return db, nil
}

// SetLogger configures the database logger. It is safe to call concurrently.
func (db *Database) SetLogger(logger Logger) {
	db.mu.Lock()
	db.logger = logger
	db.mu.Unlock()
}

// CreateCollection creates a new collection with the specified options
func (db *Database) CreateCollection(ctx context.Context, name string, opts ...CollectionOption) (*Collection, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.closed {
		return nil, ErrDatabaseClosed
	}

	if _, exists := db.collections[name]; exists {
		return nil, fmt.Errorf("collection %s already exists: %w", name, ErrCollectionExists)
	}

	if len(db.collections) >= db.config.MaxCollections {
		return nil, ErrTooManyCollections
	}

	collection, err := newCollection(ctx, name, db.storage, db.metrics, db.newWriteController(), opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection: %w", err)
	}
	collection.db = db
	if err := db.configureAsyncIndex(collection); err != nil {
		_ = collection.Close()
		_ = db.storage.DeleteCollection(name)
		return nil, fmt.Errorf("failed to configure asynchronous index: %w", err)
	}

	db.collections[name] = collection
	return collection, nil
}

// EnsureCollection gets an existing collection, or creates it with the given options.
// If the collection exists but its dimension differs from the requested dimension,
// it returns a CollectionDimensionMismatchError without modifying the existing
// collection.
func (db *Database) EnsureCollection(ctx context.Context, name string, dimension int, opts ...CollectionOption) (*Collection, error) {
	return db.ensureCollection(ctx, name, dimension, false, opts...)
}

// EnsureCollectionRecreateOnDimensionMismatch gets an existing collection, or
// creates it with the given options. If the collection exists but its dimension
// differs from the requested dimension, it is dropped and recreated.
//
// Prefer EnsureCollection unless destructive recovery from a known-bad schema is
// explicitly intended.
func (db *Database) EnsureCollectionRecreateOnDimensionMismatch(ctx context.Context, name string, dimension int, opts ...CollectionOption) (*Collection, error) {
	return db.ensureCollection(ctx, name, dimension, true, opts...)
}

func (db *Database) ensureCollection(ctx context.Context, name string, dimension int, recreateOnDimensionMismatch bool, opts ...CollectionOption) (*Collection, error) {
	if dimension <= 0 {
		return nil, ErrInvalidDimension
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	if db.closed {
		return nil, ErrDatabaseClosed
	}

	// Fast path: collection exists with correct dimension.
	if col, exists := db.collections[name]; exists {
		if col.Dimension() == dimension {
			return col, nil
		}
		if !recreateOnDimensionMismatch {
			return nil, newCollectionDimensionMismatchError(name, col.Dimension(), dimension)
		}
		if err := db.deleteCollectionLocked(col, name); err != nil {
			return nil, fmt.Errorf("failed to drop mismatched collection %q: %w", name, err)
		}
		// Fall through to create.
	}

	col, err := db.createCollectionLocked(ctx, name, ensureCollectionOptions(dimension, opts)...)
	if err == nil {
		return col, nil
	}
	if errors.Is(err, ErrCollectionExists) {
		// Another caller raced and created it. Use it if dimension matches.
		if col, getErr := db.getCollectionLocked(name); getErr == nil {
			if col.Dimension() == dimension {
				return col, nil
			}
			return nil, newCollectionDimensionMismatchError(name, col.Dimension(), dimension)
		}
	}
	return nil, err
}

func newCollectionDimensionMismatchError(name string, existing, requested int) error {
	return &CollectionDimensionMismatchError{
		Collection:         name,
		ExistingDimension:  existing,
		RequestedDimension: requested,
	}
}

func ensureCollectionOptions(dimension int, opts []CollectionOption) []CollectionOption {
	createOpts := make([]CollectionOption, 0, len(opts)+1)
	createOpts = append(createOpts, opts...)
	createOpts = append(createOpts, WithDimension(dimension))
	return createOpts
}

// createCollectionLocked creates a collection. Caller must hold db.mu.
func (db *Database) createCollectionLocked(ctx context.Context, name string, opts ...CollectionOption) (*Collection, error) {
	collection, err := newCollection(ctx, name, db.storage, db.metrics, db.newWriteController(), opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection: %w", err)
	}
	collection.db = db
	if err := db.configureAsyncIndex(collection); err != nil {
		_ = collection.Close()
		_ = db.storage.DeleteCollection(name)
		return nil, fmt.Errorf("failed to configure asynchronous index: %w", err)
	}
	db.collections[name] = collection
	return collection, nil
}

// deleteCollectionLocked deletes a collection from memory and storage.
// Caller must hold db.mu. Returns error without altering the map on failure.
func (db *Database) deleteCollectionLocked(col *Collection, name string) error {
	delete(db.collections, name)
	if err := col.Close(); err != nil {
		db.collections[name] = col
		return fmt.Errorf("failed to close collection %s: %w", name, err)
	}
	isSharded := col.config.Sharded
	if isSharded {
		for _, shardName := range shardStorageNames(name) {
			if err := db.storage.DeleteCollection(shardName); err != nil {
				return fmt.Errorf("failed to delete shard %s: %w", shardName, err)
			}
		}
		return nil
	}
	if err := db.storage.DeleteCollection(name); err != nil {
		return err
	}
	return nil
}

// getCollectionLocked returns a collection from db.collections without storage lookup.
// Caller must hold db.mu.
func (db *Database) getCollectionLocked(name string) (*Collection, error) {
	if col, exists := db.collections[name]; exists {
		return col, nil
	}
	return nil, ErrCollectionNotFound
}

// GetCollection retrieves an existing collection by name
func (db *Database) GetCollection(name string) (*Collection, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.closed {
		return nil, ErrDatabaseClosed
	}

	// Check if collection is already loaded
	collection, exists := db.collections[name]
	if exists {
		return collection, nil
	}

	// Try to load collection from storage with configuration.
	if fileEngine, ok := db.storage.(interface {
		GetCollectionWithConfig(name string) (storage.Collection, *storage.CollectionConfig, error)
	}); ok {
		var err error
		collection, err = db.loadCollectionFromStorage(context.Background(), name, fileEngine, db.bridge)
		if err != nil {
			return nil, err
		}
	} else {
		return nil, fmt.Errorf("collection %s not found", name)
	}

	// Cache the collection
	db.collections[name] = collection

	return collection, nil
}

// ListCollections returns the names of all collections as a best-effort
// compatibility helper. Use ListCollectionsWithContext when you need explicit
// error reporting from storage-backed discovery.
func (db *Database) ListCollections() []string {
	names, _ := db.ListCollectionsWithContext(context.Background())
	return names
}

// ListCollectionsWithContext returns the names of all persisted collections.
func (db *Database) ListCollectionsWithContext(ctx context.Context) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	db.mu.RLock()
	if db.closed {
		db.mu.RUnlock()
		return nil, ErrDatabaseClosed
	}
	namesMap := make(map[string]struct{}, len(db.collections))
	for name := range db.collections {
		namesMap[name] = struct{}{}
	}
	db.mu.RUnlock()

	names, err := db.storage.ListCollections()
	if err != nil {
		return nil, err
	}
	for _, name := range names {
		// Filter out shard collection names - only return parent collection names
		if _, _, ok := parseShardName(name); !ok {
			namesMap[name] = struct{}{}
		}
	}

	result := make([]string, 0, len(namesMap))
	for name := range namesMap {
		result = append(result, name)
	}
	sort.Strings(result)
	return result, nil
}

// DeleteCollection removes a collection and its persisted data.
func (db *Database) DeleteCollection(ctx context.Context, name string) error {
	db.mu.Lock()
	if db.closed {
		db.mu.Unlock()
		return ErrDatabaseClosed
	}

	collection := db.collections[name]
	isSharded := collection != nil && collection.config.Sharded
	delete(db.collections, name)
	db.mu.Unlock()

	if collection != nil {
		if err := collection.Close(); err != nil {
			return fmt.Errorf("failed to close collection %s: %w", name, err)
		}
	}

	// For sharded collections, the parent collection doesn't exist in storage.
	// Only the hidden shard children exist. Delete all shard children.
	if isSharded {
		for _, shardName := range shardStorageNames(name) {
			if err := db.storage.DeleteCollection(shardName); err != nil {
				return fmt.Errorf("failed to delete shard %s: %w", shardName, err)
			}
		}
		return nil
	}

	// Non-sharded collection: delete the normal parent storage collection.
	if err := db.storage.DeleteCollection(name); err != nil {
		return err
	}

	return nil
}

// DeleteCollections removes multiple collections by exact name.
func (db *Database) DeleteCollections(ctx context.Context, names []string) error {
	var errs []error
	for _, name := range names {
		if err := db.DeleteCollection(ctx, name); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("delete collections failed: %v", errs)
	}
	return nil
}

// Health returns the current health status
func (db *Database) Health(ctx context.Context) (*obs.HealthStatus, error) {
	return db.health.Check(ctx)
}

// HealthMonitor returns the background system health monitor. Callers can
// register additional health checks and status-change callbacks.
func (db *Database) HealthMonitor() SystemHealthMonitor {
	return db.healthMonitor
}

// Ping checks if the database is responsive and the storage engine is accessible.
func (db *Database) Ping(ctx context.Context) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return ErrDatabaseClosed
	}

	// A basic check to see if storage responds
	_, err := db.storage.ListCollections()
	return err
}

// Stats returns database statistics
func (db *Database) Stats(ctx context.Context) *DatabaseStats {
	db.mu.RLock()
	defer db.mu.RUnlock()

	stats := &DatabaseStats{
		CollectionCount: len(db.collections),
		Collections:     make(map[string]*CollectionStats),
	}

	var totalMemory int64
	for name, collection := range db.collections {
		collectionStats := collection.Stats(ctx)
		stats.Collections[name] = collectionStats
		totalMemory += collectionStats.MemoryUsage
	}

	stats.MemoryUsage = totalMemory
	return stats
}

// OptimizeCollection performs optimization on a specific collection
func (db *Database) OptimizeCollection(ctx context.Context, name string, options *OptimizationOptions) error {
	collection, err := db.GetCollection(name)
	if err != nil {
		return fmt.Errorf("collection not found: %w", err)
	}

	return collection.OptimizeCollection(ctx, options)
}

// OptimizeAllCollections performs optimization on all collections
func (db *Database) OptimizeAllCollections(ctx context.Context, options *OptimizationOptions) error {
	db.mu.RLock()
	collections := make([]*Collection, 0, len(db.collections))
	for _, collection := range db.collections {
		collections = append(collections, collection)
	}
	db.mu.RUnlock()

	var errors []error
	for _, collection := range collections {
		if err := collection.OptimizeCollection(ctx, options); err != nil {
			errors = append(errors, fmt.Errorf("failed to optimize collection %s: %w", collection.name, err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("optimization errors: %v", errors)
	}

	return nil
}

// SetGlobalMemoryLimit sets a memory limit that applies to all collections
func (db *Database) SetGlobalMemoryLimit(bytes int64) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return ErrDatabaseClosed
	}

	// Distribute memory limit across collections
	collectionCount := len(db.collections)
	if collectionCount == 0 {
		return nil
	}

	perCollectionLimit := bytes / int64(collectionCount)

	var errors []error
	for _, collection := range db.collections {
		if err := collection.SetMemoryLimit(perCollectionLimit); err != nil {
			errors = append(errors, fmt.Errorf("failed to set memory limit for collection %s: %w", collection.name, err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("memory limit errors: %v", errors)
	}

	return nil
}

// GetGlobalMemoryUsage returns total memory usage across all collections
func (db *Database) GetGlobalMemoryUsage(ctx context.Context) (*GlobalMemoryUsage, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return nil, ErrDatabaseClosed
	}

	usage := &GlobalMemoryUsage{
		Collections: make(map[string]*CollectionMemoryStats),
		Timestamp:   time.Now(),
	}

	for name, collection := range db.collections {
		memUsage, err := collection.GetMemoryUsage(ctx)
		if err != nil {
			continue // Skip collections with errors
		}

		collectionMemStats := &CollectionMemoryStats{
			Total:         memUsage.Total,
			Index:         memUsage.Indices,
			Cache:         memUsage.Caches,
			Quantized:     memUsage.Quantized,
			MemoryMapped:  memUsage.MemoryMapped,
			Limit:         memUsage.Limit,
			Available:     memUsage.Available,
			PressureLevel: calculatePressureLevel(memUsage.Total, memUsage.Limit),
			Timestamp:     memUsage.Timestamp,
		}

		usage.Collections[name] = collectionMemStats
		usage.TotalMemory += memUsage.Total
		usage.TotalIndex += memUsage.Indices
		usage.TotalCache += memUsage.Caches
		usage.TotalQuantized += memUsage.Quantized
		usage.TotalMemoryMapped += memUsage.MemoryMapped
	}

	return usage, nil
}

// TriggerGlobalGC forces garbage collection across all collections
func (db *Database) TriggerGlobalGC() error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return ErrDatabaseClosed
	}

	var errors []error
	for name, collection := range db.collections {
		if err := collection.TriggerGC(); err != nil {
			errors = append(errors, fmt.Errorf("failed to trigger GC for collection %s: %w", name, err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("GC errors: %v", errors)
	}

	return nil
}

// loadExistingCollections discovers and loads existing collections from storage
func (db *Database) loadExistingCollections(ctx context.Context, bridge *indexPersistenceBridge) error {
	fileEngine, ok := db.storage.(interface {
		ListCollections() ([]string, error)
		GetCollectionWithConfig(name string) (storage.Collection, *storage.CollectionConfig, error)
	})
	if !ok {
		return nil
	}

	names, err := fileEngine.ListCollections()
	if err != nil {
		return fmt.Errorf("failed to list collections: %w", err)
	}

	// Track which parent collections we've loaded (to avoid loading shards as separate collections)
	loadedParents := make(map[string]bool)
	loadedCollections := make(map[string]*Collection)

	for _, name := range names {
		// Skip shard collection names - they are loaded as part of the parent
		if _, _, ok := parseShardName(name); ok {
			continue
		}

		// Skip if already loaded as a parent (from a previous shard entry)
		if loadedParents[name] {
			continue
		}

		collection, err := db.loadCollectionFromStorage(ctx, name, fileEngine, bridge)
		if err != nil {
			for _, c := range loadedCollections {
				c.Close()
			}
			if bridge != nil {
				bridge.closeCachedIndexes()
			}
			return err
		}

		loadedCollections[name] = collection
		loadedParents[name] = true
	}

	db.mu.Lock()
	for name, collection := range loadedCollections {
		db.collections[name] = collection
	}
	db.mu.Unlock()
	if bridge != nil {
		bridge.closeCachedIndexes()
	}

	return nil
}

func (db *Database) loadCollectionFromStorage(ctx context.Context, name string, engine interface {
	GetCollectionWithConfig(name string) (storage.Collection, *storage.CollectionConfig, error)
}, bridge *indexPersistenceBridge) (*Collection, error) {
	// Check if this is a shard collection name - if so, skip it
	if _, _, ok := parseShardName(name); ok {
		return nil, fmt.Errorf("cannot load shard collection %s directly, load parent collection instead", name)
	}

	// Check if this collection has shards (sharded collection)
	shardNames := shardStorageNames(name)
	firstShardStorage, config, err := engine.GetCollectionWithConfig(shardNames[0])
	if err == nil {
		// Shard 0 exists - this is a sharded collection
		// Load all shards
		shardStorages := make([]storage.Collection, shardCount)
		shardStorages[0] = firstShardStorage

		for i := 1; i < shardCount; i++ {
			shardStorages[i], _, err = engine.GetCollectionWithConfig(shardNames[i])
			if err != nil {
				for j := 0; j < i; j++ {
					shardStorages[j].Close()
				}
				return nil, fmt.Errorf("collection %s is missing shard %d: %w", name, i, err)
			}
		}

		collection, err := newShardedCollectionFromStorage(ctx, name, shardStorages, config, db.metrics, db.newWriteController())
		if err != nil {
			for j := 0; j < shardCount; j++ {
				shardStorages[j].Close()
			}
			return nil, fmt.Errorf("failed to create sharded collection from storage: %w", err)
		}
		collection.db = db
		if err := db.configureAsyncIndex(collection); err != nil {
			_ = collection.Close()
			return nil, fmt.Errorf("failed to configure asynchronous index: %w", err)
		}
		return collection, nil
	}

	// Not a sharded collection - load as single collection
	storageCollection, config, err := engine.GetCollectionWithConfig(name)
	if err != nil {
		return nil, fmt.Errorf("collection %s not found", name)
	}

	// Prefer a cached index that was deserialized or rebuilt during recovery,
	// avoiding an expensive full rebuild from storage records.
	var cachedIndex index.Index
	if bridge != nil {
		cachedIndex = bridge.takeCachedIndex(name)
	}

	collection, err := newCollectionFromStorage(ctx, name, storageCollection, db.metrics, config, db.newWriteController(), cachedIndex)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection from storage: %w", err)
	}
	collection.db = db
	if err := db.configureAsyncIndex(collection); err != nil {
		_ = collection.Close()
		return nil, fmt.Errorf("failed to configure asynchronous index: %w", err)
	}

	return collection, nil
}

func (db *Database) configureAsyncIndex(collection *Collection) error {
	if db.config.AsyncIndexQueueDepth == 0 || collection == nil || collection.shards != nil || collection.config.IndexType != HNSW {
		return nil
	}
	queue, err := newAsyncIndexQueue(collection, db.config.AsyncIndexQueueDepth, db.config.AsyncIndexWorkers)
	if err != nil {
		return err
	}
	collection.asyncIndex = queue
	return nil
}

func (db *Database) newWriteController() *writeController {
	return newWriteController(db.config.MaxConcurrentWrites, db.config.MaxWriteQueueDepth)
}

func defaultMaxConcurrentWrites() int {
	procs := runtime.GOMAXPROCS(0)
	if procs < 1 {
		return 1
	}
	if procs > 2 {
		return 2
	}
	return procs
}

// Close gracefully shuts down the database
func (db *Database) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.closed {
		return nil
	}

	var errors []error

	// Stop the background health monitor before tearing down collections
	// so health checks don't access closing state.
	if db.healthMonitor != nil {
		if err := db.healthMonitor.Stop(); err != nil {
			errors = append(errors, fmt.Errorf("health monitor stop: %w", err))
		}
	}

	// Close all collections
	for _, collection := range db.collections {
		if err := collection.Close(); err != nil {
			errors = append(errors, err)
		}
	}

	// Close storage
	if err := db.storage.Close(); err != nil {
		errors = append(errors, err)
	}

	db.closed = true

	if len(errors) > 0 {
		return fmt.Errorf("errors during shutdown: %v", errors)
	}

	return nil
}

// Vacuum reclaims disk space by rewriting the underlying storage file, dropping
// deleted records and obsolete WAL frames. This is a non-blocking operation
// that only briefly pauses the database during the final swap.
func (db *Database) Vacuum(ctx context.Context) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return ErrDatabaseClosed
	}

	if v, ok := db.storage.(interface{ Vacuum(context.Context) error }); ok {
		return v.Vacuum(ctx)
	}

	return fmt.Errorf("underlying storage engine does not support Vacuum")
}

// Backup creates a point-in-time copy of the database to the specified destination
// path. It uses a non-blocking fast-forward design to ensure the copy is consistent
// without interrupting active database operations.
func (db *Database) Backup(ctx context.Context, destPath string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return ErrDatabaseClosed
	}

	if v, ok := db.storage.(interface {
		Backup(context.Context, string) error
	}); ok {
		return v.Backup(ctx, destPath)
	}

	return fmt.Errorf("underlying storage engine does not support Backup")
}

// Drop completely closes the database and destroys its underlying files from disk.
// Once a database is dropped, it cannot be recovered without a backup.
func (db *Database) Drop(ctx context.Context) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.closed {
		return ErrDatabaseClosed
	}

	// Close all collections safely
	for _, collection := range db.collections {
		collection.Close()
	}
	db.collections = make(map[string]*Collection)

	// Stop health monitor
	if db.healthMonitor != nil {
		db.healthMonitor.Stop()
	}

	db.closed = true

	if v, ok := db.storage.(interface{ Drop(context.Context) error }); ok {
		return v.Drop(ctx)
	}

	return fmt.Errorf("underlying storage engine does not support Drop")
}
