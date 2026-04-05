// Package libravdb provides a high-performance vector database library
// optimized for Go applications with HNSW indexing and LSM storage.
package libravdb

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/obs"
	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/libravdb/internal/storage/singlefile"
)

// Database represents the main vector database instance
type Database struct {
	mu          sync.RWMutex
	collections map[string]*Collection
	storage     storage.Engine
	metrics     *obs.Metrics
	health      *obs.HealthChecker
	config      *Config
	closed      bool
}

// Config holds database-wide configuration
type Config struct {
	StoragePath         string
	MetricsEnabled      bool
	TracingEnabled      bool
	MaxCollections      int
	MaxConcurrentWrites int
	MaxWriteQueueDepth  int
}

// New creates a new Database instance with the given options
func New(opts ...Option) (*Database, error) {
	config := &Config{
		StoragePath:         "./data",
		MetricsEnabled:      true,
		TracingEnabled:      false,
		MaxCollections:      100,
		MaxConcurrentWrites: defaultMaxConcurrentWrites(),
		MaxWriteQueueDepth:  32,
	}

	// Apply options
	for _, opt := range opts {
		if err := opt(config); err != nil {
			return nil, fmt.Errorf("failed to apply option: %w", err)
		}
	}

	storageEngine, err := singlefile.New(config.StoragePath)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize storage: %w", err)
	}

	// Initialize observability
	var metrics *obs.Metrics
	if config.MetricsEnabled {
		metrics = obs.NewMetrics()
	}

	db := &Database{
		collections: make(map[string]*Collection),
		storage:     storageEngine,
		metrics:     metrics,
		config:      config,
	}

	// Initialize health checker
	db.health = obs.NewHealthChecker(db)

	// Load existing collections from storage
	if err := db.loadExistingCollections(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to load existing collections: %w", err)
	}

	return db, nil
}

// CreateCollection creates a new collection with the specified options
func (db *Database) CreateCollection(ctx context.Context, name string, opts ...CollectionOption) (*Collection, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.closed {
		return nil, ErrDatabaseClosed
	}

	if _, exists := db.collections[name]; exists {
		return nil, fmt.Errorf("collection %s already exists", name)
	}

	if len(db.collections) >= db.config.MaxCollections {
		return nil, ErrTooManyCollections
	}

	collection, err := newCollection(name, db.storage, db.metrics, db.newWriteController(), opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection: %w", err)
	}
	collection.db = db

	db.collections[name] = collection
	return collection, nil
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
		collection, err = db.loadCollectionFromStorage(name, fileEngine)
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
		namesMap[name] = struct{}{}
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
	delete(db.collections, name)
	db.mu.Unlock()

	if collection != nil {
		if err := collection.Close(); err != nil {
			return fmt.Errorf("failed to close collection %s: %w", name, err)
		}
	}

	if err := db.storage.DeleteCollection(name); err != nil {
		return err
	}

	_ = ctx
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

// Stats returns database statistics
func (db *Database) Stats() *DatabaseStats {
	db.mu.RLock()
	defer db.mu.RUnlock()

	stats := &DatabaseStats{
		CollectionCount: len(db.collections),
		Collections:     make(map[string]*CollectionStats),
	}

	var totalMemory int64
	for name, collection := range db.collections {
		collectionStats := collection.Stats()
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
func (db *Database) GetGlobalMemoryUsage() (*GlobalMemoryUsage, error) {
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
		memUsage, err := collection.GetMemoryUsage()
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
			PressureLevel: "normal", // TODO: Calculate pressure level
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
func (db *Database) loadExistingCollections(ctx context.Context) error {
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

	for _, name := range names {
		collection, err := db.loadCollectionFromStorage(name, fileEngine)
		if err != nil {
			return err
		}
		db.collections[name] = collection
	}

	_ = ctx
	return nil
}

func (db *Database) loadCollectionFromStorage(name string, engine interface {
	GetCollectionWithConfig(name string) (storage.Collection, *storage.CollectionConfig, error)
}) (*Collection, error) {
	storageCollection, config, err := engine.GetCollectionWithConfig(name)
	if err != nil {
		return nil, fmt.Errorf("collection %s not found", name)
	}

	collection, err := newCollectionFromStorage(name, storageCollection, db.metrics, config, db.newWriteController())
	if err != nil {
		return nil, fmt.Errorf("failed to create collection from storage: %w", err)
	}
	collection.db = db

	return collection, nil
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
