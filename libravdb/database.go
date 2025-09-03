// Package libravdb provides a high-performance vector database library
// optimized for Go applications with HNSW indexing and LSM storage.
package libravdb

import (
	"context"
	"fmt"
	"sync"

	"github.com/xDarkicex/libravdb/internal/obs"
	"github.com/xDarkicex/libravdb/internal/storage"
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
	StoragePath    string
	MetricsEnabled bool
	TracingEnabled bool
	MaxCollections int
}

// New creates a new Database instance with the given options
func New(opts ...Option) (*Database, error) {
	config := &Config{
		StoragePath:    "./data",
		MetricsEnabled: true,
		TracingEnabled: false,
		MaxCollections: 100,
	}

	// Apply options
	for _, opt := range opts {
		if err := opt(config); err != nil {
			return nil, fmt.Errorf("failed to apply option: %w", err)
		}
	}

	// Initialize storage engine
	storageEngine, err := storage.NewLSM(config.StoragePath)
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

	collection, err := newCollection(name, db.storage, db.metrics, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection: %w", err)
	}

	db.collections[name] = collection
	return collection, nil
}

// GetCollection retrieves an existing collection by name
func (db *Database) GetCollection(name string) (*Collection, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return nil, ErrDatabaseClosed
	}

	collection, exists := db.collections[name]
	if !exists {
		return nil, fmt.Errorf("collection %s not found", name)
	}

	return collection, nil
}

// ListCollections returns the names of all collections
func (db *Database) ListCollections() []string {
	db.mu.RLock()
	defer db.mu.RUnlock()

	names := make([]string, 0, len(db.collections))
	for name := range db.collections {
		names = append(names, name)
	}
	return names
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

	for name, collection := range db.collections {
		stats.Collections[name] = collection.Stats()
	}

	return stats
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
