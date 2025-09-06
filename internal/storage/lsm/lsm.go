package lsm

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/libravdb/internal/storage/wal"
)

// Engine implements storage.Engine using WAL for persistence
// and in-memory cache for fast lookups (Phase 1 implementation)
type Engine struct {
	mu          sync.RWMutex
	basePath    string
	collections map[string]*Collection
}

// New creates a new LSM storage engine at the specified path
func New(basePath string) (storage.Engine, error) {
	// Create base directory if it doesn't exist
	if err := os.MkdirAll(basePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create storage directory: %w", err)
	}

	engine := &Engine{
		basePath:    basePath,
		collections: make(map[string]*Collection),
	}

	// Discover existing collections on startup
	if err := engine.loadExistingCollections(); err != nil {
		return nil, fmt.Errorf("failed to load existing collections: %w", err)
	}

	return engine, nil
}

// CreateCollection creates a new collection with WAL persistence
func (e *Engine) CreateCollection(name string, config interface{}) (storage.Collection, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, exists := e.collections[name]; exists {
		return nil, fmt.Errorf("collection %s already exists", name)
	}

	// Create collection directory
	collectionPath := filepath.Join(e.basePath, name)
	if err := os.MkdirAll(collectionPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create collection directory: %w", err)
	}

	// Create WAL for this collection
	walPath := filepath.Join(collectionPath, "wal.log")
	walInstance, err := wal.New(walPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create WAL: %w", err)
	}

	collection := &Collection{
		name:    name,
		path:    collectionPath,
		walPath: walPath,
		wal:     walInstance,
		cache:   make(map[string]*index.VectorEntry),
	}

	// Save configuration if provided
	if collectionConfig, ok := config.(*CollectionConfig); ok && collectionConfig != nil {
		if err := collection.saveConfig(collectionConfig); err != nil {
			collection.Close() // Clean up on failure
			os.RemoveAll(collectionPath)
			return nil, fmt.Errorf("failed to save collection config: %w", err)
		}
	}

	if err := collection.recoverFromWAL(); err != nil {
		collection.Close() // Clean up on failure
		os.RemoveAll(collectionPath)
		return nil, fmt.Errorf("failed to recover collection %s: %w", name, err)
	}

	e.collections[name] = collection
	return collection, nil
}

// GetCollection retrieves an existing collection
func (e *Engine) GetCollection(name string) (storage.Collection, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	collection, exists := e.collections[name]
	if !exists {
		return nil, fmt.Errorf("collection %s not found", name)
	}

	return collection, nil
}

// GetCollectionWithConfig retrieves an existing collection and its configuration
func (e *Engine) GetCollectionWithConfig(name string) (storage.Collection, *CollectionConfig, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	collection, exists := e.collections[name]
	if !exists {
		return nil, nil, fmt.Errorf("collection %s not found", name)
	}

	// Load configuration
	config, err := loadConfig(collection.path)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load collection config: %w", err)
	}

	return collection, config, nil
}

// Close shuts down the LSM engine and all collections
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	var errors []error

	// Close all collections
	for _, collection := range e.collections {
		if err := collection.Close(); err != nil {
			errors = append(errors, fmt.Errorf("failed to close collection %s: %w", collection.name, err))
		}
	}

	e.collections = nil

	if len(errors) > 0 {
		return fmt.Errorf("errors during engine shutdown: %v", errors)
	}

	return nil
}

// loadExistingCollections discovers and loads existing collections on startup
func (e *Engine) loadExistingCollections() error {
	entries, err := os.ReadDir(e.basePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No existing collections
		}
		return fmt.Errorf("failed to read storage directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		collectionName := entry.Name()
		collectionPath := filepath.Join(e.basePath, collectionName)
		walPath := filepath.Join(collectionPath, "wal.log")

		// Check if WAL file exists
		if _, err := os.Stat(walPath); os.IsNotExist(err) {
			continue // Skip directories without WAL
		}

		// Open WAL for this collection
		walInstance, err := wal.New(walPath)
		if err != nil {
			return fmt.Errorf("failed to open WAL for collection %s: %w", collectionName, err)
		}

		collection := &Collection{
			name:    collectionName,
			path:    collectionPath,
			walPath: walPath,
			wal:     walInstance,
			cache:   make(map[string]*index.VectorEntry),
		}

		// Recover data from WAL into cache
		if err := collection.recoverFromWAL(); err != nil {
			return fmt.Errorf("failed to recover collection %s: %w", collectionName, err)
		}

		e.collections[collectionName] = collection
	}

	return nil
}
