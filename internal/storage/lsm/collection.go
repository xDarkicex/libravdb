package lsm

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage/wal"
)

// CollectionConfig holds collection-specific configuration (mirrors libravdb.CollectionConfig)
type CollectionConfig struct {
	Dimension      int     `json:"dimension"`
	Metric         int     `json:"metric"`     // DistanceMetric as int
	IndexType      int     `json:"index_type"` // IndexType as int
	M              int     `json:"m"`
	EfConstruction int     `json:"ef_construction"`
	EfSearch       int     `json:"ef_search"`
	ML             float64 `json:"ml"`
	Version        int     `json:"version"`
}

// Collection represents a storage collection with WAL persistence and in-memory cache
type Collection struct {
	mu      sync.RWMutex
	name    string
	path    string
	walPath string
	wal     *wal.WAL
	cache   map[string]*index.VectorEntry
	closed  bool
}

// Insert persists a vector entry to WAL and updates the in-memory cache
func (c *Collection) Insert(ctx context.Context, entry *index.VectorEntry) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return fmt.Errorf("collection %s is closed", c.name)
	}

	// Create WAL entry for persistence
	walEntry := &wal.Entry{
		Operation: wal.OpInsert,
		ID:        entry.ID,
		Vector:    entry.Vector,
		Metadata:  entry.Metadata,
	}

	// Write to WAL first for durability
	if err := c.wal.Append(ctx, walEntry); err != nil {
		return fmt.Errorf("failed to write to WAL: %w", err)
	}

	// Update in-memory cache
	c.cache[entry.ID] = &index.VectorEntry{
		ID:       entry.ID,
		Vector:   make([]float32, len(entry.Vector)),
		Metadata: entry.Metadata,
	}
	copy(c.cache[entry.ID].Vector, entry.Vector)

	return nil
}

// Get retrieves a vector entry from the in-memory cache
func (c *Collection) Get(ctx context.Context, id string) (*index.VectorEntry, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, fmt.Errorf("collection %s is closed", c.name)
	}

	entry, exists := c.cache[id]
	if !exists {
		return nil, fmt.Errorf("entry %s not found", id)
	}

	// Return a copy to prevent external modifications
	return &index.VectorEntry{
		ID:       entry.ID,
		Vector:   append([]float32(nil), entry.Vector...),
		Metadata: entry.Metadata,
	}, nil
}

// Delete removes a vector entry by persisting delete operation and updating cache
func (c *Collection) Delete(ctx context.Context, id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return fmt.Errorf("collection %s is closed", c.name)
	}

	// Check if entry exists
	if _, exists := c.cache[id]; !exists {
		return fmt.Errorf("entry %s does not exist", id)
	}

	// Create WAL entry for delete operation
	walEntry := &wal.Entry{
		Operation: wal.OpDelete,
		ID:        id,
		Vector:    nil, // No need to store vector for delete
		Metadata:  nil, // No need to store metadata for delete
	}

	// Write delete operation to WAL
	if err := c.wal.Append(ctx, walEntry); err != nil {
		return fmt.Errorf("failed to write delete to WAL: %w", err)
	}

	// Remove from in-memory cache
	delete(c.cache, id)

	return nil
}

// Close shuts down the collection and releases resources
func (c *Collection) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}

	var errors []error

	// Close WAL
	if c.wal != nil {
		if err := c.wal.Close(); err != nil {
			errors = append(errors, fmt.Errorf("failed to close WAL: %w", err))
		}
	}

	// Clear cache
	c.cache = nil
	c.closed = true

	if len(errors) > 0 {
		return fmt.Errorf("errors during collection close: %v", errors)
	}

	return nil
}

// Iterate calls the provided function for each entry in the collection
func (c *Collection) Iterate(ctx context.Context, fn func(*index.VectorEntry) error) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return fmt.Errorf("collection %s is closed", c.name)
	}

	for _, entry := range c.cache {
		if err := fn(entry); err != nil {
			return err
		}
	}

	return nil
}

// saveConfig saves the collection configuration to disk
func (c *Collection) saveConfig(config *CollectionConfig) error {
	configPath := filepath.Join(c.path, "config.json")

	// Add version for future compatibility
	config.Version = 1

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(configPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// loadConfig loads the collection configuration from disk
func loadConfig(collectionPath string) (*CollectionConfig, error) {
	configPath := filepath.Join(collectionPath, "config.json")

	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			// For backward compatibility, return default config for existing collections
			// without config files
			return &CollectionConfig{
				Dimension:      3, // This should be detected from data, but fallback
				Metric:         1, // CosineDistance
				IndexType:      0, // HNSW
				M:              32,
				EfConstruction: 200,
				EfSearch:       50,
				ML:             1.0 / math.Log(2.0),
				Version:        1,
			}, nil
		}
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config CollectionConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &config, nil
}

// Stats returns collection statistics
func (c *Collection) Stats() CollectionStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return CollectionStats{
		Name:        c.name,
		EntryCount:  len(c.cache),
		WALPath:     c.walPath,
		MemoryUsage: c.estimateMemoryUsage(),
	}
}

// CollectionStats holds collection statistics
type CollectionStats struct {
	Name        string
	EntryCount  int
	WALPath     string
	MemoryUsage int64
}

// recoverFromWAL rebuilds the in-memory cache from WAL entries
func (c *Collection) recoverFromWAL() error {
	entries, err := c.wal.Read()
	if err != nil {
		return fmt.Errorf("failed to read WAL entries: %w", err)
	}

	recoveredCount := 0
	for _, entry := range entries {
		switch entry.Operation {
		case wal.OpInsert, wal.OpUpdate:
			c.cache[entry.ID] = &index.VectorEntry{
				ID:       entry.ID,
				Vector:   append([]float32(nil), entry.Vector...),
				Metadata: entry.Metadata,
			}
			recoveredCount++

		case wal.OpDelete:
			delete(c.cache, entry.ID)
			recoveredCount++

		default:
			return fmt.Errorf("unknown WAL operation: %v", entry.Operation)
		}
	}

	fmt.Printf("Recovered %d operations for collection %s (%d entries in cache)\n",
		recoveredCount, c.name, len(c.cache))

	return nil
}

// estimateMemoryUsage calculates approximate memory usage
func (c *Collection) estimateMemoryUsage() int64 {
	var usage int64

	for _, entry := range c.cache {
		// ID string
		usage += int64(len(entry.ID))
		// Vector (4 bytes per float32)
		usage += int64(len(entry.Vector) * 4)
		// Metadata (rough estimate)
		usage += 64 // Base overhead per entry
	}

	return usage
}
