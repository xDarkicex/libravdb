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
	NClusters      int     `json:"n_clusters,omitempty"`
	NProbes        int     `json:"n_probes,omitempty"`
	ML             float64 `json:"ml"`
	Version        int     `json:"version"`
	RawVectorStore string  `json:"raw_vector_store,omitempty"`
	RawStoreCap    int     `json:"raw_store_cap,omitempty"`
}

// Collection represents a storage collection with WAL persistence and in-memory cache
type Collection struct {
	mu          sync.RWMutex
	name        string
	path        string
	walPath     string
	wal         *wal.WAL
	cache       map[string]*index.VectorEntry
	ordinalToID []string
	nextOrdinal uint32
	closed      bool
}

func (c *Collection) AssignOrdinals(ctx context.Context, entries []*index.VectorEntry) error {
	_ = ctx
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, entry := range entries {
		if current, ok := c.cache[entry.ID]; ok {
			entry.Ordinal = current.Ordinal
			continue
		}
		entry.Ordinal = c.nextOrdinal
		c.nextOrdinal++
	}
	return nil
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
		Ordinal:  entry.Ordinal,
		Vector:   make([]float32, len(entry.Vector)),
		Metadata: entry.Metadata,
	}
	copy(c.cache[entry.ID].Vector, entry.Vector)
	if int(entry.Ordinal) >= len(c.ordinalToID) {
		grown := make([]string, entry.Ordinal+1)
		copy(grown, c.ordinalToID)
		c.ordinalToID = grown
	}
	c.ordinalToID[entry.Ordinal] = entry.ID

	return nil
}

// InsertBatch persists multiple vector entries with a single WAL flush/sync.
func (c *Collection) InsertBatch(ctx context.Context, entries []*index.VectorEntry) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return fmt.Errorf("collection %s is closed", c.name)
	}

	if len(entries) == 0 {
		return nil
	}

	walEntries := make([]*wal.Entry, 0, len(entries))
	cacheUpdates := make([]*index.VectorEntry, 0, len(entries))
	for _, entry := range entries {
		if entry == nil {
			return fmt.Errorf("entry cannot be nil")
		}

		walEntries = append(walEntries, &wal.Entry{
			Operation: wal.OpInsert,
			ID:        entry.ID,
			Vector:    entry.Vector,
			Metadata:  entry.Metadata,
		})

		cacheEntry := &index.VectorEntry{
			ID:       entry.ID,
			Ordinal:  entry.Ordinal,
			Vector:   make([]float32, len(entry.Vector)),
			Metadata: entry.Metadata,
		}
		copy(cacheEntry.Vector, entry.Vector)
		cacheUpdates = append(cacheUpdates, cacheEntry)
	}

	if err := c.wal.AppendBatch(ctx, walEntries); err != nil {
		return fmt.Errorf("failed to write batch to WAL: %w", err)
	}

	for _, entry := range cacheUpdates {
		c.cache[entry.ID] = entry
		if int(entry.Ordinal) >= len(c.ordinalToID) {
			grown := make([]string, entry.Ordinal+1)
			copy(grown, c.ordinalToID)
			c.ordinalToID = grown
		}
		c.ordinalToID[entry.Ordinal] = entry.ID
	}

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
		Ordinal:  entry.Ordinal,
		Vector:   append([]float32(nil), entry.Vector...),
		Metadata: entry.Metadata,
	}, nil
}

func (c *Collection) Exists(ctx context.Context, id string) (bool, error) {
	_ = ctx
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return false, fmt.Errorf("collection %s is closed", c.name)
	}

	_, exists := c.cache[id]
	return exists, nil
}

func (c *Collection) GetIDByOrdinal(ctx context.Context, ordinal uint32) (string, error) {
	_ = ctx
	c.mu.RLock()
	defer c.mu.RUnlock()
	if int(ordinal) >= len(c.ordinalToID) || c.ordinalToID[ordinal] == "" {
		return "", fmt.Errorf("ordinal %d not found", ordinal)
	}
	return c.ordinalToID[ordinal], nil
}

func (c *Collection) MemoryUsage(ctx context.Context) (int64, error) {
	_ = ctx
	c.mu.RLock()
	defer c.mu.RUnlock()

	var usage int64
	for id, entry := range c.cache {
		if entry == nil {
			continue
		}
		usage += int64(len(id))
		usage += int64(len(entry.Vector) * 4)
		for key, value := range entry.Metadata {
			usage += int64(len(key))
			switch typed := value.(type) {
			case string:
				usage += int64(len(typed))
			case []string:
				for _, item := range typed {
					usage += int64(len(item))
				}
			}
		}
	}
	return usage, nil
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

// Count returns the exact number of live entries in the collection.
func (c *Collection) Count(ctx context.Context) (int, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return 0, fmt.Errorf("collection %s is closed", c.name)
	}

	return len(c.cache), nil
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
