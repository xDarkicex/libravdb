package flat

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

// VectorEntry represents a vector entry in the flat index
type VectorEntry struct {
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata"`
}

// SearchResult represents a search result from the flat index
type SearchResult struct {
	ID       string                 `json:"id"`
	Score    float32                `json:"score"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata"`
}

// Config holds configuration for the flat index
type Config struct {
	Dimension    int                       `json:"dimension"`
	Metric       util.DistanceMetric       `json:"metric"`
	Quantization *quant.QuantizationConfig `json:"quantization,omitempty"`
}

// PersistenceMetadata holds metadata about persisted flat index
type PersistenceMetadata struct {
	Version       uint32    `json:"version"`
	NodeCount     int       `json:"node_count"`
	Dimension     int       `json:"dimension"`
	MaxLevel      int       `json:"max_level"` // Always 0 for flat index
	IndexType     string    `json:"index_type"`
	CreatedAt     time.Time `json:"created_at"`
	ChecksumCRC32 uint32    `json:"checksum_crc32"`
	FileSize      int64     `json:"file_size"`
}

// Index implements a flat (brute-force) vector index
type Index struct {
	config    *Config
	vectors   []*VectorEntry
	idToIndex map[string]int
	quantizer quant.Quantizer
	mu        sync.RWMutex
}

// NewFlat creates a new flat index
func NewFlat(config *Config) (*Index, error) {
	if config.Dimension <= 0 {
		return nil, fmt.Errorf("dimension must be positive, got %d", config.Dimension)
	}

	index := &Index{
		config:    config,
		vectors:   make([]*VectorEntry, 0),
		idToIndex: make(map[string]int),
	}

	// Initialize quantizer if configured
	if config.Quantization != nil {
		var err error
		index.quantizer, err = quant.Create(config.Quantization)
		if err != nil {
			return nil, fmt.Errorf("failed to create quantizer: %w", err)
		}
	}

	return index, nil
}

// Insert adds a vector to the index
func (idx *Index) Insert(ctx context.Context, entry *VectorEntry) error {
	if len(entry.Vector) != idx.config.Dimension {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d",
			idx.config.Dimension, len(entry.Vector))
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Check if ID already exists
	if existingIndex, exists := idx.idToIndex[entry.ID]; exists {
		// Update existing entry
		idx.vectors[existingIndex] = &VectorEntry{
			ID:       entry.ID,
			Vector:   make([]float32, len(entry.Vector)),
			Metadata: make(map[string]interface{}),
		}
		copy(idx.vectors[existingIndex].Vector, entry.Vector)
		for k, v := range entry.Metadata {
			idx.vectors[existingIndex].Metadata[k] = v
		}
		return nil
	}

	// Add new entry
	newEntry := &VectorEntry{
		ID:       entry.ID,
		Vector:   make([]float32, len(entry.Vector)),
		Metadata: make(map[string]interface{}),
	}
	copy(newEntry.Vector, entry.Vector)
	for k, v := range entry.Metadata {
		newEntry.Metadata[k] = v
	}

	idx.idToIndex[entry.ID] = len(idx.vectors)
	idx.vectors = append(idx.vectors, newEntry)

	return nil
}

// Search performs brute-force search across all vectors
func (idx *Index) Search(ctx context.Context, query []float32, k int) ([]*SearchResult, error) {
	if len(query) != idx.config.Dimension {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d",
			idx.config.Dimension, len(query))
	}

	if k <= 0 {
		return []*SearchResult{}, nil
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(idx.vectors) == 0 {
		return []*SearchResult{}, nil
	}

	// Compute distance to all vectors and collect results
	allResults := make([]*SearchResult, 0, len(idx.vectors))

	for _, entry := range idx.vectors {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		distance, err := idx.computeDistance(query, entry.Vector)
		if err != nil {
			return nil, fmt.Errorf("failed to compute distance: %w", err)
		}

		result := &SearchResult{
			ID:       entry.ID,
			Score:    distance,
			Vector:   make([]float32, len(entry.Vector)),
			Metadata: make(map[string]interface{}),
		}
		copy(result.Vector, entry.Vector)
		for k, v := range entry.Metadata {
			result.Metadata[k] = v
		}

		allResults = append(allResults, result)
	}

	// Sort results by distance (ascending for similarity search)
	for i := 0; i < len(allResults)-1; i++ {
		for j := i + 1; j < len(allResults); j++ {
			if allResults[i].Score > allResults[j].Score {
				allResults[i], allResults[j] = allResults[j], allResults[i]
			}
		}
	}

	// Return top-k results
	if k > len(allResults) {
		k = len(allResults)
	}
	results := make([]*SearchResult, k)
	copy(results, allResults[:k])

	return results, nil
}

// Delete removes a vector from the index
func (idx *Index) Delete(ctx context.Context, id string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	index, exists := idx.idToIndex[id]
	if !exists {
		return fmt.Errorf("vector with ID %s not found", id)
	}

	// Remove from vectors slice
	idx.vectors = append(idx.vectors[:index], idx.vectors[index+1:]...)

	// Update idToIndex map
	delete(idx.idToIndex, id)
	for i := index; i < len(idx.vectors); i++ {
		idx.idToIndex[idx.vectors[i].ID] = i
	}

	return nil
}

// Size returns the number of vectors in the index
func (idx *Index) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vectors)
}

// MemoryUsage estimates the memory usage of the index
func (idx *Index) MemoryUsage() int64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	var usage int64

	// Vector storage
	usage += int64(len(idx.vectors)) * int64(idx.config.Dimension) * 4 // 4 bytes per float32

	// ID storage (estimate 20 bytes per ID on average)
	usage += int64(len(idx.vectors)) * 20

	// Index map overhead (estimate 32 bytes per entry)
	usage += int64(len(idx.idToIndex)) * 32

	// Metadata storage (rough estimate)
	for _, entry := range idx.vectors {
		for k, v := range entry.Metadata {
			usage += int64(len(k)) + estimateValueSize(v)
		}
	}

	return usage
}

// Close cleans up the index resources
func (idx *Index) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.vectors = nil
	idx.idToIndex = nil
	idx.quantizer = nil

	return nil
}

// SaveToDisk persists the index to disk
func (idx *Index) SaveToDisk(ctx context.Context, path string) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	data := struct {
		Config   *Config              `json:"config"`
		Vectors  []*VectorEntry       `json:"vectors"`
		Metadata *PersistenceMetadata `json:"metadata"`
	}{
		Config:  idx.config,
		Vectors: idx.vectors,
		Metadata: &PersistenceMetadata{
			Version:   1,
			NodeCount: len(idx.vectors),
			Dimension: idx.config.Dimension,
			MaxLevel:  0, // Flat index has no levels
			IndexType: "Flat",
			CreatedAt: time.Now(),
		},
	}

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("failed to encode index data: %w", err)
	}

	// Update file size in metadata
	if stat, err := file.Stat(); err == nil {
		data.Metadata.FileSize = stat.Size()
	}

	return nil
}

// LoadFromDisk loads the index from disk
func (idx *Index) LoadFromDisk(ctx context.Context, path string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var data struct {
		Config   *Config              `json:"config"`
		Vectors  []*VectorEntry       `json:"vectors"`
		Metadata *PersistenceMetadata `json:"metadata"`
	}

	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return fmt.Errorf("failed to decode index data: %w", err)
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Restore configuration
	idx.config = data.Config

	// Restore vectors
	idx.vectors = data.Vectors
	idx.idToIndex = make(map[string]int, len(data.Vectors))
	for i, entry := range data.Vectors {
		idx.idToIndex[entry.ID] = i
	}

	// Reinitialize quantizer if needed
	if idx.config.Quantization != nil {
		idx.quantizer, err = quant.Create(idx.config.Quantization)
		if err != nil {
			return fmt.Errorf("failed to recreate quantizer: %w", err)
		}
	}

	return nil
}

// GetPersistenceMetadata returns metadata about the persisted index
func (idx *Index) GetPersistenceMetadata() *PersistenceMetadata {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	return &PersistenceMetadata{
		Version:   1,
		NodeCount: len(idx.vectors),
		Dimension: idx.config.Dimension,
		MaxLevel:  0, // Flat index has no levels
		IndexType: "Flat",
		CreatedAt: time.Now(),
	}
}

// GetConfig returns the index configuration
func (idx *Index) GetConfig() *Config {
	return idx.config
}

// computeDistance computes the distance between two vectors
func (idx *Index) computeDistance(v1, v2 []float32) (float32, error) {
	switch idx.config.Metric {
	case util.CosineDistance:
		return util.CosineDistance_func(v1, v2), nil
	case util.L2Distance:
		return util.L2Distance_func(v1, v2), nil
	case util.InnerProduct:
		return util.InnerProduct_func(v1, v2), nil
	default:
		return 0, fmt.Errorf("unsupported distance metric: %v", idx.config.Metric)
	}
}

// estimateValueSize estimates the memory size of a metadata value
func estimateValueSize(v interface{}) int64 {
	switch val := v.(type) {
	case string:
		return int64(len(val))
	case int, int32, int64, float32, float64:
		return 8
	case bool:
		return 1
	case []interface{}:
		size := int64(0)
		for _, item := range val {
			size += estimateValueSize(item)
		}
		return size
	case map[string]interface{}:
		size := int64(0)
		for k, val := range val {
			size += int64(len(k)) + estimateValueSize(val)
		}
		return size
	default:
		return 16 // Default estimate
	}
}
