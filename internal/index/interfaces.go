package index

import (
	"context"
	"time"

	"github.com/xDarkicex/libravdb/internal/index/flat"
	"github.com/xDarkicex/libravdb/internal/index/hnsw"
	"github.com/xDarkicex/libravdb/internal/index/ivfpq"
	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

// Index defines the interface for vector indexes
type Index interface {
	Insert(ctx context.Context, entry *VectorEntry) error
	Search(ctx context.Context, query []float32, k int) ([]*SearchResult, error)
	Delete(ctx context.Context, id string) error
	Size() int
	MemoryUsage() int64
	Close() error

	// NEW: Index Persistence Methods
	SaveToDisk(ctx context.Context, path string) error
	LoadFromDisk(ctx context.Context, path string) error
	GetPersistenceMetadata() *PersistenceMetadata
}

// VectorEntry represents a vector entry (avoid circular imports)
type VectorEntry struct {
	ID       string
	Vector   []float32
	Metadata map[string]interface{}
}

// SearchResult represents a search result (avoid circular imports)
type SearchResult struct {
	ID       string
	Score    float32
	Vector   []float32
	Metadata map[string]interface{}
}

// NEW: PersistenceMetadata holds metadata about persisted index
type PersistenceMetadata struct {
	Version       uint32    `json:"version"`        // Binary format version
	NodeCount     int       `json:"node_count"`     // Total number of nodes
	Dimension     int       `json:"dimension"`      // Vector dimension
	MaxLevel      int       `json:"max_level"`      // Maximum graph level
	IndexType     string    `json:"index_type"`     // Index algorithm (HNSW, etc.)
	CreatedAt     time.Time `json:"created_at"`     // When index was persisted
	ChecksumCRC32 uint32    `json:"checksum_crc32"` // File integrity checksum
	FileSize      int64     `json:"file_size"`      // Total file size in bytes
}

// IndexType represents different index algorithms
type IndexType int

const (
	IndexTypeHNSW IndexType = iota
	IndexTypeIVFPQ
	IndexTypeFlat
)

// String returns the string representation of the index type
func (it IndexType) String() string {
	switch it {
	case IndexTypeHNSW:
		return "HNSW"
	case IndexTypeIVFPQ:
		return "IVF-PQ"
	case IndexTypeFlat:
		return "Flat"
	default:
		return "Unknown"
	}
}

// HNSWConfig holds configuration for HNSW index
type HNSWConfig struct {
	Dimension      int
	M              int
	EfConstruction int
	EfSearch       int
	ML             float64
	Metric         util.DistanceMetric
	// Quantization configuration (optional)
	Quantization *quant.QuantizationConfig
}

// IVFPQConfig holds configuration for IVF-PQ index
type IVFPQConfig struct {
	Dimension     int
	NClusters     int
	NProbes       int
	Metric        util.DistanceMetric
	Quantization  *quant.QuantizationConfig
	MaxIterations int
	Tolerance     float64
	RandomSeed    int64
}

// FlatConfig holds configuration for Flat index
type FlatConfig struct {
	Dimension    int
	Metric       util.DistanceMetric
	Quantization *quant.QuantizationConfig
}

// hnswWrapper wraps the HNSW index to adapt between interface types
type hnswWrapper struct {
	index *hnsw.Index
}

// Insert adapts the interface VectorEntry to HNSW VectorEntry
func (w *hnswWrapper) Insert(ctx context.Context, entry *VectorEntry) error {
	hnswEntry := &hnsw.VectorEntry{
		ID:       entry.ID,
		Vector:   entry.Vector,
		Metadata: entry.Metadata,
	}
	return w.index.Insert(ctx, hnswEntry)
}

// Search adapts the search results from HNSW to interface types
func (w *hnswWrapper) Search(ctx context.Context, query []float32, k int) ([]*SearchResult, error) {
	hnswResults, err := w.index.Search(ctx, query, k)
	if err != nil {
		return nil, err
	}

	results := make([]*SearchResult, len(hnswResults))
	for i, r := range hnswResults {
		results[i] = &SearchResult{
			ID:       r.ID,
			Score:    r.Score,
			Vector:   r.Vector,
			Metadata: r.Metadata,
		}
	}
	return results, nil
}

// Delete delegates to the wrapped index
func (w *hnswWrapper) Delete(ctx context.Context, id string) error {
	return w.index.Delete(ctx, id)
}

// Size delegates to the wrapped index
func (w *hnswWrapper) Size() int {
	return w.index.Size()
}

// MemoryUsage delegates to the wrapped index
func (w *hnswWrapper) MemoryUsage() int64 {
	return w.index.MemoryUsage()
}

// Close delegates to the wrapped index
func (w *hnswWrapper) Close() error {
	return w.index.Close()
}

// NEW: SaveToDisk delegates persistence to the wrapped index
func (w *hnswWrapper) SaveToDisk(ctx context.Context, path string) error {
	return w.index.SaveToDisk(ctx, path)
}

// NEW: LoadFromDisk delegates loading to the wrapped index
func (w *hnswWrapper) LoadFromDisk(ctx context.Context, path string) error {
	return w.index.LoadFromDisk(ctx, path)
}

// NEW: GetPersistenceMetadata delegates to the wrapped index
func (w *hnswWrapper) GetPersistenceMetadata() *PersistenceMetadata {
	hnswMeta := w.index.GetPersistenceMetadata()
	if hnswMeta == nil {
		return nil
	}

	// Convert from HNSW metadata to interface metadata
	return &PersistenceMetadata{
		Version:       hnswMeta.Version,
		NodeCount:     hnswMeta.NodeCount,
		Dimension:     hnswMeta.Dimension,
		MaxLevel:      hnswMeta.MaxLevel,
		IndexType:     "HNSW",
		CreatedAt:     hnswMeta.CreatedAt,
		ChecksumCRC32: hnswMeta.ChecksumCRC32,
		FileSize:      hnswMeta.FileSize,
	}
}

// NewHNSW creates a new HNSW index
func NewHNSW(config *HNSWConfig) (Index, error) {
	// Convert to internal HNSW config
	hnswConfig := &hnsw.Config{
		Dimension:      config.Dimension,
		M:              config.M,
		EfConstruction: config.EfConstruction,
		EfSearch:       config.EfSearch,
		ML:             config.ML,
		Metric:         config.Metric,
		RandomSeed:     0, // Default seed for Phase 1
		Quantization:   config.Quantization,
	}

	hnswIndex, err := hnsw.NewHNSW(hnswConfig)
	if err != nil {
		return nil, err
	}

	return &hnswWrapper{index: hnswIndex}, nil
}

// ivfpqWrapper wraps the IVF-PQ index to adapt between interface types
type ivfpqWrapper struct {
	index *ivfpq.Index
}

// Insert adapts the interface VectorEntry to IVF-PQ VectorEntry
func (w *ivfpqWrapper) Insert(ctx context.Context, entry *VectorEntry) error {
	ivfpqEntry := &ivfpq.VectorEntry{
		ID:       entry.ID,
		Vector:   entry.Vector,
		Metadata: entry.Metadata,
	}
	return w.index.Insert(ctx, ivfpqEntry)
}

// Search adapts the search results from IVF-PQ to interface types
func (w *ivfpqWrapper) Search(ctx context.Context, query []float32, k int) ([]*SearchResult, error) {
	ivfpqResults, err := w.index.Search(ctx, query, k)
	if err != nil {
		return nil, err
	}

	results := make([]*SearchResult, len(ivfpqResults))
	for i, r := range ivfpqResults {
		results[i] = &SearchResult{
			ID:       r.ID,
			Score:    r.Score,
			Vector:   r.Vector,
			Metadata: r.Metadata,
		}
	}
	return results, nil
}

// Delete delegates to the wrapped index
func (w *ivfpqWrapper) Delete(ctx context.Context, id string) error {
	return w.index.Delete(ctx, id)
}

// Size delegates to the wrapped index
func (w *ivfpqWrapper) Size() int {
	return w.index.Size()
}

// MemoryUsage delegates to the wrapped index
func (w *ivfpqWrapper) MemoryUsage() int64 {
	return w.index.MemoryUsage()
}

// Close delegates to the wrapped index
func (w *ivfpqWrapper) Close() error {
	return w.index.Close()
}

// SaveToDisk delegates persistence to the wrapped index
func (w *ivfpqWrapper) SaveToDisk(ctx context.Context, path string) error {
	return w.index.SaveToDisk(ctx, path)
}

// LoadFromDisk delegates loading to the wrapped index
func (w *ivfpqWrapper) LoadFromDisk(ctx context.Context, path string) error {
	return w.index.LoadFromDisk(ctx, path)
}

// GetPersistenceMetadata delegates to the wrapped index
func (w *ivfpqWrapper) GetPersistenceMetadata() *PersistenceMetadata {
	ivfpqMeta := w.index.GetPersistenceMetadata()
	if ivfpqMeta == nil {
		return nil
	}

	// For now, return a basic metadata structure
	// TODO: Implement proper persistence metadata for IVF-PQ
	return &PersistenceMetadata{
		Version:       1,
		NodeCount:     w.index.Size(),
		Dimension:     w.index.GetConfig().Dimension,
		MaxLevel:      0, // Not applicable for IVF-PQ
		IndexType:     "IVF-PQ",
		CreatedAt:     time.Now(),
		ChecksumCRC32: 0, // TODO: Implement checksum
		FileSize:      0, // TODO: Implement file size calculation
	}
}

// NewIVFPQ creates a new IVF-PQ index
func NewIVFPQ(config *IVFPQConfig) (Index, error) {
	// Convert to internal IVF-PQ config
	ivfpqConfig := &ivfpq.Config{
		Dimension:     config.Dimension,
		NClusters:     config.NClusters,
		NProbes:       config.NProbes,
		Metric:        config.Metric,
		Quantization:  config.Quantization,
		MaxIterations: config.MaxIterations,
		Tolerance:     config.Tolerance,
		RandomSeed:    config.RandomSeed,
	}

	ivfpqIndex, err := ivfpq.NewIVFPQ(ivfpqConfig)
	if err != nil {
		return nil, err
	}

	return &ivfpqWrapper{index: ivfpqIndex}, nil
}

// flatWrapper wraps the Flat index to adapt between interface types
type flatWrapper struct {
	index *flat.Index
}

// Insert adapts the interface VectorEntry to Flat VectorEntry
func (w *flatWrapper) Insert(ctx context.Context, entry *VectorEntry) error {
	flatEntry := &flat.VectorEntry{
		ID:       entry.ID,
		Vector:   entry.Vector,
		Metadata: entry.Metadata,
	}
	return w.index.Insert(ctx, flatEntry)
}

// Search adapts the search results from Flat to interface types
func (w *flatWrapper) Search(ctx context.Context, query []float32, k int) ([]*SearchResult, error) {
	flatResults, err := w.index.Search(ctx, query, k)
	if err != nil {
		return nil, err
	}

	results := make([]*SearchResult, len(flatResults))
	for i, r := range flatResults {
		results[i] = &SearchResult{
			ID:       r.ID,
			Score:    r.Score,
			Vector:   r.Vector,
			Metadata: r.Metadata,
		}
	}
	return results, nil
}

// Delete delegates to the wrapped index
func (w *flatWrapper) Delete(ctx context.Context, id string) error {
	return w.index.Delete(ctx, id)
}

// Size delegates to the wrapped index
func (w *flatWrapper) Size() int {
	return w.index.Size()
}

// MemoryUsage delegates to the wrapped index
func (w *flatWrapper) MemoryUsage() int64 {
	return w.index.MemoryUsage()
}

// Close delegates to the wrapped index
func (w *flatWrapper) Close() error {
	return w.index.Close()
}

// SaveToDisk delegates persistence to the wrapped index
func (w *flatWrapper) SaveToDisk(ctx context.Context, path string) error {
	return w.index.SaveToDisk(ctx, path)
}

// LoadFromDisk delegates loading to the wrapped index
func (w *flatWrapper) LoadFromDisk(ctx context.Context, path string) error {
	return w.index.LoadFromDisk(ctx, path)
}

// GetPersistenceMetadata delegates to the wrapped index
func (w *flatWrapper) GetPersistenceMetadata() *PersistenceMetadata {
	flatMeta := w.index.GetPersistenceMetadata()
	if flatMeta == nil {
		return nil
	}

	// Convert from Flat metadata to interface metadata
	return &PersistenceMetadata{
		Version:       flatMeta.Version,
		NodeCount:     flatMeta.NodeCount,
		Dimension:     flatMeta.Dimension,
		MaxLevel:      flatMeta.MaxLevel,
		IndexType:     "Flat",
		CreatedAt:     flatMeta.CreatedAt,
		ChecksumCRC32: flatMeta.ChecksumCRC32,
		FileSize:      flatMeta.FileSize,
	}
}

// NewFlat creates a new Flat index
func NewFlat(config *FlatConfig) (Index, error) {
	// Convert to internal Flat config
	flatConfig := &flat.Config{
		Dimension:    config.Dimension,
		Metric:       config.Metric,
		Quantization: config.Quantization,
	}

	flatIndex, err := flat.NewFlat(flatConfig)
	if err != nil {
		return nil, err
	}

	return &flatWrapper{index: flatIndex}, nil
}
