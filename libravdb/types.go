package libravdb

import (
	"fmt"
	"time"
)

// VectorEntry represents a vector with metadata for storage/indexing
type VectorEntry struct {
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// SearchResult represents a single search result
type SearchResult struct {
	ID       string                 `json:"id"`
	Score    float32                `json:"score"`
	Vector   []float32              `json:"vector,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// SearchResults represents the complete search response
type SearchResults struct {
	Results []*SearchResult `json:"results"`
	Took    time.Duration   `json:"took"`
	Total   int             `json:"total"`
}

// DatabaseStats represents database-wide statistics
type DatabaseStats struct {
	CollectionCount int                         `json:"collection_count"`
	Collections     map[string]*CollectionStats `json:"collections"`
	MemoryUsage     int64                       `json:"memory_usage"`
	Uptime          time.Duration               `json:"uptime"`
}

// CollectionStats represents collection-specific statistics
type CollectionStats struct {
	Name        string `json:"name"`
	VectorCount int    `json:"vector_count"`
	Dimension   int    `json:"dimension"`
	IndexType   string `json:"index_type"`
	MemoryUsage int64  `json:"memory_usage"`

	// Enhanced memory statistics
	MemoryStats *CollectionMemoryStats `json:"memory_stats,omitempty"`

	// Optimization information
	OptimizationStatus *OptimizationStatus `json:"optimization_status,omitempty"`

	// Configuration information
	HasQuantization      bool `json:"has_quantization"`
	HasMemoryLimit       bool `json:"has_memory_limit"`
	MemoryMappingEnabled bool `json:"memory_mapping_enabled"`
}

func (it IndexType) String() string {
	switch it {
	case HNSW:
		return "HNSW"
	case IVFPQ:
		return "IVF-PQ"
	case Flat:
		return "Flat"
	default:
		return "Unknown"
	}
}

// FieldType represents the type of a metadata field for schema validation
type FieldType int

const (
	StringField FieldType = iota
	IntField
	FloatField
	BoolField
	TimeField
	StringArrayField
	IntArrayField
	FloatArrayField
)

// String returns the string representation of the field type
func (ft FieldType) String() string {
	switch ft {
	case StringField:
		return "string"
	case IntField:
		return "int"
	case FloatField:
		return "float"
	case BoolField:
		return "bool"
	case TimeField:
		return "time"
	case StringArrayField:
		return "string_array"
	case IntArrayField:
		return "int_array"
	case FloatArrayField:
		return "float_array"
	default:
		return "unknown"
	}
}

// CachePolicy defines cache eviction policies
type CachePolicy int

const (
	LRUCache CachePolicy = iota
	LFUCache
	FIFOCache
)

// String returns the string representation of the cache policy
func (cp CachePolicy) String() string {
	switch cp {
	case LRUCache:
		return "lru"
	case LFUCache:
		return "lfu"
	case FIFOCache:
		return "fifo"
	default:
		return "unknown"
	}
}

// MetadataSchema defines the schema for metadata fields
type MetadataSchema map[string]FieldType

// Validate checks if the metadata schema is valid
func (ms MetadataSchema) Validate() error {
	for field, fieldType := range ms {
		if field == "" {
			return fmt.Errorf("field name cannot be empty")
		}
		if fieldType < StringField || fieldType > FloatArrayField {
			return fmt.Errorf("invalid field type for field '%s': %v", field, fieldType)
		}
	}
	return nil
}

// BatchConfig configures batch operation behavior
type BatchConfig struct {
	// ChunkSize is the number of items to process in each chunk
	ChunkSize int `json:"chunk_size"`

	// MaxConcurrency is the maximum number of concurrent workers
	MaxConcurrency int `json:"max_concurrency"`

	// FailFast determines if batch operations should stop on first error
	FailFast bool `json:"fail_fast"`

	// TimeoutPerChunk is the timeout for processing each chunk
	TimeoutPerChunk time.Duration `json:"timeout_per_chunk"`
}

// DefaultBatchConfig returns sensible default batch configuration
func DefaultBatchConfig() BatchConfig {
	return BatchConfig{
		ChunkSize:       1000,
		MaxConcurrency:  4,
		FailFast:        false,
		TimeoutPerChunk: 30 * time.Second,
	}
}

// OptimizationOptions configures collection optimization behavior
type OptimizationOptions struct {
	// RebuildIndex determines if the index should be rebuilt
	RebuildIndex bool `json:"rebuild_index"`

	// OptimizeMemory determines if memory optimization should be performed
	OptimizeMemory bool `json:"optimize_memory"`

	// CompactStorage determines if storage compaction should be performed
	CompactStorage bool `json:"compact_storage"`

	// UpdateQuantization determines if quantization parameters should be retrained
	UpdateQuantization bool `json:"update_quantization"`

	// ForceIndexTypeSwitch forces switching to optimal index type regardless of current type
	ForceIndexTypeSwitch bool `json:"force_index_type_switch"`
}

// OptimizationStatus represents the current optimization state of a collection
type OptimizationStatus struct {
	// InProgress indicates if optimization is currently running
	InProgress bool `json:"in_progress"`

	// LastOptimization is the timestamp of the last completed optimization
	LastOptimization time.Time `json:"last_optimization"`

	// CanOptimize indicates if optimization can be started
	CanOptimize bool `json:"can_optimize"`

	// RecommendedOptimizations suggests which optimizations would be beneficial
	RecommendedOptimizations []string `json:"recommended_optimizations,omitempty"`
}

// CollectionMemoryStats represents memory usage statistics for a collection
type CollectionMemoryStats struct {
	// Total memory usage in bytes
	Total int64 `json:"total"`

	// Index memory usage in bytes
	Index int64 `json:"index"`

	// Cache memory usage in bytes
	Cache int64 `json:"cache"`

	// Quantized data memory usage in bytes
	Quantized int64 `json:"quantized"`

	// Memory-mapped data size in bytes
	MemoryMapped int64 `json:"memory_mapped"`

	// Memory limit in bytes (0 = no limit)
	Limit int64 `json:"limit"`

	// Available memory before hitting limit
	Available int64 `json:"available"`

	// Memory pressure level
	PressureLevel string `json:"pressure_level"`

	// Timestamp of measurement
	Timestamp time.Time `json:"timestamp"`
}

// GlobalMemoryUsage represents memory usage across all collections in the database
type GlobalMemoryUsage struct {
	// Total memory usage across all collections
	TotalMemory int64 `json:"total_memory"`

	// Total index memory usage
	TotalIndex int64 `json:"total_index"`

	// Total cache memory usage
	TotalCache int64 `json:"total_cache"`

	// Total quantized data memory usage
	TotalQuantized int64 `json:"total_quantized"`

	// Total memory-mapped data size
	TotalMemoryMapped int64 `json:"total_memory_mapped"`

	// Per-collection memory statistics
	Collections map[string]*CollectionMemoryStats `json:"collections"`

	// Timestamp of measurement
	Timestamp time.Time `json:"timestamp"`
}
