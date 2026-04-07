package libravdb

import (
	"fmt"
	"time"

	"github.com/xDarkicex/libravdb/internal/memory"
	"github.com/xDarkicex/libravdb/internal/quant"
)

// Option represents a database configuration option
type Option func(*Config) error

// WithStoragePath sets the storage path for the database
func WithStoragePath(path string) Option {
	return func(c *Config) error {
		if path == "" {
			return fmt.Errorf("storage path cannot be empty")
		}
		c.StoragePath = path
		return nil
	}
}

// WithMetrics enables or disables metrics collection
func WithMetrics(enabled bool) Option {
	return func(c *Config) error {
		c.MetricsEnabled = enabled
		return nil
	}
}

// WithTracing enables or disables distributed tracing
func WithTracing(enabled bool) Option {
	return func(c *Config) error {
		c.TracingEnabled = enabled
		return nil
	}
}

// WithMaxCollections sets the maximum number of collections
func WithMaxCollections(max int) Option {
	return func(c *Config) error {
		if max <= 0 {
			return fmt.Errorf("max collections must be positive")
		}
		c.MaxCollections = max
		return nil
	}
}

// WithMaxConcurrentWrites bounds collection write execution parallelism.
func WithMaxConcurrentWrites(max int) Option {
	return func(c *Config) error {
		if max <= 0 {
			return fmt.Errorf("max concurrent writes must be positive")
		}
		c.MaxConcurrentWrites = max
		return nil
	}
}

// WithMaxWriteQueueDepth bounds queued writers waiting for collection admission.
func WithMaxWriteQueueDepth(depth int) Option {
	return func(c *Config) error {
		if depth < 0 {
			return fmt.Errorf("max write queue depth must be non-negative")
		}
		c.MaxWriteQueueDepth = depth
		return nil
	}
}

// CollectionOption represents a collection configuration option
type CollectionOption func(*CollectionConfig) error

// WithDimension sets the vector dimension for the collection
func WithDimension(dim int) CollectionOption {
	return func(c *CollectionConfig) error {
		if dim <= 0 {
			return fmt.Errorf("dimension must be positive")
		}
		c.Dimension = dim
		return nil
	}
}

// WithMetric sets the distance metric for the collection
func WithMetric(metric DistanceMetric) CollectionOption {
	return func(c *CollectionConfig) error {
		c.Metric = metric
		return nil
	}
}

// WithHNSW configures HNSW index parameters
func WithHNSW(m, efConstruction, efSearch int) CollectionOption {
	return func(c *CollectionConfig) error {
		if m <= 0 || efConstruction <= 0 || efSearch <= 0 {
			return fmt.Errorf("HNSW parameters must be positive")
		}
		c.IndexType = HNSW
		c.M = m
		c.EfConstruction = efConstruction
		c.EfSearch = efSearch
		return nil
	}
}

// WithRawVectorStoreMemory keeps raw vector payloads in the default in-memory store.
func WithRawVectorStoreMemory() CollectionOption {
	return func(c *CollectionConfig) error {
		c.RawVectorStore = "memory"
		return nil
	}
}

// WithRawVectorStoreSlabby stores raw vector payloads in a slabby-backed fixed-size store.
func WithRawVectorStoreSlabby(segmentCapacity int) CollectionOption {
	return func(c *CollectionConfig) error {
		if segmentCapacity <= 0 {
			return fmt.Errorf("slabby segment capacity must be positive")
		}
		c.RawVectorStore = "slabby"
		c.RawStoreCap = segmentCapacity
		return nil
	}
}

// WithIndexPersistence enables or disables index persistence
func WithIndexPersistence(enabled bool) CollectionOption {
	return func(c *CollectionConfig) error {
		c.AutoSave = enabled
		if enabled {
			c.SaveInterval = 5 * time.Minute // Default interval
		}
		return nil
	}
}

// WithPersistencePath sets the path for automatic index saves
func WithPersistencePath(path string) CollectionOption {
	return func(c *CollectionConfig) error {
		if path == "" {
			return fmt.Errorf("persistence path cannot be empty")
		}
		c.SavePath = path
		return nil
	}
}

// WithSaveInterval sets the interval between automatic saves
func WithSaveInterval(interval time.Duration) CollectionOption {
	return func(c *CollectionConfig) error {
		if interval <= 0 {
			return fmt.Errorf("save interval must be positive")
		}
		c.SaveInterval = interval
		return nil
	}
}

// WithQuantization enables vector quantization for the collection
func WithQuantization(config *quant.QuantizationConfig) CollectionOption {
	return func(c *CollectionConfig) error {
		if config == nil {
			return fmt.Errorf("quantization config cannot be nil")
		}
		if err := config.Validate(); err != nil {
			return fmt.Errorf("invalid quantization config: %w", err)
		}
		c.Quantization = config
		return nil
	}
}

// WithProductQuantization enables Product Quantization with specified parameters
func WithProductQuantization(codebooks, bits int, trainRatio float64) CollectionOption {
	return func(c *CollectionConfig) error {
		config := &quant.QuantizationConfig{
			Type:       quant.ProductQuantization,
			Codebooks:  codebooks,
			Bits:       bits,
			TrainRatio: trainRatio,
			CacheSize:  1000, // Default cache size
		}
		if err := config.Validate(); err != nil {
			return fmt.Errorf("invalid product quantization config: %w", err)
		}
		c.Quantization = config
		return nil
	}
}

// WithScalarQuantization enables Scalar Quantization with specified parameters
func WithScalarQuantization(bits int, trainRatio float64) CollectionOption {
	return func(c *CollectionConfig) error {
		config := &quant.QuantizationConfig{
			Type:       quant.ScalarQuantization,
			Bits:       bits,
			TrainRatio: trainRatio,
		}
		if err := config.Validate(); err != nil {
			return fmt.Errorf("invalid scalar quantization config: %w", err)
		}
		c.Quantization = config
		return nil
	}
}

// WithFlat configures the collection to use a Flat (brute-force) index
func WithFlat() CollectionOption {
	return func(c *CollectionConfig) error {
		c.IndexType = Flat
		return nil
	}
}

// WithIVFPQ configures the collection to use an IVF-PQ index.
func WithIVFPQ(nClusters, nProbes int) CollectionOption {
	return func(c *CollectionConfig) error {
		if nClusters <= 0 {
			return fmt.Errorf("IVF-PQ cluster count must be positive")
		}
		if nProbes <= 0 || nProbes > nClusters {
			return fmt.Errorf("IVF-PQ probe count must be between 1 and %d", nClusters)
		}
		c.IndexType = IVFPQ
		c.NClusters = nClusters
		c.NProbes = nProbes
		return nil
	}
}

// WithAutoIndexSelection enables automatic index type selection based on collection size.
// Small collections (<2000 vectors) use Flat, medium collections use HNSW, large collections use IVF-PQ.
// The thresholds can be customized via WithAutoIndexThresholds.
func WithAutoIndexSelection(enabled bool) CollectionOption {
	return func(c *CollectionConfig) error {
		c.AutoIndexSelection = enabled
		return nil
	}
}

// WithAutoIndexThresholds overrides the default auto-index selection thresholds.
// This allows tuning when HNSW or IVF-PQ is selected instead of Flat.
// hnswThreshold: vector count at which HNSW is selected over Flat (default: 10000)
// ivfpqThreshold: vector count at which IVF-PQ is selected over HNSW (default: 1000000)
func WithAutoIndexThresholds(hnswThreshold, ivfpqThreshold int) CollectionOption {
	return func(c *CollectionConfig) error {
		if hnswThreshold < 0 {
			return fmt.Errorf("hnsw threshold must be non-negative, got %d", hnswThreshold)
		}
		if ivfpqThreshold < hnswThreshold {
			return fmt.Errorf("ivfpq threshold (%d) must be >= hnsw threshold (%d)", ivfpqThreshold, hnswThreshold)
		}
		c.AutoIndexThresholds.HNSWThreshold = hnswThreshold
		c.AutoIndexThresholds.IVFPQThreshold = ivfpqThreshold
		return nil
	}
}

// WithMemoryLimit sets the maximum memory usage for the collection in bytes
func WithMemoryLimit(bytes int64) CollectionOption {
	return func(c *CollectionConfig) error {
		if bytes < 0 {
			return fmt.Errorf("memory limit must be non-negative, got %d", bytes)
		}
		c.MemoryLimit = bytes
		return nil
	}
}

// WithCachePolicy sets the cache eviction policy for the collection
func WithCachePolicy(policy CachePolicy) CollectionOption {
	return func(c *CollectionConfig) error {
		c.CachePolicy = policy
		return nil
	}
}

// WithMemoryMapping enables or disables memory mapping for large indices
func WithMemoryMapping(enabled bool) CollectionOption {
	return func(c *CollectionConfig) error {
		c.EnableMMapping = enabled
		return nil
	}
}

// WithMemoryConfig sets advanced memory management configuration
func WithMemoryConfig(config *memory.MemoryConfig) CollectionOption {
	return func(c *CollectionConfig) error {
		if config == nil {
			return fmt.Errorf("memory config cannot be nil")
		}
		if config.MaxMemory < 0 {
			return fmt.Errorf("max memory must be non-negative, got %d", config.MaxMemory)
		}
		if config.MonitorInterval <= 0 {
			return fmt.Errorf("monitor interval must be positive, got %v", config.MonitorInterval)
		}
		c.MemoryConfig = config
		return nil
	}
}

// WithMetadataSchema sets the schema for metadata fields with type validation
func WithMetadataSchema(schema MetadataSchema) CollectionOption {
	return func(c *CollectionConfig) error {
		if err := schema.Validate(); err != nil {
			return fmt.Errorf("invalid metadata schema: %w", err)
		}
		c.MetadataSchema = schema
		return nil
	}
}

// WithIndexedFields specifies which metadata fields should be indexed for faster filtering
func WithIndexedFields(fields ...string) CollectionOption {
	return func(c *CollectionConfig) error {
		if len(fields) == 0 {
			return fmt.Errorf("at least one field must be specified")
		}

		// Validate that all fields exist in schema if schema is set
		if c.MetadataSchema != nil {
			for _, field := range fields {
				if _, exists := c.MetadataSchema[field]; !exists {
					return fmt.Errorf("indexed field '%s' not found in metadata schema", field)
				}
			}
		}

		c.IndexedFields = fields
		return nil
	}
}

// WithBatchConfig sets the batch processing configuration
func WithBatchConfig(config BatchConfig) CollectionOption {
	return func(c *CollectionConfig) error {
		if config.ChunkSize <= 0 {
			return fmt.Errorf("batch chunk size must be positive, got %d", config.ChunkSize)
		}
		if config.MaxConcurrency <= 0 {
			return fmt.Errorf("batch max concurrency must be positive, got %d", config.MaxConcurrency)
		}
		if config.TimeoutPerChunk <= 0 {
			return fmt.Errorf("batch timeout per chunk must be positive, got %v", config.TimeoutPerChunk)
		}
		c.BatchConfig = config
		return nil
	}
}

// WithBatchChunkSize sets the chunk size for batch operations
func WithBatchChunkSize(size int) CollectionOption {
	return func(c *CollectionConfig) error {
		if size <= 0 {
			return fmt.Errorf("batch chunk size must be positive, got %d", size)
		}
		c.BatchConfig.ChunkSize = size
		return nil
	}
}

// WithBatchConcurrency sets the maximum concurrency for batch operations
func WithBatchConcurrency(concurrency int) CollectionOption {
	return func(c *CollectionConfig) error {
		if concurrency <= 0 {
			return fmt.Errorf("batch concurrency must be positive, got %d", concurrency)
		}
		c.BatchConfig.MaxConcurrency = concurrency
		return nil
	}
}

// WithSharding enables sharding for the collection.
// Sharding splits the collection into multiple shards for parallel writes.
// Only HNSW and Flat index types support sharding.
// IVFPQ and AutoIndexSelection do not support sharding.
func WithSharding(enabled bool) CollectionOption {
	return func(c *CollectionConfig) error {
		c.Sharded = enabled
		return nil
	}
}
