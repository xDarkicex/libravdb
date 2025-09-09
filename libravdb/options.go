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

// WithAutoIndexSelection enables automatic index type selection based on collection size
// Small collections (<10K vectors) use Flat, medium collections use HNSW, large collections use IVF-PQ
func WithAutoIndexSelection(enabled bool) CollectionOption {
	return func(c *CollectionConfig) error {
		c.AutoIndexSelection = enabled
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
