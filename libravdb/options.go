package libravdb

import (
	"fmt"
	"time"

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
