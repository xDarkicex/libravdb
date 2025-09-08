package libravdb

import "fmt"

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

// TODO: finish implementation
// WithIndexPersistence enables or disables index persistence
func WithIndexPersistence(enabled bool) CollectionOption {
	// For Week 1: Just store the option, implementation comes later
	return func(c *CollectionConfig) error {
		// We'll add persistence config to CollectionConfig in Week 3
		// For now, this is a no-op that validates the option works
		_ = enabled // Use the parameter to avoid unused variable warning
		return nil
	}
}
