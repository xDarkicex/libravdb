package quant

import (
	"context"
	"fmt"
)

// QuantizationType represents the type of quantization algorithm
type QuantizationType int

const (
	// ProductQuantization uses k-means clustering on vector subspaces
	ProductQuantization QuantizationType = iota
	// ScalarQuantization uses linear quantization to fixed-point representation
	ScalarQuantization
	// FiniteScalarQuantization uses codebook-free bounded scalar quantization
	FiniteScalarQuantization
)

// String returns the string representation of the quantization type
func (qt QuantizationType) String() string {
	switch qt {
	case ProductQuantization:
		return "product"
	case ScalarQuantization:
		return "scalar"
	case FiniteScalarQuantization:
		return "fsq"
	default:
		return "unknown"
	}
}

// QuantizationConfig holds configuration for quantization algorithms
type QuantizationConfig struct {
	// Type specifies the quantization algorithm to use
	Type QuantizationType `json:"type"`

	// Codebooks specifies the number of codebooks for Product Quantization
	Codebooks int `json:"codebooks,omitempty"`

	// Bits specifies bits per component/codebook entry
	Bits int `json:"bits"`

	// TrainRatio specifies the ratio of data to use for training (0.0-1.0)
	TrainRatio float64 `json:"train_ratio"`

	// CacheSize specifies the size of the codebook cache
	CacheSize int `json:"cache_size,omitempty"`

	// Levels specifies the per-coordinate FSQ level cycle. If omitted for FSQ,
	// a uniform level count derived from Bits is used.
	Levels []int `json:"levels,omitempty"`
}

// Validate checks if the quantization configuration is valid
func (qc *QuantizationConfig) Validate() error {
	if qc.Bits < 1 || qc.Bits > 16 {
		return fmt.Errorf("bits must be between 1 and 16, got %d", qc.Bits)
	}

	if qc.TrainRatio <= 0.0 || qc.TrainRatio > 1.0 {
		return fmt.Errorf("train_ratio must be between 0.0 and 1.0, got %f", qc.TrainRatio)
	}

	switch qc.Type {
	case ProductQuantization:
		if qc.Codebooks < 1 {
			return fmt.Errorf("codebooks must be positive for product quantization, got %d", qc.Codebooks)
		}
		if qc.CacheSize < 0 {
			return fmt.Errorf("cache_size must be non-negative, got %d", qc.CacheSize)
		}
	case ScalarQuantization:
		// No additional validation needed for scalar quantization
	case FiniteScalarQuantization:
		for _, level := range qc.Levels {
			if level < 2 {
				return fmt.Errorf("fsq levels must be >= 2, got %d", level)
			}
		}
	default:
		return fmt.Errorf("unsupported quantization type: %v", qc.Type)
	}

	return nil
}

// DefaultConfig returns a default quantization configuration
func DefaultConfig(qType QuantizationType) *QuantizationConfig {
	switch qType {
	case ProductQuantization:
		return &QuantizationConfig{
			Type:       ProductQuantization,
			Codebooks:  8,
			Bits:       8,
			TrainRatio: 0.1,
			CacheSize:  1000,
		}
	case ScalarQuantization:
		return &QuantizationConfig{
			Type:       ScalarQuantization,
			Bits:       8,
			TrainRatio: 0.1,
		}
	case FiniteScalarQuantization:
		return &QuantizationConfig{
			Type:       FiniteScalarQuantization,
			Bits:       8,
			TrainRatio: 0.1,
		}
	default:
		return nil
	}
}

// Quantizer defines the interface for vector quantization algorithms
type Quantizer interface {
	// Train trains the quantizer using the provided vectors
	Train(ctx context.Context, vectors [][]float32) error

	// Configure sets the quantization configuration
	Configure(config *QuantizationConfig) error

	// Compress compresses a vector into quantized representation
	Compress(vector []float32) ([]byte, error)

	// Decompress decompresses quantized data back to vector
	Decompress(data []byte) ([]float32, error)

	// Distance computes distance between two compressed vectors
	Distance(compressed1, compressed2 []byte) (float32, error)

	// DistanceToQuery computes distance from compressed vector to query vector.
	// state is an optional precomputed state returned by PrepareQuery.
	DistanceToQuery(compressed []byte, query []float32, state any) (float32, error)

	// PrepareQuery precomputes distance tables or other state for a query.
	// Returns a state object that should be passed to DistanceToQuery.
	PrepareQuery(query []float32) any

	// CodeSize returns the byte length of a single compressed vector for
	// the current trained configuration. Returns 0 if the quantizer has
	// no compressed form (e.g. not trained or no quantization). This is
	// the canonical width every index layer must use to size retained
	// storage; callers MUST NOT recompute it from config fields.
	CodeSize() int

	// Dimension returns the vector dimension the quantizer was trained
	// for, or 0 if not trained. Used by persistence to validate that the
	// embedded quantizer agrees with the outer index config.
	Dimension() int

	// SerializeState returns a portable byte payload describing the
	// trained quantizer state. The bytes are self-describing and may be
	// passed to DeserializeState to restore an identical trained state.
	// Returns an error if the quantizer is not trained.
	SerializeState() ([]byte, error)

	// DeserializeState restores a previously-SerializeState'd quantizer
	// into a trained state. After this call, IsTrained returns true and
	// the quantizer is ready for Compress / Distance / etc.
	DeserializeState(data []byte) error

	// CompressionRatio returns the compression ratio achieved
	CompressionRatio() float32

	// MemoryUsage returns the memory usage in bytes
	MemoryUsage() int64

	// IsTrained returns whether the quantizer has been trained
	IsTrained() bool

	// Config returns the current configuration
	Config() *QuantizationConfig

	// Close cleans up resources
	Close() error
}

// CentroidProvider is an optional interface implemented by quantizers that explicitly model
// centroids (like ProductQuantizer or IVF coarse quantizers). It provides exact mathematical
// distance between packed centroid IDs, allowing for exact cardinality pruning over codebooks.
type CentroidProvider interface {
	// CentroidDistance computes the exact mathematical distance between two packed centroid IDs.
	// For PQ, the ID is typically packed as `subspace << bits | centroid`.
	CentroidDistance(a, b uint32) float32

	// MaxResidualBound returns the maximum possible length (L2 norm) of any quantized residual
	// produced by this codebook. Used for exact cardinality pruning of clusters.
	MaxResidualBound() float32
}
// QuantizerFactory creates quantizer instances
type QuantizerFactory interface {
	// Create creates a new quantizer instance
	Create(config *QuantizationConfig) (Quantizer, error)

	// Supports returns true if the factory supports the given quantization type
	Supports(qType QuantizationType) bool

	// Name returns the name of the quantizer implementation
	Name() string
}
