package quant

import (
	"context"
	"fmt"
	"math"
	"sync"
)

// ScalarQuantizer implements Scalar Quantization algorithm
// Uses linear quantization to map floating-point values to fixed-point representation
type ScalarQuantizer struct {
	mu sync.RWMutex

	// Configuration
	config *QuantizationConfig

	// Training state
	trained   bool
	dimension int

	// Quantization parameters per dimension
	minValues []float32 // Minimum value per dimension
	maxValues []float32 // Maximum value per dimension
	scales    []float32 // Scale factor per dimension: (max - min) / (2^bits - 1)
	offsets   []float32 // Offset per dimension: min value

	// Quantization levels
	maxLevel uint32 // Maximum quantization level: 2^bits - 1

	// Memory usage tracking
	memoryUsage int64
}

// NewScalarQuantizer creates a new Scalar Quantizer instance
func NewScalarQuantizer() *ScalarQuantizer {
	return &ScalarQuantizer{
		trained: false,
	}
}

// Configure sets the quantization configuration
func (sq *ScalarQuantizer) Configure(config *QuantizationConfig) error {
	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}

	if err := config.Validate(); err != nil {
		return fmt.Errorf("invalid config: %w", err)
	}

	if config.Type != ScalarQuantization {
		return fmt.Errorf("expected ScalarQuantization type, got %s", config.Type.String())
	}

	sq.mu.Lock()
	defer sq.mu.Unlock()

	sq.config = config
	sq.maxLevel = (1 << config.Bits) - 1 // 2^bits - 1

	return nil
}

// Train computes min/max ranges for each dimension from training vectors
func (sq *ScalarQuantizer) Train(ctx context.Context, vectors [][]float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no training vectors provided")
	}

	if sq.config == nil {
		return fmt.Errorf("quantizer not configured")
	}

	sq.mu.Lock()
	defer sq.mu.Unlock()

	// Initialize dimensions
	sq.dimension = len(vectors[0])

	// Validate all vectors have same dimension
	for i, vec := range vectors {
		if len(vec) != sq.dimension {
			return fmt.Errorf("vector %d has dimension %d, expected %d", i, len(vec), sq.dimension)
		}
	}

	// Sample training vectors based on train ratio
	numTraining := int(float64(len(vectors)) * sq.config.TrainRatio)
	if numTraining < 1 {
		numTraining = len(vectors) // Use all vectors if too few
	}

	trainingVectors := sq.sampleVectors(vectors, numTraining)

	// Initialize min/max arrays
	sq.minValues = make([]float32, sq.dimension)
	sq.maxValues = make([]float32, sq.dimension)
	sq.scales = make([]float32, sq.dimension)
	sq.offsets = make([]float32, sq.dimension)

	// Initialize with first vector
	if len(trainingVectors) > 0 {
		copy(sq.minValues, trainingVectors[0])
		copy(sq.maxValues, trainingVectors[0])
	}

	// Find min/max for each dimension
	for _, vec := range trainingVectors {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		for d := 0; d < sq.dimension; d++ {
			if vec[d] < sq.minValues[d] {
				sq.minValues[d] = vec[d]
			}
			if vec[d] > sq.maxValues[d] {
				sq.maxValues[d] = vec[d]
			}
		}
	}

	// Compute scales and offsets for each dimension
	for d := 0; d < sq.dimension; d++ {
		range_ := sq.maxValues[d] - sq.minValues[d]
		if range_ == 0 {
			// Handle constant dimensions
			sq.scales[d] = 1.0
			sq.offsets[d] = sq.minValues[d]
		} else {
			sq.scales[d] = range_ / float32(sq.maxLevel)
			sq.offsets[d] = sq.minValues[d]
		}
	}

	sq.trained = true
	sq.updateMemoryUsage()

	return nil
}

// Compress compresses a vector using linear quantization
func (sq *ScalarQuantizer) Compress(vector []float32) ([]byte, error) {
	sq.mu.RLock()
	defer sq.mu.RUnlock()

	if !sq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}

	if len(vector) != sq.dimension {
		return nil, fmt.Errorf("vector dimension %d does not match expected %d",
			len(vector), sq.dimension)
	}

	// Calculate number of bytes needed
	bitsPerValue := sq.config.Bits
	totalBits := sq.dimension * bitsPerValue
	numBytes := (totalBits + 7) / 8 // Round up to nearest byte

	compressed := make([]byte, numBytes)
	bitOffset := 0

	// Quantize each dimension
	for d := 0; d < sq.dimension; d++ {
		// Clamp value to trained range
		value := vector[d]
		if value < sq.minValues[d] {
			value = sq.minValues[d]
		} else if value > sq.maxValues[d] {
			value = sq.maxValues[d]
		}

		// Linear quantization: (value - offset) / scale
		normalized := (value - sq.offsets[d]) / sq.scales[d]
		quantized := uint32(normalized + 0.5) // Round to nearest integer

		// Ensure quantized value is within bounds
		if quantized > sq.maxLevel {
			quantized = sq.maxLevel
		}

		// Pack the quantized value into compressed bytes
		sq.packBits(compressed, bitOffset, bitsPerValue, quantized)
		bitOffset += bitsPerValue
	}

	return compressed, nil
}

// Decompress decompresses quantized data back to a vector
func (sq *ScalarQuantizer) Decompress(data []byte) ([]float32, error) {
	sq.mu.RLock()
	defer sq.mu.RUnlock()

	if !sq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}

	vector := make([]float32, sq.dimension)
	bitOffset := 0
	bitsPerValue := sq.config.Bits

	// Decompress each dimension
	for d := 0; d < sq.dimension; d++ {
		// Extract quantized value from compressed data
		quantized := sq.unpackBits(data, bitOffset, bitsPerValue)
		bitOffset += bitsPerValue

		// Dequantize: offset + quantized * scale
		value := sq.offsets[d] + float32(quantized)*sq.scales[d]
		vector[d] = value
	}

	return vector, nil
}

// Distance computes Euclidean distance between two compressed vectors
func (sq *ScalarQuantizer) Distance(compressed1, compressed2 []byte) (float32, error) {
	sq.mu.RLock()
	defer sq.mu.RUnlock()

	if !sq.trained {
		return 0, fmt.Errorf("quantizer not trained")
	}

	distance := float32(0)
	bitOffset := 0
	bitsPerValue := sq.config.Bits

	// Compute distance for each dimension using quantized values directly
	for d := 0; d < sq.dimension; d++ {
		// Extract quantized values from both compressed vectors
		q1 := sq.unpackBits(compressed1, bitOffset, bitsPerValue)
		q2 := sq.unpackBits(compressed2, bitOffset, bitsPerValue)
		bitOffset += bitsPerValue

		// Compute distance in quantized space and scale back
		diff := float32(int32(q1) - int32(q2)) // Use signed difference
		scaledDiff := diff * sq.scales[d]
		distance += scaledDiff * scaledDiff
	}

	return float32(math.Sqrt(float64(distance))), nil
}

// DistanceToQuery computes distance from compressed vector to query vector
func (sq *ScalarQuantizer) DistanceToQuery(compressed []byte, query []float32) (float32, error) {
	sq.mu.RLock()
	defer sq.mu.RUnlock()

	if !sq.trained {
		return 0, fmt.Errorf("quantizer not trained")
	}

	if len(query) != sq.dimension {
		return 0, fmt.Errorf("query dimension %d does not match expected %d",
			len(query), sq.dimension)
	}

	distance := float32(0)
	bitOffset := 0
	bitsPerValue := sq.config.Bits

	// Compute distance for each dimension
	for d := 0; d < sq.dimension; d++ {
		// Extract quantized value from compressed vector
		quantized := sq.unpackBits(compressed, bitOffset, bitsPerValue)
		bitOffset += bitsPerValue

		// Dequantize the compressed value
		dequantized := sq.offsets[d] + float32(quantized)*sq.scales[d]

		// Compute distance to query value
		diff := query[d] - dequantized
		distance += diff * diff
	}

	return float32(math.Sqrt(float64(distance))), nil
}

// Helper functions

func (sq *ScalarQuantizer) sampleVectors(vectors [][]float32, n int) [][]float32 {
	if n >= len(vectors) {
		return vectors
	}

	// Simple sampling - take every k-th vector for deterministic behavior
	step := len(vectors) / n
	if step < 1 {
		step = 1
	}

	sampled := make([][]float32, 0, n)
	for i := 0; i < len(vectors) && len(sampled) < n; i += step {
		sampled = append(sampled, vectors[i])
	}

	return sampled
}

func (sq *ScalarQuantizer) packBits(data []byte, bitOffset, numBits int, value uint32) {
	for i := 0; i < numBits; i++ {
		byteIdx := (bitOffset + i) / 8
		bitIdx := (bitOffset + i) % 8

		if byteIdx >= len(data) {
			return
		}

		if (value>>i)&1 == 1 {
			data[byteIdx] |= 1 << bitIdx
		}
	}
}

func (sq *ScalarQuantizer) unpackBits(data []byte, bitOffset, numBits int) uint32 {
	value := uint32(0)
	for i := 0; i < numBits; i++ {
		byteIdx := (bitOffset + i) / 8
		bitIdx := (bitOffset + i) % 8

		if byteIdx >= len(data) {
			break
		}

		if (data[byteIdx]>>bitIdx)&1 == 1 {
			value |= 1 << i
		}
	}
	return value
}

func (sq *ScalarQuantizer) updateMemoryUsage() {
	usage := int64(0)

	// Min/max values, scales, and offsets
	usage += int64(len(sq.minValues) * 4) // 4 bytes per float32
	usage += int64(len(sq.maxValues) * 4) // 4 bytes per float32
	usage += int64(len(sq.scales) * 4)    // 4 bytes per float32
	usage += int64(len(sq.offsets) * 4)   // 4 bytes per float32

	sq.memoryUsage = usage
}

// Interface implementation

func (sq *ScalarQuantizer) CompressionRatio() float32 {
	if !sq.trained {
		return 0
	}

	originalBits := sq.dimension * 32 // 32 bits per float32
	compressedBits := sq.dimension * sq.config.Bits

	return float32(originalBits) / float32(compressedBits)
}

func (sq *ScalarQuantizer) MemoryUsage() int64 {
	sq.mu.RLock()
	defer sq.mu.RUnlock()
	return sq.memoryUsage
}

func (sq *ScalarQuantizer) IsTrained() bool {
	sq.mu.RLock()
	defer sq.mu.RUnlock()
	return sq.trained
}

func (sq *ScalarQuantizer) Config() *QuantizationConfig {
	sq.mu.RLock()
	defer sq.mu.RUnlock()
	if sq.config == nil {
		return nil
	}

	// Return a copy to prevent external modification
	configCopy := *sq.config
	return &configCopy
}

// ScalarQuantizerFactory creates ScalarQuantizer instances
type ScalarQuantizerFactory struct{}

func NewScalarQuantizerFactory() *ScalarQuantizerFactory {
	return &ScalarQuantizerFactory{}
}

func (f *ScalarQuantizerFactory) Create(config *QuantizationConfig) (Quantizer, error) {
	if config.Type != ScalarQuantization {
		return nil, fmt.Errorf("unsupported quantization type: %s", config.Type.String())
	}

	sq := NewScalarQuantizer()
	if err := sq.Configure(config); err != nil {
		return nil, err
	}

	return sq, nil
}

func (f *ScalarQuantizerFactory) Supports(qType QuantizationType) bool {
	return qType == ScalarQuantization
}

func (f *ScalarQuantizerFactory) Name() string {
	return "ScalarQuantizer"
}
