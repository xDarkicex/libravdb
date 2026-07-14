package quant

import (
	"context"
	"fmt"
	"math"
	"sync"
)

const fsqEpsilon = 1e-3

// FSQQuantizer implements Finite Scalar Quantization with no learned codebook.
// It uses per-channel min/max statistics to map arbitrary vectors into [-1, 1],
// then applies the CHEAP/FSQ bound -> round -> scale transform.
type FSQQuantizer struct {
	config        *QuantizationConfig
	minValues     []float32
	maxValues     []float32
	levels        []int
	bitWidths     []int
	halfLevels    []float32
	levelOffsets  []float32
	levelShifts   []float32
	decodeScales  []float32
	decodeOffsets []float32
	dimension     int
	memoryUsage   int64
	mu            sync.RWMutex
	trained       bool
}

func NewFSQQuantizer() *FSQQuantizer {
	return &FSQQuantizer{}
}

func (fq *FSQQuantizer) Configure(config *QuantizationConfig) error {
	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}
	if err := config.Validate(); err != nil {
		return fmt.Errorf("invalid config: %w", err)
	}
	if config.Type != FiniteScalarQuantization {
		return fmt.Errorf("expected FiniteScalarQuantization type, got %s", config.Type.String())
	}

	fq.mu.Lock()
	defer fq.mu.Unlock()

	configCopy := *config
	if len(config.Levels) > 0 {
		configCopy.Levels = append([]int(nil), config.Levels...)
	}
	fq.config = &configCopy
	return nil
}

func (fq *FSQQuantizer) Train(ctx context.Context, vectors [][]float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no training vectors provided")
	}
	if fq.config == nil {
		return fmt.Errorf("quantizer not configured")
	}

	fq.mu.Lock()
	defer fq.mu.Unlock()

	fq.dimension = len(vectors[0])
	for i, vec := range vectors {
		if len(vec) != fq.dimension {
			return fmt.Errorf("vector %d has dimension %d, expected %d", i, len(vec), fq.dimension)
		}
	}

	numTraining := int(float64(len(vectors)) * fq.config.TrainRatio)
	if numTraining < 1 {
		numTraining = len(vectors)
	}
	trainingVectors := sampleVectors(vectors, numTraining)

	fq.minValues = make([]float32, fq.dimension)
	fq.maxValues = make([]float32, fq.dimension)
	copy(fq.minValues, trainingVectors[0])
	copy(fq.maxValues, trainingVectors[0])

	for _, vec := range trainingVectors {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		for d := 0; d < fq.dimension; d++ {
			if vec[d] < fq.minValues[d] {
				fq.minValues[d] = vec[d]
			}
			if vec[d] > fq.maxValues[d] {
				fq.maxValues[d] = vec[d]
			}
		}
	}

	fq.levels = make([]int, fq.dimension)
	fq.bitWidths = make([]int, fq.dimension)
	fq.halfLevels = make([]float32, fq.dimension)
	fq.levelOffsets = make([]float32, fq.dimension)
	fq.levelShifts = make([]float32, fq.dimension)
	fq.decodeScales = make([]float32, fq.dimension)
	fq.decodeOffsets = make([]float32, fq.dimension)
	for d := 0; d < fq.dimension; d++ {
		level := fq.levelForDimensionLocked(d)
		fq.levels[d] = level
		fq.bitWidths[d] = bitsForLevel(level)
		halfL, offset, shift := fsqLevelParams(level)
		fq.halfLevels[d] = halfL
		fq.levelOffsets[d] = offset
		fq.levelShifts[d] = shift
		halfWidth := float32(level / 2)
		if halfWidth > 0 && fq.maxValues[d] > fq.minValues[d] {
			fq.decodeScales[d] = (fq.maxValues[d] - fq.minValues[d]) / (2 * halfWidth)
		}
		fq.decodeOffsets[d] = fq.minValues[d]
	}

	fq.trained = true
	fq.memoryUsage = int64((len(fq.minValues)+len(fq.maxValues)+len(fq.halfLevels)+len(fq.levelOffsets)+len(fq.levelShifts)+len(fq.decodeScales)+len(fq.decodeOffsets))*4 +
		len(fq.levels)*8 + len(fq.bitWidths)*8)
	return nil
}

func (fq *FSQQuantizer) Compress(vector []float32) ([]byte, error) {
	fq.mu.RLock()
	defer fq.mu.RUnlock()

	if !fq.trained {
		return nil, NewQuantizationError(ErrQuantNotTrained, "FSQQuantizer", "", "quantizer not trained")
	}
	if len(vector) != fq.dimension {
		return nil, fmt.Errorf("vector dimension %d does not match expected %d", len(vector), fq.dimension)
	}

	totalBits := fq.totalBitsLocked()
	compressed := make([]byte, (totalBits+7)/8)
	bitOffset := 0
	for d := 0; d < fq.dimension; d++ {
		code := fq.quantizeCodeLocked(vector[d], d)
		packBits(compressed, bitOffset, fq.bitWidths[d], code)
		bitOffset += fq.bitWidths[d]
	}
	return compressed, nil
}

func (fq *FSQQuantizer) Decompress(data []byte) ([]float32, error) {
	fq.mu.RLock()
	defer fq.mu.RUnlock()

	if !fq.trained {
		return nil, NewQuantizationError(ErrQuantNotTrained, "FSQQuantizer", "", "quantizer not trained")
	}

	vector := make([]float32, fq.dimension)
	bitOffset := 0
	for d := 0; d < fq.dimension; d++ {
		code, err := unpackBits(data, bitOffset, fq.bitWidths[d])
		if err != nil {
			return nil, err
		}
		bitOffset += fq.bitWidths[d]
		vector[d] = fq.decodeCodeLocked(code, d)
	}
	return vector, nil
}

func (fq *FSQQuantizer) Distance(compressed1, compressed2 []byte) (float32, error) {
	fq.mu.RLock()
	defer fq.mu.RUnlock()

	if !fq.trained {
		return 0, NewQuantizationError(ErrQuantNotTrained, "FSQQuantizer", "", "quantizer not trained")
	}

	var sum float32
	bitOffset := 0
	for d := 0; d < fq.dimension; d++ {
		c1, err := unpackBits(compressed1, bitOffset, fq.bitWidths[d])
		if err != nil {
			return 0, err
		}
		c2, err := unpackBits(compressed2, bitOffset, fq.bitWidths[d])
		if err != nil {
			return 0, err
		}
		bitOffset += fq.bitWidths[d]
		diff := fq.decodeCodeLocked(c1, d) - fq.decodeCodeLocked(c2, d)
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum))), nil
}

func (fq *FSQQuantizer) PrepareQuery(query []float32) any {
	return nil
}

func (fq *FSQQuantizer) DistanceToQuery(compressed []byte, query []float32, state any) (float32, error) {
	fq.mu.RLock()
	defer fq.mu.RUnlock()

	if !fq.trained {
		return 0, NewQuantizationError(ErrQuantNotTrained, "FSQQuantizer", "", "quantizer not trained")
	}
	if len(query) != fq.dimension {
		return 0, fmt.Errorf("query dimension %d does not match expected %d", len(query), fq.dimension)
	}

	var sum float32
	bitOffset := 0
	for d := 0; d < fq.dimension; d++ {
		code, err := unpackBits(compressed, bitOffset, fq.bitWidths[d])
		if err != nil {
			return 0, err
		}
		bitOffset += fq.bitWidths[d]
		diff := query[d] - fq.decodeCodeLocked(code, d)
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum))), nil
}

func (fq *FSQQuantizer) CompressionRatio() float32 {
	fq.mu.RLock()
	defer fq.mu.RUnlock()
	if !fq.trained {
		return 0
	}
	return float32(fq.dimension*32) / float32(fq.totalBitsLocked())
}

func (fq *FSQQuantizer) MemoryUsage() int64 {
	fq.mu.RLock()
	defer fq.mu.RUnlock()
	return fq.memoryUsage
}

func (fq *FSQQuantizer) IsTrained() bool {
	fq.mu.RLock()
	defer fq.mu.RUnlock()
	return fq.trained
}

func (fq *FSQQuantizer) Config() *QuantizationConfig {
	fq.mu.RLock()
	defer fq.mu.RUnlock()
	if fq.config == nil {
		return nil
	}
	configCopy := *fq.config
	if fq.config.Levels != nil {
		configCopy.Levels = append([]int(nil), fq.config.Levels...)
	}
	return &configCopy
}

func (fq *FSQQuantizer) Close() error {
	return nil
}

func (fq *FSQQuantizer) levelForDimensionLocked(d int) int {
	if len(fq.config.Levels) > 0 {
		return fq.config.Levels[d%len(fq.config.Levels)]
	}
	return 1 << fq.config.Bits
}

func (fq *FSQQuantizer) totalBitsLocked() int {
	total := 0
	for _, bits := range fq.bitWidths {
		total += bits
	}
	return total
}

func (fq *FSQQuantizer) quantizeCodeLocked(value float32, d int) uint32 {
	normalized := fq.normalizeLocked(value, d)
	zhat := fsqBoundAndRound(normalized, fq.halfLevels[d], fq.levelOffsets[d], fq.levelShifts[d])
	code := int(zhat + float32(fq.levels[d]/2))
	if code < 0 {
		code = 0
	}
	if code >= fq.levels[d] {
		code = fq.levels[d] - 1
	}
	return uint32(code)
}

func (fq *FSQQuantizer) decodeCodeLocked(code uint32, d int) float32 {
	level := fq.levels[d]
	if int(code) >= level {
		code = uint32(level - 1)
	}
	return fq.decodeOffsets[d] + float32(code)*fq.decodeScales[d]
}

func (fq *FSQQuantizer) normalizeLocked(value float32, d int) float32 {
	minValue := fq.minValues[d]
	maxValue := fq.maxValues[d]
	if maxValue <= minValue {
		return 0
	}
	if value < minValue {
		value = minValue
	} else if value > maxValue {
		value = maxValue
	}
	return 2*((value-minValue)/(maxValue-minValue)) - 1
}

func (fq *FSQQuantizer) denormalizeLocked(value float32, d int) float32 {
	minValue := fq.minValues[d]
	maxValue := fq.maxValues[d]
	if maxValue <= minValue {
		return minValue
	}
	return ((value + 1) * 0.5 * (maxValue - minValue)) + minValue
}

func fsqLevelParams(level int) (float32, float32, float32) {
	halfL := float32(level-1) * (1 - fsqEpsilon) / 2
	if halfL <= 0 {
		return 0, 0, 0
	}
	offset := float32(0)
	if level%2 == 0 {
		offset = 0.5
	}
	shift := float32(math.Tanh(float64(offset / halfL)))
	return halfL, offset, shift
}

func fsqBoundAndRound(value float32, halfL, offset, shift float32) float32 {
	if halfL <= 0 {
		return 0
	}
	bounded := float32(math.Tanh(float64(value+shift)))*halfL - offset
	return float32(math.Round(float64(bounded)))
}

func bitsForLevel(level int) int {
	bits := 0
	value := level - 1
	for value > 0 {
		bits++
		value >>= 1
	}
	if bits == 0 {
		return 1
	}
	return bits
}

func sampleVectors(vectors [][]float32, n int) [][]float32 {
	if n >= len(vectors) {
		return vectors
	}
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

func packBits(data []byte, bitOffset, numBits int, value uint32) {
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

func unpackBits(data []byte, bitOffset, numBits int) (uint32, error) {
	value := uint32(0)
	for i := 0; i < numBits; i++ {
		byteIdx := (bitOffset + i) / 8
		bitIdx := (bitOffset + i) % 8
		if byteIdx >= len(data) {
			return 0, fmt.Errorf("insufficient data: expected %d bits, got %d bytes", numBits, len(data))
		}
		if (data[byteIdx]>>bitIdx)&1 == 1 {
			value |= 1 << i
		}
	}
	return value, nil
}

type FSQQuantizerFactory struct{}

func NewFSQQuantizerFactory() *FSQQuantizerFactory {
	return &FSQQuantizerFactory{}
}

func (f *FSQQuantizerFactory) Create(config *QuantizationConfig) (Quantizer, error) {
	if config.Type != FiniteScalarQuantization {
		return nil, fmt.Errorf("unsupported quantization type: %s", config.Type.String())
	}
	fq := NewFSQQuantizer()
	if err := fq.Configure(config); err != nil {
		return nil, err
	}
	return fq, nil
}

func (f *FSQQuantizerFactory) Supports(qType QuantizationType) bool {
	return qType == FiniteScalarQuantization
}

func (f *FSQQuantizerFactory) Name() string {
	return "FSQQuantizer"
}
