package quant

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// ProductQuantizer implements Product Quantization (PQ) algorithm
type ProductQuantizer struct {
	mu sync.RWMutex

	// Configuration
	config *QuantizationConfig

	// Training state
	trained   bool
	dimension int
	subspaces int           // Number of subspaces (codebooks)
	subDim    int           // Dimension of each subspace
	centroids [][][]float32 // [subspace][centroid][dimension]

	// Distance lookup tables for fast computation
	distanceTables [][]float32 // [subspace][centroid] -> distance to query subvector
	queryVector    []float32   // Current query vector for distance tables

	// Memory usage tracking
	memoryUsage int64
}

// NewProductQuantizer creates a new Product Quantizer instance
func NewProductQuantizer() *ProductQuantizer {
	return &ProductQuantizer{
		trained: false,
	}
}

// Configure sets the quantization configuration
func (pq *ProductQuantizer) Configure(config *QuantizationConfig) error {
	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}

	if err := config.Validate(); err != nil {
		return fmt.Errorf("invalid config: %w", err)
	}

	if config.Type != ProductQuantization {
		return fmt.Errorf("expected ProductQuantization type, got %s", config.Type.String())
	}

	pq.mu.Lock()
	defer pq.mu.Unlock()

	pq.config = config
	pq.subspaces = config.Codebooks

	return nil
}

// Train trains the quantizer using k-means clustering on vector subspaces
func (pq *ProductQuantizer) Train(ctx context.Context, vectors [][]float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no training vectors provided")
	}

	if pq.config == nil {
		return fmt.Errorf("quantizer not configured")
	}

	pq.mu.Lock()
	defer pq.mu.Unlock()

	// Initialize dimensions
	pq.dimension = len(vectors[0])
	pq.subDim = pq.dimension / pq.subspaces

	if pq.dimension%pq.subspaces != 0 {
		return fmt.Errorf("dimension %d must be divisible by number of codebooks %d",
			pq.dimension, pq.subspaces)
	}

	// Validate all vectors have same dimension
	for i, vec := range vectors {
		if len(vec) != pq.dimension {
			return fmt.Errorf("vector %d has dimension %d, expected %d", i, len(vec), pq.dimension)
		}
	}

	// Sample training vectors based on train ratio
	numTraining := int(float64(len(vectors)) * pq.config.TrainRatio)
	if numTraining < pq.subspaces {
		numTraining = len(vectors) // Use all vectors if too few
	}

	trainingVectors := pq.sampleVectors(vectors, numTraining)

	// Initialize centroids for each subspace
	numCentroids := 1 << pq.config.Bits // 2^bits centroids per codebook
	pq.centroids = make([][][]float32, pq.subspaces)

	for s := 0; s < pq.subspaces; s++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Extract subvectors for this subspace
		subvectors := make([][]float32, len(trainingVectors))
		for i, vec := range trainingVectors {
			start := s * pq.subDim
			end := start + pq.subDim
			subvectors[i] = vec[start:end]
		}

		// Train codebook for this subspace using k-means
		centroids, err := pq.trainCodebook(ctx, subvectors, numCentroids)
		if err != nil {
			return fmt.Errorf("failed to train codebook for subspace %d: %w", s, err)
		}

		pq.centroids[s] = centroids
	}

	// Initialize distance tables
	pq.distanceTables = make([][]float32, pq.subspaces)
	for s := 0; s < pq.subspaces; s++ {
		pq.distanceTables[s] = make([]float32, numCentroids)
	}

	pq.trained = true
	pq.updateMemoryUsage()

	return nil
}

// trainCodebook trains a single codebook using k-means clustering
func (pq *ProductQuantizer) trainCodebook(ctx context.Context, vectors [][]float32, k int) ([][]float32, error) {
	if len(vectors) == 0 {
		return nil, fmt.Errorf("no vectors to train codebook")
	}

	dim := len(vectors[0])
	if k > len(vectors) {
		k = len(vectors) // Can't have more centroids than vectors
	}

	// Initialize centroids randomly
	centroids := make([][]float32, k)
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < k; i++ {
		centroids[i] = make([]float32, dim)
		// Initialize with random vector from training set
		randIdx := rand.Intn(len(vectors))
		copy(centroids[i], vectors[randIdx])
	}

	// K-means iterations
	maxIterations := 100
	tolerance := 1e-6

	for iter := 0; iter < maxIterations; iter++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Assignment step: assign each vector to nearest centroid
		assignments := make([]int, len(vectors))
		for i, vec := range vectors {
			minDist := float32(math.Inf(1))
			bestCentroid := 0

			for j, centroid := range centroids {
				dist := pq.euclideanDistance(vec, centroid)
				if dist < minDist {
					minDist = dist
					bestCentroid = j
				}
			}
			assignments[i] = bestCentroid
		}

		// Update step: recompute centroids
		newCentroids := make([][]float32, k)
		counts := make([]int, k)

		for i := 0; i < k; i++ {
			newCentroids[i] = make([]float32, dim)
		}

		for i, vec := range vectors {
			centroidIdx := assignments[i]
			counts[centroidIdx]++
			for d := 0; d < dim; d++ {
				newCentroids[centroidIdx][d] += vec[d]
			}
		}

		// Average and check convergence
		converged := true
		for i := 0; i < k; i++ {
			if counts[i] > 0 {
				for d := 0; d < dim; d++ {
					newCentroids[i][d] /= float32(counts[i])
				}

				// Check convergence
				if pq.euclideanDistance(centroids[i], newCentroids[i]) > float32(tolerance) {
					converged = false
				}
			} else {
				// Empty cluster - reinitialize with random vector
				randIdx := rand.Intn(len(vectors))
				copy(newCentroids[i], vectors[randIdx])
				converged = false
			}
		}

		centroids = newCentroids

		if converged {
			break
		}
	}

	return centroids, nil
}

// Compress compresses a vector using the trained codebooks
func (pq *ProductQuantizer) Compress(vector []float32) ([]byte, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if !pq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}

	if len(vector) != pq.dimension {
		return nil, fmt.Errorf("vector dimension %d does not match expected %d",
			len(vector), pq.dimension)
	}

	// Calculate number of bytes needed
	bitsPerCode := pq.config.Bits
	totalBits := pq.subspaces * bitsPerCode
	numBytes := (totalBits + 7) / 8 // Round up to nearest byte

	compressed := make([]byte, numBytes)
	bitOffset := 0

	// Quantize each subspace
	for s := 0; s < pq.subspaces; s++ {
		// Extract subvector
		start := s * pq.subDim
		end := start + pq.subDim
		subvector := vector[start:end]

		// Find nearest centroid
		minDist := float32(math.Inf(1))
		bestCode := 0

		for c, centroid := range pq.centroids[s] {
			dist := pq.euclideanDistance(subvector, centroid)
			if dist < minDist {
				minDist = dist
				bestCode = c
			}
		}

		// Pack the code into compressed bytes
		pq.packBits(compressed, bitOffset, bitsPerCode, uint32(bestCode))
		bitOffset += bitsPerCode
	}

	return compressed, nil
}

// Decompress decompresses quantized data back to a vector
func (pq *ProductQuantizer) Decompress(data []byte) ([]float32, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if !pq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}

	vector := make([]float32, pq.dimension)
	bitOffset := 0
	bitsPerCode := pq.config.Bits

	// Decompress each subspace
	for s := 0; s < pq.subspaces; s++ {
		// Extract code from compressed data
		code := pq.unpackBits(data, bitOffset, bitsPerCode)
		bitOffset += bitsPerCode

		if int(code) >= len(pq.centroids[s]) {
			return nil, fmt.Errorf("invalid code %d for subspace %d", code, s)
		}

		// Copy centroid to output vector
		start := s * pq.subDim
		centroid := pq.centroids[s][code]
		copy(vector[start:start+pq.subDim], centroid)
	}

	return vector, nil
}

// Distance computes distance between two compressed vectors
func (pq *ProductQuantizer) Distance(compressed1, compressed2 []byte) (float32, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if !pq.trained {
		return 0, fmt.Errorf("quantizer not trained")
	}

	distance := float32(0)
	bitOffset := 0
	bitsPerCode := pq.config.Bits

	// Compute distance for each subspace
	for s := 0; s < pq.subspaces; s++ {
		// Extract codes from both compressed vectors
		code1 := pq.unpackBits(compressed1, bitOffset, bitsPerCode)
		code2 := pq.unpackBits(compressed2, bitOffset, bitsPerCode)
		bitOffset += bitsPerCode

		if int(code1) >= len(pq.centroids[s]) || int(code2) >= len(pq.centroids[s]) {
			return 0, fmt.Errorf("invalid codes for subspace %d", s)
		}

		// Compute distance between centroids
		centroid1 := pq.centroids[s][code1]
		centroid2 := pq.centroids[s][code2]
		subDist := pq.euclideanDistance(centroid1, centroid2)
		distance += subDist * subDist // Squared Euclidean distance
	}

	return float32(math.Sqrt(float64(distance))), nil
}

// DistanceToQuery computes distance from compressed vector to query vector
func (pq *ProductQuantizer) DistanceToQuery(compressed []byte, query []float32) (float32, error) {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	if !pq.trained {
		return 0, fmt.Errorf("quantizer not trained")
	}

	if len(query) != pq.dimension {
		return 0, fmt.Errorf("query dimension %d does not match expected %d",
			len(query), pq.dimension)
	}

	// Update distance tables if query changed
	if !pq.vectorsEqual(pq.queryVector, query) {
		pq.updateDistanceTables(query)
		pq.queryVector = make([]float32, len(query))
		copy(pq.queryVector, query)
	}

	distance := float32(0)
	bitOffset := 0
	bitsPerCode := pq.config.Bits

	// Compute distance using precomputed tables
	for s := 0; s < pq.subspaces; s++ {
		code := pq.unpackBits(compressed, bitOffset, bitsPerCode)
		bitOffset += bitsPerCode

		if int(code) >= len(pq.distanceTables[s]) {
			return 0, fmt.Errorf("invalid code %d for subspace %d", code, s)
		}

		subDist := pq.distanceTables[s][code]
		distance += subDist * subDist // Squared distance
	}

	return float32(math.Sqrt(float64(distance))), nil
}

// updateDistanceTables precomputes distance tables for fast query processing
func (pq *ProductQuantizer) updateDistanceTables(query []float32) {
	for s := 0; s < pq.subspaces; s++ {
		start := s * pq.subDim
		end := start + pq.subDim
		querySubvector := query[start:end]

		for c, centroid := range pq.centroids[s] {
			pq.distanceTables[s][c] = pq.euclideanDistance(querySubvector, centroid)
		}
	}
}

// Helper functions

func (pq *ProductQuantizer) euclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}

	sum := float32(0)
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

func (pq *ProductQuantizer) vectorsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (pq *ProductQuantizer) sampleVectors(vectors [][]float32, n int) [][]float32 {
	if n >= len(vectors) {
		return vectors
	}

	rand.Seed(time.Now().UnixNano())
	indices := rand.Perm(len(vectors))[:n]

	sampled := make([][]float32, n)
	for i, idx := range indices {
		sampled[i] = vectors[idx]
	}

	return sampled
}

func (pq *ProductQuantizer) packBits(data []byte, bitOffset, numBits int, value uint32) {
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

func (pq *ProductQuantizer) unpackBits(data []byte, bitOffset, numBits int) uint32 {
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

func (pq *ProductQuantizer) updateMemoryUsage() {
	usage := int64(0)

	// Centroids memory
	for _, subspace := range pq.centroids {
		for _, centroid := range subspace {
			usage += int64(len(centroid) * 4) // 4 bytes per float32
		}
	}

	// Distance tables memory
	for _, table := range pq.distanceTables {
		usage += int64(len(table) * 4) // 4 bytes per float32
	}

	// Query vector memory
	if pq.queryVector != nil {
		usage += int64(len(pq.queryVector) * 4)
	}

	pq.memoryUsage = usage
}

// Interface implementation

func (pq *ProductQuantizer) CompressionRatio() float32 {
	if !pq.trained {
		return 0
	}

	originalBits := pq.dimension * 32 // 32 bits per float32
	compressedBits := pq.subspaces * pq.config.Bits

	return float32(originalBits) / float32(compressedBits)
}

func (pq *ProductQuantizer) MemoryUsage() int64 {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	return pq.memoryUsage
}

func (pq *ProductQuantizer) IsTrained() bool {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	return pq.trained
}

func (pq *ProductQuantizer) Config() *QuantizationConfig {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	if pq.config == nil {
		return nil
	}

	// Return a copy to prevent external modification
	configCopy := *pq.config
	return &configCopy
}

// ProductQuantizerFactory creates ProductQuantizer instances
type ProductQuantizerFactory struct{}

func NewProductQuantizerFactory() *ProductQuantizerFactory {
	return &ProductQuantizerFactory{}
}

func (f *ProductQuantizerFactory) Create(config *QuantizationConfig) (Quantizer, error) {
	if config.Type != ProductQuantization {
		return nil, fmt.Errorf("unsupported quantization type: %s", config.Type.String())
	}

	pq := NewProductQuantizer()
	if err := pq.Configure(config); err != nil {
		return nil, err
	}

	return pq, nil
}

func (f *ProductQuantizerFactory) Supports(qType QuantizationType) bool {
	return qType == ProductQuantization
}

func (f *ProductQuantizerFactory) Name() string {
	return "ProductQuantizer"
}
