package quant

import (
	"github.com/xDarkicex/memory"

	"context"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// ProductQuantizer implements Product Quantization (PQ) algorithm
type ProductQuantizer struct {
	config      *QuantizationConfig
	centroids   [][][]float32
	dimension   int
	subspaces   int
	subDim      int
	memoryUsage int64
	mu          sync.RWMutex
	trained     bool
}

// CodeSize returns the byte length of a single compressed vector for
// product quantization: ceil(subspaces * bits / 8). Returns 0 if the
// quantizer has not been trained yet.
func (pq *ProductQuantizer) CodeSize() int {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	if !pq.trained || pq.config == nil {
		return 0
	}
	return (pq.subspaces*pq.config.Bits + 7) / 8
}

func (pq *ProductQuantizer) Dimension() int {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	if !pq.trained {
		return 0
	}
	return pq.dimension
}

func (pq *ProductQuantizer) SerializeState() ([]byte, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	if !pq.trained {
		return nil, fmt.Errorf("ProductQuantizer not trained")
	}
	buf := make([]byte, 0, 8*1024)
	w := &pqStateWriter{buf: buf}
	w.u32(uint32(pq.dimension))
	w.u32(uint32(pq.subspaces))
	cps := 0
	if pq.subspaces > 0 && len(pq.centroids) > 0 {
		cps = len(pq.centroids[0])
	}
	w.u32(uint32(cps))
	w.u32(uint32(pq.subDim))
	w.u32(uint32(pq.config.Bits))
	w.f64(pq.config.TrainRatio)
	w.u32(uint32(pq.config.CacheSize))
	for _, ss := range pq.centroids {
		for _, c := range ss {
			for _, v := range c {
				w.f32(v)
			}
		}
	}
	return w.buf, nil
}

func (pq *ProductQuantizer) DeserializeState(data []byte) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	if len(data) < 32 {
		return fmt.Errorf("PQ DeserializeState: too short (%d < 32)", len(data))
	}
	r := &pqStateReader{buf: data}
	dim, err := r.u32()
	if err != nil {
		return fmt.Errorf("PQ dim: %w", err)
	}
	subsp, err := r.u32()
	if err != nil {
		return fmt.Errorf("PQ subsp: %w", err)
	}
	cps, err := r.u32()
	if err != nil {
		return fmt.Errorf("PQ cps: %w", err)
	}
	subDim, err := r.u32()
	if err != nil {
		return fmt.Errorf("PQ subDim: %w", err)
	}
	bitsV, err := r.u32()
	if err != nil {
		return fmt.Errorf("PQ bits: %w", err)
	}
	tr, err := r.f64()
	if err != nil {
		return fmt.Errorf("PQ trainRatio: %w", err)
	}
	cs, err := r.u32()
	if err != nil {
		return fmt.Errorf("PQ cacheSize: %w", err)
	}
	// Validate header fields.
	if dim < 1 || dim > 65536 {
		return fmt.Errorf("PQ dim %d invalid", dim)
	}
	if subsp < 1 || subsp > 1024 {
		return fmt.Errorf("PQ subsp %d invalid", subsp)
	}
	if subDim < 1 || subDim > 65536 {
		return fmt.Errorf("PQ subDim %d invalid", subDim)
	}
	if int(subsp)*int(subDim) != int(dim) {
		return fmt.Errorf("PQ geometry mismatch: %d*%d != %d", subsp, subDim, dim)
	}
	if cps < 1 || cps > 65536 {
		return fmt.Errorf("PQ cps %d invalid", cps)
	}
	if int64(bitsV) < 1 || int64(bitsV) > 16 {
		return fmt.Errorf("PQ bits %d invalid", bitsV)
	}
	if math.IsNaN(tr) || math.IsInf(tr, 0) || tr <= 0 || tr > 1 {
		return fmt.Errorf("PQ trainRatio %f invalid", tr)
	}
	// Calculate exact expected length with int64 to avoid uint32 wrap.
	fps := int64(cps) * int64(subDim)
	tf := int64(subsp) * fps
	if int64(subsp) > 0 && tf/int64(subsp) != fps {
		return fmt.Errorf("PQ state overflow: subsp=%d cps=%d subDim=%d", subsp, cps, subDim)
	}
	tfb := tf * 4
	if tf > 0 && tfb/tf != 4 {
		return fmt.Errorf("PQ state overflow: totalFloats=%d", tf)
	}
	expected := int64(32) + tfb
	const qCeil = int64(1 << 26)
	if expected > qCeil {
		return fmt.Errorf("PQ state %d bytes exceeds ceiling %d", expected, qCeil)
	}
	if len(data) != int(expected) {
		return fmt.Errorf("PQ DeserializeState: len=%d expected=%d", len(data), expected)
	}
	// All fields validated; commit.
	pq.dimension = int(dim)
	pq.subspaces = int(subsp)
	pq.subDim = int(subDim)
	pq.config = &QuantizationConfig{Type: ProductQuantization, Codebooks: int(subsp), Bits: int(bitsV), TrainRatio: tr, CacheSize: int(cs)}
	pq.centroids = make([][][]float32, int(subsp))
	for s := uint32(0); s < subsp; s++ {
		pq.centroids[s] = make([][]float32, cps)
		for c := uint32(0); c < cps; c++ {
			pq.centroids[s][c] = make([]float32, subDim)
			for d := uint32(0); d < subDim; d++ {
				v, err := r.f32()
				if err != nil {
					return fmt.Errorf("PQ centroids[%d][%d][%d]: %w", s, c, d, err)
				}
				pq.centroids[s][c][d] = v
			}
		}
	}
	pq.trained = true
	pq.updateMemoryUsage()
	return nil
}

type pqStateWriter struct{ buf []byte }

func (w *pqStateWriter) u32(v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	w.buf = append(w.buf, b[:]...)
}
func (w *pqStateWriter) f32(v float32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
	w.buf = append(w.buf, b[:]...)
}
func (w *pqStateWriter) f64(v float64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
	w.buf = append(w.buf, b[:]...)
}

type pqStateReader struct {
	buf []byte
	pos int
}

func (r *pqStateReader) need(n int) error {
	if r.pos+n > len(r.buf) {
		return fmt.Errorf("truncated at pos %d, need %d", r.pos, n)
	}
	return nil
}
func (r *pqStateReader) u32() (uint32, error) {
	if err := r.need(4); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint32(r.buf[r.pos:])
	r.pos += 4
	return v, nil
}
func (r *pqStateReader) f32() (float32, error) {
	if err := r.need(4); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint32(r.buf[r.pos:])
	r.pos += 4
	return math.Float32frombits(v), nil
}
func (r *pqStateReader) f64() (float64, error) {
	if err := r.need(8); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint64(r.buf[r.pos:])
	r.pos += 8
	return math.Float64frombits(v), nil
}

// PrepareQuery precomputes distance tables for a query so that concurrent
// DistanceToQuery calls with the same query only read from the cache.
func (pq *ProductQuantizer) PrepareQuery(query []float32) any {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if !pq.trained || len(query) != pq.dimension {
		return nil
	}

	return pq.buildDistanceTables(query)
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

	pool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  64 * 1024 * 1024, // 64MB hard limit
		SlabSize:  2 * 1024 * 1024,  // 2MB slabs
		SlabCount: 4,
		Prealloc:  false,
	}, 64)
	if err != nil {
		return fmt.Errorf("failed to create memory pool for kmeans: %w", err)
	}
	defer pool.Free()

	// Pre-allocate scratch arrays off-heap once for all subspaces
	assignments := memory.MustPoolSlice[int](pool, len(trainingVectors))
	assignments = assignments[:len(trainingVectors)]

	newCentroids := memory.MustPoolSlice[[]float32](pool, numCentroids)
	newCentroids = newCentroids[:numCentroids]
	for i := 0; i < numCentroids; i++ {
		newCentroids[i] = memory.MustPoolSlice[float32](pool, pq.subDim)
		newCentroids[i] = newCentroids[i][:pq.subDim]
	}

	counts := memory.MustPoolSlice[int](pool, numCentroids)
	counts = counts[:numCentroids]

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
		centroids, err := pq.trainCodebook(ctx, subvectors, numCentroids, assignments, newCentroids, counts)
		if err != nil {
			return fmt.Errorf("failed to train codebook for subspace %d: %w", s, err)
		}

		pq.centroids[s] = centroids
	}

	// Initialization of distance tables is removed since they are now computed dynamically

	pq.trained = true
	pq.updateMemoryUsage()

	return nil
}

// trainCodebook trains a single codebook using k-means clustering
func (pq *ProductQuantizer) trainCodebook(ctx context.Context, vectors [][]float32, k int, assignments []int, newCentroids [][]float32, counts []int) ([][]float32, error) {
	if len(vectors) == 0 {
		return nil, fmt.Errorf("no vectors to train codebook")
	}

	dim := len(vectors[0])
	if k > len(vectors) {
		k = len(vectors) // Can't have more centroids than vectors
	}

	// Initialize centroids randomly
	centroids := make([][]float32, k)

	localRand := rand.New(rand.NewSource(time.Now().UnixNano()))

	for i := 0; i < k; i++ {
		centroids[i] = make([]float32, dim)
		// Initialize with random vector from training set
		randIdx := localRand.Intn(len(vectors))
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

		// Zero scratch arrays instead of reallocating
		for i := 0; i < k; i++ {
			counts[i] = 0
			for d := 0; d < dim; d++ {
				newCentroids[i][d] = 0
			}
		}

		// Assignment step: assign each vector to nearest centroid
		for i, vec := range vectors {
			minDist := float32(math.Inf(1))
			bestCentroid := 0

			for j, centroid := range centroids {
				dist := pq.squaredEuclideanDistance(vec, centroid)
				if dist < minDist {
					minDist = dist
					bestCentroid = j
				}
			}
			assignments[i] = bestCentroid
		}

		// Update step: recompute centroids
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

		for i := 0; i < k; i++ {
			copy(centroids[i], newCentroids[i])
		}

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
		return nil, NewQuantizationError(ErrQuantNotTrained, "ProductQuantizer", "", "quantizer not trained")
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
	for i := range compressed {
		compressed[i] = 0
	}
	bitOffset := 0

	// Quantize each subspace
	for s := 0; s < pq.subspaces; s++ {
		// Extract subvector
		start := s * pq.subDim
		end := start + pq.subDim
		subvector := vector[start:end]

		// Find nearest centroid using squared distance — sqrt is
		// monotonic, so argmin(sqrt(d)) ≡ argmin(d).
		minDist := float32(math.Inf(1))
		bestCode := 0

		for c, centroid := range pq.centroids[s] {
			dist := pq.squaredEuclideanDistance(subvector, centroid)
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
		return nil, NewQuantizationError(ErrQuantNotTrained, "ProductQuantizer", "", "quantizer not trained")
	}

	vector := make([]float32, pq.dimension)
	bitOffset := 0
	bitsPerCode := pq.config.Bits

	// Decompress each subspace
	for s := 0; s < pq.subspaces; s++ {
		// Extract code from compressed data
		code, err := pq.unpackBits(data, bitOffset, bitsPerCode)
		if err != nil {
			return nil, err
		}
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
		return 0, NewQuantizationError(ErrQuantNotTrained, "ProductQuantizer", "", "quantizer not trained")
	}

	distance := float32(0)
	bitOffset := 0
	bitsPerCode := pq.config.Bits

	// Compute distance for each subspace
	for s := 0; s < pq.subspaces; s++ {
		// Extract codes from both compressed vectors
		code1, err := pq.unpackBits(compressed1, bitOffset, bitsPerCode)
		if err != nil {
			return 0, err
		}
		code2, err := pq.unpackBits(compressed2, bitOffset, bitsPerCode)
		if err != nil {
			return 0, err
		}
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

// DistanceToQuery computes distance from compressed vector to query vector.
// state is an optional precomputed distance table returned by PrepareQuery.
func (pq *ProductQuantizer) DistanceToQuery(compressed []byte, query []float32, state any) (float32, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if !pq.trained {
		return 0, NewQuantizationError(ErrQuantNotTrained, "ProductQuantizer", "", "quantizer not trained")
	}

	if len(query) != pq.dimension {
		return 0, fmt.Errorf("query dimension %d does not match expected %d",
			len(query), pq.dimension)
	}

	var distanceTables [][]float32
	if state != nil {
		if dt, ok := state.([][]float32); ok {
			distanceTables = dt
		}
	}

	if distanceTables == nil {
		distanceTables = pq.buildDistanceTablesUnsafe(query)
	}

	distance := float32(0)
	bitOffset := 0
	bitsPerCode := pq.config.Bits

	// Compute distance using precomputed tables
	for s := 0; s < pq.subspaces; s++ {
		code, err := pq.unpackBits(compressed, bitOffset, bitsPerCode)
		if err != nil {
			return 0, err
		}
		bitOffset += bitsPerCode

		if int(code) >= len(distanceTables[s]) {
			return 0, fmt.Errorf("invalid code %d for subspace %d", code, s)
		}

		subDist := distanceTables[s][code]
		distance += subDist * subDist // Squared distance
	}

	return float32(math.Sqrt(float64(distance))), nil
}

// buildDistanceTables precomputes distance tables for fast query processing
func (pq *ProductQuantizer) buildDistanceTables(query []float32) [][]float32 {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	return pq.buildDistanceTablesUnsafe(query)
}

func (pq *ProductQuantizer) buildDistanceTablesUnsafe(query []float32) [][]float32 {
	tables := make([][]float32, pq.subspaces)
	for s := 0; s < pq.subspaces; s++ {
		tables[s] = make([]float32, len(pq.centroids[s]))
		start := s * pq.subDim
		end := start + pq.subDim
		querySubvector := query[start:end]

		for c, centroid := range pq.centroids[s] {
			tables[s][c] = pq.euclideanDistance(querySubvector, centroid)
		}
	}
	return tables
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

// squaredEuclideanDistance returns the squared Euclidean distance.
// sqrt is monotonic, so argmin(sqrt(d)) ≡ argmin(d). Dropping sqrt
// saves one instruction per centroid per subspace in Compress and
// k-means assignment.
func (pq *ProductQuantizer) squaredEuclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}
	sum := float32(0)
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
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

		mask := byte(1 << bitIdx)
		if (value>>i)&1 == 1 {
			data[byteIdx] |= mask
		} else {
			data[byteIdx] &= ^mask
		}
	}
}

func (pq *ProductQuantizer) unpackBits(data []byte, bitOffset, numBits int) (uint32, error) {
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

func (pq *ProductQuantizer) updateMemoryUsage() {
	usage := int64(0)

	// Centroids memory
	for _, subspace := range pq.centroids {
		for _, centroid := range subspace {
			usage += int64(len(centroid) * 4) // 4 bytes per float32
		}
	}

	// distanceTables and queryVector are now removed so they don't consume memory here

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

// Close releases resources used by the quantizer
func (pq *ProductQuantizer) Close() error {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	return nil
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

// GetCodebooks returns the trained PQ codebooks for persistence.
// Returns nil if the quantizer is not trained.
func (pq *ProductQuantizer) GetCodebooks() [][][]float32 {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	if !pq.trained {
		return nil
	}
	return pq.centroids
}

// SetCodebooks restores trained PQ codebooks from persistence.
func (pq *ProductQuantizer) SetCodebooks(codebooks [][][]float32, dimension, subspaces, subDim int) {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	pq.dimension = dimension
	pq.subspaces = subspaces
	pq.subDim = subDim
	pq.centroids = codebooks

	// distanceTables are no longer pre-allocated here
	pq.trained = true
	pq.updateMemoryUsage()
}

// CentroidDistance implements CentroidProvider.
// It computes the Euclidean distance between two packed centroid IDs.
// For PQ, the ID is packed as: subspace << bits | centroid.
func (pq *ProductQuantizer) CentroidDistance(a, b uint32) float32 {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	mask := uint32((1 << pq.config.Bits) - 1)
	subA := int(a >> pq.config.Bits)
	subB := int(b >> pq.config.Bits)
	centA := int(a & mask)
	centB := int(b & mask)

	if subA != subB {
		// Distance between centroids in different subspaces is mathematically undefined
		// in the context of single-subspace comparison. Return MaxFloat32.
		return math.MaxFloat32
	}

	if subA >= pq.subspaces || centA >= len(pq.centroids[subA]) || centB >= len(pq.centroids[subB]) {
		return math.MaxFloat32
	}

	vecA := pq.centroids[subA][centA]
	vecB := pq.centroids[subA][centB]

	var dist float32
	// Use manual loop instead of SIMD if it's just a small subDim vector
	for i := 0; i < pq.subDim; i++ {
		diff := vecA[i] - vecB[i]
		dist += diff * diff
	}
	return float32(math.Sqrt(float64(dist)))
}

// MaxResidualBound implements CentroidProvider.
// It returns the maximum possible length (L2 norm) of any quantized residual produced by this codebook.
func (pq *ProductQuantizer) MaxResidualBound() float32 {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	var maxNormSq float32

	for i := 0; i < pq.subspaces; i++ {
		var maxSubspaceNormSq float32
		for j := 0; j < len(pq.centroids[i]); j++ {
			var normSq float32
			for k := 0; k < pq.subDim; k++ {
				val := pq.centroids[i][j][k]
				normSq += val * val
			}
			if normSq > maxSubspaceNormSq {
				maxSubspaceNormSq = normSq
			}
		}
		maxNormSq += maxSubspaceNormSq
	}

	return float32(math.Sqrt(float64(maxNormSq)))
}
