package ivfpq

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/xDarkicex/memory"

	indexmodel "github.com/xDarkicex/libravdb/internal/index/model"
	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

// ErrHydrationConflict reports that the active IVF-PQ generation changed
// while DeserializeFromBytes was staging a replacement. Retrying against a
// quiescent index is safe; committing the stale replacement is not.
var ErrHydrationConflict = errors.New("IVF-PQ hydration conflicts with a live mutation")

// Config holds configuration for IVF-PQ index
type Config struct {
	Quantization *quant.QuantizationConfig
	Dimension    int
	NClusters    int
	NProbes      int
	Metric       util.DistanceMetric
	// RecordPoolBytes caps the off-heap budget for retained inverted-list
	// records (ordinals + compressed codes). It is a hard ceiling: when
	// the index would exceed it, allocation fails closed and Insert
	// returns an error rather than silently dropping the record. Zero
	// means "use DefaultRecordPoolBytes". Callers MUST NOT inject their
	// own *memory.Pool; the index owns its pool exclusively so that
	// Close() can deterministically release every byte.
	RecordPoolBytes uint64
	MaxIterations   int
	Tolerance       float64
	RandomSeed      int64
}

// DefaultRecordPoolBytes is the default off-heap ceiling for retained
// inverted-list records when Config.RecordPoolBytes is left zero. Sized
// to comfortably hold ~1M PQ-8 codes at 4 codebooks × 8 bits per vector
// plus their uint32 ordinals.
const DefaultRecordPoolBytes uint64 = 256 * 1024 * 1024

// VectorEntry = indexmodel.VectorEntry
type VectorEntry = indexmodel.VectorEntry

// SearchResult represents a search result
type SearchResult struct {
	Ordinal uint32
	Score   float32
}

// DefaultConfig returns a default IVF-PQ configuration
func DefaultConfig(dimension int) *Config {
	// Rule of thumb: sqrt(N) clusters for N vectors, but start with reasonable defaults
	nClusters := int(math.Max(64, math.Min(4096, float64(dimension))))

	return &Config{
		Dimension:     dimension,
		NClusters:     nClusters,
		NProbes:       min(16, nClusters/4), // Probe 25% of clusters by default
		Metric:        util.L2Distance,
		Quantization:  quant.DefaultConfig(quant.ProductQuantization),
		MaxIterations: 100,
		Tolerance:     1e-4,
		RandomSeed:    time.Now().UnixNano(),
	}
}

// AutoTuneConfig automatically tunes IVF-PQ parameters based on dataset characteristics
func AutoTuneConfig(dimension int, estimatedVectors int, targetMemoryMB int) *Config {
	// Automatic cluster count tuning based on dataset size
	var nClusters int
	if estimatedVectors < 1000 {
		nClusters = max(4, estimatedVectors/50) // Small datasets: fewer clusters
	} else if estimatedVectors < 100000 {
		nClusters = int(math.Sqrt(float64(estimatedVectors))) // Medium datasets: sqrt(N)
	} else {
		nClusters = int(math.Pow(float64(estimatedVectors), 0.4)) // Large datasets: N^0.4
	}

	// Clamp cluster count to reasonable bounds
	nClusters = max(4, min(nClusters, 16384))

	// Automatic probe count tuning for accuracy vs speed trade-off
	var nProbes int
	if estimatedVectors < 10000 {
		nProbes = max(1, nClusters/2) // Small datasets: probe more clusters for accuracy
	} else if estimatedVectors < 1000000 {
		nProbes = max(1, nClusters/4) // Medium datasets: balanced approach
	} else {
		nProbes = max(1, nClusters/8) // Large datasets: probe fewer for speed
	}

	// Clamp probe count
	nProbes = max(1, min(nProbes, nClusters))

	// Auto-tune quantization based on memory constraints
	var quantConfig *quant.QuantizationConfig
	if targetMemoryMB > 0 {
		// Estimate memory usage and adjust quantization accordingly
		estimatedMemoryMB := (estimatedVectors * dimension * 4) / (1024 * 1024) // 4 bytes per float32

		if estimatedMemoryMB > targetMemoryMB {
			// Need aggressive quantization
			quantConfig = &quant.QuantizationConfig{
				Type:       quant.ProductQuantization,
				Codebooks:  max(4, dimension/16), // More aggressive subspace division
				Bits:       4,                    // Lower bits for more compression
				TrainRatio: 0.1,
				CacheSize:  1000,
			}
		} else if estimatedMemoryMB > targetMemoryMB/2 {
			// Moderate quantization
			quantConfig = &quant.QuantizationConfig{
				Type:       quant.ProductQuantization,
				Codebooks:  max(4, dimension/8),
				Bits:       6,
				TrainRatio: 0.15,
				CacheSize:  2000,
			}
		} else {
			// Light quantization for better accuracy
			quantConfig = &quant.QuantizationConfig{
				Type:       quant.ProductQuantization,
				Codebooks:  max(4, dimension/4),
				Bits:       8,
				TrainRatio: 0.2,
				CacheSize:  5000,
			}
		}
	} else {
		// Default quantization when no memory constraint
		quantConfig = quant.DefaultConfig(quant.ProductQuantization)
	}

	return &Config{
		Dimension:     dimension,
		NClusters:     nClusters,
		NProbes:       nProbes,
		Metric:        util.L2Distance,
		Quantization:  quantConfig,
		MaxIterations: 100,
		Tolerance:     1e-4,
		RandomSeed:    time.Now().UnixNano(),
	}
}

type clusterSegment struct {
	ordinals []uint32
	codes    []byte
	used     uint32
}

type clusterStorage struct {
	segments        []*clusterSegment
	count           uint64
	segmentCapacity uint32
	// codeWidth is the byte length of a single compressed code for every
	// record currently stored. It is set from the trained quantizer at
	// construction AND updated after Train / DeserializeFromBytes so the
	// index never retains codes at a stale width. The width is canonical:
	// derived from idx.gen.quantizer.CodeSize() and never recomputed from
	// config fields alone.
	codeWidth uint32
}

// setCodeWidth updates the canonical code width used by append, delete,
// and serialization. Caller MUST pass a width derived from the trained
// quantizer (idx.gen.quantizer.CodeSize()). A zero width is valid only when
// the index has no quantizer.
func (s *clusterStorage) setCodeWidth(width uint32) {
	if s.codeWidth == width {
		return
	}
	// Width change invalidates any existing per-record code bytes. Caller
	// is responsible for clearing storage before changing the width.
	s.codeWidth = width
}

func (s *clusterStorage) append(ordinal uint32, code []byte, pool *memory.Pool) error {
	// Fail closed on any code-width mismatch, including the zero-width case.
	// A wrong-width or uninitialized code must never be published as a valid
	// ordinal — that would silently corrupt downstream distance computations.
	if s.codeWidth == 0 {
		if len(code) != 0 {
			return fmt.Errorf("clusterStorage.append: codeWidth=0 but code has %d bytes", len(code))
		}
	} else {
		if len(code) != int(s.codeWidth) {
			return fmt.Errorf("clusterStorage.append: codeWidth=%d but code has %d bytes", s.codeWidth, len(code))
		}
	}

	var seg *clusterSegment
	if len(s.segments) == 0 {
		var err error
		seg, err = s.allocateSegment(pool)
		if err != nil {
			return err
		}
		s.segments = append(s.segments, seg)
	} else {
		seg = s.segments[len(s.segments)-1]
		if seg.used >= s.segmentCapacity {
			var err error
			seg, err = s.allocateSegment(pool)
			if err != nil {
				return err
			}
			s.segments = append(s.segments, seg)
		}
	}

	seg.ordinals[seg.used] = ordinal
	if s.codeWidth > 0 {
		copy(seg.codes[seg.used*s.codeWidth:], code)
	}
	seg.used++
	s.count++
	return nil
}

// clear logically resets every segment without freeing the underlying pool
// memory. The retained segment backing arrays remain valid for reuse by a
// subsequent append call. Used by hydration paths that want to overwrite
// the in-memory inverted lists in place.
func (s *clusterStorage) clear() {
	for _, seg := range s.segments {
		seg.used = 0
	}
	s.count = 0
}

func (s *clusterStorage) allocateSegment(pool *memory.Pool) (*clusterSegment, error) {
	ordSlice, err := memory.PoolSlice[uint32](pool, int(s.segmentCapacity))
	if err != nil {
		return nil, err
	}
	ordinals := ordSlice[:int(s.segmentCapacity)]

	var codes []byte
	if s.codeWidth > 0 {
		codeSlice, err := memory.PoolSlice[byte](pool, int(s.segmentCapacity*s.codeWidth))
		if err != nil {
			return nil, err
		}
		codes = codeSlice[:int(s.segmentCapacity*s.codeWidth)]
	}

	return &clusterSegment{
		ordinals: ordinals,
		codes:    codes,
		used:     0,
	}, nil
}

func (s *clusterStorage) deleteByOrdinal(ordinal uint32) bool {
	var targetSegIdx, targetOffset int
	var found bool

	for i, seg := range s.segments {
		for j := 0; j < int(seg.used); j++ {
			if seg.ordinals[j] == ordinal {
				targetSegIdx = i
				targetOffset = j
				found = true
				break
			}
		}
		if found {
			break
		}
	}
	if !found {
		return false
	}

	lastSegIdx := len(s.segments) - 1
	lastSeg := s.segments[lastSegIdx]
	lastOffset := int(lastSeg.used) - 1

	if targetSegIdx == lastSegIdx && targetOffset == lastOffset {
		lastSeg.used--
		s.count--
	} else {
		targetSeg := s.segments[targetSegIdx]
		targetSeg.ordinals[targetOffset] = lastSeg.ordinals[lastOffset]
		if s.codeWidth > 0 {
			copy(targetSeg.codes[targetOffset*int(s.codeWidth):], lastSeg.codes[lastOffset*int(s.codeWidth):(lastOffset+1)*int(s.codeWidth)])
		}
		lastSeg.used--
		s.count--
	}
	return true
}

// Cluster represents a single inverted list cluster
type Cluster struct {
	storage       *clusterStorage
	Centroid      []float32
	ID            int
	mutex         sync.RWMutex
	centroidNorm2 float32
	centroidNorm  float32
}

// generation owns one IVF-PQ state lifetime. Searches pin it while it is
// readable; writes mutate it under Index.RLock. Hydration stages a replacement
// and may publish it only if this generation's mutation epoch is unchanged.
type generation struct {
	pool      *memory.Pool
	poolCfg   memory.AllocatorConfig
	clusters  []*Cluster
	quantizer quant.Quantizer
	config    *Config
	size      atomic.Int64
	mutation  atomic.Uint64
	trained   bool
	refs      atomic.Int32
	retired   atomic.Bool
	freed     atomic.Bool // set by drainAndFree for test verification
	id        uint64
}

var genIDSeq atomic.Uint64

func newGeneration(pool *memory.Pool, poolCfg memory.AllocatorConfig, clusters []*Cluster, q quant.Quantizer, config *Config, sz int, trained bool) *generation {
	g := &generation{
		pool: pool, poolCfg: poolCfg, clusters: clusters, quantizer: q, config: config, trained: trained,
		id: genIDSeq.Add(1),
	}
	g.refs.Store(1) // Index owner ref
	g.size.Store(int64(sz))
	return g
}

func (g *generation) acquire() { g.refs.Add(1) }
func (g *generation) release() {
	if g.refs.Add(-1) == 0 && g.retired.Load() {
		if g.quantizer != nil {
			g.quantizer.Close()
		}
		g.pool.Free()
		g.freed.Store(true)
	}
}

// Index implements IVF-PQ. The active generation holds all mutable state
// and is swapped atomically during hydration.
type Index struct {
	gen          *generation
	distanceFunc util.DistanceFunc
	searchStats  *SearchStats
	config       *Config // convenience mirror of gen.config
	scratchPool  *sync.Pool
	mutex        sync.RWMutex
	adaptiveMode atomic.Bool
	queryTiers   [4]ivfPoolTier
	rand         *rand.Rand
}

// codeSize returns the size of the compressed code for a single vector.
// It delegates to the trained quantizer's CodeSize so scalar, PQ, and FSQ
// all report their actual byte width through a single canonical source.
// Returns 0 when the index has no trained quantizer.
func (idx *Index) codeSize() int {
	if idx.gen.quantizer == nil {
		return 0
	}
	return idx.gen.quantizer.CodeSize()
}

// SearchStats tracks search performance for adaptive optimization
type SearchStats struct {
	lastAdjustment time.Time
	totalSearches  int64
	totalLatencyMs int64
	accuracySum    float64
	currentProbes  int
	mutex          sync.RWMutex
}

type candidate struct {
	ordinal     uint32
	distance    float32
	clusterDist float32
}

// ivfHeapElement is a max-heap node storing a candidate entry and its distance.
type ivfHeapElement struct {
	ordinal  uint32
	distance float32
}

// ivfUpHeap bubbles the element at i up to restore max-heap property.
func ivfUpHeap(h []ivfHeapElement, i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h[parent].distance >= h[i].distance {
			break
		}
		h[parent], h[i] = h[i], h[parent]
		i = parent
	}
}

// ivfDownHeap sifts the element at i down to restore max-heap property.
func ivfDownHeap(h []ivfHeapElement, i, n int) {
	for {
		largest := i
		left := 2*i + 1
		right := 2*i + 2
		if left < n && h[left].distance > h[largest].distance {
			largest = left
		}
		if right < n && h[right].distance > h[largest].distance {
			largest = right
		}
		if largest == i {
			break
		}
		h[i], h[largest] = h[largest], h[i]
		i = largest
	}
}

// ivfUserDataOffset is the byte offset within a ShardedFreeList slot where user
// data begins. The memory package's SFL metadata occupies offsets 0–43 (Hyaline
// chain at 0/8/16/24/32, structIdx+shardIdx at 40); 8-byte aligned to 48.
const ivfUserDataOffset = 48

// ivfHeapSlot binds an off-heap slot to its originating pool so that free()
// routes to the correct tier by construction.
type ivfHeapSlot struct {
	pool *memory.ShardedFreeList
	slot []byte
}

func (hs *ivfHeapSlot) free() { hs.pool.Deallocate(hs.slot) }

// Power-of-2 tier table. Each tier's slot is sized for its maxK.
type ivfPoolTier struct {
	pool *memory.ShardedFreeList
	maxK int
	once sync.Once
}

// acquireIVFHeapSlot returns an ivfHeapSlot paired with a []ivfHeapElement buffer
// backed by the appropriate off-heap tier. Returns nil, nil if k exceeds the
// largest tier — caller must fall back to Go heap allocation.
func (idx *Index) acquireIVFHeapSlot(k int) (*ivfHeapSlot, []ivfHeapElement) {
	for i := range idx.queryTiers {
		if k > idx.queryTiers[i].maxK {
			continue
		}
		tier := &idx.queryTiers[i]
		tier.once.Do(func() {
			slotSize := uint64(ivfUserDataOffset + tier.maxK*16)
			pool, err := memory.NewShardedFreeList(memory.FreeListConfig{
				PoolSize:  16 * 1024 * 1024,
				SlotSize:  slotSize,
				SlabSize:  1 * 1024 * 1024,
				SlabCount: 16,
				Prealloc:  true,
			}, 64, 16)
			if err != nil {
				panic("ivfpq: failed to create query pool tier: " + err.Error())
			}
			tier.pool = pool
		})
		slot, err := tier.pool.Allocate()
		if err != nil {
			return nil, nil
		}
		ptr := unsafe.Add(unsafe.Pointer(unsafe.SliceData(slot)), ivfUserDataOffset)
		heapBuf := unsafe.Slice((*ivfHeapElement)(ptr), tier.maxK)[:k]
		return &ivfHeapSlot{slot: slot, pool: tier.pool}, heapBuf
	}
	return nil, nil
}

// NewIVFPQ creates a new IVF-PQ index
func NewIVFPQ(config *Config) (*Index, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	if config.Dimension <= 0 {
		return nil, fmt.Errorf("dimension must be positive, got %d", config.Dimension)
	}

	if config.NClusters <= 0 {
		return nil, fmt.Errorf("number of clusters must be positive, got %d", config.NClusters)
	}

	if config.NProbes <= 0 || config.NProbes > config.NClusters {
		return nil, fmt.Errorf("number of probes must be between 1 and %d, got %d", config.NClusters, config.NProbes)
	}

	// Create quantizer if quantization is enabled
	var quantizer quant.Quantizer
	if config.Quantization != nil {
		var err error
		quantizer, err = quant.Create(config.Quantization)
		if err != nil {
			return nil, fmt.Errorf("failed to create quantizer: %w", err)
		}
	}

	// Get distance function
	distanceFunc, err := util.GetDistanceFunc(config.Metric)
	if err != nil {
		return nil, fmt.Errorf("failed to get distance function: %w", err)
	}

	// Initialize recordPool with a config-driven ceiling. PoolSize is the
	// hard mmap budget; SlabSize is the per-slab granularity and must be
	// ≤ PoolSize. We derive a sensible SlabSize as a power-of-two fraction
	// of the budget so callers cannot accidentally under-provision or
	// over-provision the slab list.
	poolBudget := config.RecordPoolBytes
	if poolBudget == 0 {
		poolBudget = DefaultRecordPoolBytes
	}
	if poolBudget < 1024*1024 {
		return nil, fmt.Errorf("RecordPoolBytes must be >= 1 MiB, got %d", poolBudget)
	}
	slabSize := poolBudget
	// Halve slab size until it fits inside 64 MiB so that segment allocations
	// for one cluster never starve the rest of the slab list.
	for slabSize > 64*1024*1024 {
		slabSize /= 2
	}
	recordPool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize: poolBudget,
		SlabSize: slabSize,
		// SlabCount is descriptor capacity, not extra capacity. Choose
		// enough descriptors that the index can grow within the budget
		// without descriptor-table exhaustion. 16 is enough for ≤16
		// in-flight allocations per concurrent writer; deeper parallelism
		// is still safe because Pool.Allocate grows the descriptor table.
		SlabCount:     16,
		Prealloc:      false,
		MadviseRandom: true,
	}, 64)
	if err != nil {
		return nil, fmt.Errorf("failed to create record pool: %w", err)
	}

	// Code width comes from the trained quantizer; the index must NOT
	// recompute it from config fields (scalar, PQ, and FSQ all differ).
	// A quantizer that returns 0 here means it is not yet trained, which
	// is fine — the storage codeWidth stays 0 until Insert or
	// DeserializeFromBytes produces a trained quantizer.
	codeWidth := 0
	if quantizer != nil {
		codeWidth = quantizer.CodeSize()
	}

	// Initialize clusters
	clusters := make([]*Cluster, config.NClusters)
	for i := 0; i < config.NClusters; i++ {
		clusters[i] = &Cluster{
			ID:       i,
			Centroid: make([]float32, config.Dimension),
			storage: &clusterStorage{
				segments:        make([]*clusterSegment, 0),
				segmentCapacity: 1024,
				codeWidth:       uint32(codeWidth),
			},
		}
	}

	scratchPool := &sync.Pool{
		New: func() any {
			a, _ := memory.NewArena(1024*1024, 64)
			return a
		},
	}

	poolCfg := memory.AllocatorConfig{
		PoolSize: poolBudget, SlabSize: slabSize, SlabCount: 16, Prealloc: false, MadviseRandom: true,
	}
	gen := newGeneration(recordPool, poolCfg, clusters, quantizer, config, 0, false)
	return &Index{
		gen:          gen,
		config:       config,
		distanceFunc: distanceFunc,
		scratchPool:  scratchPool,
		rand:         rand.New(rand.NewSource(config.RandomSeed)),
		searchStats: &SearchStats{
			currentProbes:  config.NProbes,
			lastAdjustment: time.Now(),
		},
		queryTiers: [4]ivfPoolTier{
			{maxK: 16},
			{maxK: 128},
			{maxK: 1024},
			{maxK: 4096},
		},
	}, nil
}

// Train trains the IVF-PQ index using k-means clustering
func (idx *Index) Train(ctx context.Context, vectors [][]float32) error {
	// Training mutates centroids, the quantizer, and every cluster's code
	// width. Keep it exclusive so hydration cannot stage from one training
	// state and commit over another, and so readers never observe a partially
	// trained generation.
	idx.mutex.Lock()
	defer idx.mutex.Unlock()
	if idx.gen == nil {
		return fmt.Errorf("index closed")
	}

	if len(vectors) == 0 {
		return fmt.Errorf("no training vectors provided")
	}

	if len(vectors) < idx.config.NClusters {
		return fmt.Errorf("need at least %d training vectors for %d clusters, got %d",
			idx.config.NClusters, idx.config.NClusters, len(vectors))
	}

	// Validate vector dimensions
	for i, vec := range vectors {
		if len(vec) != idx.config.Dimension {
			return fmt.Errorf("vector %d has dimension %d, expected %d", i, len(vec), idx.config.Dimension)
		}
	}

	// Perform k-means clustering to train coarse quantizer
	if err := idx.trainCoarseQuantizer(ctx, vectors); err != nil {
		return fmt.Errorf("failed to train coarse quantizer: %w", err)
	}

	// Train fine quantizer (PQ) if enabled
	if idx.gen.quantizer != nil {
		if err := idx.gen.quantizer.Train(ctx, vectors); err != nil {
			return fmt.Errorf("failed to train fine quantizer: %w", err)
		}
	}

	// Now that the fine quantizer is trained, propagate its canonical
	// code width to every cluster's storage. Without this, the storage
	// keeps its NewIVFPQ-time width (0 for an untrained quantizer) and
	// the next Insert would either silently drop codes or reject them.
	width := uint32(idx.codeSize())
	for _, c := range idx.gen.clusters {
		c.storage.setCodeWidth(width)
	}
	idx.gen.trained = true
	idx.gen.mutation.Add(1)
	return nil
}

// trainCoarseQuantizer performs k-means clustering to create cluster centroids
func (idx *Index) trainCoarseQuantizer(ctx context.Context, vectors [][]float32) error {
	// Initialize centroids using k-means++
	if err := idx.initializeCentroids(vectors); err != nil {
		return fmt.Errorf("failed to initialize centroids: %w", err)
	}

	prevInertia := math.Inf(1)

	for iter := 0; iter < idx.config.MaxIterations; iter++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Assignment step: assign each vector to nearest centroid.
		assignments := make([]int, len(vectors))
		totalInertia, err := idx.assignVectorsToClusters(ctx, vectors, assignments)
		if err != nil {
			return fmt.Errorf("failed during assignment step: %w", err)
		}

		// Check for convergence
		if math.Abs(prevInertia-totalInertia)/prevInertia < idx.config.Tolerance {
			break
		}
		prevInertia = totalInertia

		// Update step: recompute centroids
		if err := idx.updateCentroids(vectors, assignments); err != nil {
			return fmt.Errorf("failed to update centroids: %w", err)
		}
	}

	return nil
}

func (idx *Index) assignVectorsToClusters(ctx context.Context, vectors [][]float32, assignments []int) (float64, error) {
	// Precompute norms for fast assignment scores
	centroidNorm2 := make([]float32, len(idx.gen.clusters))
	centroidNorm := make([]float32, len(idx.gen.clusters))
	for j, cluster := range idx.gen.clusters {
		var sum float32
		for _, v := range cluster.Centroid {
			sum += v * v
		}
		centroidNorm2[j] = sum
		centroidNorm[j] = float32(math.Sqrt(float64(sum)))
	}

	workers := parallelismFor(len(vectors))
	if workers == 1 {
		totalInertia := float64(0)
		for i, vec := range vectors {
			bestCluster := 0
			bestScore := float32(math.Inf(-1))

			for j, cluster := range idx.gen.clusters {
				score := idx.computeAssignmentScore(vec, cluster.Centroid, centroidNorm2[j], centroidNorm[j])
				if score > bestScore {
					bestScore = score
					bestCluster = j
				}
			}

			assignments[i] = bestCluster
			totalInertia += float64(idx.distanceFunc(vec, idx.gen.clusters[bestCluster].Centroid))
		}
		return totalInertia, nil
	}

	chunkSize := (len(vectors) + workers - 1) / workers
	inertias := make([]float64, workers)
	errCh := make(chan error, workers)
	var wg sync.WaitGroup

	for worker := 0; worker < workers; worker++ {
		start := worker * chunkSize
		if start >= len(vectors) {
			break
		}
		end := min(start+chunkSize, len(vectors))

		wg.Add(1)
		go func(worker, start, end int) {
			defer wg.Done()

			localInertia := float64(0)
			for i := start; i < end; i++ {
				select {
				case <-ctx.Done():
					errCh <- ctx.Err()
					return
				default:
				}

				vec := vectors[i]
				bestCluster := 0
				bestScore := float32(math.Inf(-1))
				for j, cluster := range idx.gen.clusters {
					score := idx.computeAssignmentScore(vec, cluster.Centroid, centroidNorm2[j], centroidNorm[j])
					if score > bestScore {
						bestScore = score
						bestCluster = j
					}
				}

				assignments[i] = bestCluster
				localInertia += float64(idx.distanceFunc(vec, idx.gen.clusters[bestCluster].Centroid))
			}

			inertias[worker] = localInertia
		}(worker, start, end)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		if err != nil {
			return 0, err
		}
	}

	totalInertia := float64(0)
	for _, inertia := range inertias {
		totalInertia += inertia
	}
	return totalInertia, nil
}

// initializeCentroids initializes cluster centroids using k-means++
// with running-min tracking: O(N·k·dim) instead of O(N·k²·dim).
func (idx *Index) initializeCentroids(vectors [][]float32) error {
	nClusters := idx.config.NClusters
	if len(vectors) < nClusters {
		return fmt.Errorf("not enough vectors for initialization")
	}

	// Choose first centroid randomly.
	firstIdx := idx.rand.Intn(len(vectors))
	copy(idx.gen.clusters[0].Centroid, vectors[firstIdx])

	// minDist[i] tracks the squared distance from vector i to its nearest
	// already-chosen centroid. Updated incrementally as each new centroid
	// is selected — we only compute distance to the new centroid, not all k.
	minDist := make([]float64, len(vectors))
	totalDist := float64(0)
	for i, vec := range vectors {
		d := float64(idx.distanceFunc(vec, idx.gen.clusters[0].Centroid))
		minDist[i] = d * d
		totalDist += minDist[i]
	}

	// Choose remaining centroids using k-means++ (proportional to squared distance).
	for k := 1; k < nClusters; k++ {
		// Select next centroid via roulette-wheel selection.
		target := idx.rand.Float64() * totalDist
		cumulative := float64(0)
		chosenIdx := 0
		for i, d := range minDist {
			cumulative += d
			if cumulative >= target {
				chosenIdx = i
				break
			}
		}
		copy(idx.gen.clusters[k].Centroid, vectors[chosenIdx])

		// Update running-min distances: only compare against the new centroid.
		totalDist = 0
		newCentroid := idx.gen.clusters[k].Centroid
		for i, vec := range vectors {
			d := float64(idx.distanceFunc(vec, newCentroid))
			d2 := d * d
			if d2 < minDist[i] {
				minDist[i] = d2
			}
			totalDist += minDist[i]
		}
	}

	return nil
}

// updateCentroids recomputes cluster centroids based on current assignments
func (idx *Index) updateCentroids(vectors [][]float32, assignments []int) error {
	// Reset centroids
	for _, cluster := range idx.gen.clusters {
		for i := range cluster.Centroid {
			cluster.Centroid[i] = 0
		}
	}

	// Count vectors per cluster
	counts := make([]int, idx.config.NClusters)

	// Sum vectors for each cluster
	for i, vec := range vectors {
		clusterID := assignments[i]
		counts[clusterID]++

		for j, val := range vec {
			idx.gen.clusters[clusterID].Centroid[j] += val
		}
	}

	// Compute averages (avoid division by zero)
	for i, cluster := range idx.gen.clusters {
		if counts[i] > 0 {
			for j := range cluster.Centroid {
				cluster.Centroid[j] /= float32(counts[i])
			}
		} else {
			// Reinitialize empty clusters randomly
			randomIdx := idx.rand.Intn(len(vectors))
			copy(cluster.Centroid, vectors[randomIdx])
		}
		// Precompute norms for fast assignment scores
		var norm2 float32
		for _, v := range cluster.Centroid {
			norm2 += v * v
		}
		cluster.centroidNorm2 = norm2
		cluster.centroidNorm = float32(math.Sqrt(float64(norm2)))
	}

	return nil
}

// computeAssignmentScore computes a metric-specific score to maximize for cluster assignment.
// This avoids expensive sqrt or ||x|| computations where possible.
func (idx *Index) computeAssignmentScore(vec, centroid []float32, norm2, norm float32) float32 {
	switch util.DistanceMetric(idx.config.Metric) {
	case util.L2Distance:
		// argmin(||x-c||²) ≡ argmax(dot(x,c) - ||c||²/2)
		return dotProduct(vec, centroid) - norm2*0.5
	case util.InnerProduct:
		// IP is maximized when dot product is maximized
		return dotProduct(vec, centroid)
	case util.CosineDistance:
		// Cosine similarity = dot(x,c) / (||x|| * ||c||)
		// Since ||x|| is constant, we maximize dot(x,c) / ||c||
		if norm == 0 {
			return float32(math.Inf(-1))
		}
		return dotProduct(vec, centroid) / norm
	default:
		// Fallback for unknown metrics: use actual distance function (negated so smaller distance = higher score)
		return -idx.distanceFunc(vec, centroid)
	}
}

// assignToCluster finds the best cluster for a vector
func (idx *Index) assignToCluster(vector []float32) (int, error) {
	if !idx.gen.trained {
		return 0, fmt.Errorf("assignToCluster: %w", util.ErrNotTrained)
	}

	bestCluster := 0
	bestScore := float32(math.Inf(-1))

	for i, cluster := range idx.gen.clusters {
		score := idx.computeAssignmentScore(vector, cluster.Centroid, cluster.centroidNorm2, cluster.centroidNorm)
		if score > bestScore {
			bestScore = score
			bestCluster = i
		}
	}

	return bestCluster, nil
}

// findProbeClusters finds the top-k closest clusters for search probing
func (idx *Index) findProbeClusters(query []float32) ([]int, error) {
	if !idx.gen.trained {
		return nil, fmt.Errorf("Search: %w", util.ErrNotTrained)
	}

	distances := make([]clusterDistance, len(idx.gen.clusters))

	for i, cluster := range idx.gen.clusters {
		distance := idx.distanceFunc(query, cluster.Centroid)
		distances[i] = clusterDistance{id: i, distance: distance}
	}

	// Sort by distance and take top NProbes
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	probes := make([]int, min(idx.config.NProbes, len(distances)))
	for i := range probes {
		probes[i] = distances[i].id
	}

	return probes, nil
}

// IsTrained returns whether the index has been trained
func (idx *Index) IsTrained() bool {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	if idx.gen == nil {
		return false
	}
	return idx.gen.trained
}

// GetConfig returns the index configuration
func (idx *Index) GetConfig() *Config {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	if idx.config == nil {
		return nil
	}
	return cloneConfig(idx.config)
}

// GetClusterInfo returns information about clusters
func (idx *Index) GetClusterInfo() []ClusterInfo {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	if idx.gen == nil {
		return nil
	}
	info := make([]ClusterInfo, len(idx.gen.clusters))
	for i, cluster := range idx.gen.clusters {
		cluster.mutex.RLock()
		info[i] = ClusterInfo{
			ID:       cluster.ID,
			Size:     int(cluster.storage.count),
			Centroid: make([]float32, len(cluster.Centroid)),
		}
		copy(info[i].Centroid, cluster.Centroid)
		cluster.mutex.RUnlock()
	}

	return info
}

// ClusterInfo provides information about a cluster
type ClusterInfo struct {
	Centroid []float32
	ID       int
	Size     int
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func dotProduct(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

type clusterDistance struct {
	id       int
	distance float32
}

func siftDownClusterDist(h []clusterDistance, i, n int) {
	for {
		largest := i
		left := 2*i + 1
		right := 2*i + 2
		if left < n && h[left].distance > h[largest].distance {
			largest = left
		}
		if right < n && h[right].distance > h[largest].distance {
			largest = right
		}
		if largest == i {
			return
		}
		h[i], h[largest] = h[largest], h[i]
		i = largest
	}
}

// Insert adds a vector entry to the index with enhanced quantization support
func (idx *Index) Insert(ctx context.Context, entry *VectorEntry) error {
	if entry == nil {
		return fmt.Errorf("entry cannot be nil")
	}

	if len(entry.Vector) == 0 {
		return fmt.Errorf("vector cannot be empty")
	}

	if len(entry.Vector) != idx.config.Dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d",
			len(entry.Vector), idx.config.Dimension)
	}

	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	if idx.gen == nil {
		return fmt.Errorf("index closed")
	}
	gen := idx.gen
	if !gen.trained {
		return fmt.Errorf("Insert: %w", util.ErrNotTrained)
	}
	// cluster assignment, quantization, append, and size update all happen
	// under the Index RLock so a concurrent hydrate cannot swap gen mid-operation.

	clusterID, err := idx.assignToCluster(entry.Vector)
	if err != nil {
		return fmt.Errorf("failed to assign vector to cluster: %w", err)
	}

	codeSize := 0
	var compressed []byte
	if gen.quantizer != nil && gen.quantizer.IsTrained() {
		codeSize = idx.codeSize()
		compressed, err = gen.quantizer.Compress(entry.Vector)
		if err != nil {
			return fmt.Errorf("failed to compress vector: %w", err)
		}
		if len(compressed) != codeSize {
			return fmt.Errorf("compressed code size = %d, want %d", len(compressed), codeSize)
		}
	}

	cluster := gen.clusters[clusterID]
	cluster.mutex.Lock()
	err = cluster.storage.append(entry.Ordinal, compressed, gen.pool)
	cluster.mutex.Unlock()
	if err != nil {
		return fmt.Errorf("failed to append to cluster storage: %w", err)
	}

	gen.size.Add(1)
	gen.mutation.Add(1)
	return nil
}

// BatchInsert adds multiple vector entries to the index in parallel
func (idx *Index) BatchInsert(ctx context.Context, entries []*VectorEntry) error {
	if len(entries) == 0 {
		return nil
	}

	// Maximum entries to process in one arena to prevent memory exhaustion
	const maxChunkSize = 1000

	for i := 0; i < len(entries); i += maxChunkSize {
		end := min(i+maxChunkSize, len(entries))
		chunk := entries[i:end]

		if err := idx.batchInsertChunk(ctx, chunk); err != nil {
			return err
		}
	}

	return nil
}

func (idx *Index) batchInsertChunk(ctx context.Context, entries []*VectorEntry) error {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	if idx.gen == nil {
		return fmt.Errorf("index closed")
	}
	gen := idx.gen
	if !gen.trained {
		return fmt.Errorf("Insert: %w", util.ErrNotTrained)
	}
	workers := parallelismFor(len(entries))

	type processedEntry struct {
		ordinal     uint32
		clusterID   int
		sourceIndex int
	}

	arena := idx.scratchPool.Get().(*memory.Arena)
	defer func() {
		arena.Reset()
		idx.scratchPool.Put(arena)
	}()

	processedSlice, err := memory.ArenaSlice[processedEntry](arena, len(entries))
	if err != nil {
		return err
	}
	processed := processedSlice[:len(entries)]
	errs := make([]error, len(entries))

	codeSize := 0
	if gen.quantizer != nil && gen.quantizer.IsTrained() {
		codeSize = idx.codeSize()
	}
	codesSlice, err := memory.ArenaSlice[byte](arena, len(entries)*codeSize)
	if err != nil {
		return err
	}
	codes := codesSlice[:len(entries)*codeSize]

	var wg sync.WaitGroup
	chunkSize := (len(entries) + workers - 1) / workers

	for worker := 0; worker < workers; worker++ {
		start := worker * chunkSize
		if start >= len(entries) {
			break
		}
		end := min(start+chunkSize, len(entries))

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				select {
				case <-ctx.Done():
					errs[i] = ctx.Err()
					return
				default:
				}

				entry := entries[i]
				if entry == nil {
					errs[i] = fmt.Errorf("entry cannot be nil")
					continue
				}

				if len(entry.Vector) != gen.config.Dimension {
					errs[i] = fmt.Errorf("vector dimension %d does not match index dimension %d",
						len(entry.Vector), gen.config.Dimension)
					continue
				}

				clusterID, err := idx.assignToCluster(entry.Vector)
				if err != nil {
					errs[i] = fmt.Errorf("failed to assign vector to cluster: %w", err)
					continue
				}

				if codeSize > 0 {
					code, err := gen.quantizer.Compress(entry.Vector)
					if err != nil {
						errs[i] = fmt.Errorf("failed to compress vector: %w", err)
						continue
					}
					if len(code) != codeSize {
						errs[i] = fmt.Errorf("compressed code size = %d, want %d", len(code), codeSize)
						continue
					}
					copy(codes[i*codeSize:(i+1)*codeSize], code)
				}

				processed[i] = processedEntry{
					ordinal:     entry.Ordinal,
					clusterID:   clusterID,
					sourceIndex: i,
				}
			}
		}(start, end)
	}

	wg.Wait()

	for _, e := range errs {
		if e != nil {
			return e
		}
	}

	countsSlice, err := memory.ArenaSlice[int](arena, len(gen.clusters))
	if err != nil {
		return fmt.Errorf("arena allocate counts: %w", err)
	}
	counts := countsSlice[:len(gen.clusters)]
	clear(counts)

	for _, p := range processed {
		counts[p.clusterID]++
	}

	clusterUpdates := make([][]processedEntry, len(gen.clusters))

	for i, c := range counts {
		if c > 0 {
			slice, err := memory.ArenaSlice[processedEntry](arena, c)
			if err != nil {
				return fmt.Errorf("arena allocate cluster %d updates: %w", i, err)
			}
			clusterUpdates[i] = slice[:0]
		}
	}

	for _, p := range processed {
		clusterUpdates[p.clusterID] = append(clusterUpdates[p.clusterID], processedEntry{
			ordinal:     p.ordinal,
			clusterID:   p.clusterID,
			sourceIndex: p.sourceIndex,
		})
	}

	for clusterID, updates := range clusterUpdates {
		if len(updates) == 0 {
			continue
		}

		cluster := gen.clusters[clusterID]
		cluster.mutex.Lock()

		for _, p := range updates {
			var code []byte
			if codeSize > 0 {
				origIndex := p.sourceIndex
				code = codes[origIndex*codeSize : (origIndex+1)*codeSize]
			}
			err := cluster.storage.append(p.ordinal, code, gen.pool)
			if err != nil {
				cluster.mutex.Unlock()
				return fmt.Errorf("failed to append to cluster %d storage: %w", clusterID, err)
			}
		}

		cluster.mutex.Unlock()
	}

	gen.size.Add(int64(len(entries)))
	gen.mutation.Add(1)
	return nil
}

// findProbeClusters returns the top probeClusters for a query, using only
// generation-owned state. No Index dereference.
func findProbeClusters(gen *generation, query []float32, arena *memory.Arena) ([]int, []float32, error) {
	clusters := gen.clusters
	nProbes := gen.config.NProbes
	distances, err := memory.ArenaSlice[clusterDistance](arena, len(clusters))
	if err != nil {
		return nil, nil, fmt.Errorf("arena allocate distances: %w", err)
	}
	distances = distances[:len(clusters)]
	workers := parallelismFor(len(clusters))
	if workers == 1 {
		for i, cluster := range clusters {
			d := euclideanDist(query, cluster.Centroid)
			distances[i] = clusterDistance{id: i, distance: d}
		}
	} else {
		chunkSize := (len(clusters) + workers - 1) / workers
		var wg sync.WaitGroup
		for worker := 0; worker < workers; worker++ {
			start := worker * chunkSize
			if start >= len(clusters) {
				break
			}
			end := min(start+chunkSize, len(clusters))
			wg.Add(1)
			go func(start, end int) {
				defer wg.Done()
				for i := start; i < end; i++ {
					d := euclideanDist(query, clusters[i].Centroid)
					distances[i] = clusterDistance{id: i, distance: d}
				}
			}(start, end)
		}
		wg.Wait()
	}
	probeCount := nProbes
	if probeCount > len(distances) {
		probeCount = len(distances)
	}
	heap := distances[:probeCount]
	for i := probeCount/2 - 1; i >= 0; i-- {
		siftDownClusterDist(heap, i, probeCount)
	}
	for i := probeCount; i < len(distances); i++ {
		if distances[i].distance < heap[0].distance {
			heap[0] = distances[i]
			siftDownClusterDist(heap, 0, probeCount)
		}
	}
	probes, _ := memory.ArenaSlice[int](arena, probeCount)
	probes = probes[:probeCount]
	probeDists, _ := memory.ArenaSlice[float32](arena, probeCount)
	probeDists = probeDists[:probeCount]
	for i := probeCount - 1; i >= 0; i-- {
		probes[i] = heap[0].id
		probeDists[i] = heap[0].distance
		heap[0] = heap[probeCount-1]
		probeCount--
		siftDownClusterDist(heap, 0, probeCount)
	}
	return probes, probeDists, nil
}

func euclideanDist(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return float32(math.Sqrt(float64(sum)))
}

// collectCandidatesGen scans probe clusters using only the captured
// generation and precomputed codeSize.
func collectCandidatesGen(ctx context.Context, query []float32, probeClusters []int, clusterDistances []float32, k int, arena *memory.Arena, gen *generation, queryState any, codeSize int, filter interface {
	Test(idx uint64) bool
}) ([]candidate, error) {
	workers := parallelismFor(len(probeClusters))
	if workers == 1 {
		return collectCandidatesSeqGen(ctx, query, probeClusters, clusterDistances, k, arena, gen, queryState, codeSize, filter)
	}
	chunkSize := (len(probeClusters) + workers - 1) / workers
	results := make([][]candidate, workers)
	errCh := make(chan error, workers)
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		start := worker * chunkSize
		if start >= len(probeClusters) {
			break
		}
		end := min(start+chunkSize, len(probeClusters))
		wg.Add(1)
		go func(worker, start, end int) {
			defer wg.Done()
			local, err := collectCandidatesSeqGen(ctx, query, probeClusters[start:end], clusterDistances[start:end], k, arena, gen, queryState, codeSize, filter)
			if err != nil {
				errCh <- err
				return
			}
			results[worker] = local
		}(worker, start, end)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			return nil, err
		}
	}
	return mergeSortedWorkerResults(results, k, arena)
}

func collectCandidatesSeqGen(ctx context.Context, query []float32, probeClusters []int, clusterDistances []float32, k int, arena *memory.Arena, gen *generation, queryState any, codeSize int, filter interface {
	Test(idx uint64) bool
}) ([]candidate, error) {
	heapBuf, err := memory.ArenaSlice[ivfHeapElement](arena, k)
	if err != nil {
		return nil, fmt.Errorf("arena allocate heap buf: %w", err)
	}
	heapBuf = heapBuf[:k]
	count := 0

	for _, clusterID := range probeClusters {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		cluster := gen.clusters[clusterID]
		cluster.mutex.RLock()
		for _, seg := range cluster.storage.segments {
			for j := uint32(0); j < seg.used; j++ {
				ordinal := seg.ordinals[j]
				if filter != nil && !filter.Test(uint64(ordinal)) {
					continue
				}
				var distance float32
				if codeSize > 0 {
					compressed := seg.codes[int(j)*codeSize : int(j+1)*codeSize]
					distance, err = gen.quantizer.DistanceToQuery(compressed, query, queryState)
					if err != nil {
						cluster.mutex.RUnlock()
						return nil, err
					}
				}
				if count < k {
					heapBuf[count] = ivfHeapElement{ordinal: ordinal, distance: distance}
					ivfUpHeap(heapBuf, count)
					count++
				} else if distance < heapBuf[0].distance {
					heapBuf[0] = ivfHeapElement{ordinal: ordinal, distance: distance}
					ivfDownHeap(heapBuf, 0, count)
				}
			}
		}
		cluster.mutex.RUnlock()
	}
	candidates, err := memory.ArenaSlice[candidate](arena, count)
	if err != nil {
		return nil, fmt.Errorf("arena allocate candidates: %w", err)
	}
	candidates = candidates[:count]
	for i := count - 1; i >= 0; i-- {
		elem := heapBuf[0]
		count--
		heapBuf[0] = heapBuf[count]
		ivfDownHeap(heapBuf, 0, count)
		candidates[i] = candidate{ordinal: elem.ordinal, distance: elem.distance}
	}
	return candidates, nil
}

// Search finds the k nearest neighbors to the query vector.
// Captures the active generation under RLock, pins it, then uses only
// generation-owned state for the entire execution.
func (idx *Index) Search(ctx context.Context, query []float32, k int, filter interface {
	Test(idx uint64) bool
}) ([]*SearchResult, error) {
	startTime := time.Now()

	idx.mutex.RLock()
	if idx.gen == nil {
		idx.mutex.RUnlock()
		return nil, fmt.Errorf("index closed")
	}
	if !idx.gen.trained {
		idx.mutex.RUnlock()
		return nil, fmt.Errorf("Search: %w", util.ErrNotTrained)
	}
	gen := idx.gen
	gen.acquire()
	idx.mutex.RUnlock()
	defer gen.release()

	if len(query) != gen.config.Dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d",
			len(query), gen.config.Dimension)
	}
	if k <= 0 {
		return nil, fmt.Errorf("k must be positive, got %d: %w", k, util.ErrInvalidK)
	}
	if k > 4096 {
		return nil, fmt.Errorf("k %d exceeds maximum allowed search result limit of 4096", k)
	}

	idx.adjustProbeCount()

	arena := idx.scratchPool.Get().(*memory.Arena)
	defer func() {
		arena.Reset()
		idx.scratchPool.Put(arena)
	}()

	cs := 0
	if gen.quantizer != nil && gen.quantizer.IsTrained() {
		cs = gen.quantizer.CodeSize()
	}

	probeClusters, clusterDistances, err := findProbeClusters(gen, query, arena)
	if err != nil {
		return nil, fmt.Errorf("failed to find probe clusters: %w", err)
	}

	var queryState any
	if gen.quantizer != nil {
		queryState = gen.quantizer.PrepareQuery(query)
	}

	candidates, err := collectCandidatesGen(ctx, query, probeClusters, clusterDistances, k, arena, gen, queryState, cs, filter)
	if err != nil {
		return nil, err
	}

	// Candidates are already top-k in ascending distance order from the heap.
	results := make([]*SearchResult, len(candidates))

	for i, cand := range candidates {
		results[i] = &SearchResult{
			Ordinal: cand.ordinal,
			Score:   cand.distance,
		}
	}

	// Record search statistics for adaptive optimization
	latencyMs := time.Since(startTime).Milliseconds()

	// Estimate accuracy based on result quality (simplified metric)
	accuracy := 1.0
	if len(results) > 0 {
		accuracy = math.Min(1.0, float64(len(results))/float64(k))
	}

	idx.recordSearchStats(latencyMs, accuracy)

	return results, nil
}

// mergeElem is a k-way merge heap node: (distance, source worker, position within worker).
type mergeElem struct {
	distance float32
	worker   int
	pos      int
}

func mergeDownHeap(h []mergeElem, i, n int) {
	for {
		smallest := i
		left := 2*i + 1
		right := 2*i + 2
		if left < n && h[left].distance < h[smallest].distance {
			smallest = left
		}
		if right < n && h[right].distance < h[smallest].distance {
			smallest = right
		}
		if smallest == i {
			break
		}
		h[i], h[smallest] = h[smallest], h[i]
		i = smallest
	}
}

// mergeSortedWorkerResults merges W pre-sorted (ascending distance) candidate
// arrays into a single top-k candidate slice via k-way merge.
// Complexity: O(W·k log W) instead of the previous O(W·k log k) re-heaping.
func mergeSortedWorkerResults(results [][]candidate, k int, arena *memory.Arena) ([]candidate, error) {
	// Count non-empty workers and compute total available results.
	active := 0
	total := 0
	for _, batch := range results {
		if len(batch) > 0 {
			active++
			total += len(batch)
		}
	}
	if active == 0 {
		return nil, nil
	}
	if k > total {
		k = total
	}

	mergeHeap, err := memory.ArenaSlice[mergeElem](arena, active)
	if err != nil {
		return nil, fmt.Errorf("arena allocate mergeHeap: %w", err)
	}
	mergeHeap = mergeHeap[:0]
	pos, err := memory.ArenaSlice[int](arena, len(results))
	if err != nil {
		return nil, fmt.Errorf("arena allocate pos: %w", err)
	}
	pos = pos[:len(results)]

	for w, batch := range results {
		if len(batch) == 0 {
			continue
		}
		mergeHeap = append(mergeHeap, mergeElem{
			distance: batch[0].distance,
			worker:   w,
			pos:      0,
		})
	}
	// Heapify: build min-heap in O(W).
	for i := len(mergeHeap)/2 - 1; i >= 0; i-- {
		mergeDownHeap(mergeHeap, i, len(mergeHeap))
	}

	candidates, err := memory.ArenaSlice[candidate](arena, k)
	if err != nil {
		return nil, fmt.Errorf("arena allocate candidates: %w", err)
	}
	candidates = candidates[:0]
	for len(candidates) < k && len(mergeHeap) > 0 {
		root := mergeHeap[0]
		w := root.worker
		p := root.pos

		candidates = append(candidates, results[w][p])

		// Advance this worker's cursor; if more elements remain, replace root.
		pos[w] = p + 1
		if pos[w] < len(results[w]) {
			mergeHeap[0] = mergeElem{
				distance: results[w][pos[w]].distance,
				worker:   w,
				pos:      pos[w],
			}
		} else {
			// Worker exhausted — swap with last and shrink.
			mergeHeap[0] = mergeHeap[len(mergeHeap)-1]
			mergeHeap = mergeHeap[:len(mergeHeap)-1]
		}
		mergeDownHeap(mergeHeap, 0, len(mergeHeap))
	}
	return candidates, nil
}

func parallelismFor(items int) int {
	if items <= 1 {
		return 1
	}

	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if workers > items {
		workers = items
	}
	return workers
}

// EnableAdaptiveSearch enables adaptive probe count adjustment based on performance
func (idx *Index) EnableAdaptiveSearch() {
	idx.adaptiveMode.Store(true)
}

// DisableAdaptiveSearch disables adaptive probe count adjustment
func (idx *Index) DisableAdaptiveSearch() {
	idx.adaptiveMode.Store(false)
}

// GetSearchStats returns current search statistics
func (idx *Index) GetSearchStats() SearchStats {
	idx.searchStats.mutex.RLock()
	defer idx.searchStats.mutex.RUnlock()
	return SearchStats{
		totalSearches:  idx.searchStats.totalSearches,
		totalLatencyMs: idx.searchStats.totalLatencyMs,
		accuracySum:    idx.searchStats.accuracySum,
		currentProbes:  idx.searchStats.currentProbes,
		lastAdjustment: idx.searchStats.lastAdjustment,
	}
}

// adjustProbeCount adaptively adjusts the number of probes based on search performance
func (idx *Index) adjustProbeCount() {
	if !idx.adaptiveMode.Load() {
		return
	}

	if !idx.searchStats.mutex.TryLock() {
		return
	}
	defer idx.searchStats.mutex.Unlock()

	// Only adjust every 100 searches or after 30 seconds
	if idx.searchStats.totalSearches%100 != 0 &&
		time.Since(idx.searchStats.lastAdjustment) < 30*time.Second {
		return
	}

	if idx.searchStats.totalSearches < 10 {
		return // Need more data
	}

	avgLatencyMs := float64(idx.searchStats.totalLatencyMs) / float64(idx.searchStats.totalSearches)
	avgAccuracy := idx.searchStats.accuracySum / float64(idx.searchStats.totalSearches)

	// Target: < 50ms latency, > 0.9 accuracy
	targetLatencyMs := 50.0
	targetAccuracy := 0.9

	currentProbes := idx.searchStats.currentProbes
	newProbes := currentProbes

	if avgLatencyMs > targetLatencyMs && avgAccuracy > targetAccuracy {
		// Too slow but accurate - reduce probes
		newProbes = max(1, currentProbes-1)
	} else if avgLatencyMs < targetLatencyMs*0.5 && avgAccuracy < targetAccuracy {
		// Fast but inaccurate - increase probes
		newProbes = min(idx.config.NClusters, currentProbes+1)
	}

	if newProbes != currentProbes {
		idx.searchStats.currentProbes = newProbes
		idx.searchStats.lastAdjustment = time.Now()

		// Reset stats for next adjustment period
		idx.searchStats.totalSearches = 0
		idx.searchStats.totalLatencyMs = 0
		idx.searchStats.accuracySum = 0
	}
}

// recordSearchStats records statistics for a search operation
func (idx *Index) recordSearchStats(latencyMs int64, accuracy float64) {
	if !idx.adaptiveMode.Load() {
		return
	}

	if !idx.searchStats.mutex.TryLock() {
		return
	}
	defer idx.searchStats.mutex.Unlock()

	idx.searchStats.totalSearches++
	idx.searchStats.totalLatencyMs += latencyMs
	idx.searchStats.accuracySum += accuracy
}

// Delete is unsupported, use DeleteByOrdinal
func (idx *Index) Delete(ctx context.Context, id string) error {
	return fmt.Errorf("Delete(id) unsupported by IVF-PQ, use DeleteByOrdinal")
}

// DeleteByOrdinal removes a vector entry from the index by its ordinal
func (idx *Index) DeleteByOrdinal(ctx context.Context, ordinal uint32) error {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	if idx.gen == nil {
		return fmt.Errorf("index closed")
	}
	gen := idx.gen
	for _, cluster := range gen.clusters {
		cluster.mutex.Lock()
		if cluster.storage.deleteByOrdinal(ordinal) {
			cluster.mutex.Unlock()
			gen.size.Add(-1)
			gen.mutation.Add(1)
			return nil
		}
		cluster.mutex.Unlock()
	}

	return fmt.Errorf("entry with ordinal %d: %w", ordinal, util.ErrNotFound)
}

// Size returns the number of vectors in the index
func (idx *Index) Size() int {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	if idx.gen == nil {
		return 0
	}
	return int(idx.gen.size.Load())
}

// MemoryUsage returns the estimated memory usage in bytes
func (idx *Index) MemoryUsage() int64 {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	if idx.gen == nil {
		return 0
	}
	var usage int64

	// Cluster centroids
	usage += int64(len(idx.gen.clusters) * idx.config.Dimension * 4) // float32 = 4 bytes

	// Vector entries and compressed vectors
	for _, cluster := range idx.gen.clusters {
		cluster.mutex.RLock()

		usage += int64(len(cluster.storage.segments)) * int64(cluster.storage.segmentCapacity) * 4                                // uint32 = 4 bytes
		usage += int64(len(cluster.storage.segments)) * int64(cluster.storage.segmentCapacity) * int64(cluster.storage.codeWidth) // byte = 1 byte

		cluster.mutex.RUnlock()
	}

	// Quantizer memory usage
	if idx.gen.quantizer != nil {
		usage += idx.gen.quantizer.MemoryUsage()
	}

	return usage
}

// Close closes the index and releases resources
func (idx *Index) Close() error {
	idx.mutex.Lock()
	old := idx.gen
	if old == nil {
		idx.mutex.Unlock()
		return nil
	}
	idx.gen = nil
	for i := range idx.queryTiers {
		if idx.queryTiers[i].pool != nil {
			idx.queryTiers[i].pool.Free()
			idx.queryTiers[i].pool = nil
		}
	}
	idx.mutex.Unlock()

	// Retire the detached generation outside the lock. Pinned searches
	// hold refs; the final release frees the pool and quantizer.
	if old != nil {
		old.retired.Store(true)
		old.release()
	}
	return nil
}

// PersistenceMetadata holds metadata about a persisted IVF-PQ index.
type PersistenceMetadata struct {
	NumClusters    int
	NumSubspaces   int
	NumCentroids   int
	CompressedSize int64
}

// GetPersistenceMetadata returns metadata about the persisted index state,
// or nil if the index is not trained.
func (idx *Index) GetPersistenceMetadata() *PersistenceMetadata {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	if idx.gen == nil || !idx.gen.trained {
		return nil
	}
	meta := &PersistenceMetadata{
		NumClusters: len(idx.gen.clusters),
	}
	if idx.config.Quantization != nil && idx.gen.quantizer != nil {
		meta.NumSubspaces = idx.config.Quantization.Codebooks
		meta.NumCentroids = 1 << idx.config.Quantization.Bits
	}
	for _, cluster := range idx.gen.clusters {
		cluster.mutex.RLock()
		meta.CompressedSize += int64(cluster.storage.count * uint64(cluster.storage.codeWidth))
		cluster.mutex.RUnlock()
	}
	return meta
}
