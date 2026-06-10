package hnsw

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	internalmemory "github.com/xDarkicex/libravdb/internal/memory"
	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/memory"
	"unsafe"
)

// VectorEntry represents a vector entry for HNSW indexing
type VectorEntry struct {
	ID       string
	Ordinal  uint32
	Vector   []float32
	Metadata map[string]interface{}
	Version  uint64
}

// SearchResult represents a search result from HNSW
type SearchResult struct {
	Ordinal  uint32
	ID       string
	Score    float32
	Vector   []float32
	Metadata map[string]interface{}
	Version  uint64
}

type VectorProvider interface {
	GetByOrdinal(ordinal uint32) ([]float32, error)
	Distance(query []float32, ordinal uint32) (float32, error)
}

// Index implements the HNSW algorithm for approximate nearest neighbor search
type Index struct {
	mu                   sync.RWMutex
	config               *Config
	nodes                []*Node
	entryPoint           *Node
	maxLevel             int
	levelGenerator       *rand.Rand
	distance             util.DistanceFunc
	size                 int
	nextOrdinal          uint32 // monotonic counter for standalone (no-provider) inserts
	provider             VectorProvider
	idToIndex            map[string]uint32 // Legacy standalone HNSW path
	ordinalToID          map[uint32]string // Legacy standalone HNSW path
	entryPointCandidates []uint32          // High-level nodes for entry point selection
	// Performance optimizations
	neighborSelector  *NeighborSelector // Optimized neighbor selection
	searchScratchPool sync.Pool
	scratchPool       *sync.Pool
	linkSFL           *memory.ShardedFreeList
	link0SFL          *memory.ShardedFreeList
	rawVectorStore    RawVectorStore
	// Quantization support
	quantizer           quant.Quantizer
	trainingVectors     [][]float32 // Vectors collected for quantizer training
	quantizationTrained bool
	// Memory mapping support
	memoryMapped     bool
	mmapPath         string
	mmapSize         int64
	vecMmap          *internalmemory.MemoryMap // Mmap for raw vectors
	pqMmap           *internalmemory.MemoryMap // Mmap for compressed vectors
	originalMemUsage int64             // Memory usage before mapping
}

// Config holds HNSW configuration parameters
type Config struct {
	Dimension      int
	M              int     // Maximum number of bi-directional links for each node
	EfConstruction int     // Size of dynamic candidate list
	EfSearch       int     // Size of dynamic candidate list during search
	ML             float64 // Level generation factor (1/ln(2))
	Metric         util.DistanceMetric
	RandomSeed     int64 // For reproducible tests
	Provider       VectorProvider
	RawVectorStore string
	RawStoreCap    int
	// Quantization configuration (optional)
	Quantization *quant.QuantizationConfig
}

// NewHNSW creates a new HNSW index
func NewHNSW(config *Config) (*Index, error) {
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid HNSW config: %w", err)
	}

	distanceFunc, err := util.GetDistanceFunc(config.Metric)
	if err != nil {
		return nil, fmt.Errorf("unsupported distance metric: %w", err)
	}

	slack := config.M / 4
	if slack < 4 {
		slack = 4
	}
	linkSFL, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  512 * 1024 * 1024,
		SlotSize:  uint64(48 + (config.M+slack)*4),
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 8,
		Prealloc:  false,
	}, 64)
	if err != nil {
		return nil, fmt.Errorf("failed to create linkSFL: %w", err)
	}

	slack0 := (config.M * 2) / 4
	if slack0 < 4 {
		slack0 = 4
	}
	link0SFL, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  512 * 1024 * 1024,
		SlotSize:  uint64(48 + (config.M*2+slack0)*4),
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 8,
		Prealloc:  false,
	}, 64)
	if err != nil {
		return nil, fmt.Errorf("failed to create link0SFL: %w", err)
	}

	scratchPool := &sync.Pool{
		New: func() any {
			a, _ := memory.NewArena(1024 * 1024)
			return a
		},
	}

	index := &Index{
		config:               config,
		nodes:                make([]*Node, 0),
		levelGenerator:       rand.New(rand.NewSource(config.RandomSeed)),
		distance:             distanceFunc,
		provider:             config.Provider,
		idToIndex:            make(map[string]uint32),
		ordinalToID:          make(map[uint32]string),
		entryPointCandidates: make([]uint32, 0),
		trainingVectors:      make([][]float32, 0),
		quantizationTrained:  false,
		nextOrdinal:          0,
		linkSFL:              linkSFL,
		link0SFL:             link0SFL,
		scratchPool:          scratchPool,
	}
	index.searchScratchPool.New = func() interface{} {
		return &searchScratch{}
	}
	switch config.RawVectorStore {
	case "", RawVectorStoreMemory:
		index.rawVectorStore = NewInMemoryRawVectorStore(config.Dimension)
	case RawVectorStoreSlabby:
		store, err := NewSlabbyRawVectorStore(config.Dimension, config.RawStoreCap)
		if err != nil {
			return nil, fmt.Errorf("failed to create slabby raw vector store: %w", err)
		}
		index.rawVectorStore = store
	default:
		return nil, fmt.Errorf("unsupported raw vector store backend: %s", config.RawVectorStore)
	}

	// Initialize quantizer if quantization is configured
	if config.Quantization != nil {
		quantizer, err := quant.Create(config.Quantization)
		if err != nil {
			return nil, fmt.Errorf("failed to create quantizer: %w", err)
		}
		index.quantizer = quantizer
	}

	return index, nil
}

func (h *Index) Insert(ctx context.Context, entry *VectorEntry) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	return h.insertSingle(ctx, entry)
}

// insertSingle handles single vector insertion (must be called with lock held)
func (h *Index) insertSingle(ctx context.Context, entry *VectorEntry) error {
	if entry == nil {
		return fmt.Errorf("entry cannot be nil")
	}
	if len(entry.Vector) == 0 {
		return fmt.Errorf("vector cannot be empty")
	}
	if len(entry.Vector) != h.config.Dimension {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", h.config.Dimension, len(entry.Vector))
	}

	if entry.ID != "" {
		if _, exists := h.idToIndex[entry.ID]; exists {
			return fmt.Errorf("node with ID '%s' already exists", entry.ID)
		}
	}

	// Handle quantization training collection
	if h.quantizer != nil && !h.quantizationTrained {
		// Collect vectors for training
		vectorCopy := make([]float32, len(entry.Vector))
		copy(vectorCopy, entry.Vector)
		h.trainingVectors = append(h.trainingVectors, vectorCopy)

		// Train quantizer when we have enough training data
		if len(h.trainingVectors) >= h.getTrainingThreshold() {
			if err := h.trainQuantizer(ctx); err != nil {
				return fmt.Errorf("failed to train quantizer: %w", err)
			}
		}
	}

	// Create new node with optimized memory allocation
	level := h.generateLevel()
	ordinal := entry.Ordinal
	if h.provider == nil {
		ordinal = h.nextOrdinal
	}
	if int(ordinal) < len(h.nodes) && h.nodes[ordinal] != nil {
		return fmt.Errorf("node with ordinal %d already exists", ordinal)
	}
	node := &Node{
		Ordinal: ordinal,
		Level:   level,
		Links:   h.newNodeLinks(level, h.config.M),
	}

	// Handle vector storage (quantized or original)
	if h.quantizer != nil && h.quantizationTrained {
		// Compress the vector
		compressed, err := h.quantizer.Compress(entry.Vector)
		if err != nil {
			return fmt.Errorf("failed to compress vector: %w", err)
		}
		node.CompressedVector = compressed
	} else if h.rawVectorStore != nil {
		_, err := h.rawVectorStore.Put(entry.Vector)
		if err != nil {
			return fmt.Errorf("failed to store raw vector: %w", err)
		}
	}

	nodeID := node.Ordinal
	if int(nodeID) >= len(h.nodes) {
		h.ensureNodeCapacity(int(nodeID) + 1)
	}
	h.nodes[nodeID] = node
	if h.provider == nil {
		h.nextOrdinal++
	}

	if entry.ID != "" {
		h.idToIndex[entry.ID] = nodeID
		h.ordinalToID[nodeID] = entry.ID
	}

	// Add to entry point candidates if level is high enough
	// Using level >= 2 as threshold for entry point candidates
	if level >= 2 {
		h.entryPointCandidates = append(h.entryPointCandidates, nodeID)
	}

	// If this is the first node, set it as entry point
	if h.entryPoint == nil {
		h.entryPoint = node
		h.maxLevel = level
		h.size++
		return nil
	}

	// Delegate to insertion logic in insert.go
	if err := h.insertNode(ctx, node, nodeID, entry.Vector); err != nil {
		h.nodes[nodeID] = nil
		if entry.ID != "" {
			delete(h.idToIndex, entry.ID)
			delete(h.ordinalToID, nodeID)
		}

		if level >= 2 && len(h.entryPointCandidates) > 0 {
			lastIdx := len(h.entryPointCandidates) - 1
			if h.entryPointCandidates[lastIdx] == nodeID {
				h.entryPointCandidates = h.entryPointCandidates[:lastIdx]
			}
		}

		return fmt.Errorf("failed to insert node: %w", err)
	}

	h.size++

	// Update entry point if necessary
	if level > h.maxLevel {
		h.entryPoint = node
		h.maxLevel = level
	}

	return nil
}

func (h *Index) newNodeLinks(level int, baseM int) [][]uint32 {
	links := make([][]uint32, level+1)
	for i := 0; i <= level; i++ {
		capacity := levelMaxLinks(baseM, i)
		slack := capacity / 4
		if slack < 4 {
			slack = 4
		}
		maxCapacity := capacity + slack

		var slot []byte
		var err error
		if i == 0 {
			slot, err = h.link0SFL.Allocate()
		} else {
			slot, err = h.linkSFL.Allocate()
		}
		if err != nil {
			panic(fmt.Sprintf("hnsw: failed to allocate links: %v", err))
		}
		ptr := unsafe.Pointer(&slot[48])
		// Slice up to maxCapacity, but start with len 0
		links[i] = unsafe.Slice((*uint32)(ptr), maxCapacity)[:0]
	}
	return links
}

func initialNodeLinkCapacity(baseM int, level int) int {
	maxLinks := levelMaxLinks(baseM, level)
	if level == 0 {
		return maxLinks + max(8, maxLinks/2)
	}
	return maxLinks
}

// BatchInsert provides optimized batch insertion for better performance with large datasets
func (h *Index) BatchInsert(ctx context.Context, entries []*VectorEntry) error {
	if len(entries) == 0 {
		return nil
	}

	// Use optimized batch processing for all batch sizes
	return h.batchInsertOptimized(ctx, entries)
}

// batchInsertOptimized handles large batch insertions with memory optimization
func (h *Index) batchInsertOptimized(ctx context.Context, entries []*VectorEntry) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Pre-allocate space for nodes to avoid repeated slice growth
	expectedSize := len(h.nodes) + len(entries)
	if h.provider != nil {
		for _, entry := range entries {
			if entry != nil && int(entry.Ordinal)+1 > expectedSize {
				expectedSize = int(entry.Ordinal) + 1
			}
		}
	}
	if expectedSize > len(h.nodes) {
		h.ensureNodeCapacity(expectedSize)
	}

	// Process entries in chunks to manage memory usage
	chunkSize := 100 // Process 100 entries at a time
	for i := 0; i < len(entries); i += chunkSize {
		end := min(i+chunkSize, len(entries))
		chunk := entries[i:end]

		// Check context cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Process chunk
		for _, entry := range chunk {
			if err := h.insertSingle(ctx, entry); err != nil {
				return fmt.Errorf("failed to insert entry %s in batch: %w", entry.ID, err)
			}
		}
	}

	return nil
}

// Search finds the k nearest neighbors to the query vector
func (h *Index) Search(ctx context.Context, query []float32, k int) ([]*SearchResult, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if k <= 0 {
		return nil, fmt.Errorf("k must be positive, got %d: %w", k, util.ErrInvalidK)
	}
	if k > 4096 {
		return nil, fmt.Errorf("k %d exceeds maximum allowed search result limit of 4096", k)
	}

	if h.size == 0 {
		return nil, fmt.Errorf("%w", util.ErrEmptyIndex)
	}

	if len(query) != h.config.Dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d",
			len(query), h.config.Dimension)
	}

	// Phase 1: Search from top level to level 1
	ep := h.entryPoint
	for level := h.maxLevel; level > 0; level-- {
		candidate := h.greedySearchLevel(query, ep, level)
		if candidate != nil {
			ep = h.nodes[candidate.ID]
		}
	}

	// Phase 2: Search level 0 with ef
	ef := max(h.config.EfSearch, k)
	candidates := h.searchLevel(query, ep, ef, 0)

	// Convert to results and limit to k
	results := make([]*SearchResult, 0, min(k, len(candidates)))
	for i, candidate := range candidates {
		if i >= k {
			break
		}

		node := h.nodes[candidate.ID]
		if node == nil {
			continue
		}

		// Decompression is off the hot path — the collection layer hydrates
		// vectors from storage. Standalone path reads from off-heap rawVectorStore.
		var resultVector []float32
		if h.provider == nil {
			resultVector, _ = h.getNodeVector(node)
		}

		results = append(results, &SearchResult{
			Ordinal: node.Ordinal,
			ID:      h.ordinalToID[node.Ordinal],
			Score:   candidate.Distance,
			Vector:  resultVector,
		})
	}

	return results, nil
}

// Size returns the number of vectors in the index
func (h *Index) Size() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.size
}

// MemoryUsage returns approximate memory usage in bytes
func (h *Index) MemoryUsage() int64 {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.calculateMemoryUsage()
}

func (h *Index) RawVectorStoreProfile() map[string]any {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.rawVectorStore == nil {
		return nil
	}
	profile := h.rawVectorStore.Profile()
	return map[string]any{
		"backend":              profile.Backend,
		"vector_count":         profile.VectorCount,
		"dimension":            profile.Dimension,
		"bytes_per_vector":     profile.BytesPerVector,
		"memory_usage":         profile.MemoryUsage,
		"reserved_bytes":       profile.ReservedBytes,
		"reserved_data_bytes":  profile.ReservedDataBytes,
		"reserved_meta_bytes":  profile.ReservedMetaBytes,
		"reserved_guard_bytes": profile.ReservedGuardBytes,
		"live_bytes":           profile.LiveBytes,
		"free_bytes":           profile.FreeBytes,
		"capacity_utilization": profile.CapacityUtilization,
	}
}

// calculateMemoryUsage calculates memory usage without acquiring locks (must be called with lock held)
func (h *Index) calculateMemoryUsage() int64 {
	// If memory mapped, return minimal in-memory usage
	if h.memoryMapped {
		var usage int64

		// Count only essential in-memory structures
		usage += int64(len(h.nodes) * 8)                // Pointer overhead
		usage += int64(len(h.idToIndex) * 16)           // Map overhead
		usage += int64(len(h.entryPointCandidates) * 4) // uint32 slice

		// Add quantizer memory usage if present
		if h.quantizer != nil {
			usage += h.quantizer.MemoryUsage()
		}

		return usage
	}

	var usage int64
	for _, node := range h.nodes {
		if node == nil {
			continue
		}
		// Vector data (original or compressed)
		if node.CompressedVector != nil {
			usage += int64(len(node.CompressedVector))
		}

		// Links
		for _, links := range node.Links {
			usage += int64(len(links) * 4) // 4 bytes per uint32
		}

		// Node overhead (approximate)
		usage += 32
	}

	// Add quantizer memory usage if present
	if h.quantizer != nil {
		usage += h.quantizer.MemoryUsage()
	}
	if h.rawVectorStore != nil {
		usage += h.rawVectorStore.MemoryUsage()
	}

	// Add training vectors memory usage if still collecting
	if h.trainingVectors != nil {
		for _, vec := range h.trainingVectors {
			usage += int64(len(vec) * 4)
		}
	}

	return usage
}

// Close shuts down the index
func (h *Index) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Free off-heap link storage.
	if h.linkSFL != nil {
		h.linkSFL.Free()
		h.linkSFL = nil
	}
	if h.link0SFL != nil {
		h.link0SFL.Free()
		h.link0SFL = nil
	}

	// Clear all data structures
	h.nodes = nil
	h.entryPoint = nil
	h.size = 0
	h.nextOrdinal = 0
	h.idToIndex = make(map[string]uint32)
	h.ordinalToID = make(map[uint32]string)
	if h.rawVectorStore != nil {
		_ = h.rawVectorStore.Reset()
	}

	return nil
}

// generateLevel returns a random level for a new node
func (h *Index) generateLevel() int {
	level := 0
	for h.levelGenerator.Float64() < h.config.ML && level < 16 { // Cap at 16 levels
		level++
	}
	return level
}

// findNodeID returns the stable node index without scanning h.nodes.
func (h *Index) findNodeID(target *Node) uint32 {
	if target == nil {
		return ^uint32(0)
	}
	if int(target.Ordinal) >= len(h.nodes) || h.nodes[target.Ordinal] != target {
		return ^uint32(0)
	}
	return target.Ordinal
}

func (h *Index) ensureNodeCapacity(minSize int) {
	if minSize <= len(h.nodes) {
		return
	}
	grown := make([]*Node, minSize, nextNodeCapacity(cap(h.nodes), minSize))
	copy(grown, h.nodes)
	h.nodes = grown
}

func nextNodeCapacity(currentCap, minSize int) int {
	if minSize <= currentCap {
		return currentCap
	}
	newSize := currentCap
	if newSize < 16 {
		newSize = 16
	}
	for newSize < minSize {
		if newSize < 1024 {
			newSize *= 2
		} else {
			newSize += newSize / 2
		}
	}
	return newSize
}

// validate checks if the configuration is valid
func (c *Config) validate() error {
	if c.Dimension <= 0 {
		return fmt.Errorf("dimension must be positive")
	}
	if c.M <= 0 {
		return fmt.Errorf("M must be positive")
	}
	if c.EfConstruction <= 0 {
		return fmt.Errorf("EfConstruction must be positive")
	}
	if c.EfSearch <= 0 {
		return fmt.Errorf("EfSearch must be positive")
	}
	if c.ML <= 0 {
		return fmt.Errorf("ML must be positive")
	}

	// Validate quantization config if present
	if c.Quantization != nil {
		if err := c.Quantization.Validate(); err != nil {
			return fmt.Errorf("invalid quantization config: %w", err)
		}
	}

	return nil
}

// getTrainingThreshold returns the minimum number of vectors needed for training
func (h *Index) getTrainingThreshold() int {
	if h.config.Quantization == nil {
		return 0
	}

	// Use a reasonable threshold based on quantization type
	switch h.config.Quantization.Type {
	case quant.ProductQuantization:
		// PQ needs more training data for k-means clustering, but keep it reasonable
		return max(100, h.config.Quantization.Codebooks*32)
	case quant.ScalarQuantization:
		// Scalar quantization needs less training data
		return max(50, h.config.Dimension*2)
	default:
		return 100
	}
}

// trainQuantizer trains the quantizer using collected training vectors
func (h *Index) trainQuantizer(ctx context.Context) error {
	if h.quantizer == nil || len(h.trainingVectors) == 0 {
		return fmt.Errorf("no quantizer or training data available")
	}

	// Use a subset of training vectors based on TrainRatio
	trainRatio := h.config.Quantization.TrainRatio
	if trainRatio <= 0 || trainRatio > 1 {
		trainRatio = 0.1 // Default to 10%
	}

	trainCount := int(float64(len(h.trainingVectors)) * trainRatio)
	if trainCount < 1 {
		trainCount = len(h.trainingVectors)
	}

	trainingSet := h.trainingVectors[:trainCount]

	if err := h.quantizer.Train(ctx, trainingSet); err != nil {
		return fmt.Errorf("quantizer training failed: %w", err)
	}

	h.quantizationTrained = true

	// Clear training vectors to free memory
	h.trainingVectors = nil

	return nil
}

// getNodeVector returns the vector for a node, handling quantization
func (h *Index) getNodeVector(node *Node) ([]float32, error) {
	if node == nil {
		return nil, fmt.Errorf("node is nil")
	}
	if node.CompressedVector != nil && h.quantizer != nil {
		return h.quantizer.Decompress(node.CompressedVector)
	}
	if h.provider != nil {
		if vec, err := h.provider.GetByOrdinal(node.Ordinal); err == nil && vec != nil {
			return vec, nil
		}
	}
	if h.rawVectorStore != nil {
		ref := VectorRef{
			Kind:  VectorEncodingRaw,
			Slot:  node.Ordinal,
			Bytes: uint32(h.config.Dimension * 4),
			Valid: true,
		}
		return h.rawVectorStore.Get(ref)
	}
	return nil, fmt.Errorf("vector unavailable for ordinal %d", node.Ordinal)
}

// getNodeVectorLocal retrieves a node's vector from local storage only.
// It deliberately skips h.provider to avoid re-entrant lock acquisition
// during serialization (engine holds e.mu.Lock; provider.GetByOrdinal
// would try e.mu.RLock → deadlock).
func (h *Index) getNodeVectorLocal(node *Node) ([]float32, error) {
	if node == nil {
		return nil, fmt.Errorf("node is nil")
	}
	if node.CompressedVector != nil && h.quantizer != nil {
		return h.quantizer.Decompress(node.CompressedVector)
	}
	if h.rawVectorStore != nil {
		ref := VectorRef{
			Kind:  VectorEncodingRaw,
			Slot:  node.Ordinal,
			Bytes: uint32(h.config.Dimension * 4),
			Valid: true,
		}
		vec, err := h.rawVectorStore.Get(ref)
		if err == nil && vec != nil {
			return vec, nil
		}
	}
	return nil, fmt.Errorf("vector for ordinal %d not in local storage", node.Ordinal)
}

// SnapshotVectorsFromProvider copies all node vectors from the provider into
// rawVectorStore so that subsequent serialization can proceed without calling
// back into the provider. Must be called before SerializeToBytes when a
// provider is set and the caller cannot re-enter the provider.
func (h *Index) SnapshotVectorsFromProvider(ctx context.Context) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.provider == nil {
		return nil
	}
	if h.rawVectorStore == nil {
		h.rawVectorStore = NewInMemoryRawVectorStore(h.config.Dimension)
	}

	for _, node := range h.nodes {
		if node == nil || node.CompressedVector != nil {
			continue
		}
		// Check if already present in rawVectorStore.
		ref := VectorRef{
			Kind:  VectorEncodingRaw,
			Slot:  node.Ordinal,
			Bytes: uint32(h.config.Dimension * 4),
			Valid: true,
		}
		if vec, err := h.rawVectorStore.Get(ref); err == nil && vec != nil {
			continue
		}
		vec, err := h.provider.GetByOrdinal(node.Ordinal)
		if err != nil || vec == nil {
			return fmt.Errorf("snapshot vector ordinal %d: %w", node.Ordinal, err)
		}
		if _, err := h.rawVectorStore.Put(vec); err != nil {
			return fmt.Errorf("snapshot store ordinal %d: %w", node.Ordinal, err)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}
	return nil
}

// computeDistance computes distance between vectors, handling quantization
func (h *Index) computeDistance(vec1, vec2 []float32, node1, node2 *Node) (float32, error) {
	// If both nodes are quantized, use quantized distance computation
	if node1 != nil && node2 != nil &&
		node1.CompressedVector != nil && node2.CompressedVector != nil &&
		h.quantizer != nil {
		return h.quantizer.Distance(node1.CompressedVector, node2.CompressedVector)
	}

	// If one is a query vector and the other is quantized
	if node2 != nil && node2.CompressedVector != nil && h.quantizer != nil {
		return h.quantizer.DistanceToQuery(node2.CompressedVector, vec1)
	}

	if vec1 == nil && node1 != nil {
		var err error
		vec1, err = h.getNodeVector(node1)
		if err != nil {
			return 0, err
		}
	}
	if vec2 == nil && node2 != nil {
		var err error
		vec2, err = h.getNodeVector(node2)
		if err != nil {
			return 0, err
		}
	}

	return h.distance(vec1, vec2), nil
}

func (h *Index) Delete(ctx context.Context, id string) error {
	return h.deleteNode(ctx, id)
}

func (h *Index) DeleteByOrdinal(ctx context.Context, ordinal uint32) error {
	return h.deleteNodeByOrdinal(ctx, ordinal)
}

// MemoryMappable interface implementation

// CanMemoryMap returns true if the index can be memory mapped
func (h *Index) CanMemoryMap() bool {
	h.mu.RLock()
	defer h.mu.RUnlock()

	// Can memory map if we have nodes and are not already mapped
	return h.size > 0 && !h.memoryMapped
}

// EstimateSize returns the estimated size in bytes if memory mapped
func (h *Index) EstimateSize() int64 {
	h.mu.RLock()
	defer h.mu.RUnlock()

	return h.calculateMemoryUsage()
}

// EnableMemoryMapping enables memory mapping for the index
func (h *Index) EnableMemoryMapping(basePath string) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.memoryMapped {
		return fmt.Errorf("index is already memory mapped")
	}

	if h.size == 0 {
		return fmt.Errorf("cannot memory map empty index")
	}

	// Store current memory usage for comparison (calculate inline to avoid deadlock)
	h.originalMemUsage = h.calculateMemoryUsage()

	// Create a temporary file path for the memory-mapped index
	mmapPath := fmt.Sprintf("%s/hnsw_index_%p.mmap", basePath, h)

	// Save the index to the memory-mapped file (use lock-free version since we already hold the lock)
	if err := h.saveToDiskWithoutLock(context.Background(), mmapPath); err != nil {
		return fmt.Errorf("failed to save index for memory mapping: %w", err)
	}

	// Get file size
	stat, err := os.Stat(mmapPath)
	if err != nil {
		return fmt.Errorf("failed to stat memory-mapped file: %w", err)
	}

	h.mmapPath = mmapPath
	h.mmapSize = stat.Size()
	h.memoryMapped = true

	// Clear in-memory data structures to free memory
	if h.rawVectorStore != nil {
		vecMmapPath := fmt.Sprintf("%s/hnsw_vectors_%p.mmap", basePath, h)
		mmapStore, err := h.createMmapRawVectorStore(vecMmapPath)
		if err != nil {
			return fmt.Errorf("failed to create memory-mapped raw vector store: %w", err)
		}
		h.vecMmap = mmapStore.mmap
		_ = h.rawVectorStore.Reset()
		h.rawVectorStore = mmapStore
	} else if h.quantizer != nil {
		pqMmapPath := fmt.Sprintf("%s/hnsw_pq_%p.mmap", basePath, h)
		pqMmap, err := h.createMmapCompressedVectorStore(pqMmapPath)
		if err != nil {
			return fmt.Errorf("failed to create memory-mapped compressed vector store: %w", err)
		}
		h.pqMmap = pqMmap
	}

	// Clear training vectors
	h.trainingVectors = nil

	return nil
}

// DisableMemoryMapping disables memory mapping and loads data back to RAM
func (h *Index) DisableMemoryMapping() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if !h.memoryMapped {
		return fmt.Errorf("index is not memory mapped")
	}

	// Unmap and clean up vector mmap files first so loadFromDiskImpl can create new stores
	if h.vecMmap != nil {
		path := h.vecMmap.Path()
		h.vecMmap.Close()
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			fmt.Printf("Failed to remove vector memory-mapped file %s: %v\n", path, err)
		}
		h.vecMmap = nil
		h.rawVectorStore = nil
	}
	if h.pqMmap != nil {
		path := h.pqMmap.Path()
		h.pqMmap.Close()
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			fmt.Printf("Failed to remove pq memory-mapped file %s: %v\n", path, err)
		}
		h.pqMmap = nil
	}

	// Reload the index from the memory-mapped file
	if err := h.loadFromDiskImpl(context.Background(), h.mmapPath); err != nil {
		return fmt.Errorf("failed to reload index from memory mapping: %w", err)
	}

	// Clean up the memory-mapped file
	if err := os.Remove(h.mmapPath); err != nil && !os.IsNotExist(err) {
		// Log error but don't fail the operation
		fmt.Printf("Failed to remove memory-mapped file %s: %v\n", h.mmapPath, err)
	}

	h.memoryMapped = false
	h.mmapPath = ""
	h.mmapSize = 0
	h.originalMemUsage = 0

	return nil
}

// IsMemoryMapped returns true if currently using memory mapping
func (h *Index) IsMemoryMapped() bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.memoryMapped
}

// MemoryMappedSize returns the size of memory-mapped data
func (h *Index) MemoryMappedSize() int64 {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.memoryMapped {
		return h.mmapSize
	}
	return 0
}

// SaveToDisk persists the HNSW index to disk in binary format
func (h *Index) SaveToDisk(ctx context.Context, path string) error {
	return h.saveToDiskImpl(ctx, path)
}

// LoadFromDisk rebuilds HNSW index from disk
func (h *Index) LoadFromDisk(ctx context.Context, path string) error {
	return h.loadFromDiskImpl(ctx, path)
}

// SerializeToBytes serializes the index to an in-memory byte slice using the
// same binary format as SaveToDisk.
func (h *Index) SerializeToBytes() ([]byte, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	var buf bytes.Buffer
	writer := bufio.NewWriter(&buf)

	if err := h.writeHeader(writer); err != nil {
		return nil, fmt.Errorf("failed to write header: %w", err)
	}
	if err := h.writeConfig(writer); err != nil {
		return nil, fmt.Errorf("failed to write config: %w", err)
	}
	if err := h.writeNodes(writer); err != nil {
		return nil, fmt.Errorf("failed to write nodes: %w", err)
	}
	if err := h.writeLinks(writer); err != nil {
		return nil, fmt.Errorf("failed to write links: %w", err)
	}
	if err := h.writeMetadata(writer); err != nil {
		return nil, fmt.Errorf("failed to write metadata: %w", err)
	}

	writer.Flush()
	return buf.Bytes(), nil
}

// DeserializeFromBytes restores the index from an in-memory byte slice.
func (h *Index) DeserializeFromBytes(ctx context.Context, data []byte) error {
	reader := bufio.NewReader(bytes.NewReader(data))

	if err := h.readHeader(reader); err != nil {
		return fmt.Errorf("failed to read header: %w", err)
	}
	if err := h.readConfig(reader); err != nil {
		return fmt.Errorf("failed to read config: %w", err)
	}

	// Always restore a local raw vector store during deserialization so the
	// index can answer searches even before external storage is populated.
	if h.rawVectorStore == nil {
		switch h.config.RawVectorStore {
		case "", RawVectorStoreMemory:
			h.rawVectorStore = NewInMemoryRawVectorStore(h.config.Dimension)
		case RawVectorStoreSlabby:
			store, err := NewSlabbyRawVectorStore(h.config.Dimension, h.config.RawStoreCap)
			if err != nil {
				return fmt.Errorf("failed to create slabby raw vector store: %w", err)
			}
			h.rawVectorStore = store
		default:
			return fmt.Errorf("unsupported raw vector store backend: %s", h.config.RawVectorStore)
		}
	}

	if err := h.readNodes(ctx, reader); err != nil {
		return fmt.Errorf("failed to read nodes: %w", err)
	}
	if err := h.readLinks(ctx, reader); err != nil {
		return fmt.Errorf("failed to read links: %w", err)
	}
	if err := h.readMetadata(reader); err != nil {
		return fmt.Errorf("failed to read metadata: %w", err)
	}

	return h.rebuildIndexState()
}

// GetPersistenceMetadata returns metadata about the current index state
func (h *Index) GetPersistenceMetadata() *HNSWPersistenceMetadata {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.size == 0 {
		return nil // No metadata for empty index
	}

	return &HNSWPersistenceMetadata{
		Version:       FormatVersion,
		NodeCount:     h.size,
		Dimension:     h.config.Dimension,
		MaxLevel:      h.maxLevel,
		CreatedAt:     time.Now(),
		ChecksumCRC32: h.calculateCRC32(),
		FileSize:      h.estimateFileSize(),
	}
}

// Helper methods for metadata
func (h *Index) estimateFileSize() int64 {
	// Rough estimate of serialized size
	var size int64

	// Header + config
	size += 64

	// Nodes
	for _, node := range h.nodes {
		if node != nil {
			vectorBytes := int64(0)
			switch {
			case node.CompressedVector != nil:
				vectorBytes = int64(len(node.CompressedVector))
			case h.provider == nil:
				vectorBytes = int64(h.config.Dimension * 4)
			}
			size += vectorBytes + 24
		}
	}

	// Links
	for _, node := range h.nodes {
		if node != nil {
			for _, connections := range node.Links {
				size += int64(len(connections) * 4) // 4 bytes per uint32
			}
		}
	}

	return size
}

func (h *Index) freeNodeLinks(node *Node) {
	if node == nil || len(node.Links) == 0 {
		return
	}
	for i, links := range node.Links {
		ptr := unsafe.Pointer(unsafe.SliceData(links))
		if ptr == nil {
			continue
		}
		basePtr := unsafe.Pointer(uintptr(ptr) - 48)
		slack := h.config.M / 4
		if slack < 4 {
			slack = 4
		}
		slack0 := (h.config.M * 2) / 4
		if slack0 < 4 {
			slack0 = 4
		}

		if i == 0 {
			slot := unsafe.Slice((*byte)(basePtr), int(48+(h.config.M*2+slack0)*4))
			_ = h.link0SFL.Deallocate(slot)
		} else {
			slot := unsafe.Slice((*byte)(basePtr), int(48+(h.config.M+slack)*4))
			_ = h.linkSFL.Deallocate(slot)
		}
	}
}
