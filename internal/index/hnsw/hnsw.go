package hnsw

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

// VectorEntry represents a vector entry for HNSW indexing
type VectorEntry struct {
	ID       string
	Ordinal  uint32
	Vector   []float32
	Metadata map[string]interface{}
}

// SearchResult represents a search result from HNSW
type SearchResult struct {
	Ordinal  uint32
	ID       string
	Score    float32
	Vector   []float32
	Metadata map[string]interface{}
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
	provider             VectorProvider
	idToIndex            map[string]uint32 // Legacy standalone HNSW path
	ordinalToID          map[uint32]string // Legacy standalone HNSW path
	entryPointCandidates []uint32          // High-level nodes for entry point selection
	// Performance optimizations
	neighborSelector  *NeighborSelector // Optimized neighbor selection
	searchScratchPool sync.Pool
	rawVectorStore    RawVectorStore
	// Quantization support
	quantizer           quant.Quantizer
	trainingVectors     [][]float32 // Vectors collected for quantizer training
	quantizationTrained bool
	// Memory mapping support
	memoryMapped     bool
	mmapPath         string
	mmapSize         int64
	originalMemUsage int64 // Memory usage before mapping
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
	}
	index.searchScratchPool.New = func() interface{} {
		return &searchScratch{}
	}
	if config.Provider == nil {
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
		ordinal = uint32(len(h.nodes))
	}
	if int(ordinal) < len(h.nodes) && h.nodes[ordinal] != nil {
		return fmt.Errorf("node with ordinal %d already exists", ordinal)
	}
	node := &Node{
		Ordinal: ordinal,
		Level:   level,
		Links:   newNodeLinks(level, h.config.M),
	}

	// Handle vector storage (quantized or original)
	if h.quantizer != nil && h.quantizationTrained {
		// Compress the vector
		compressed, err := h.quantizer.Compress(entry.Vector)
		if err != nil {
			return fmt.Errorf("failed to compress vector: %w", err)
		}
		node.CompressedVector = compressed
	} else if h.provider == nil {
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

	if entry.ID != "" {
		h.idToIndex[entry.ID] = nodeID
		if h.provider == nil {
			h.ordinalToID[nodeID] = entry.ID
		}
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

func newNodeLinks(level int, baseM int) [][]uint32 {
	links := make([][]uint32, level+1)
	totalCapacity := 0
	for i := 0; i <= level; i++ {
		totalCapacity += initialNodeLinkCapacity(baseM, i)
	}

	backing := make([]uint32, 0, totalCapacity)
	offset := 0
	for i := 0; i <= level; i++ {
		capacity := initialNodeLinkCapacity(baseM, i)
		links[i] = backing[offset : offset : offset+capacity]
		offset += capacity
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

	// For small batches, use regular insertion
	if len(entries) <= 10 {
		h.mu.Lock()
		defer h.mu.Unlock()

		for _, entry := range entries {
			if err := h.insertSingle(ctx, entry); err != nil {
				return fmt.Errorf("failed to insert entry %s: %w", entry.ID, err)
			}
		}
		return nil
	}

	// For larger batches, use optimized batch processing
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

	if h.size == 0 {
		return nil, fmt.Errorf("index is empty")
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
	ef := max(h.config.EfSearch, k) // Using builtin max function
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

		var resultVector []float32
		if node.CompressedVector != nil && h.quantizer != nil {
			var err error
			resultVector, err = h.quantizer.Decompress(node.CompressedVector)
			if err != nil {
				resultVector = nil
			}
		} else if h.provider == nil {
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

	// Clear all data structures
	h.nodes = nil
	h.entryPoint = nil
	h.size = 0
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
	grown := make([]*Node, nextNodeCapacity(len(h.nodes), minSize))
	copy(grown, h.nodes)
	h.nodes = grown
}

func nextNodeCapacity(currentLen, minSize int) int {
	if minSize <= currentLen {
		return currentLen
	}
	newSize := currentLen
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
	if node.CompressedVector != nil && h.quantizer != nil {
		return h.quantizer.Decompress(node.CompressedVector)
	}
	if h.provider != nil {
		return h.provider.GetByOrdinal(node.Ordinal)
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
	// Keep essential structures but clear large data
	for _, node := range h.nodes {
		if node != nil {
			node.CompressedVector = nil
		}
	}
	if h.rawVectorStore != nil {
		_ = h.rawVectorStore.Reset()
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
		MaxLevel:      h.getMaxLevel(),
		CreatedAt:     time.Now(),
		ChecksumCRC32: h.calculateCRC32(),
		FileSize:      h.estimateFileSize(),
	}
}

// Helper methods for metadata
func (h *Index) getMaxLevel() int {
	maxLevel := 0
	for _, node := range h.nodes {
		if node != nil && node.Level > maxLevel {
			maxLevel = node.Level
		}
	}
	return maxLevel
}

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
