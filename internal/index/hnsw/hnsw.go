package hnsw

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

// VectorEntry represents a vector entry for HNSW indexing
type VectorEntry struct {
	ID       string
	Vector   []float32
	Metadata map[string]interface{}
}

// SearchResult represents a search result from HNSW
type SearchResult struct {
	ID       string
	Score    float32
	Vector   []float32
	Metadata map[string]interface{}
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
	idToIndex            map[string]uint32 // O(1) ID to node index lookup
	entryPointCandidates []uint32          // High-level nodes for entry point selection
	// Quantization support
	quantizer           quant.Quantizer
	trainingVectors     [][]float32 // Vectors collected for quantizer training
	quantizationTrained bool
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
		idToIndex:            make(map[string]uint32),
		entryPointCandidates: make([]uint32, 0),
		trainingVectors:      make([][]float32, 0),
		quantizationTrained:  false,
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

	// Check for duplicate ID
	if _, exists := h.idToIndex[entry.ID]; exists {
		return fmt.Errorf("node with ID '%s' already exists", entry.ID)
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

	// Create new node
	level := h.generateLevel()
	node := &Node{
		ID:       entry.ID,
		Level:    level,
		Metadata: entry.Metadata,
		Links:    make([][]uint32, level+1),
	}

	// Handle vector storage (quantized or original)
	if h.quantizer != nil && h.quantizationTrained {
		// Compress the vector
		compressed, err := h.quantizer.Compress(entry.Vector)
		if err != nil {
			return fmt.Errorf("failed to compress vector: %w", err)
		}
		node.CompressedVector = compressed
		// Don't store original vector to save memory
		node.Vector = nil
	} else {
		// Store original vector
		node.Vector = make([]float32, len(entry.Vector))
		copy(node.Vector, entry.Vector)
	}

	// Initialize empty link lists for each level
	for i := 0; i <= level; i++ {
		node.Links[i] = make([]uint32, 0, h.config.M)
	}

	nodeID := uint32(len(h.nodes))
	h.nodes = append(h.nodes, node)

	// Add to ID mapping
	h.idToIndex[entry.ID] = nodeID

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
	if err := h.insertNode(ctx, node, nodeID); err != nil {
		// Rollback: remove the node we just added and clean up mappings
		h.nodes = h.nodes[:len(h.nodes)-1]
		delete(h.idToIndex, entry.ID)

		// Remove from entry point candidates if it was added
		if level >= 2 && len(h.entryPointCandidates) > 0 {
			// Remove the last added candidate (which would be this node)
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
		candidates := h.searchLevel(query, ep, 1, level) // This calls the method from search.go
		if len(candidates) > 0 {
			ep = h.nodes[candidates[0].ID]
		}
	}

	// Phase 2: Search level 0 with ef
	ef := max(h.config.EfSearch, k) // Using builtin max function
	candidates := h.searchLevel(query, ep, ef, 0)

	// Convert to results and limit to k
	results := make([]*SearchResult, 0, min(k, len(candidates))) // Using builtin min function
	for i, candidate := range candidates {
		if i >= k {
			break
		}

		node := h.nodes[candidate.ID]

		// Get the vector for the result (decompressed if quantized)
		var resultVector []float32
		if node.CompressedVector != nil && h.quantizer != nil {
			var err error
			resultVector, err = h.quantizer.Decompress(node.CompressedVector)
			if err != nil {
				// If decompression fails, return nil vector
				resultVector = nil
			}
		} else {
			resultVector = node.Vector
		}

		results = append(results, &SearchResult{
			ID:       node.ID,
			Score:    candidate.Distance,
			Vector:   resultVector,
			Metadata: node.Metadata,
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

	var usage int64
	for _, node := range h.nodes {
		// Vector data (original or compressed)
		if node.CompressedVector != nil {
			usage += int64(len(node.CompressedVector))
		} else if node.Vector != nil {
			usage += int64(len(node.Vector) * 4) // 4 bytes per float32
		}

		// Links
		for _, links := range node.Links {
			usage += int64(len(links) * 4) // 4 bytes per uint32
		}

		// Node overhead (approximate)
		usage += 64
	}

	// Add quantizer memory usage if present
	if h.quantizer != nil {
		usage += h.quantizer.MemoryUsage()
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

// findNodeID finds the ID of a node (helper function)
func (h *Index) findNodeID(target *Node) uint32 {
	for i, node := range h.nodes {
		if node == target {
			return uint32(i)
		}
	}
	return ^uint32(0) // Not found
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
		// PQ needs more training data for k-means clustering
		return max(1000, h.config.Quantization.Codebooks*256)
	case quant.ScalarQuantization:
		// Scalar quantization needs less training data
		return max(100, h.config.Dimension*10)
	default:
		return 1000
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
	return node.Vector, nil
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

	// Fall back to regular distance computation
	return h.distance(vec1, vec2), nil
}

func (h *Index) Delete(ctx context.Context, id string) error {
	return h.deleteNode(ctx, id)
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
			size += int64(len(node.ID)) + int64(len(node.Vector)*4) + 16
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
