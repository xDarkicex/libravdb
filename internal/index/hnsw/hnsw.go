package hnsw

import (
	"context"
	"fmt"
	"math/rand"
	"sync"

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

	// Create new node
	level := h.generateLevel()
	node := &Node{
		ID:       entry.ID,
		Vector:   make([]float32, len(entry.Vector)),
		Level:    level,
		Metadata: entry.Metadata,
		Links:    make([][]uint32, level+1),
	}
	copy(node.Vector, entry.Vector)

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
		results = append(results, &SearchResult{
			ID:       h.nodes[candidate.ID].ID,
			Score:    candidate.Distance,
			Vector:   h.nodes[candidate.ID].Vector,
			Metadata: h.nodes[candidate.ID].Metadata,
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
		// Vector data
		usage += int64(len(node.Vector) * 4) // 4 bytes per float32

		// Links
		for _, links := range node.Links {
			usage += int64(len(links) * 4) // 4 bytes per uint32
		}

		// Node overhead (approximate)
		usage += 64
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
	return nil
}

func (h *Index) Delete(ctx context.Context, id string) error {
	return h.deleteNode(ctx, id)
}
