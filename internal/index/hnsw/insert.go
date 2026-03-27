package hnsw

import (
	"context"
	"fmt"

	"github.com/xDarkicex/libravdb/internal/util"
)

// insertNode implements the optimized HNSW insertion algorithm
func (h *Index) insertNode(ctx context.Context, node *Node, nodeID uint32) error {
	// Handle the second node (simple connection to entry point)
	if h.size == 1 {
		entryID := h.findNodeID(h.entryPoint)
		if entryID != ^uint32(0) && node.Level >= 0 {
			node.Links[0] = append(node.Links[0], entryID)
			h.entryPoint.Links[0] = append(h.entryPoint.Links[0], nodeID)
		}
		return nil
	}

	// Initialize neighbor selector if not already done
	if h.neighborSelector == nil {
		h.neighborSelector = NewNeighborSelector(h.config.M, 2.0)
	}

	// Get the vector for search (original or decompressed)
	searchVector, err := h.getNodeVector(node)
	if err != nil {
		return fmt.Errorf("failed to get node vector for search: %w", err)
	}

	// Phase 1: Search from top level down to node.Level + 1 with ef=1 (greedy search)
	entryPoints := h.fallbackEntryPoints(searchVector, h.entryPoint)

	for level := h.maxLevel; level > node.Level; level-- {
		currentNode := h.pickEntryNode(entryPoints)
		if currentNode == nil {
			currentNode = h.entryPoint
		}
		entryPoints = h.searchLevel(searchVector, currentNode, 1, level)
		if len(entryPoints) == 0 {
			entryPoints = h.fallbackEntryPoints(searchVector, currentNode)
		}
	}

	// Phase 2: From node.Level down to 0, search with efConstruction and connect
	for level := node.Level; level >= 0; level-- {
		// Search for efConstruction candidates
		currentNode := h.pickEntryNode(entryPoints)
		if currentNode == nil {
			currentNode = h.entryPoint
		}
		candidates := h.searchLevel(searchVector, currentNode, h.config.EfConstruction, level)
		if len(candidates) == 0 {
			candidates = h.fallbackEntryPoints(searchVector, currentNode)
		}

		// Select M neighbors using optimized heuristic
		selected := h.neighborSelector.SelectNeighborsOptimized(searchVector, candidates, level, h)
		if len(selected) == 0 {
			selected = h.fallbackEntryPoints(searchVector, currentNode)
		}

		// Connect bidirectionally
		h.connectBidirectionalOptimized(nodeID, selected, level)

		// Prune connections of neighbors if they exceed maxM
		h.pruneNeighborConnectionsOptimized(selected, level)

		// Update entry points for next level
		entryPoints = selected
	}

	return nil
}

func (h *Index) pickEntryNode(entryPoints []*util.Candidate) *Node {
	if len(entryPoints) == 0 {
		return nil
	}

	entryID := entryPoints[0].ID
	if int(entryID) >= len(h.nodes) {
		return nil
	}
	return h.nodes[entryID]
}

func (h *Index) fallbackEntryPoints(searchVector []float32, node *Node) []*util.Candidate {
	if node == nil {
		return nil
	}

	entryID := h.findNodeID(node)
	if entryID == ^uint32(0) || int(entryID) >= len(h.nodes) {
		return nil
	}

	distance := h.computeDistanceOptimized(searchVector, node)
	if distance < 0 {
		distance = 0
	}

	return []*util.Candidate{{
		ID:       entryID,
		Distance: distance,
	}}
}

// Legacy method for backward compatibility - delegates to optimized version
func (h *Index) selectNeighborsHeuristic(queryVector []float32, candidates []*util.Candidate, level int) []*util.Candidate {
	if h.neighborSelector == nil {
		h.neighborSelector = NewNeighborSelector(h.config.M, 2.0)
	}
	return h.neighborSelector.SelectNeighborsOptimized(queryVector, candidates, level, h)
}

// Legacy method for backward compatibility - delegates to optimized version
func (h *Index) connectBidirectional(nodeID uint32, neighbors []*util.Candidate, level int) {
	h.connectBidirectionalOptimized(nodeID, neighbors, level)
}

// connectBidirectionalOptimized creates bidirectional connections with better memory management
func (h *Index) connectBidirectionalOptimized(nodeID uint32, neighbors []*util.Candidate, level int) {
	node := h.nodes[nodeID]

	// Pre-allocate slice capacity to avoid reallocations
	if cap(node.Links[level]) < len(neighbors) {
		newLinks := make([]uint32, len(node.Links[level]), len(neighbors)+h.config.M)
		copy(newLinks, node.Links[level])
		node.Links[level] = newLinks
	}

	for _, neighbor := range neighbors {
		// Add neighbor to node's links
		node.Links[level] = append(node.Links[level], neighbor.ID)

		// Add node to neighbor's links
		neighborNode := h.nodes[neighbor.ID]
		// Only add the connection if the neighbor has this level
		if level < len(neighborNode.Links) {
			// Pre-allocate capacity for neighbor's links too
			if cap(neighborNode.Links[level]) < len(neighborNode.Links[level])+1 {
				newLinks := make([]uint32, len(neighborNode.Links[level]), len(neighborNode.Links[level])+h.config.M)
				copy(newLinks, neighborNode.Links[level])
				neighborNode.Links[level] = newLinks
			}
			neighborNode.Links[level] = append(neighborNode.Links[level], nodeID)
		}
	}
}

// pruneNeighborConnectionsOptimized ensures neighbors don't exceed maxM connections using optimized algorithm
func (h *Index) pruneNeighborConnectionsOptimized(neighbors []*util.Candidate, level int) {
	if h.neighborSelector == nil {
		h.neighborSelector = NewNeighborSelector(h.config.M, 2.0)
	}

	for _, neighbor := range neighbors {
		if err := h.neighborSelector.PruneConnections(neighbor.ID, level, h); err != nil {
			// Log error but continue - this is not critical for correctness
			continue
		}
	}
}

// Legacy methods for backward compatibility
// pruneNeighborConnections ensures neighbors don't exceed maxM connections
func (h *Index) pruneNeighborConnections(neighbors []*util.Candidate, level int) {
	h.pruneNeighborConnectionsOptimized(neighbors, level)
}
