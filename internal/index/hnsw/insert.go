package hnsw

import (
	"context"

	"github.com/xDarkicex/libravdb/internal/util"
)

// insertNode implements the complete HNSW insertion algorithm
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

	// Phase 1: Search from top level down to node.Level + 1 with ef=1 (greedy search)
	entryPoints := []*util.Candidate{{ID: h.findNodeID(h.entryPoint), Distance: 0}}

	for level := h.maxLevel; level > node.Level; level-- {
		entryPoints = h.searchLevel(node.Vector, h.nodes[entryPoints[0].ID], 1, level)
	}

	// Phase 2: From node.Level down to 0, search with efConstruction and connect
	for level := node.Level; level >= 0; level-- {
		// Search for efConstruction candidates
		candidates := h.searchLevel(node.Vector, h.nodes[entryPoints[0].ID], h.config.EfConstruction, level)

		// Select M neighbors using heuristic
		selected := h.selectNeighborsHeuristic(node.Vector, candidates, level)

		// Connect bidirectionally
		h.connectBidirectional(nodeID, selected, level)

		// Prune connections of neighbors if they exceed maxM
		h.pruneNeighborConnections(selected, level)

		// Update entry points for next level
		entryPoints = selected
	}

	return nil
}

// selectNeighborsHeuristic implements the heuristic neighbor selection
// This prevents clustering and maintains graph navigability
func (h *Index) selectNeighborsHeuristic(queryVector []float32, candidates []*util.Candidate, level int) []*util.Candidate {
	maxM := h.config.M
	if level == 0 && len(candidates) > maxM {
		// Level 0 can have more connections for better recall
		maxM = maxM * 2
	}

	if len(candidates) <= maxM {
		return candidates
	}

	// Sort candidates by distance
	selected := make([]*util.Candidate, 0, maxM)
	remaining := make([]*util.Candidate, len(candidates))
	copy(remaining, candidates)

	for len(selected) < maxM && len(remaining) > 0 {
		bestIdx := 0
		bestCandidate := remaining[0]

		// Find the best candidate using the heuristic
		for i, candidate := range remaining {
			if h.shouldSelectCandidate(queryVector, candidate, selected) {
				if candidate.Distance < bestCandidate.Distance ||
					!h.shouldSelectCandidate(queryVector, bestCandidate, selected) {
					bestIdx = i
					bestCandidate = candidate
				}
			}
		}

		selected = append(selected, bestCandidate)
		// Remove selected candidate from remaining
		remaining[bestIdx] = remaining[len(remaining)-1]
		remaining = remaining[:len(remaining)-1]
	}

	return selected
}

// shouldSelectCandidate checks if a candidate should be selected based on the heuristic
func (h *Index) shouldSelectCandidate(queryVector []float32, candidate *util.Candidate, selected []*util.Candidate) bool {
	candidateVector := h.nodes[candidate.ID].Vector

	// Check pruning condition: don't select if there's a closer neighbor
	// that makes this candidate redundant
	for _, sel := range selected {
		selectedVector := h.nodes[sel.ID].Vector

		// If the distance from candidate to selected neighbor is less than
		// the distance from candidate to query, then candidate is redundant
		distCandidateToSelected := h.distance(candidateVector, selectedVector)
		if distCandidateToSelected < candidate.Distance {
			return false
		}
	}

	return true
}

// connectBidirectional creates bidirectional connections between node and neighbors
func (h *Index) connectBidirectional(nodeID uint32, neighbors []*util.Candidate, level int) {
	node := h.nodes[nodeID]

	for _, neighbor := range neighbors {
		// Add neighbor to node's links
		node.Links[level] = append(node.Links[level], neighbor.ID)

		// Add node to neighbor's links
		neighborNode := h.nodes[neighbor.ID]
		neighborNode.Links[level] = append(neighborNode.Links[level], nodeID)
	}
}

// pruneNeighborConnections ensures neighbors don't exceed maxM connections
func (h *Index) pruneNeighborConnections(neighbors []*util.Candidate, level int) {
	maxM := h.config.M
	if level == 0 {
		maxM = maxM * 2 // Level 0 can have more connections
	}

	for _, neighbor := range neighbors {
		neighborNode := h.nodes[neighbor.ID]

		if len(neighborNode.Links[level]) > maxM {
			// Need to prune - select best maxM connections
			candidates := make([]*util.Candidate, 0, len(neighborNode.Links[level]))

			for _, linkID := range neighborNode.Links[level] {
				distance := h.distance(neighborNode.Vector, h.nodes[linkID].Vector)
				candidates = append(candidates, &util.Candidate{
					ID:       linkID,
					Distance: distance,
				})
			}

			// Select best connections using heuristic
			selected := h.selectNeighborsHeuristic(neighborNode.Vector, candidates, level)

			// Update the neighbor's links
			newLinks := make([]uint32, 0, len(selected))
			for _, sel := range selected {
				newLinks = append(newLinks, sel.ID)
			}
			neighborNode.Links[level] = newLinks
		}
	}
}
