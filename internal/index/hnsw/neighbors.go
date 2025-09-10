package hnsw

import (
	"sort"

	"github.com/xDarkicex/libravdb/internal/util"
)

// NeighborSelector implements optimized neighbor selection algorithms
type NeighborSelector struct {
	maxConnections  int
	levelMultiplier float64
}

// NewNeighborSelector creates a new optimized neighbor selector
func NewNeighborSelector(maxConnections int, levelMultiplier float64) *NeighborSelector {
	return &NeighborSelector{
		maxConnections:  maxConnections,
		levelMultiplier: levelMultiplier,
	}
}

// SelectNeighborsOptimized implements an optimized neighbor selection algorithm
// that prevents clustering and maintains graph navigability with better performance
func (ns *NeighborSelector) SelectNeighborsOptimized(
	queryVector []float32,
	candidates []*util.Candidate,
	level int,
	index *Index,
) []*util.Candidate {
	maxM := ns.maxConnections
	if level == 0 {
		// Level 0 can have more connections for better recall
		maxM = int(float64(maxM) * ns.levelMultiplier)
	}

	if len(candidates) <= maxM {
		return candidates
	}

	// Pre-sort candidates by distance for better performance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Distance < candidates[j].Distance
	})

	// Use a more efficient selection algorithm
	return ns.selectWithSimpleHeuristic(queryVector, candidates, maxM, index)
}

// selectWithSimpleHeuristic uses a simplified but efficient heuristic
// that provides good graph quality while being much faster than the complex version
func (ns *NeighborSelector) selectWithSimpleHeuristic(
	queryVector []float32,
	candidates []*util.Candidate,
	maxM int,
	index *Index,
) []*util.Candidate {
	if len(candidates) <= maxM {
		return candidates
	}

	selected := make([]*util.Candidate, 0, maxM)

	// Always select the closest candidate
	selected = append(selected, candidates[0])

	// For remaining selections, use distance-based selection with simple diversity check
	for i := 1; i < len(candidates) && len(selected) < maxM; i++ {
		candidate := candidates[i]

		// Simple diversity check: ensure candidate is not too close to already selected nodes
		shouldSelect := true
		candidateNode := index.nodes[candidate.ID]
		candidateVector, err := index.getNodeVector(candidateNode)
		if err != nil {
			continue // Skip if we can't get the vector
		}

		// Check against a limited number of already selected nodes for efficiency
		checkLimit := min(len(selected), 3) // Only check against 3 closest selected nodes
		for j := 0; j < checkLimit; j++ {
			selectedNode := index.nodes[selected[j].ID]
			selectedVector, err := index.getNodeVector(selectedNode)
			if err != nil {
				continue
			}

			// Simple distance check - if candidate is much closer to selected node
			// than to query, it might be redundant
			distToSelected := index.distance(candidateVector, selectedVector)
			if distToSelected < candidate.Distance*0.8 { // 80% threshold
				shouldSelect = false
				break
			}
		}

		if shouldSelect {
			selected = append(selected, candidate)
		}
	}

	// If we still need more candidates and have remaining ones, just take them by distance
	for i := 1; i < len(candidates) && len(selected) < maxM; i++ {
		candidate := candidates[i]
		// Check if already selected
		alreadySelected := false
		for _, sel := range selected {
			if sel.ID == candidate.ID {
				alreadySelected = true
				break
			}
		}
		if !alreadySelected {
			selected = append(selected, candidate)
		}
	}

	return selected
}

// PruneConnections optimizes the connections of a node to maintain graph quality
func (ns *NeighborSelector) PruneConnections(
	nodeID uint32,
	level int,
	index *Index,
) error {
	node := index.nodes[nodeID]
	if level >= len(node.Links) {
		return nil
	}

	maxM := ns.maxConnections
	if level == 0 {
		maxM = int(float64(maxM) * ns.levelMultiplier)
	}

	if len(node.Links[level]) <= maxM {
		return nil // No pruning needed
	}

	// Get node vector for distance calculations
	nodeVector, err := index.getNodeVector(node)
	if err != nil {
		return err
	}

	// Create candidates from current connections
	candidates := make([]*util.Candidate, 0, len(node.Links[level]))
	for _, linkID := range node.Links[level] {
		linkNode := index.nodes[linkID]
		linkVector, err := index.getNodeVector(linkNode)
		if err != nil {
			continue // Skip if we can't get the vector
		}

		distance := index.distance(nodeVector, linkVector)
		candidates = append(candidates, &util.Candidate{
			ID:       linkID,
			Distance: distance,
		})
	}

	// Select best connections
	selected := ns.SelectNeighborsOptimized(nodeVector, candidates, level, index)

	// Update the node's connections
	newLinks := make([]uint32, 0, len(selected))
	for _, sel := range selected {
		newLinks = append(newLinks, sel.ID)
	}
	node.Links[level] = newLinks

	return nil
}
