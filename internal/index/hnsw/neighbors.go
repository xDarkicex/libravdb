package hnsw

import (
	"slices"

	"github.com/xDarkicex/libravdb/internal/util"
)

// NeighborSelector implements optimized neighbor selection algorithms
type NeighborSelector struct {
	maxConnections  int
	levelMultiplier float64
}

func compareCandidatePtrs(a, b *util.Candidate) int {
	if a.Distance < b.Distance {
		return -1
	}
	if a.Distance > b.Distance {
		return 1
	}
	if a.ID < b.ID {
		return -1
	}
	if a.ID > b.ID {
		return 1
	}
	return 0
}

func compareCandidateValues(a, b util.Candidate) int {
	if a.Distance < b.Distance {
		return -1
	}
	if a.Distance > b.Distance {
		return 1
	}
	if a.ID < b.ID {
		return -1
	}
	if a.ID > b.ID {
		return 1
	}
	return 0
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
	slices.SortFunc(candidates, compareCandidatePtrs)

	// Bound the heuristic working set so insertion cost remains proportional to
	// the graph degree rather than the full efConstruction frontier.
	candidateLimit := maxM * 4
	if level > 0 {
		candidateLimit = maxM * 2
	}
	if candidateLimit < maxM {
		candidateLimit = maxM
	}
	if len(candidates) > candidateLimit {
		candidates = candidates[:candidateLimit]
	}

	// Use a more efficient selection algorithm
	return ns.selectWithSimpleHeuristic(queryVector, candidates, maxM, index)
}

func (ns *NeighborSelector) SelectNeighborsOptimizedValues(
	queryVector []float32,
	candidates []util.Candidate,
	level int,
	index *Index,
) []util.Candidate {
	maxM := ns.maxConnections
	if level == 0 {
		maxM = int(float64(maxM) * ns.levelMultiplier)
	}

	if len(candidates) <= maxM {
		return candidates
	}

	slices.SortFunc(candidates, compareCandidateValues)

	candidateLimit := maxM * 4
	if level > 0 {
		candidateLimit = maxM * 2
	}
	if candidateLimit < maxM {
		candidateLimit = maxM
	}
	if len(candidates) > candidateLimit {
		candidates = candidates[:candidateLimit]
	}

	return ns.selectWithSimpleHeuristicValues(queryVector, candidates, maxM, index)
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
	selectedVectors := make([][]float32, 0, min(maxM, 4))

	// Always select the closest candidate
	selected = append(selected, candidates[0])
	if vector, ok := index.nodeVectorForHeuristic(candidates[0].ID); ok {
		selectedVectors = append(selectedVectors, vector)
	} else {
		selectedVectors = append(selectedVectors, nil)
	}

	// For remaining selections, use distance-based selection with simple diversity check
	for i := 1; i < len(candidates) && len(selected) < maxM; i++ {
		candidate := candidates[i]

		// Simple diversity check: ensure candidate is not too close to already selected nodes
		shouldSelect := true
		candidateVector, ok := index.nodeVectorForHeuristic(candidate.ID)
		if !ok {
			continue // Skip if we can't get the vector
		}

		// Check against a limited number of already selected nodes for efficiency
		checkLimit := min(len(selected), 3) // Only check against 3 closest selected nodes
		for j := 0; j < checkLimit; j++ {
			selectedVector := selectedVectors[j]
			if selectedVector == nil {
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
			if len(selectedVectors) < maxM {
				selectedVectors = append(selectedVectors, candidateVector)
			}
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

func (ns *NeighborSelector) selectWithSimpleHeuristicValues(
	queryVector []float32,
	candidates []util.Candidate,
	maxM int,
	index *Index,
) []util.Candidate {
	if len(candidates) <= maxM {
		return candidates
	}

	selectedCount := 1
	var selectedVectorBuf [4][]float32
	selectedVectors := selectedVectorBuf[:0]

	if vector, ok := index.nodeVectorForHeuristic(candidates[0].ID); ok {
		selectedVectors = append(selectedVectors, vector)
	} else {
		selectedVectors = append(selectedVectors, nil)
	}

	for i := 1; i < len(candidates) && selectedCount < maxM; i++ {
		candidate := candidates[i]
		shouldSelect := true
		candidateVector, ok := index.nodeVectorForHeuristic(candidate.ID)
		if !ok {
			continue
		}

		checkLimit := min(selectedCount, 3)
		for j := 0; j < checkLimit; j++ {
			selectedVector := selectedVectors[j]
			if selectedVector == nil {
				continue
			}
			distToSelected := index.distance(candidateVector, selectedVector)
			if distToSelected < candidate.Distance*0.8 {
				shouldSelect = false
				break
			}
		}

		if shouldSelect {
			candidates[selectedCount] = candidate
			selectedCount++
			if len(selectedVectors) < len(selectedVectorBuf) {
				selectedVectors = append(selectedVectors, candidateVector)
			}
		}
	}

	for i := 1; i < len(candidates) && selectedCount < maxM; i++ {
		candidate := candidates[i]
		alreadySelected := false
		for j := 0; j < selectedCount; j++ {
			if candidates[j].ID == candidate.ID {
				alreadySelected = true
				break
			}
		}
		if !alreadySelected {
			candidates[selectedCount] = candidate
			selectedCount++
		}
	}

	return candidates[:selectedCount]
}

// PruneConnections optimizes the connections of a node to maintain graph quality
func (ns *NeighborSelector) PruneConnections(
	nodeID uint32,
	level int,
	index *Index,
) error {
	scratch := index.acquireSearchScratch()
	defer index.releaseSearchScratch(scratch)

	node := index.nodes[nodeID]
	if node == nil {
		return nil
	}
	if level >= len(node.Links) {
		return nil
	}

	maxM := ns.maxConnections
	if level == 0 {
		maxM = int(float64(maxM) * ns.levelMultiplier)
	}

	overflowSlack := levelOverflowSlack(maxM)
	nodeVector, err := index.getNodeVector(node)
	if err != nil {
		return err
	}

	originalLinks := node.Links[level]

	// Create candidates from current live connections and compact stale links.
	if cap(scratch.pruneBuf) < len(originalLinks) {
		scratch.pruneBuf = make([]util.Candidate, 0, len(originalLinks))
	} else {
		scratch.pruneBuf = scratch.pruneBuf[:0]
	}
	candidates := scratch.pruneBuf
	liveLinks := make([]uint32, 0, len(originalLinks))
	for _, linkID := range originalLinks {
		if int(linkID) >= len(index.nodes) {
			continue
		}
		linkNode := index.nodes[linkID]
		if linkNode == nil {
			continue
		}
		linkVector, err := index.getNodeVector(linkNode)
		if err != nil {
			continue
		}

		distance := index.distance(nodeVector, linkVector)
		liveLinks = append(liveLinks, linkID)
		candidates = append(candidates, util.Candidate{
			ID:       linkID,
			Distance: distance,
		})
	}

	if len(candidates) <= maxM+overflowSlack && len(liveLinks) == len(originalLinks) {
		return nil
	}

	keepIDs := make(map[uint32]struct{}, min(len(candidates), maxM))
	if len(candidates) <= maxM {
		newLinks := node.Links[level][:0]
		for _, candidate := range candidates {
			newLinks = append(newLinks, candidate.ID)
			keepIDs[candidate.ID] = struct{}{}
		}
		node.Links[level] = newLinks
		ns.removeDroppedBacklinks(nodeID, level, liveLinks, keepIDs, index)
		return nil
	}

	// Pruning runs on every insertion and is a major write-path hot loop.
	// Keeping the closest neighbors is much cheaper here than re-running the
	// full diversity heuristic, while still preserving bounded graph degree.
	slices.SortFunc(candidates, compareCandidateValues)

	// Update the node's connections
	newLinks := node.Links[level][:0]
	for _, sel := range candidates[:maxM] {
		newLinks = append(newLinks, sel.ID)
		keepIDs[sel.ID] = struct{}{}
	}
	node.Links[level] = newLinks
	ns.removeDroppedBacklinks(nodeID, level, liveLinks, keepIDs, index)

	return nil
}

func (ns *NeighborSelector) removeDroppedBacklinks(
	nodeID uint32,
	level int,
	original []uint32,
	keepIDs map[uint32]struct{},
	index *Index,
) {
	for _, linkID := range original {
		if _, keep := keepIDs[linkID]; keep {
			continue
		}
		index.removeConnection(linkID, nodeID, level)
	}
}

func (h *Index) nodeVectorForHeuristic(nodeID uint32) ([]float32, bool) {
	if int(nodeID) >= len(h.nodes) {
		return nil, false
	}
	node := h.nodes[nodeID]
	if node == nil {
		return nil, false
	}
	vector, err := h.getNodeVector(node)
	if err != nil {
		return nil, false
	}
	return vector, true
}
