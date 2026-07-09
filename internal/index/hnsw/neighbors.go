package hnsw

import (
	"fmt"
	"runtime"
	"slices"
	"sync/atomic"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/memory"
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
	var selectedVectorBuf [128][]float32
	selectedVectors := selectedVectorBuf[:0]
	if maxM > len(selectedVectorBuf) {
		selectedVectors = make([][]float32, 0, maxM)
	}

	// Stack-allocated bitset for up to 32768 candidates
	var picked [512]uint64
	setPicked := func(idx int) {
		if idx < len(picked)*64 {
			picked[idx/64] |= (1 << (idx % 64))
		}
	}
	isPicked := func(idx int) bool {
		if idx < len(picked)*64 {
			return (picked[idx/64] & (1 << (idx % 64))) != 0
		}
		return false
	}

	// Always select the closest candidate
	selected = append(selected, candidates[0])
	setPicked(0)
	if vector, ok := index.nodeVectorForHeuristic(candidates[0].ID); ok {
		selectedVectors = append(selectedVectors, vector)
	} else {
		selectedVectors = append(selectedVectors, nil)
	}

	for i := 1; i < len(candidates) && len(selected) < maxM; i++ {
		candidate := candidates[i]
		shouldSelect := true
		candidateVector, ok := index.nodeVectorForHeuristic(candidate.ID)
		if !ok {
			continue // Skip if we can't get the vector
		}

		for j := 0; j < len(selected); j++ {
			selectedVector := selectedVectors[j]
			if selectedVector == nil {
				continue
			}
			distToSelected := index.distance(candidateVector, selectedVector)
			if distToSelected < candidate.Distance {
				shouldSelect = false
				break
			}
		}

		if shouldSelect {
			selected = append(selected, candidate)
			setPicked(i)
			selectedVectors = append(selectedVectors, candidateVector)
		}
	}

	// If we still need more candidates and have remaining ones, just take them by distance
	for i := 1; i < len(candidates) && len(selected) < maxM; i++ {
		if !isPicked(i) {
			selected = append(selected, candidates[i])
			setPicked(i)
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

	var selectedBuf [128]util.Candidate
	selected := selectedBuf[:0]
	if maxM > len(selectedBuf) {
		selected = make([]util.Candidate, 0, maxM)
	}

	var selectedVectorBuf [128][]float32
	selectedVectors := selectedVectorBuf[:0]
	if maxM > len(selectedVectorBuf) {
		selectedVectors = make([][]float32, 0, maxM)
	}

	var picked [512]uint64
	setPicked := func(idx int) {
		if idx < len(picked)*64 {
			picked[idx/64] |= (1 << (idx % 64))
		}
	}
	isPicked := func(idx int) bool {
		if idx < len(picked)*64 {
			return (picked[idx/64] & (1 << (idx % 64))) != 0
		}
		return false
	}

	selected = append(selected, candidates[0])
	setPicked(0)

	if vector, ok := index.nodeVectorForHeuristic(candidates[0].ID); ok {
		selectedVectors = append(selectedVectors, vector)
	} else {
		selectedVectors = append(selectedVectors, nil)
	}

	for i := 1; i < len(candidates) && len(selected) < maxM; i++ {
		candidate := candidates[i]
		shouldSelect := true
		candidateVector, ok := index.nodeVectorForHeuristic(candidate.ID)
		if !ok {
			continue
		}

		for j := 0; j < len(selected); j++ {
			selectedVector := selectedVectors[j]
			if selectedVector == nil {
				continue
			}
			distToSelected := index.distance(candidateVector, selectedVector)
			if distToSelected < candidate.Distance {
				shouldSelect = false
				break
			}
		}

		if shouldSelect {
			selected = append(selected, candidate)
			setPicked(i)
			selectedVectors = append(selectedVectors, candidateVector)
		}
	}

	for i := 1; i < len(candidates) && len(selected) < maxM; i++ {
		if !isPicked(i) {
			selected = append(selected, candidates[i])
			setPicked(i)
		}
	}

	copy(candidates, selected)
	return candidates[:len(selected)]
}

// PruneConnections optimizes the connections of a node to maintain graph quality
func (ns *NeighborSelector) PruneConnections(
	nodeID uint32,
	level int,
	index *Index,
) error {
	scratch := index.acquireSearchScratch()
	defer index.releaseSearchScratch(scratch)

	node := index.nodes.Get(nodeID)
	if node == nil {
		return nil
	}
	if level >= (node.Level + 1) {
		return nil
	}

	maxM := ns.maxConnections
	if level == 0 {
		maxM = int(float64(maxM) * ns.levelMultiplier)
	}

	nodeVector, err := index.getNodeVector(node)
	if err != nil {
		return err
	}

	for !index.acquirePruneLock(node) {
		runtime.Gosched()
	}

	originalLinks := index.getNodeLinks(node, level)

	// Create candidates from current live connections and compact stale links.
	// Always re-allocate from the arena — the previous buffer is stale after Reset.
	pruneBuf, err := memory.ArenaSlice[util.Candidate](scratch.arena, len(originalLinks))
	if err != nil {
		index.releasePruneLock(node)
		return fmt.Errorf("arena allocate pruneBuf: %w", err)
	}
	scratch.pruneBuf = pruneBuf[:0]
	candidates := scratch.pruneBuf
	liveLinks, err := memory.ArenaSlice[uint32](scratch.arena, len(originalLinks))
	if err != nil {
		index.releasePruneLock(node)
		return fmt.Errorf("arena allocate liveLinks: %w", err)
	}
	liveLinks = liveLinks[:0]
	for _, linkID := range originalLinks {
		if int(linkID) >= index.nodes.Len() {
			continue
		}
		linkNode := index.nodes.Get(linkID)
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

	if len(candidates) <= maxM && len(liveLinks) == len(originalLinks) {
		index.releasePruneLock(node)
		return nil
	}

	maxCapacity := linkArrayCapacity(index.config.M, level)
	slice := unsafe.Slice(node.Links[level], maxCapacity)

	keepIDs, _ := memory.ArenaSlice[uint32](scratch.arena, min(len(candidates), maxM))
	keepIDs = keepIDs[:0]
	if len(candidates) <= maxM {
		// Write directly to fixed-size array
		for i, candidate := range candidates {
			atomic.StoreUint32(&slice[i], candidate.ID)
			keepIDs = append(keepIDs, candidate.ID)
		}
		// Sentinel the rest
		for i := len(candidates); i < maxCapacity; i++ {
			if atomic.LoadUint32(&slice[i]) == SentinelNodeID {
				break
			}
			atomic.StoreUint32(&slice[i], SentinelNodeID)
		}
		atomic.StoreUint32(&node.LinkCounts[level], uint32(len(candidates)))
		atomic.StoreUint32(&node.LinkHeuristic[level], 0)

		index.releasePruneLock(node)
		ns.removeDroppedBacklinks(nodeID, level, liveLinks, keepIDs, index)
		return nil
	}

	// Qdrant applies the same diversity heuristic when an existing link
	// container overflows. Closest-only pruning is faster, but it clusters
	// backlinks and forces wider beams to recover recall.
	slices.SortFunc(candidates, compareCandidateValues)
	selected := ns.selectWithSimpleHeuristicValues(nodeVector, candidates, maxM, index)

	// Update the node's connections
	// Update the node's connections in the fixed array
	for i, sel := range selected {
		atomic.StoreUint32(&slice[i], sel.ID)
		keepIDs = append(keepIDs, sel.ID)
	}
	// Sentinel the rest
	for i := len(selected); i < maxCapacity; i++ {
		if atomic.LoadUint32(&slice[i]) == SentinelNodeID {
			break
		}
		atomic.StoreUint32(&slice[i], SentinelNodeID)
	}
	atomic.StoreUint32(&node.LinkCounts[level], uint32(len(selected)))
	atomic.StoreUint32(&node.LinkHeuristic[level], uint32(len(selected)))

	index.releasePruneLock(node)
	ns.removeDroppedBacklinks(nodeID, level, liveLinks, keepIDs, index)

	return nil
}

func (ns *NeighborSelector) connectLinkWithHeuristic(
	targetID uint32,
	newID uint32,
	level int,
	index *Index,
) bool {
	if int(targetID) >= index.nodes.Len() || int(newID) >= index.nodes.Len() {
		return false
	}
	targetNode := index.nodes.Get(targetID)
	if targetNode == nil || level >= (targetNode.Level+1) {
		return false
	}

	targetVector, ok := index.nodeVectorForHeuristic(targetID)
	if !ok {
		return index.appendWithSpinlock(targetNode, targetNode.Links[level], newID, index.config.M, level)
	}
	newVector, ok := index.nodeVectorForHeuristic(newID)
	if !ok {
		return false
	}
	newDistance := index.distance(targetVector, newVector)

	maxM := levelMaxLinks(index.config.M, level)
	maxCapacity := linkArrayCapacity(index.config.M, level)

	var originalBuf [256]uint32
	original := originalBuf[:0]
	if maxCapacity > len(originalBuf) {
		original = make([]uint32, 0, maxCapacity)
	}

	var candidateBuf [257]util.Candidate
	candidates := candidateBuf[:0]
	if maxCapacity+1 > len(candidateBuf) {
		candidates = make([]util.Candidate, 0, maxCapacity+1)
	}

	var droppedBuf [256]uint32
	dropped := droppedBuf[:0]
	if maxCapacity > len(droppedBuf) {
		dropped = make([]uint32, 0, maxCapacity)
	}

	for !index.acquirePruneLock(targetNode) {
		runtime.Gosched()
	}

	accepted := false
	func() {
		defer index.releasePruneLock(targetNode)

		slice := unsafe.Slice(targetNode.Links[level], maxCapacity)
		count := int(atomic.LoadUint32(&targetNode.LinkCounts[level]))
		if count > maxCapacity {
			count = maxCapacity
		}
		heuristicCount := int(atomic.LoadUint32(&targetNode.LinkHeuristic[level]))

		for i := 0; i < count; i++ {
			linkID := atomic.LoadUint32(&slice[i])
			if linkID == SentinelNodeID {
				continue
			}
			if linkID == newID {
				return
			}
			if int(linkID) >= index.nodes.Len() || index.nodes.Get(linkID) == nil {
				continue
			}
			original = append(original, linkID)
		}

		if len(original) < maxM {
			atomic.StoreUint32(&slice[len(original)], newID)
			atomic.StoreUint32(&targetNode.LinkCounts[level], uint32(len(original)+1))
			atomic.StoreUint32(&targetNode.LinkHeuristic[level], 0)
			accepted = true
			return
		}

		if heuristicCount >= len(original) && len(original) > 0 {
			worstID := original[len(original)-1]
			worstVector, ok := index.nodeVectorForHeuristic(worstID)
			if ok && newDistance >= index.distance(targetVector, worstVector) {
				return
			}
			var selectedIDBuf [256]uint32
			selectedIDs := selectedIDBuf[:0]
			if maxM > len(selectedIDBuf) {
				selectedIDs = make([]uint32, 0, maxM)
			}

			newInserted := false
			newAccepted := false
			tryInsertNew := func() bool {
				for _, selectedID := range selectedIDs {
					selectedVector, ok := index.nodeVectorForHeuristic(selectedID)
					if !ok {
						continue
					}
					if index.distance(newVector, selectedVector) < newDistance {
						newInserted = true
						return false
					}
				}
				selectedIDs = append(selectedIDs, newID)
				newInserted = true
				newAccepted = true
				return true
			}

			for _, linkID := range original {
				linkVector, ok := index.nodeVectorForHeuristic(linkID)
				if !ok {
					continue
				}
				linkDistance := index.distance(targetVector, linkVector)
				if !newInserted && (newDistance < linkDistance || (newDistance == linkDistance && newID < linkID)) {
					if !tryInsertNew() {
						return
					}
					if len(selectedIDs) >= maxM {
						break
					}
				}

				if newAccepted && index.distance(linkVector, newVector) < linkDistance {
					continue
				}
				selectedIDs = append(selectedIDs, linkID)
				if len(selectedIDs) >= maxM {
					break
				}
			}
			if !newInserted && len(selectedIDs) < maxM {
				_ = tryInsertNew()
			}
			if !newAccepted {
				return
			}

			for i, selectedID := range selectedIDs {
				atomic.StoreUint32(&slice[i], selectedID)
			}
			for i := len(selectedIDs); i < maxCapacity; i++ {
				if atomic.LoadUint32(&slice[i]) == SentinelNodeID {
					break
				}
				atomic.StoreUint32(&slice[i], SentinelNodeID)
			}
			atomic.StoreUint32(&targetNode.LinkCounts[level], uint32(len(selectedIDs)))
			atomic.StoreUint32(&targetNode.LinkHeuristic[level], uint32(len(selectedIDs)))
			accepted = true

			for _, linkID := range original {
				if !uint32SliceContains(selectedIDs, linkID) {
					dropped = append(dropped, linkID)
				}
			}
			return
		}

		for _, linkID := range original {
			linkVector, ok := index.nodeVectorForHeuristic(linkID)
			if !ok {
				continue
			}
			candidates = append(candidates, util.Candidate{
				ID:       linkID,
				Distance: index.distance(targetVector, linkVector),
			})
		}
		candidates = append(candidates, util.Candidate{
			ID:       newID,
			Distance: newDistance,
		})
		if len(candidates) == 0 {
			atomic.StoreUint32(&targetNode.LinkCounts[level], 0)
			return
		}

		slices.SortFunc(candidates, compareCandidateValues)
		selected := ns.selectWithSimpleHeuristicValues(targetVector, candidates, maxM, index)

		for i, selectedCandidate := range selected {
			atomic.StoreUint32(&slice[i], selectedCandidate.ID)
			if selectedCandidate.ID == newID {
				accepted = true
			}
		}
		for i := len(selected); i < maxCapacity; i++ {
			if atomic.LoadUint32(&slice[i]) == SentinelNodeID {
				break
			}
			atomic.StoreUint32(&slice[i], SentinelNodeID)
		}
		atomic.StoreUint32(&targetNode.LinkCounts[level], uint32(len(selected)))
		atomic.StoreUint32(&targetNode.LinkHeuristic[level], uint32(len(selected)))

		for _, linkID := range original {
			if !candidateValuesContainID(selected, linkID) {
				dropped = append(dropped, linkID)
			}
		}
	}()

	for _, linkID := range dropped {
		index.removeConnection(targetID, linkID, level)
	}
	return accepted
}

func (ns *NeighborSelector) removeDroppedBacklinks(
	nodeID uint32,
	level int,
	original []uint32,
	keepIDs []uint32,
	index *Index,
) {
	for _, linkID := range original {
		keep := false
		for _, k := range keepIDs {
			if k == linkID {
				keep = true
				break
			}
		}
		if keep {
			continue
		}
		index.removeConnection(nodeID, linkID, level)
	}
}

func candidateValuesContainID(candidates []util.Candidate, id uint32) bool {
	for _, candidate := range candidates {
		if candidate.ID == id {
			return true
		}
	}
	return false
}

func uint32SliceContains(values []uint32, id uint32) bool {
	for _, value := range values {
		if value == id {
			return true
		}
	}
	return false
}

func (h *Index) nodeVectorForHeuristic(nodeID uint32) ([]float32, bool) {
	if int(nodeID) >= h.nodes.Len() {
		return nil, false
	}
	node := h.nodes.Get(nodeID)
	if node == nil {
		return nil, false
	}
	if node.Vector != nil {
		return node.Vector, true
	}
	vector, err := h.getNodeVector(node)
	return vector, err == nil && vector != nil
}
