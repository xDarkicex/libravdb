package hnsw

import (
	"fmt"
	"runtime"
	"slices"
	"sync/atomic"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/libravdb/internal/util/simd"
	"github.com/xDarkicex/memory"
)

// NeighborSelector implements optimized neighbor selection algorithms
type NeighborSelector struct {
	maxConnections  int
	levelMultiplier float64
}

// level0LinkMultiplier lets level 0 use a small portion of the preallocated
// link slack. With M=16 this keeps 36 links: enough for stable ef=200 recall
// on the current benchmark, without paying for the full 40-link capacity.
const level0LinkMultiplier = 2.25

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

	usePtrSIMD := index.useHeuristicPtrSIMD()
	var selectedPtrBuf [128]unsafe.Pointer
	selectedPtrs := selectedPtrBuf[:0]
	if maxM > len(selectedPtrBuf) {
		selectedPtrs = make([]unsafe.Pointer, 0, maxM)
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

	if vector, ptr, ok := index.nodeVectorAndPtrForHeuristic(candidates[0].ID); ok {
		selectedVectors = append(selectedVectors, vector)
		selectedPtrs = append(selectedPtrs, ptr)
	} else {
		selectedVectors = append(selectedVectors, nil)
		selectedPtrs = append(selectedPtrs, nil)
	}

	for i := 1; i < len(candidates) && len(selected) < maxM; i++ {
		candidate := candidates[i]
		candidateVector, candidatePtr, ok := index.nodeVectorAndPtrForHeuristic(candidate.ID)
		if !ok {
			continue
		}

		shouldSelect := !index.rejectBySelectedHeuristic(
			candidateVector,
			selectedVectors,
			selectedPtrs,
			candidate.Distance,
			usePtrSIMD,
		)

		if shouldSelect {
			selected = append(selected, candidate)
			setPicked(i)
			selectedVectors = append(selectedVectors, candidateVector)
			selectedPtrs = append(selectedPtrs, candidatePtr)
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

func (h *Index) useHeuristicPtrSIMD() bool {
	return h.config != nil &&
		h.config.Metric == util.L2Distance &&
		h.quantizer == nil &&
		h.provider == nil &&
		simd.HasL2Batch8Ptr()
}

func (h *Index) rejectBySelectedHeuristic(
	candidateVector []float32,
	selectedVectors [][]float32,
	selectedPtrs []unsafe.Pointer,
	cutoff float32,
	usePtrSIMD bool,
) bool {
	relaxedCutoff := h.relaxedHeuristicCutoff(cutoff)
	if usePtrSIMD && len(selectedPtrs) == len(selectedVectors) {
		j := 0
		for j+7 < len(selectedPtrs) {
			p0 := selectedPtrs[j]
			p1 := selectedPtrs[j+1]
			p2 := selectedPtrs[j+2]
			p3 := selectedPtrs[j+3]
			p4 := selectedPtrs[j+4]
			p5 := selectedPtrs[j+5]
			p6 := selectedPtrs[j+6]
			p7 := selectedPtrs[j+7]
			if p0 != nil && p1 != nil && p2 != nil && p3 != nil && p4 != nil && p5 != nil && p6 != nil && p7 != nil {
				if h.useHeuristicPredicate {
					if simd.L2AnyLessThan8Ptr(candidateVector, p0, p1, p2, p3, p4, p5, p6, p7, relaxedCutoff) != 0 {
						return true
					}
					j += 8
					continue
				}
				d0, d1, d2, d3, d4, d5, d6, d7 := simd.L2Distance8Ptr(candidateVector, p0, p1, p2, p3, p4, p5, p6, p7)
				if d0 < relaxedCutoff || d1 < relaxedCutoff || d2 < relaxedCutoff || d3 < relaxedCutoff ||
					d4 < relaxedCutoff || d5 < relaxedCutoff || d6 < relaxedCutoff || d7 < relaxedCutoff {
					return true
				}
				j += 8
				continue
			}
			selectedVector := selectedVectors[j]
			if selectedVector != nil && h.distance(candidateVector, selectedVector) < relaxedCutoff {
				return true
			}
			j++
		}
		for j+3 < len(selectedPtrs) {
			p0 := selectedPtrs[j]
			p1 := selectedPtrs[j+1]
			p2 := selectedPtrs[j+2]
			p3 := selectedPtrs[j+3]
			if p0 != nil && p1 != nil && p2 != nil && p3 != nil {
				d0, d1, d2, d3 := simd.L2Distance4Ptr(candidateVector, p0, p1, p2, p3)
				if d0 < relaxedCutoff || d1 < relaxedCutoff || d2 < relaxedCutoff || d3 < relaxedCutoff {
					return true
				}
				j += 4
				continue
			}
			selectedVector := selectedVectors[j]
			if selectedVector != nil && h.distance(candidateVector, selectedVector) < relaxedCutoff {
				return true
			}
			j++
		}
		for ; j < len(selectedVectors); j++ {
			selectedVector := selectedVectors[j]
			if selectedVector != nil && h.distance(candidateVector, selectedVector) < relaxedCutoff {
				return true
			}
		}
		return false
	}

	for _, selectedVector := range selectedVectors {
		if selectedVector == nil {
			continue
		}
		if h.distance(candidateVector, selectedVector) < relaxedCutoff {
			return true
		}
	}
	return false
}

func (h *Index) pruneAlphaSquared() float32 {
	if h == nil || h.config == nil || h.config.PruneAlpha <= 1 {
		return 1
	}
	return h.config.PruneAlpha * h.config.PruneAlpha
}

func (h *Index) relaxedHeuristicCutoff(cutoff float32) float32 {
	alphaSquared := h.pruneAlphaSquared()
	if alphaSquared <= 1 {
		return cutoff
	}
	return cutoff / alphaSquared
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
	if capacity := linkArrayCapacity(index.config.M, level); maxM > capacity {
		maxM = capacity
	}

	nodeVector, err := index.getNodeVector(node)
	if err != nil {
		return err
	}

	for !index.acquirePruneLock(node) {
		runtime.Gosched()
	}
	if index.nodes.Get(nodeID) != node || level > node.Level || node.Links[level] == nil {
		index.releasePruneLock(node)
		return nil
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
	liveLinks, candidates = index.appendHeuristicCandidatesFromIDs(nodeVector, originalLinks, liveLinks, candidates)

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
	newRelaxedDistance := index.relaxedHeuristicCutoff(newDistance)

	maxCapacity := linkArrayCapacity(index.config.M, level)
	maxM := ns.maxConnections
	if level == 0 {
		maxM = int(float64(maxM) * ns.levelMultiplier)
	}
	if maxM > maxCapacity {
		maxM = maxCapacity
	}

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

		// Deletion unpublishes a node before taking PruneLock and reclaiming
		// its link arrays. A writer may have captured targetNode before that
		// unpublish, so revalidate both identity and storage under the lock.
		if index.nodes.Get(targetID) != targetNode || targetNode.Links[level] == nil {
			return
		}
		if index.nodes.Get(newID) == nil {
			return
		}

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

		// Defer expensive diversity pruning into the fixed overflow slack.
		// Immediate backlink pruning makes high-M construction pay O(M^3)
		// work on nearly every accepted edge. The slack is preallocated for
		// this exact purpose: extra edges are visible to search, improve
		// routing during construction, and get collapsed by the next full
		// prune once the array reaches physical capacity.
		if len(original) < maxCapacity {
			atomic.StoreUint32(&slice[len(original)], newID)
			atomic.StoreUint32(&targetNode.LinkCounts[level], uint32(len(original)+1))
			atomic.StoreUint32(&targetNode.LinkHeuristic[level], 0)
			index.markRepairDirty(targetNode, level)
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
					if index.distance(newVector, selectedVector) < newRelaxedDistance {
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

				if newAccepted {
					if index.distance(linkVector, newVector) < index.relaxedHeuristicCutoff(linkDistance) {
						continue
					}
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

		_, candidates = index.appendHeuristicCandidatesFromIDs(targetVector, original, nil, candidates)
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

func (h *Index) appendHeuristicCandidatesFromIDs(
	queryVector []float32,
	ids []uint32,
	liveLinks []uint32,
	candidates []util.Candidate,
) ([]uint32, []util.Candidate) {
	trackLiveLinks := liveLinks != nil || cap(liveLinks) > 0
	if h.useHeuristicPtrSIMD() {
		var idBuf [8]uint32
		var ptrBuf [8]unsafe.Pointer
		for i := 0; i < len(ids); {
			n := 0
			for i < len(ids) && n < len(idBuf) {
				id := ids[i]
				i++
				_, ptr, ok := h.nodeVectorAndPtrForHeuristic(id)
				if !ok || ptr == nil {
					continue
				}
				idBuf[n] = id
				ptrBuf[n] = ptr
				n++
			}
			if n == 0 {
				continue
			}
			if n == 8 {
				d0, d1, d2, d3, d4, d5, d6, d7 := simd.L2Distance8Ptr(
					queryVector,
					ptrBuf[0], ptrBuf[1], ptrBuf[2], ptrBuf[3],
					ptrBuf[4], ptrBuf[5], ptrBuf[6], ptrBuf[7],
				)
				if trackLiveLinks {
					liveLinks = append(liveLinks, idBuf[0], idBuf[1], idBuf[2], idBuf[3], idBuf[4], idBuf[5], idBuf[6], idBuf[7])
				}
				candidates = append(candidates,
					util.Candidate{ID: idBuf[0], Distance: d0},
					util.Candidate{ID: idBuf[1], Distance: d1},
					util.Candidate{ID: idBuf[2], Distance: d2},
					util.Candidate{ID: idBuf[3], Distance: d3},
					util.Candidate{ID: idBuf[4], Distance: d4},
					util.Candidate{ID: idBuf[5], Distance: d5},
					util.Candidate{ID: idBuf[6], Distance: d6},
					util.Candidate{ID: idBuf[7], Distance: d7},
				)
				continue
			}
			j := 0
			if n >= 4 {
				d0, d1, d2, d3 := simd.L2Distance4Ptr(queryVector, ptrBuf[0], ptrBuf[1], ptrBuf[2], ptrBuf[3])
				if trackLiveLinks {
					liveLinks = append(liveLinks, idBuf[0], idBuf[1], idBuf[2], idBuf[3])
				}
				candidates = append(candidates,
					util.Candidate{ID: idBuf[0], Distance: d0},
					util.Candidate{ID: idBuf[1], Distance: d1},
					util.Candidate{ID: idBuf[2], Distance: d2},
					util.Candidate{ID: idBuf[3], Distance: d3},
				)
				j = 4
			}
			for ; j < n; j++ {
				vector := unsafe.Slice((*float32)(ptrBuf[j]), len(queryVector))
				if trackLiveLinks {
					liveLinks = append(liveLinks, idBuf[j])
				}
				candidates = append(candidates, util.Candidate{
					ID:       idBuf[j],
					Distance: h.distance(queryVector, vector),
				})
			}
		}
		return liveLinks, candidates
	}

	for _, id := range ids {
		vector, ok := h.nodeVectorForHeuristic(id)
		if !ok {
			continue
		}
		if trackLiveLinks {
			liveLinks = append(liveLinks, id)
		}
		candidates = append(candidates, util.Candidate{
			ID:       id,
			Distance: h.distance(queryVector, vector),
		})
	}
	return liveLinks, candidates
}

func (h *Index) nodeVectorForHeuristic(nodeID uint32) ([]float32, bool) {
	vector, _, ok := h.nodeVectorAndPtrForHeuristic(nodeID)
	return vector, ok
}

func (h *Index) nodeVectorAndPtrForHeuristic(nodeID uint32) ([]float32, unsafe.Pointer, bool) {
	if int(nodeID) >= h.nodes.Len() {
		return nil, nil, false
	}
	node := h.nodes.Get(nodeID)
	if node == nil {
		return nil, nil, false
	}
	if node.Vector != nil {
		return node.Vector, node.VectorPtr, true
	}
	vector, err := h.getNodeVector(node)
	if err != nil || vector == nil {
		return nil, nil, false
	}
	var ptr unsafe.Pointer
	if len(vector) > 0 {
		ptr = unsafe.Pointer(&vector[0])
	}
	return vector, ptr, true
}
