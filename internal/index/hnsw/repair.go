package hnsw

import (
	"runtime"
	"slices"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/util"
)

const repairLevelMask uint32 = (1 << MaxLevel) - 1

func (h *Index) startRepairWorker() {
	if h == nil || h.repairCh == nil || h.repairStop != nil {
		return
	}
	h.repairStop = make(chan struct{})
	h.repairDone = make(chan struct{})
	go h.repairWorker()
}

func (h *Index) stopRepairWorker() {
	if h == nil || h.repairStop == nil {
		return
	}
	close(h.repairStop)
	<-h.repairDone
	h.repairStop = nil
	h.repairDone = nil
}

func (h *Index) markRepairDirty(node *Node, level int) {
	if h == nil || h.repairCh == nil || node == nil || level < 0 || level >= MaxLevel {
		return
	}
	bit := uint32(1) << uint(level)
	for {
		old := atomic.LoadUint32(&node.RepairMask)
		if old&bit != 0 {
			break
		}
		if atomic.CompareAndSwapUint32(&node.RepairMask, old, old|bit) {
			break
		}
	}
	h.enqueueRepair(node)
}

func (h *Index) enqueueRepair(node *Node) {
	if h == nil || h.repairCh == nil || node == nil {
		return
	}
	if !atomic.CompareAndSwapUint32(&node.RepairQueued, 0, 1) {
		return
	}
	select {
	case h.repairCh <- node.Ordinal:
	default:
		h.repairOverflow.Store(true)
		atomic.StoreUint32(&node.RepairQueued, 0)
	}
}

func (h *Index) repairWorker() {
	defer close(h.repairDone)

	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()

	batchSize := h.config.repairBatchSize()
	for {
		select {
		case nodeID := <-h.repairCh:
			h.repairNode(nodeID)
			for i := 1; i < batchSize; i++ {
				select {
				case nodeID = <-h.repairCh:
					h.repairNode(nodeID)
				default:
					i = batchSize
				}
			}
		case <-ticker.C:
			if h.repairOverflow.CompareAndSwap(true, false) {
				h.scanDirtyRepairs(batchSize)
			}
		case <-h.repairStop:
			return
		}
	}
}

// FlushRepairs runs queued and overflow-tracked repair work synchronously.
// A limit <= 0 drains all currently visible repair work.
func (h *Index) FlushRepairs(limit int) int {
	if h == nil || h.repairCh == nil {
		return 0
	}
	processed := 0
	for limit <= 0 || processed < limit {
		select {
		case nodeID := <-h.repairCh:
			h.repairNode(nodeID)
			processed++
		default:
			remaining := 0
			if limit > 0 {
				remaining = limit - processed
			}
			scanned := h.scanDirtyRepairs(remaining)
			processed += scanned
			if scanned == 0 {
				h.repairOverflow.Store(false)
				return processed
			}
		}
	}
	return processed
}

func (h *Index) scanDirtyRepairs(limit int) int {
	if h == nil || h.nodes == nil {
		return 0
	}
	processed := 0
	nodeCount := h.nodes.Len()
	for i := 0; i < nodeCount; i++ {
		if limit > 0 && processed >= limit {
			return processed
		}
		node := h.nodes.Get(uint32(i))
		if node == nil || atomic.LoadUint32(&node.RepairMask)&repairLevelMask == 0 {
			continue
		}
		if !atomic.CompareAndSwapUint32(&node.RepairQueued, 0, 1) {
			continue
		}
		h.repairNode(node.Ordinal)
		processed++
	}
	return processed
}

func (h *Index) repairNode(nodeID uint32) {
	if h == nil || h.nodes == nil || h.neighborSelector == nil || int(nodeID) >= h.nodes.Len() {
		return
	}
	node := h.nodes.Get(nodeID)
	if node == nil {
		return
	}

	mask := atomic.SwapUint32(&node.RepairMask, 0) & repairLevelMask
	if mask == 0 {
		atomic.StoreUint32(&node.RepairQueued, 0)
		return
	}

	maxLevel := node.Level
	if maxLevel >= MaxLevel {
		maxLevel = MaxLevel - 1
	}
	for level := 0; level <= maxLevel; level++ {
		if mask&(uint32(1)<<uint(level)) == 0 {
			continue
		}
		if !h.levelNeedsRepair(node, level) {
			continue
		}
		h.repairConnections(nodeID, level)
	}

	atomic.StoreUint32(&node.RepairQueued, 0)
	if atomic.LoadUint32(&node.RepairMask)&repairLevelMask != 0 {
		h.enqueueRepair(node)
	}
}

func (h *Index) levelNeedsRepair(node *Node, level int) bool {
	if h == nil || node == nil || level < 0 || level >= MaxLevel {
		return false
	}
	count := int(atomic.LoadUint32(&node.LinkCounts[level]))
	if count == 0 {
		return false
	}
	maxLinks := h.repairMaxLinks(level)
	return count > maxLinks
}

func (h *Index) repairMaxLinks(level int) int {
	maxLinks := h.config.M
	if level == 0 {
		maxLinks = int(float64(maxLinks) * h.config.level0LinkMultiplier())
	}
	if capacity := linkArrayCapacity(h.config.M, level); maxLinks > capacity {
		maxLinks = capacity
	}
	return maxLinks
}

func (h *Index) repairConnections(nodeID uint32, level int) bool {
	if h == nil || h.nodes == nil || h.neighborSelector == nil || int(nodeID) >= h.nodes.Len() {
		return false
	}
	node := h.nodes.Get(nodeID)
	if node == nil || level < 0 || level >= MaxLevel || level > node.Level {
		return false
	}
	nodeVector, err := h.getNodeVector(node)
	if err != nil {
		return false
	}

	maxM := h.repairMaxLinks(level)
	if maxM <= 0 {
		return false
	}
	candidateLimit := maxM * 4
	if candidateLimit < maxM {
		candidateLimit = maxM
	}
	if candidateLimit > 512 {
		candidateLimit = 512
	}

	var idBuf [512]uint32
	ids := idBuf[:0]
	addID := func(id uint32) bool {
		if id == SentinelNodeID || id == nodeID || len(ids) >= candidateLimit || int(id) >= h.nodes.Len() {
			return false
		}
		if uint32SliceContains(ids, id) {
			return false
		}
		if h.nodes.Get(id) == nil {
			return false
		}
		ids = append(ids, id)
		return true
	}

	links := h.getNodeLinks(node, level)
	backlinks := h.getNodeBacklinks(node, level)
	directEnd := 0
	for _, id := range links {
		addID(id)
	}
	for _, id := range backlinks {
		addID(id)
	}
	directEnd = len(ids)

	for i := 0; i < directEnd && len(ids) < candidateLimit; i++ {
		neighbor := h.nodes.Get(ids[i])
		if neighbor == nil || level > neighbor.Level {
			continue
		}
		for _, id := range h.getNodeLinks(neighbor, level) {
			if len(ids) >= candidateLimit {
				break
			}
			addID(id)
		}
		for _, id := range h.getNodeBacklinks(neighbor, level) {
			if len(ids) >= candidateLimit {
				break
			}
			addID(id)
		}
	}

	var candidateBuf [512]util.Candidate
	candidates := candidateBuf[:0]
	addCandidate := func(id uint32) {
		if len(candidates) >= candidateLimit || candidateValuesContainID(candidates, id) {
			return
		}
		vector, ok := h.nodeVectorForHeuristic(id)
		if !ok {
			return
		}
		candidates = append(candidates, util.Candidate{
			ID:       id,
			Distance: h.distance(nodeVector, vector),
		})
	}
	for _, id := range ids {
		addCandidate(id)
	}

	for !h.acquirePruneLock(node) {
		runtime.Gosched()
	}

	maxCapacity := linkArrayCapacity(h.config.M, level)
	slice := unsafe.Slice(node.Links[level], maxCapacity)
	count := int(atomic.LoadUint32(&node.LinkCounts[level]))
	if count > maxCapacity {
		count = maxCapacity
	}
	var originalBuf [512]uint32
	original := originalBuf[:0]
	for i := 0; i < count; i++ {
		id := atomic.LoadUint32(&slice[i])
		if id == SentinelNodeID || int(id) >= h.nodes.Len() || h.nodes.Get(id) == nil {
			continue
		}
		original = append(original, id)
		addCandidate(id)
	}

	if len(candidates) == 0 {
		h.releasePruneLock(node)
		return false
	}
	slices.SortFunc(candidates, compareCandidateValues)
	selected := h.neighborSelector.selectWithSimpleHeuristicValues(nodeVector, candidates, maxM, h)

	var keepBuf [512]uint32
	keepIDs := keepBuf[:0]
	for i, candidate := range selected {
		atomic.StoreUint32(&slice[i], candidate.ID)
		keepIDs = append(keepIDs, candidate.ID)
	}
	for i := len(selected); i < maxCapacity; i++ {
		if atomic.LoadUint32(&slice[i]) == SentinelNodeID {
			break
		}
		atomic.StoreUint32(&slice[i], SentinelNodeID)
	}
	atomic.StoreUint32(&node.LinkCounts[level], uint32(len(selected)))
	atomic.StoreUint32(&node.LinkHeuristic[level], uint32(len(selected)))
	h.releasePruneLock(node)

	for _, keepID := range keepIDs {
		neighbor := h.nodes.Get(keepID)
		if neighbor != nil && level <= neighbor.Level {
			h.appendWithSpinlock(neighbor, neighbor.Backlinks[level], nodeID, h.config.M, level)
		}
	}
	h.neighborSelector.removeDroppedBacklinks(nodeID, level, original, keepIDs, h)
	return true
}
