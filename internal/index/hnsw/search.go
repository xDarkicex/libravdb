package hnsw

import (
	"github.com/xDarkicex/libravdb/internal/util"
)

type searchScratch struct {
	visitedMarks []uint32
	visitMark    uint32
	maxHeapBuf   []util.Candidate
	minHeapBuf   []util.Candidate
	pruneBuf     []util.Candidate
}

type candidateMinHeap struct {
	items []util.Candidate
}

func (h candidateMinHeap) Len() int { return len(h.items) }
func (h candidateMinHeap) Less(i, j int) bool {
	if h.items[i].Distance == h.items[j].Distance {
		return h.items[i].ID < h.items[j].ID
	}
	return h.items[i].Distance < h.items[j].Distance
}
func (h candidateMinHeap) Swap(i, j int) {
	h.items[i], h.items[j] = h.items[j], h.items[i]
}
func (h *candidateMinHeap) PushCandidate(c util.Candidate) {
	h.items = append(h.items, c)
	h.siftUp(len(h.items) - 1)
}
func (h *candidateMinHeap) PopCandidate() util.Candidate {
	n := len(h.items) - 1
	item := h.items[0]
	h.items[0] = h.items[n]
	h.items = h.items[:n]
	if len(h.items) > 0 {
		h.siftDown(0)
	}
	return item
}
func (h *candidateMinHeap) siftUp(idx int) {
	for idx > 0 {
		parent := (idx - 1) / 2
		if h.items[parent].Distance < h.items[idx].Distance ||
			(h.items[parent].Distance == h.items[idx].Distance && h.items[parent].ID <= h.items[idx].ID) {
			break
		}
		h.items[parent], h.items[idx] = h.items[idx], h.items[parent]
		idx = parent
	}
}
func (h *candidateMinHeap) siftDown(idx int) {
	for {
		left := idx*2 + 1
		right := left + 1
		smallest := idx

		if left < len(h.items) && (h.items[left].Distance < h.items[smallest].Distance ||
			(h.items[left].Distance == h.items[smallest].Distance && h.items[left].ID < h.items[smallest].ID)) {
			smallest = left
		}
		if right < len(h.items) && (h.items[right].Distance < h.items[smallest].Distance ||
			(h.items[right].Distance == h.items[smallest].Distance && h.items[right].ID < h.items[smallest].ID)) {
			smallest = right
		}
		if smallest == idx {
			return
		}
		h.items[idx], h.items[smallest] = h.items[smallest], h.items[idx]
		idx = smallest
	}
}

type candidateMaxHeap struct {
	items []util.Candidate
}

func (h candidateMaxHeap) Len() int { return len(h.items) }
func (h candidateMaxHeap) Less(i, j int) bool {
	if h.items[i].Distance == h.items[j].Distance {
		return h.items[i].ID > h.items[j].ID
	}
	return h.items[i].Distance > h.items[j].Distance
}
func (h candidateMaxHeap) Swap(i, j int) {
	h.items[i], h.items[j] = h.items[j], h.items[i]
}
func (h *candidateMaxHeap) PushCandidate(c util.Candidate) {
	h.items = append(h.items, c)
	h.siftUp(len(h.items) - 1)
}
func (h *candidateMaxHeap) PopCandidate() util.Candidate {
	n := len(h.items) - 1
	item := h.items[0]
	h.items[0] = h.items[n]
	h.items = h.items[:n]
	if len(h.items) > 0 {
		h.siftDown(0)
	}
	return item
}
func (h *candidateMaxHeap) Top() *util.Candidate {
	if len(h.items) == 0 {
		return nil
	}
	return &h.items[0]
}
func (h *candidateMaxHeap) siftUp(idx int) {
	for idx > 0 {
		parent := (idx - 1) / 2
		if h.items[parent].Distance > h.items[idx].Distance ||
			(h.items[parent].Distance == h.items[idx].Distance && h.items[parent].ID >= h.items[idx].ID) {
			break
		}
		h.items[parent], h.items[idx] = h.items[idx], h.items[parent]
		idx = parent
	}
}
func (h *candidateMaxHeap) siftDown(idx int) {
	for {
		left := idx*2 + 1
		right := left + 1
		largest := idx

		if left < len(h.items) && (h.items[left].Distance > h.items[largest].Distance ||
			(h.items[left].Distance == h.items[largest].Distance && h.items[left].ID > h.items[largest].ID)) {
			largest = left
		}
		if right < len(h.items) && (h.items[right].Distance > h.items[largest].Distance ||
			(h.items[right].Distance == h.items[largest].Distance && h.items[right].ID > h.items[largest].ID)) {
			largest = right
		}
		if largest == idx {
			return
		}
		h.items[idx], h.items[largest] = h.items[largest], h.items[idx]
		idx = largest
	}
}

func (h *Index) acquireSearchScratch() *searchScratch {
	scratch := h.searchScratchPool.Get().(*searchScratch)
	nodeCount := len(h.nodes)
	if cap(scratch.visitedMarks) < nodeCount {
		newCap := cap(scratch.visitedMarks)
		if newCap == 0 {
			newCap = 1024
		}
		for newCap < nodeCount {
			newCap *= 2
		}

		grown := make([]uint32, nodeCount, newCap)
		copy(grown, scratch.visitedMarks)
		scratch.visitedMarks = grown
	} else {
		scratch.visitedMarks = scratch.visitedMarks[:nodeCount]
	}
	scratch.visitMark++
	if scratch.visitMark == 0 {
		for i := range scratch.visitedMarks {
			scratch.visitedMarks[i] = 0
		}
		scratch.visitMark = 1
	}
	return scratch
}

func (h *Index) releaseSearchScratch(scratch *searchScratch) {
	h.searchScratchPool.Put(scratch)
}

// greedySearchLevel performs the ef=1 greedy descent used by HNSW on upper layers.
func (h *Index) greedySearchLevel(query []float32, entryPoint *Node, level int) *util.Candidate {
	candidate, ok := h.greedySearchLevelValue(query, entryPoint, level)
	if !ok {
		return nil
	}
	return &util.Candidate{
		ID:       candidate.ID,
		Distance: candidate.Distance,
	}
}

func (h *Index) greedySearchLevelValue(query []float32, entryPoint *Node, level int) (util.Candidate, bool) {
	if entryPoint == nil {
		return util.Candidate{}, false
	}

	scratch := h.acquireSearchScratch()
	defer h.releaseSearchScratch(scratch)

	visited := scratch.visitedMarks[:len(h.nodes)]
	visitMark := scratch.visitMark

	current := entryPoint
	currentID := h.findNodeID(current)
	if currentID == ^uint32(0) || int(currentID) >= len(h.nodes) {
		return util.Candidate{}, false
	}

	currentDistance := h.computeDistanceOptimized(query, current)
	if currentDistance < 0 {
		return util.Candidate{}, false
	}
	visited[currentID] = visitMark

	for {
		improved := false
		if level >= len(current.Links) {
			break
		}

		for _, neighborID := range current.Links[level] {
			if int(neighborID) >= len(h.nodes) || visited[neighborID] == visitMark {
				continue
			}
			visited[neighborID] = visitMark

			neighborNode := h.nodes[neighborID]
			if neighborNode == nil {
				continue
			}

			neighborDistance := h.computeDistanceOptimized(query, neighborNode)
			if neighborDistance < 0 {
				continue
			}

			if neighborDistance < currentDistance {
				current = neighborNode
				currentID = neighborID
				currentDistance = neighborDistance
				improved = true
			}
		}

		if !improved {
			break
		}
	}

	return util.Candidate{
		ID:       currentID,
		Distance: currentDistance,
	}, true
}

// searchLevel performs optimized search at a specific level
func (h *Index) searchLevel(query []float32, entryPoint *Node, ef int, level int) []*util.Candidate {
	return h.searchLevelWithOptions(query, entryPoint, ef, level, true)
}

func (h *Index) searchLevelForConstruction(query []float32, entryPoint *Node, ef int, level int) []util.Candidate {
	scratch := h.acquireSearchScratch()
	defer h.releaseSearchScratch(scratch)

	return h.searchLevelValuesWithScratch(query, entryPoint, ef, level, false, scratch)
}

func (h *Index) searchAndSelectForConstruction(
	query []float32,
	entryPoint *Node,
	ef int,
	level int,
	maxM int,
) []util.Candidate {
	scratch := h.acquireSearchScratch()
	defer h.releaseSearchScratch(scratch)

	selected := h.searchAndSelectForConstructionWithScratch(query, entryPoint, ef, level, maxM, scratch)
	if len(selected) == 0 {
		return nil
	}
	result := make([]util.Candidate, len(selected))
	copy(result, selected)
	return result
}

func (h *Index) searchAndSelectForConstructionWithScratch(
	query []float32,
	entryPoint *Node,
	ef int,
	level int,
	maxM int,
	scratch *searchScratch,
) []util.Candidate {
	workingSet := h.searchLevelScratchValues(query, entryPoint, ef, level, scratch)
	if len(workingSet) == 0 {
		return nil
	}

	selected := h.neighborSelector.SelectNeighborsOptimizedValues(query, workingSet, level, h)
	if len(selected) == 0 {
		return nil
	}
	if len(selected) > maxM {
		selected = selected[:maxM]
	}
	return selected
}

func (h *Index) searchLevelWithOptions(query []float32, entryPoint *Node, ef int, level int, sortResults bool) []*util.Candidate {
	scratch := h.acquireSearchScratch()
	defer h.releaseSearchScratch(scratch)

	values := h.searchLevelValuesWithScratch(query, entryPoint, ef, level, sortResults, scratch)
	if len(values) == 0 {
		return []*util.Candidate{}
	}

	result := make([]*util.Candidate, len(values))
	for i := range values {
		candidateCopy := values[i]
		result[i] = &util.Candidate{
			ID:       candidateCopy.ID,
			Distance: candidateCopy.Distance,
		}
	}
	return result
}

func (h *Index) searchLevelValuesWithScratch(query []float32, entryPoint *Node, ef int, level int, sortResults bool, scratch *searchScratch) []util.Candidate {
	if !sortResults {
		values := h.searchLevelScratchValues(query, entryPoint, ef, level, scratch)
		if len(values) == 0 {
			return nil
		}
		result := make([]util.Candidate, len(values))
		copy(result, values)
		return result
	}

	values := h.searchLevelScratchValues(query, entryPoint, ef, level, scratch)
	if len(values) == 0 {
		return nil
	}

	resultLen := len(values)
	result := make([]util.Candidate, resultLen)
	heap := candidateMaxHeap{items: values}
	for i := resultLen - 1; i >= 0; i-- {
		result[i] = heap.PopCandidate()
	}

	return result
}

func (h *Index) searchLevelScratchValues(query []float32, entryPoint *Node, ef int, level int, scratch *searchScratch) []util.Candidate {
	if ef <= 0 {
		return nil
	}

	visited := scratch.visitedMarks[:len(h.nodes)]
	visitMark := scratch.visitMark
	maxCap := ef * 2
	if cap(scratch.maxHeapBuf) < maxCap {
		scratch.maxHeapBuf = make([]util.Candidate, 0, maxCap)
	} else {
		scratch.maxHeapBuf = scratch.maxHeapBuf[:0]
	}
	minCap := max(maxCap, ef*8)
	if cap(scratch.minHeapBuf) < minCap {
		scratch.minHeapBuf = make([]util.Candidate, 0, minCap)
	} else {
		scratch.minHeapBuf = scratch.minHeapBuf[:0]
	}
	candidates := &candidateMaxHeap{items: scratch.maxHeapBuf}
	w := &candidateMinHeap{items: scratch.minHeapBuf}

	// Initialize with entry point
	entryID := h.findNodeID(entryPoint)
	if entryID == ^uint32(0) || entryID >= uint32(len(visited)) {
		return nil
	}

	// Compute distance handling quantization
	distance := h.computeDistanceOptimized(query, entryPoint)
	if distance < 0 {
		return nil // Error in distance computation
	}

	candidate := util.Candidate{ID: entryID, Distance: distance}

	candidates.PushCandidate(candidate)
	w.PushCandidate(candidate)
	visited[entryID] = visitMark

	for w.Len() > 0 {
		current := w.PopCandidate()

		// Early termination condition - optimized for large datasets
		if candidates.Len() >= ef && current.Distance > candidates.Top().Distance {
			break
		}

		// Explore neighbors
		currentNode := h.nodes[current.ID]
		if currentNode == nil {
			continue
		}
		if level < len(currentNode.Links) {
			// Process neighbors in batches for better cache locality
			neighbors := currentNode.Links[level]
			for _, neighborID := range neighbors {
				if neighborID < uint32(len(visited)) && visited[neighborID] != visitMark {
					visited[neighborID] = visitMark

					// Compute distance with optimized method
					neighborNode := h.nodes[neighborID]
					if neighborNode == nil {
						continue
					}
					neighborDistance := h.computeDistanceOptimized(query, neighborNode)
					if neighborDistance < 0 {
						continue // Skip if distance computation failed
					}

					neighborCandidate := util.Candidate{
						ID:       neighborID,
						Distance: neighborDistance,
					}

					// Add to candidates if it's one of the ef closest
					if candidates.Len() < ef || neighborDistance < candidates.Top().Distance {
						candidates.PushCandidate(neighborCandidate)
						w.PushCandidate(neighborCandidate)

						// Remove furthest if we exceed ef
						if candidates.Len() > ef {
							candidates.PopCandidate()
						}
					}
				}
			}
		}
	}
	return candidates.items
}

// computeDistanceOptimized provides optimized distance computation with error handling
func (h *Index) computeDistanceOptimized(query []float32, node *Node) float32 {
	if node == nil {
		return -1
	}
	if node.CompressedVector != nil && h.quantizer != nil {
		distance, err := h.quantizer.DistanceToQuery(node.CompressedVector, query)
		if err != nil {
			// Fall back to decompressed vector
			vec, decompErr := h.quantizer.Decompress(node.CompressedVector)
			if decompErr != nil {
				return -1 // Signal error
			}
			return h.distance(query, vec)
		}
		return distance
	}
	if h.provider != nil {
		distance, err := h.provider.Distance(query, node.Ordinal)
		if err == nil {
			return distance
		}
	}
	vec, err := h.getNodeVector(node)
	if err == nil && vec != nil {
		return h.distance(query, vec)
	}
	return -1 // Signal error - no vector available
}
