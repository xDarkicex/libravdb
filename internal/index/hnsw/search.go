package hnsw

import (
	"context"
	"fmt"

	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/memory"
)

type searchScratch struct {
	arena        *memory.Arena
	visitedMarks []uint32
	maxHeapBuf   []util.Candidate
	minHeapBuf   []util.Candidate
	pruneBuf     []util.Candidate
	visitMark    uint32
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
func (h *candidateMaxHeap) Top() util.Candidate {
	if len(h.items) == 0 {
		return util.Candidate{}
	}
	return h.items[0]
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
	return h.acquireSearchScratchWithNodeCount(len(h.nodes))
}

func (h *Index) acquireSearchScratchWithNodeCount(nodeCount int) *searchScratch {
	scratch := h.searchScratchPool.Get().(*searchScratch)

	// Ensure the Arena is sized for visitedMarks + heap bufs (~320 KB headroom).
	needed := uint64(nodeCount*4 + 320*1024)
	if scratch.arena == nil || scratch.arena.Remaining() < needed {
		if scratch.arena != nil {
			scratch.arena.Free()
		}
		a, err := memory.NewArena(needed)
		if err != nil {
			// mmap failure is fatal — the scratch is unusable.
			panic("hnsw: failed to allocate search scratch arena: " + err.Error())
		}
		scratch.arena = a
	} else {
		scratch.arena.Reset()
	}

	marks, err := memory.ArenaSlice[uint32](scratch.arena, nodeCount)
	if err != nil {
		panic("hnsw: failed to allocate visited marks from arena: " + err.Error())
	}
	scratch.visitedMarks = marks

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
	// Arena.Reset() rewinds the bump pointer, keeping the mmap'd region
	// so the next acquireSearchScratch can reuse it without a new mmap.
	scratch.arena.Reset()
	h.searchScratchPool.Put(scratch)
}

// greedySearchLevel performs the ef=1 greedy descent used by HNSW on upper layers.
func (h *Index) greedySearchLevel(ctx context.Context, query []float32, entryPoint *Node, level int, queryState any) (*util.Candidate, error) {
	candidate, ok, err := h.greedySearchLevelValue(ctx, query, entryPoint, level, queryState)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, nil
	}
	return &util.Candidate{
		ID:       candidate.ID,
		Distance: candidate.Distance,
	}, nil
}

func (h *Index) greedySearchLevelValue(ctx context.Context, query []float32, entryPoint *Node, level int, queryState any) (util.Candidate, bool, error) {
	if entryPoint == nil {
		return util.Candidate{}, false, nil
	}

	scratch := h.acquireSearchScratch()
	defer h.releaseSearchScratch(scratch)

	visited := scratch.visitedMarks[:len(h.nodes)]
	visitMark := scratch.visitMark

	current := entryPoint
	currentID := h.findNodeID(current)
	if currentID == ^uint32(0) || int(currentID) >= len(h.nodes) {
		return util.Candidate{}, false, nil
	}

	currentDistance, err := h.computeDistanceOptimized(query, current, queryState)
	if err != nil {
		return util.Candidate{}, false, err
	}
	visited[currentID] = visitMark

	for {
		select {
		case <-ctx.Done():
			return util.Candidate{}, false, ctx.Err()
		default:
		}

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

			neighborDistance, err := h.computeDistanceOptimized(query, neighborNode, queryState)
			if err != nil {
				return util.Candidate{}, false, err
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
	}, true, nil
}

func (h *Index) searchLevel(ctx context.Context, query []float32, entryPoint *Node, ef int, level int, queryState any) ([]*util.Candidate, error) {
	return h.searchLevelWithOptions(ctx, query, entryPoint, ef, level, true, queryState)
}

func (h *Index) searchLevelForConstruction(query []float32, entryPoint *Node, ef int, level int, queryState any) ([]util.Candidate, error) {
	scratch := h.acquireSearchScratch()
	defer h.releaseSearchScratch(scratch)

	return h.searchLevelValuesWithScratch(context.Background(), query, entryPoint, ef, level, false, scratch, queryState)
}

// searchAndSelectForConstruction finds neighbors at a specific level and selects the best ones for construction
func (h *Index) searchAndSelectForConstruction(query []float32, entryPoint *Node, ef int, level int, maxM int, queryState any) ([]util.Candidate, error) {
	scratch := h.acquireSearchScratch()
	defer h.releaseSearchScratch(scratch)

	return h.searchAndSelectForConstructionWithScratch(query, entryPoint, ef, level, maxM, scratch, queryState)
}

func (h *Index) searchAndSelectForConstructionWithScratch(
	query []float32,
	entryPoint *Node,
	ef int,
	level int,
	maxM int,
	scratch *searchScratch,
	queryState any,
) ([]util.Candidate, error) {
	workingSet, err := h.searchLevelScratchValues(context.Background(), query, entryPoint, ef, level, scratch, queryState)
	if err != nil {
		return nil, err
	}
	if len(workingSet) == 0 {
		return nil, nil
	}

	selected := h.neighborSelector.SelectNeighborsOptimizedValues(query, workingSet, level, h)
	if len(selected) == 0 {
		return nil, nil
	}
	if len(selected) > maxM {
		selected = selected[:maxM]
	}
	return selected, nil
}

func (h *Index) searchLevelWithOptions(ctx context.Context, query []float32, entryPoint *Node, ef int, level int, sortResults bool, queryState any) ([]*util.Candidate, error) {
	scratch := h.acquireSearchScratch()
	defer h.releaseSearchScratch(scratch)

	values, err := h.searchLevelValuesWithScratch(ctx, query, entryPoint, ef, level, sortResults, scratch, queryState)
	if err != nil {
		return nil, err
	}
	if len(values) == 0 {
		return []*util.Candidate{}, nil
	}

	result := make([]*util.Candidate, len(values))
	for i := range values {
		candidateCopy := values[i]
		result[i] = &util.Candidate{
			ID:       candidateCopy.ID,
			Distance: candidateCopy.Distance,
		}
	}
	return result, nil
}

func (h *Index) searchLevelValuesWithScratch(ctx context.Context, query []float32, entryPoint *Node, ef int, level int, sortResults bool, scratch *searchScratch, queryState any) ([]util.Candidate, error) {
	if !sortResults {
		values, err := h.searchLevelScratchValues(ctx, query, entryPoint, ef, level, scratch, queryState)
		if err != nil {
			return nil, err
		}
		if len(values) == 0 {
			return nil, nil
		}
		result := make([]util.Candidate, len(values))
		copy(result, values)
		return result, nil
	}

	values, err := h.searchLevelScratchValues(ctx, query, entryPoint, ef, level, scratch, queryState)
	if err != nil {
		return nil, err
	}
	if len(values) == 0 {
		return nil, nil
	}

	resultLen := len(values)
	result := make([]util.Candidate, resultLen)
	heap := candidateMaxHeap{items: values}
	for i := resultLen - 1; i >= 0; i-- {
		result[i] = heap.PopCandidate()
	}

	return result, nil
}

func (h *Index) searchLevelScratchValues(ctx context.Context, query []float32, entryPoint *Node, ef int, level int, scratch *searchScratch, queryState any) ([]util.Candidate, error) {
	if ef <= 0 {
		return nil, nil
	}

	visited := scratch.visitedMarks[:len(h.nodes)]
	visitMark := scratch.visitMark
	// ArenaSlice always re-allocates from the arena because the previous
	// search's arena was Reset() in releaseSearchScratch — old slice
	// pointers are stale after the arena is rewound.
	maxCap := ef * 2
	maxHeap, err := memory.ArenaSlice[util.Candidate](scratch.arena, maxCap)
	if err != nil {
		panic("hnsw: arena maxHeapBuf: " + err.Error())
	}
	scratch.maxHeapBuf = maxHeap
	minCap := max(maxCap, ef*8)
	minHeap, err := memory.ArenaSlice[util.Candidate](scratch.arena, minCap)
	if err != nil {
		panic("hnsw: arena minHeapBuf: " + err.Error())
	}
	scratch.minHeapBuf = minHeap
	candidates := &candidateMaxHeap{items: scratch.maxHeapBuf}
	w := &candidateMinHeap{items: scratch.minHeapBuf}

	// Initialize with entry point
	entryID := h.findNodeID(entryPoint)
	if entryID == ^uint32(0) || entryID >= uint32(len(visited)) {
		return nil, nil
	}

	// Compute distance handling quantization
	distance, err := h.computeDistanceOptimized(query, entryPoint, queryState)
	if err != nil {
		return nil, err // Error in distance computation
	}

	candidate := util.Candidate{ID: entryID, Distance: distance}

	candidates.PushCandidate(candidate)
	w.PushCandidate(candidate)
	visited[entryID] = visitMark

	for w.Len() > 0 {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

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
					neighborDistance, err := h.computeDistanceOptimized(query, neighborNode, queryState)
					if err != nil {
						return nil, err // Signal error
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
	return candidates.items, nil
}

// computeDistanceOptimized provides optimized distance computation with error handling
func (h *Index) computeDistanceOptimized(query []float32, node *Node, queryState any) (float32, error) {
	if node == nil {
		return -1, fmt.Errorf("node is nil")
	}
	if node.CompressedVector != nil && h.quantizer != nil {
		distance, err := h.quantizer.DistanceToQuery(node.CompressedVector, query, queryState)
		if err != nil {
			// Fall back to decompressed vector
			vec, decompErr := h.quantizer.Decompress(node.CompressedVector)
			if decompErr != nil {
				return -1, fmt.Errorf("decompression failed: %w", decompErr)
			}
			return h.distance(query, vec), nil
		}
		return distance, nil
	}
	if h.provider != nil {
		distance, err := h.provider.Distance(query, node.Ordinal)
		if err == nil {
			return distance, nil
		}
	}
	vec, err := h.getNodeVector(node)
	if err == nil && vec != nil {
		return h.distance(query, vec), nil
	}
	if err != nil {
		return -1, err
	}
	return -1, fmt.Errorf("no vector available")
}
