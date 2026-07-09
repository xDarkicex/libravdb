package hnsw

import (
	"context"
	"fmt"
	"runtime"
	"slices"
	"sync/atomic"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/libravdb/internal/util/simd"
	"github.com/xDarkicex/memory"
	"golang.org/x/sys/cpu"
)

type searchScratch struct {
	arena         *memory.Arena
	arenaBytes    uint64
	visitedMarks  []uint32
	maxHeapBuf    []util.Candidate
	minHeapBuf    []util.Candidate
	pruneBuf      []util.Candidate
	inFlightBuf   []uint32
	prefetchedIDs []uint32
	prefetchPtrs  []unsafe.Pointer
	prefetchVecs  [][]float32
	visitMark     uint32
}

func (s *searchScratch) nextVisitMark() uint32 {
	s.visitMark++
	if s.visitMark == 0 {
		for i := range s.visitedMarks {
			s.visitedMarks[i] = 0
		}
		s.visitMark = 1
	}
	return s.visitMark
}

type candidateMinHeap struct {
	items []util.Candidate
}

func (h candidateMinHeap) Len() int { return len(h.items) }

func (h *candidateMinHeap) PushCandidate(c util.Candidate) {
	items := h.items
	idx := len(items)
	h.items = append(items, c)
	items = h.items // update items

	for idx > 0 {
		parent := (idx - 1) / 2
		p := items[parent]
		if p.Distance < c.Distance || (p.Distance == c.Distance && p.ID <= c.ID) {
			break
		}
		items[idx] = p
		idx = parent
	}
	items[idx] = c
}

func (h *candidateMinHeap) PopCandidate() util.Candidate {
	items := h.items
	n := len(items) - 1
	result := items[0]
	c := items[n]
	h.items = items[:n]
	if n == 0 {
		return result
	}

	idx := 0
	for {
		left := idx*2 + 1
		if left >= n {
			break
		}
		right := left + 1
		smallest := left
		lItem := items[left]

		if right < n {
			rItem := items[right]
			if rItem.Distance < lItem.Distance || (rItem.Distance == lItem.Distance && rItem.ID < lItem.ID) {
				smallest = right
				lItem = rItem
			}
		}

		if lItem.Distance > c.Distance || (lItem.Distance == c.Distance && lItem.ID >= c.ID) {
			break
		}

		items[idx] = lItem
		idx = smallest
	}
	items[idx] = c
	return result
}

// CandidateMode selects the candidate tracking data structure used during
// search. "heap" remains the production default until the admission shootout is
// stable across alternating multi-count runs. "unsorted" keeps the cached-worst
// array path available for targeted throughput/recall testing.
var CandidateMode = "heap"

// unsortedTopK tracks the K closest candidates found so far using an
// unsorted array with a cached worst-element index. For small K (ef ≤ 200),
// this dominates sorted slices and binary heaps because:
//
//   - ~85-90% of pushes are rejects: a single float compare against cached worst
//   - ~10-15% are accepts: replace worst in-place + one full rescan (K compares)
//
// The full rescan over a contiguous 800-byte block touching 2 cache lines is
// faster than a binary heap's log₂K ≈ 7 levels of branch-mispredicting
// bubble-up/down, especially on wide OoO cores (Apple Firestorm, Intel/
// AMD). USearch's sorted_buffer_gt uses the same strategy for K < 128.
type unsortedTopK struct {
	items    []util.Candidate // backing array: len == logical size, cap >= maxSize
	maxSize  int              // target K — stop accepting new entries at this size
	worstIdx int              // index of the element with the largest distance
}

func (u unsortedTopK) Len() int { return len(u.items) }

func (u *unsortedTopK) PushCandidate(c util.Candidate) {
	if u.Len() < u.maxSize {
		// Not yet full — append and track worst.
		u.items = append(u.items, c)
		if c.Distance > u.items[u.worstIdx].Distance {
			u.worstIdx = u.Len() - 1
		}
		return
	}

	// Full: only accept if better than the current worst.
	if c.Distance >= u.items[u.worstIdx].Distance {
		return // reject — this is the ~85-90% fast path
	}

	// Replace worst and rescan for the new worst.
	u.items[u.worstIdx] = c
	u.worstIdx = 0
	worstDist := u.items[0].Distance
	for i := 1; i < len(u.items); i++ {
		if d := u.items[i].Distance; d > worstDist {
			worstDist = d
			u.worstIdx = i
		}
	}
}

// Top returns the worst (furthest) candidate.
func (u *unsortedTopK) Top() util.Candidate {
	if len(u.items) == 0 {
		return util.Candidate{}
	}
	return u.items[u.worstIdx]
}

func (u *unsortedTopK) Items() []util.Candidate { return u.items }

// PopCandidate removes and returns the worst candidate.
func (u *unsortedTopK) PopCandidate() util.Candidate {
	n := len(u.items) - 1
	removed := u.items[u.worstIdx]

	// Swap last element into the removed slot, then shrink.
	if u.worstIdx != n {
		u.items[u.worstIdx] = u.items[n]
	}
	u.items = u.items[:n]

	// Rescan for the new worst (only needed when something remains).
	if n > 0 {
		u.worstIdx = 0
		worstDist := u.items[0].Distance
		for i := 1; i < n; i++ {
			if d := u.items[i].Distance; d > worstDist {
				worstDist = d
				u.worstIdx = i
			}
		}
	}
	return removed
}

// candidateMaxHeap is a standard binary max-heap over a slice. It is retained
// for the cold sort-results path (searchLevelValuesWithScratch, sortResults=true).
// The hot search loop uses unsortedTopK instead.
type candidateMaxHeap struct {
	items []util.Candidate
}

func (h candidateMaxHeap) Len() int { return len(h.items) }

func (h *candidateMaxHeap) PushCandidate(c util.Candidate) {
	items := h.items
	idx := len(items)
	h.items = append(items, c)
	items = h.items

	for idx > 0 {
		parent := (idx - 1) / 2
		p := items[parent]
		if p.Distance > c.Distance || (p.Distance == c.Distance && p.ID >= c.ID) {
			break
		}
		items[idx] = p
		idx = parent
	}
	items[idx] = c
}

func (h *candidateMaxHeap) PopCandidate() util.Candidate {
	items := h.items
	n := len(items) - 1
	result := items[0]
	c := items[n]
	h.items = items[:n]
	if n == 0 {
		return result
	}

	idx := 0
	for {
		left := idx*2 + 1
		if left >= n {
			break
		}
		right := left + 1
		largest := left
		lItem := items[left]

		if right < n {
			rItem := items[right]
			if rItem.Distance > lItem.Distance || (rItem.Distance == lItem.Distance && rItem.ID > lItem.ID) {
				largest = right
				lItem = rItem
			}
		}

		if lItem.Distance < c.Distance || (lItem.Distance == c.Distance && lItem.ID <= c.ID) {
			break
		}

		items[idx] = lItem
		idx = largest
	}
	items[idx] = c
	return result
}

func (h *candidateMaxHeap) ReplaceTop(c util.Candidate) {
	items := h.items
	n := len(items)
	if n == 0 {
		return
	}

	idx := 0
	for {
		left := idx*2 + 1
		if left >= n {
			break
		}
		right := left + 1
		largest := left
		lItem := items[left]

		if right < n {
			rItem := items[right]
			if rItem.Distance > lItem.Distance || (rItem.Distance == lItem.Distance && rItem.ID > lItem.ID) {
				largest = right
				lItem = rItem
			}
		}

		if lItem.Distance < c.Distance || (lItem.Distance == c.Distance && lItem.ID <= c.ID) {
			break
		}

		items[idx] = lItem
		idx = largest
	}
	items[idx] = c
}

func (h *candidateMaxHeap) Items() []util.Candidate { return h.items }

func (h *candidateMaxHeap) Top() util.Candidate {
	if len(h.items) == 0 {
		return util.Candidate{}
	}
	return h.items[0]
}

func admitCandidateMaxHeap(candidates *candidateMaxHeap, working *candidateMinHeap, ef int, id uint32, distance float32) {
	candidate := util.Candidate{ID: id, Distance: distance}
	if len(candidates.items) >= ef {
		worst := candidates.items[0]
		if distance >= worst.Distance {
			return
		}
		candidates.ReplaceTop(candidate)
		working.PushCandidate(candidate)
		return
	}

	candidates.PushCandidate(candidate)
	working.PushCandidate(candidate)
}

func admitBatch4MaxHeap(candidates *candidateMaxHeap, working *candidateMinHeap, ef int, ids []uint32, d0, d1, d2, d3 float32) {
	if len(candidates.items) >= ef {
		worst := candidates.items[0].Distance
		if d0 >= worst && d1 >= worst && d2 >= worst && d3 >= worst {
			return
		}
	}
	admitCandidateMaxHeap(candidates, working, ef, ids[0], d0)
	admitCandidateMaxHeap(candidates, working, ef, ids[1], d1)
	admitCandidateMaxHeap(candidates, working, ef, ids[2], d2)
	admitCandidateMaxHeap(candidates, working, ef, ids[3], d3)
}

func admitCandidateUnsorted(candidates *unsortedTopK, working *candidateMinHeap, ef int, id uint32, distance float32) {
	if len(candidates.items) >= ef {
		worst := candidates.items[candidates.worstIdx]
		if distance >= worst.Distance {
			return
		}
	}

	candidate := util.Candidate{ID: id, Distance: distance}
	candidates.PushCandidate(candidate)
	working.PushCandidate(candidate)
	if len(candidates.items) > ef {
		candidates.PopCandidate()
	}
}

func admitBatch4Unsorted(candidates *unsortedTopK, working *candidateMinHeap, ef int, ids []uint32, d0, d1, d2, d3 float32) {
	if len(candidates.items) >= ef {
		worst := candidates.items[candidates.worstIdx].Distance
		if d0 >= worst && d1 >= worst && d2 >= worst && d3 >= worst {
			return
		}
	}
	admitCandidateUnsorted(candidates, working, ef, ids[0], d0)
	admitCandidateUnsorted(candidates, working, ef, ids[1], d1)
	admitCandidateUnsorted(candidates, working, ef, ids[2], d2)
	admitCandidateUnsorted(candidates, working, ef, ids[3], d3)
}

func (h *Index) acquireSearchScratch() *searchScratch {
	return h.acquireSearchScratchWithEF(0)
}

func (h *Index) acquireSearchScratchWithEF(ef int) *searchScratch {
	return h.acquireSearchScratchWithNodeCountAndEF(h.nodes.Len(), ef)
}

func (h *Index) acquireSearchScratchWithNodeCount(nodeCount int) *searchScratch {
	return h.acquireSearchScratchWithNodeCountAndEF(nodeCount, 0)
}

func (h *Index) acquireSearchScratchWithNodeCountAndEF(nodeCount int, ef int) *searchScratch {
	scratch := h.searchScratchPool.Get().(*searchScratch)

	maxCap, minCap := searchHeapCaps(nodeCount, ef)
	prefetchCap := max(128, linkArrayCapacity(h.config.M, 0)*2)
	candidateBytes := uint64(maxCap+minCap) * uint64(unsafe.Sizeof(util.Candidate{}))
	// Ensure the Arena is sized for visitedMarks, prefetch ID buffer, and the
	// two candidate frontiers used by this search. ef may be much larger than
	// the default headroom during quality sweeps and user-tuned high-recall
	// searches, so fixed scratch sizing can panic under valid configurations.
	needed := uint64(nodeCount*4+prefetchCap*4) + candidateBytes + 64*1024 + 8*64
	if needed < 320*1024 {
		needed = 320 * 1024
	}
	if scratch.arena == nil || scratch.arenaBytes < needed {
		if scratch.arena != nil {
			scratch.arena.Free()
			scratch.arena = nil
			scratch.arenaBytes = 0
		}
		a, err := memory.NewArena(needed, 64)
		if err != nil {
			// mmap failure is fatal — the scratch is unusable.
			panic("hnsw: failed to allocate search scratch arena: " + err.Error())
		}
		scratch.arena = a
		scratch.arenaBytes = needed
	} else {
		scratch.arena.Reset()
	}

	marks, err := memory.ArenaSlice[uint32](scratch.arena, nodeCount)
	if err != nil {
		panic("hnsw: failed to allocate visited marks from arena: " + err.Error())
	}
	scratch.visitedMarks = marks[:nodeCount]

	prefetchBuf, err := memory.ArenaSlice[uint32](scratch.arena, prefetchCap)
	if err != nil {
		panic("hnsw: arena prefetchBuf: " + err.Error())
	}
	scratch.prefetchedIDs = prefetchBuf[:0]

	if cap(scratch.prefetchVecs) < prefetchCap {
		scratch.prefetchVecs = make([][]float32, 0, prefetchCap)
	} else {
		scratch.prefetchVecs = scratch.prefetchVecs[:0]
	}
	if cap(scratch.prefetchPtrs) < prefetchCap {
		scratch.prefetchPtrs = make([]unsafe.Pointer, 0, prefetchCap)
	} else {
		scratch.prefetchPtrs = scratch.prefetchPtrs[:0]
	}

	maxHeap, err := memory.ArenaSlice[util.Candidate](scratch.arena, maxCap)
	if err != nil {
		panic("hnsw: arena maxHeapBuf: " + err.Error())
	}
	scratch.maxHeapBuf = maxHeap[:0]
	minHeap, err := memory.ArenaSlice[util.Candidate](scratch.arena, minCap)
	if err != nil {
		panic("hnsw: arena minHeapBuf: " + err.Error())
	}
	scratch.minHeapBuf = minHeap[:0]

	return scratch
}

func searchHeapCaps(nodeCount int, ef int) (int, int) {
	if ef <= 0 {
		ef = 1
	}
	if nodeCount <= 0 {
		nodeCount = 1
	}
	maxCap := min(ef*2, nodeCount)
	if maxCap < ef && ef <= nodeCount {
		maxCap = ef
	}
	minCap := min(max(maxCap, ef*8), nodeCount)
	return maxCap, minCap
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

	scratch := h.acquireSearchScratchWithEF(1)
	defer h.releaseSearchScratch(scratch)

	nodeCount := len(scratch.visitedMarks)
	visited := scratch.visitedMarks[:nodeCount]
	visitMark := scratch.nextVisitMark()

	current := entryPoint
	currentID := h.findNodeID(current)
	if currentID == ^uint32(0) || int(currentID) >= nodeCount {
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
		if level >= (current.Level + 1) {
			break
		}

		connections := h.getNodeLinks(current, level)
		for _, neighborID := range connections {
			if int(neighborID) >= nodeCount || visited[neighborID] == visitMark {
				continue
			}
			visited[neighborID] = visitMark
			neighborNode := h.nodes.Get(neighborID)
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
		backlinks := h.getNodeBacklinks(current, level)
		for _, neighborID := range backlinks {
			if int(neighborID) >= nodeCount || visited[neighborID] == visitMark {
				continue
			}
			visited[neighborID] = visitMark
			neighborNode := h.nodes.Get(neighborID)
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

func (h *Index) searchLevel(ctx context.Context, query []float32, entryPoint *Node, ef int, level int, queryState any, filter interface{ Test(idx uint64) bool }) ([]*util.Candidate, error) {
	return h.searchLevelWithOptions(ctx, query, entryPoint, ef, level, true, queryState, filter)
}

func (h *Index) searchLevelForConstruction(query []float32, entryPoint *Node, ef int, level int, queryState any) ([]util.Candidate, error) {
	scratch := h.acquireSearchScratchWithEF(ef)
	defer h.releaseSearchScratch(scratch)

	return h.searchLevelValuesWithScratch(context.Background(), query, entryPoint, ef, level, false, scratch, queryState, nil)
}

// searchAndSelectForConstruction finds neighbors at a specific level and selects the best ones for construction
func (h *Index) searchAndSelectForConstruction(query []float32, entryPoint *Node, ef int, level int, maxM int, queryState any) ([]util.Candidate, error) {
	scratch := h.acquireSearchScratchWithEF(ef)
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
	workingSet, err := h.searchLevelScratchValues(context.Background(), query, entryPoint, ef, level, scratch, queryState, nil)
	if err != nil {
		return nil, err
	}

	// Evaluate in-flight nodes to resolve concurrency paradox perfectly.
	// This static snapshot of concurrent insertions allows mutually unaware nodes
	// to establish bidirectional edges without global locks.
	if h.inFlightNodes != nil {
		scratch.inFlightBuf = h.inFlightNodes.GetSnapshot(scratch.inFlightBuf)
		for _, inFlightID := range scratch.inFlightBuf {
			inFlightNode := h.nodes.Get(inFlightID)
			if inFlightNode == nil || inFlightNode.Level < level || atomic.LoadUint32(&inFlightNode.InFlight) == 0 {
				continue
			}
			distance, err := h.computeDistanceOptimized(query, inFlightNode, queryState)
			if err == nil && distance > 0 { // distance 0 is likely the querying node itself
				workingSet = append(workingSet, util.Candidate{ID: inFlightID, Distance: distance})
			}
		}
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

func (h *Index) searchLevelWithOptions(ctx context.Context, query []float32, entryPoint *Node, ef int, level int, sortResults bool, queryState any, filter interface{ Test(idx uint64) bool }) ([]*util.Candidate, error) {
	scratch := h.acquireSearchScratchWithEF(ef)
	defer h.releaseSearchScratch(scratch)

	values, err := h.searchLevelValuesWithScratch(ctx, query, entryPoint, ef, level, sortResults, scratch, queryState, filter)
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

func (h *Index) searchLevelValuesWithScratch(ctx context.Context, query []float32, entryPoint *Node, ef int, level int, sortResults bool, scratch *searchScratch, queryState any, filter interface{ Test(idx uint64) bool }) ([]util.Candidate, error) {
	if !sortResults {
		values, err := h.searchLevelScratchValues(ctx, query, entryPoint, ef, level, scratch, queryState, filter)
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

	values, err := h.searchLevelScratchValues(ctx, query, entryPoint, ef, level, scratch, queryState, filter)
	if err != nil {
		return nil, err
	}
	if len(values) == 0 {
		return nil, nil
	}

	slices.SortFunc(values, compareCandidateValues)
	return values, nil
}

func (h *Index) searchLevelScratchValues(ctx context.Context, query []float32, entryPoint *Node, ef int, level int, scratch *searchScratch, queryState any, filter interface{ Test(idx uint64) bool }) ([]util.Candidate, error) {
	if ef <= 0 {
		return nil, nil
	}

	nodeCount := len(scratch.visitedMarks)
	visited := scratch.visitedMarks[:nodeCount]
	visitMark := scratch.nextVisitMark()
	scratch.maxHeapBuf = scratch.maxHeapBuf[:0]
	scratch.minHeapBuf = scratch.minHeapBuf[:0]
	heapMode := CandidateMode == "heap"
	heapCandidates := candidateMaxHeap{items: scratch.maxHeapBuf}
	unsortedCandidates := unsortedTopK{items: scratch.maxHeapBuf, maxSize: ef}
	w := &candidateMinHeap{items: scratch.minHeapBuf}
	useRawNEONPtrL2 := h.config.Metric == util.L2Distance && runtime.GOARCH == "arm64" && h.quantizer == nil && h.provider == nil
	useNEONBatchL2 := h.config.Metric == util.L2Distance && runtime.GOARCH == "arm64" && !useRawNEONPtrL2
	useAVX2BatchL2 := h.config.Metric == util.L2Distance && runtime.GOARCH == "amd64" && cpu.X86.HasAVX2 && cpu.X86.HasFMA
	useSIMDBatchL2 := useNEONBatchL2 || useAVX2BatchL2

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

	if heapMode {
		heapCandidates.PushCandidate(candidate)
	} else {
		unsortedCandidates.PushCandidate(candidate)
	}
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
		if heapMode {
			if len(heapCandidates.items) >= ef && current.Distance > heapCandidates.items[0].Distance {
				break
			}
		} else {
			if len(unsortedCandidates.items) >= ef && current.Distance > unsortedCandidates.items[unsortedCandidates.worstIdx].Distance {
				break
			}
		}

		// Explore neighbors
		currentNode := h.nodes.Get(current.ID)
		if currentNode == nil {
			continue
		}
		if level < (currentNode.Level + 1) {
			// Process neighbors in batches for better cache locality
			neighbors := h.getNodeLinks(currentNode, level)
			// Pass 1: Gather and logically prefetch
			scratch.prefetchedIDs = scratch.prefetchedIDs[:0]
			scratch.prefetchPtrs = scratch.prefetchPtrs[:0]
			scratch.prefetchVecs = scratch.prefetchVecs[:0]

			for _, neighborID := range neighbors {
				if neighborID < uint32(len(visited)) && visited[neighborID] != visitMark && neighborID != SentinelNodeID {
					visited[neighborID] = visitMark

					node := h.nodes.Get(neighborID)
					if node != nil {
						if useRawNEONPtrL2 {
							ptr := node.VectorPtr
							if ptr == nil {
								continue
							}
							scratch.prefetchedIDs = append(scratch.prefetchedIDs, neighborID)
							scratch.prefetchPtrs = append(scratch.prefetchPtrs, ptr)
							simd.PrefetchL1(ptr)
							continue
						}
						scratch.prefetchedIDs = append(scratch.prefetchedIDs, neighborID)

						// Raw float traversal owns direct vector views on Node.
						// Quantized/provider nodes intentionally fall through to
						// computeDistanceOptimized in the scoring pass.
						vec := node.Vector
						scratch.prefetchVecs = append(scratch.prefetchVecs, vec)
						if vec != nil && len(vec) > 0 {
							if useNEONBatchL2 {
								simd.PrefetchL1(unsafe.Pointer(&vec[0]))
							} else {
								_ = vec[0]
							}
						}
					}
				}
			}
			backlinks := h.getNodeBacklinks(currentNode, level)
			for _, neighborID := range backlinks {
				if neighborID < uint32(len(visited)) && visited[neighborID] != visitMark && neighborID != SentinelNodeID {
					visited[neighborID] = visitMark

					node := h.nodes.Get(neighborID)
					if node != nil {
						if useRawNEONPtrL2 {
							ptr := node.VectorPtr
							if ptr == nil {
								continue
							}
							scratch.prefetchedIDs = append(scratch.prefetchedIDs, neighborID)
							scratch.prefetchPtrs = append(scratch.prefetchPtrs, ptr)
							simd.PrefetchL1(ptr)
							continue
						}
						scratch.prefetchedIDs = append(scratch.prefetchedIDs, neighborID)

						vec := node.Vector
						scratch.prefetchVecs = append(scratch.prefetchVecs, vec)
						if vec != nil && len(vec) > 0 {
							if useNEONBatchL2 {
								simd.PrefetchL1(unsafe.Pointer(&vec[0]))
							} else {
								_ = vec[0]
							}
						}
					}
				}
			}

			if useRawNEONPtrL2 {
				for i := 0; i < len(scratch.prefetchedIDs); {
					if i+3 < len(scratch.prefetchedIDs) {
						d0, d1, d2, d3 := simd.L2Distance4PtrNEON(
							query,
							scratch.prefetchPtrs[i],
							scratch.prefetchPtrs[i+1],
							scratch.prefetchPtrs[i+2],
							scratch.prefetchPtrs[i+3],
						)
						batchIDs := scratch.prefetchedIDs[i : i+4]
						if heapMode {
							admitBatch4MaxHeap(&heapCandidates, w, ef, batchIDs, d0, d1, d2, d3)
						} else {
							admitBatch4Unsorted(&unsortedCandidates, w, ef, batchIDs, d0, d1, d2, d3)
						}
						i += 4
						continue
					}

					neighborID := scratch.prefetchedIDs[i]
					vec := unsafe.Slice((*float32)(scratch.prefetchPtrs[i]), len(query))
					neighborDistance := h.distance(query, vec)
					if heapMode {
						admitCandidateMaxHeap(&heapCandidates, w, ef, neighborID, neighborDistance)
					} else {
						admitCandidateUnsorted(&unsortedCandidates, w, ef, neighborID, neighborDistance)
					}
					i++
				}
				continue
			}

			for i := 0; i < len(scratch.prefetchedIDs); {
				if useSIMDBatchL2 && i+3 < len(scratch.prefetchedIDs) {
					v0 := scratch.prefetchVecs[i]
					v1 := scratch.prefetchVecs[i+1]
					v2 := scratch.prefetchVecs[i+2]
					v3 := scratch.prefetchVecs[i+3]
					if v0 != nil && v1 != nil && v2 != nil && v3 != nil {
						var d0, d1, d2, d3 float32
						if useNEONBatchL2 {
							d0, d1, d2, d3 = simd.L2Distance4NEON(query, v0, v1, v2, v3)
						} else {
							d0, d1, d2, d3 = simd.L2Distance4AVX2(query, v0, v1, v2, v3)
						}
						batchIDs := scratch.prefetchedIDs[i : i+4]
						if heapMode {
							admitBatch4MaxHeap(&heapCandidates, w, ef, batchIDs, d0, d1, d2, d3)
						} else {
							admitBatch4Unsorted(&unsortedCandidates, w, ef, batchIDs, d0, d1, d2, d3)
						}
						i += 4
						continue
					}
				}

				neighborID := scratch.prefetchedIDs[i]
				vec := scratch.prefetchVecs[i]

				var neighborDistance float32
				var err error

				if vec != nil {
					neighborDistance = h.distance(query, vec)
				} else {
					neighborNode := h.nodes.Get(neighborID)
					if neighborNode == nil {
						i++
						continue
					}
					neighborDistance, err = h.computeDistanceOptimized(query, neighborNode, queryState)
					if err != nil {
						return nil, err
					}
				}

				if heapMode {
					admitCandidateMaxHeap(&heapCandidates, w, ef, neighborID, neighborDistance)
				} else {
					admitCandidateUnsorted(&unsortedCandidates, w, ef, neighborID, neighborDistance)
				}
				i++
			}
		}
	}
	if heapMode {
		return heapCandidates.items, nil
	}
	return unsortedCandidates.items, nil
}

func (h *Index) normalizeQuantizedDistance(distance float32) float32 {
	if h.config != nil && h.config.Metric == util.L2Distance {
		return distance * distance
	}
	return distance
}

// computeDistanceOptimized provides optimized distance computation with error handling
func (h *Index) computeDistanceOptimized(query []float32, node *Node, queryState any) (float32, error) {
	if node == nil {
		return -1, fmt.Errorf("node is nil")
	}
	if node.CompressedVector == nil && node.Vector != nil {
		return h.distance(query, node.Vector), nil
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
		return h.normalizeQuantizedDistance(distance), nil
	}
	if h.provider != nil {
		distance, err := h.provider.Distance(query, node.Ordinal)
		if err == nil {
			return distance, nil
		}
	}
	if node.Vector != nil {
		return h.distance(query, node.Vector), nil
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
