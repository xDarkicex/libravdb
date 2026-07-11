package hnsw

import (
	"context"
	"fmt"
	"math/bits"
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
	slot          uint8
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

const candidateHeapFanout = 4

func (h candidateMinHeap) Len() int { return len(h.items) }

func (h *candidateMinHeap) PushCandidate(c util.Candidate) {
	items := h.items
	idx := len(items)
	h.items = append(items, c)
	items = h.items // update items

	for idx > 0 {
		parent := (idx - 1) / candidateHeapFanout
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
		first := idx*candidateHeapFanout + 1
		if first >= n {
			break
		}
		smallest := first
		child := items[first]

		childIdx := first + 1
		if childIdx < n {
			item := items[childIdx]
			if item.Distance < child.Distance || (item.Distance == child.Distance && item.ID < child.ID) {
				smallest = childIdx
				child = item
			}
		}
		childIdx = first + 2
		if childIdx < n {
			item := items[childIdx]
			if item.Distance < child.Distance || (item.Distance == child.Distance && item.ID < child.ID) {
				smallest = childIdx
				child = item
			}
		}
		childIdx = first + 3
		if childIdx < n {
			item := items[childIdx]
			if item.Distance < child.Distance || (item.Distance == child.Distance && item.ID < child.ID) {
				smallest = childIdx
				child = item
			}
		}

		if child.Distance > c.Distance || (child.Distance == c.Distance && child.ID >= c.ID) {
			break
		}

		items[idx] = child
		idx = smallest
	}
	items[idx] = c
	return result
}

// CandidateMode selects the candidate tracking data structure used during
// search. "heap" is the current production default. "unsorted" and
// "reservoir" remain available for targeted throughput/recall testing.
var CandidateMode = "heap"

// unsortedTopK tracks the K closest candidates found so far using an
// unsorted array with a cached worst-element index. For small K (ef ≤ 200),
// this can beat sorted slices and binary heaps when rejects dominate because:
//
//   - ~85-90% of pushes are rejects: a single float compare against cached worst
//   - ~10-15% are accepts: replace worst in-place + one full rescan (K compares)
//
// The full rescan over a contiguous 800-byte block touching 2 cache lines is
// faster than a binary heap's log₂K ≈ 7 levels of branch-mispredicting
// bubble-up/down, especially on wide OoO cores (Apple Firestorm, Intel/
// AMD). On the current HNSW benchmarks, the binary heap remains faster because
// accepted replacements are still common enough for full rescans to cost more
// than heap repair.
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

// reservoirTopK follows Elasticsearch's bulk collector shape: append accepted
// candidates into a 2*K reservoir, then compact back to K with selection. The
// threshold can be stale between compactions, which may over-admit work into the
// frontier, but it never drops candidates that could belong to the exact top-K.
type reservoirTopK struct {
	items     []util.Candidate
	maxSize   int
	threshold util.Candidate
	full      bool
}

func (r reservoirTopK) Len() int {
	if len(r.items) < r.maxSize {
		return len(r.items)
	}
	return r.maxSize
}

func (r *reservoirTopK) Full() bool { return r.full }

func (r *reservoirTopK) Threshold() util.Candidate { return r.threshold }

func (r *reservoirTopK) PushCandidate(c util.Candidate) bool {
	if r.maxSize <= 0 {
		return false
	}
	if len(r.items) < r.maxSize {
		r.items = append(r.items, c)
		if len(r.items) == r.maxSize {
			r.refreshThreshold()
		}
		return true
	}
	if !candidateBetter(c, r.threshold) {
		return false
	}
	r.items = append(r.items, c)
	if len(r.items) == cap(r.items) {
		r.compact()
	}
	return true
}

func (r *reservoirTopK) Items() []util.Candidate {
	r.compact()
	return r.items
}

func (r *reservoirTopK) compact() {
	if len(r.items) <= r.maxSize {
		if len(r.items) == r.maxSize {
			r.refreshThreshold()
		}
		return
	}
	selectTopKCandidates(r.items, r.maxSize)
	r.items = r.items[:r.maxSize]
	r.refreshThreshold()
}

func (r *reservoirTopK) refreshThreshold() {
	if len(r.items) < r.maxSize {
		r.full = false
		return
	}
	worstIdx := 0
	worst := r.items[0]
	for i := 1; i < r.maxSize; i++ {
		item := r.items[i]
		if candidateWorse(item, worst) {
			worstIdx = i
			worst = item
		}
	}
	r.threshold = r.items[worstIdx]
	r.full = true
}

func candidateBetter(a, b util.Candidate) bool {
	return a.Distance < b.Distance || (a.Distance == b.Distance && a.ID < b.ID)
}

func candidateWorse(a, b util.Candidate) bool {
	return a.Distance > b.Distance || (a.Distance == b.Distance && a.ID > b.ID)
}

func candidateWorseOrEqual(a, b util.Candidate) bool {
	return a.Distance > b.Distance || (a.Distance == b.Distance && a.ID >= b.ID)
}

func selectTopKCandidates(items []util.Candidate, k int) {
	if k <= 0 || k >= len(items) {
		return
	}
	target := k - 1
	left, right := 0, len(items)-1
	for left < right {
		pivot := partitionCandidates(items, left, right, medianCandidateIndex(items, left, right))
		if pivot == target {
			return
		}
		if target < pivot {
			right = pivot - 1
		} else {
			left = pivot + 1
		}
	}
}

func medianCandidateIndex(items []util.Candidate, left, right int) int {
	mid := left + (right-left)/2
	a, b, c := items[left], items[mid], items[right]
	if candidateBetter(b, a) {
		if candidateBetter(c, b) {
			return mid
		}
		if candidateBetter(c, a) {
			return right
		}
		return left
	}
	if candidateBetter(c, a) {
		return left
	}
	if candidateBetter(c, b) {
		return right
	}
	return mid
}

func partitionCandidates(items []util.Candidate, left, right, pivotIdx int) int {
	pivot := items[pivotIdx]
	items[pivotIdx], items[right] = items[right], items[pivotIdx]
	store := left
	for i := left; i < right; i++ {
		if candidateBetter(items[i], pivot) {
			items[store], items[i] = items[i], items[store]
			store++
		}
	}
	items[right], items[store] = items[store], items[right]
	return store
}

// candidateMaxHeap is a standard max-heap over a slice. It is the current
// hot-path default for candidate tracking.
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
		parent := (idx - 1) / candidateHeapFanout
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
		first := idx*candidateHeapFanout + 1
		if first >= n {
			break
		}
		largest := first
		child := items[first]

		childIdx := first + 1
		if childIdx < n {
			item := items[childIdx]
			if item.Distance > child.Distance || (item.Distance == child.Distance && item.ID > child.ID) {
				largest = childIdx
				child = item
			}
		}
		childIdx = first + 2
		if childIdx < n {
			item := items[childIdx]
			if item.Distance > child.Distance || (item.Distance == child.Distance && item.ID > child.ID) {
				largest = childIdx
				child = item
			}
		}
		childIdx = first + 3
		if childIdx < n {
			item := items[childIdx]
			if item.Distance > child.Distance || (item.Distance == child.Distance && item.ID > child.ID) {
				largest = childIdx
				child = item
			}
		}

		if child.Distance < c.Distance || (child.Distance == c.Distance && child.ID <= c.ID) {
			break
		}

		items[idx] = child
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
		first := idx*candidateHeapFanout + 1
		if first >= n {
			break
		}
		largest := first
		child := items[first]

		childIdx := first + 1
		if childIdx < n {
			item := items[childIdx]
			if item.Distance > child.Distance || (item.Distance == child.Distance && item.ID > child.ID) {
				largest = childIdx
				child = item
			}
		}
		childIdx = first + 2
		if childIdx < n {
			item := items[childIdx]
			if item.Distance > child.Distance || (item.Distance == child.Distance && item.ID > child.ID) {
				largest = childIdx
				child = item
			}
		}
		childIdx = first + 3
		if childIdx < n {
			item := items[childIdx]
			if item.Distance > child.Distance || (item.Distance == child.Distance && item.ID > child.ID) {
				largest = childIdx
				child = item
			}
		}

		if child.Distance < c.Distance || (child.Distance == c.Distance && child.ID <= c.ID) {
			break
		}

		items[idx] = child
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

	if len(candidates.items) >= ef {
		if d0 < candidates.items[0].Distance {
			candidate := util.Candidate{ID: ids[0], Distance: d0}
			candidates.ReplaceTop(candidate)
			working.PushCandidate(candidate)
		}
	} else {
		candidate := util.Candidate{ID: ids[0], Distance: d0}
		candidates.PushCandidate(candidate)
		working.PushCandidate(candidate)
	}

	if len(candidates.items) >= ef {
		if d1 < candidates.items[0].Distance {
			candidate := util.Candidate{ID: ids[1], Distance: d1}
			candidates.ReplaceTop(candidate)
			working.PushCandidate(candidate)
		}
	} else {
		candidate := util.Candidate{ID: ids[1], Distance: d1}
		candidates.PushCandidate(candidate)
		working.PushCandidate(candidate)
	}

	if len(candidates.items) >= ef {
		if d2 < candidates.items[0].Distance {
			candidate := util.Candidate{ID: ids[2], Distance: d2}
			candidates.ReplaceTop(candidate)
			working.PushCandidate(candidate)
		}
	} else {
		candidate := util.Candidate{ID: ids[2], Distance: d2}
		candidates.PushCandidate(candidate)
		working.PushCandidate(candidate)
	}

	if len(candidates.items) >= ef {
		if d3 < candidates.items[0].Distance {
			candidate := util.Candidate{ID: ids[3], Distance: d3}
			candidates.ReplaceTop(candidate)
			working.PushCandidate(candidate)
		}
		return
	}
	candidate := util.Candidate{ID: ids[3], Distance: d3}
	candidates.PushCandidate(candidate)
	working.PushCandidate(candidate)
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

func admitCandidateReservoir(candidates *reservoirTopK, working *candidateMinHeap, id uint32, distance float32) {
	candidate := util.Candidate{ID: id, Distance: distance}
	if candidates.PushCandidate(candidate) {
		working.PushCandidate(candidate)
	}
}

func admitBatch4Reservoir(candidates *reservoirTopK, working *candidateMinHeap, ids []uint32, d0, d1, d2, d3 float32) {
	if candidates.Full() {
		threshold := candidates.Threshold()
		if candidateWorseOrEqual(util.Candidate{ID: ids[0], Distance: d0}, threshold) &&
			candidateWorseOrEqual(util.Candidate{ID: ids[1], Distance: d1}, threshold) &&
			candidateWorseOrEqual(util.Candidate{ID: ids[2], Distance: d2}, threshold) &&
			candidateWorseOrEqual(util.Candidate{ID: ids[3], Distance: d3}, threshold) {
			return
		}
	}
	admitCandidateReservoir(candidates, working, ids[0], d0)
	admitCandidateReservoir(candidates, working, ids[1], d1)
	admitCandidateReservoir(candidates, working, ids[2], d2)
	admitCandidateReservoir(candidates, working, ids[3], d3)
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
	var scratch *searchScratch
	for scratch == nil {
		free := h.searchScratchFree.Load()
		if free == 0 {
			runtime.Gosched()
			continue
		}
		slot := bits.TrailingZeros64(free)
		bit := uint64(1) << slot
		if h.searchScratchFree.CompareAndSwap(free, free&^bit) {
			scratch = &h.searchScratches[slot]
		}
	}
	h.prepareSearchScratch(scratch, nodeCount, ef)
	return scratch
}

func (h *Index) prepareSearchScratch(scratch *searchScratch, nodeCount int, ef int) {
	maxCap, minCap := searchHeapCaps(nodeCount, ef)
	prefetchCap := max(128, linkArrayCapacity(h.config.M, 0)*2)
	candidateBytes := uint64(maxCap+minCap) * uint64(unsafe.Sizeof(util.Candidate{}))
	prefetchBytes := uint64(prefetchCap) * uint64(
		unsafe.Sizeof(uint32(0))+unsafe.Sizeof(unsafe.Pointer(nil))+unsafe.Sizeof([]float32(nil)),
	)
	// Ensure the Arena is sized for visitedMarks, prefetch ID buffer, and the
	// two candidate frontiers used by this search. ef may be much larger than
	// the default headroom during quality sweeps and user-tuned high-recall
	// searches, so fixed scratch sizing can panic under valid configurations.
	needed := uint64(nodeCount*4) + prefetchBytes + candidateBytes + 64*1024 + 8*64
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

	prefetchVecs, err := memory.ArenaSlice[[]float32](scratch.arena, prefetchCap)
	if err != nil {
		panic("hnsw: arena prefetchVecs: " + err.Error())
	}
	scratch.prefetchVecs = prefetchVecs[:0]
	prefetchPtrs, err := memory.ArenaSlice[unsafe.Pointer](scratch.arena, prefetchCap)
	if err != nil {
		panic("hnsw: arena prefetchPtrs: " + err.Error())
	}
	scratch.prefetchPtrs = prefetchPtrs[:0]

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
	bit := uint64(1) << scratch.slot
	if previous := h.searchScratchFree.Or(bit); previous&bit != 0 {
		panic("hnsw: search scratch released twice")
	}
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
	if h.inFlightNodes != nil && h.inFlightNodes.Active() > 1 {
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
	candidateMode := CandidateMode
	heapMode := candidateMode != "unsorted" && candidateMode != "reservoir"
	reservoirMode := candidateMode == "reservoir"
	heapCandidates := candidateMaxHeap{items: scratch.maxHeapBuf}
	unsortedCandidates := unsortedTopK{items: scratch.maxHeapBuf, maxSize: ef}
	reservoirCandidates := reservoirTopK{items: scratch.maxHeapBuf, maxSize: ef}
	w := &candidateMinHeap{items: scratch.minHeapBuf}
	useRawNEONPtrL2 := h.config.Metric == util.L2Distance && runtime.GOARCH == "arm64" && h.quantizer == nil && h.provider == nil
	useNEONBatchL2 := h.config.Metric == util.L2Distance && runtime.GOARCH == "arm64" && !useRawNEONPtrL2
	useAVX2BatchL2 := h.config.Metric == util.L2Distance && runtime.GOARCH == "amd64" && cpu.X86.HasAVX2 && cpu.X86.HasFMA
	useSIMDBatchL2 := useNEONBatchL2 || useAVX2BatchL2
	var done <-chan struct{}
	if ctx != nil {
		done = ctx.Done()
	}

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

	switch {
	case heapMode:
		heapCandidates.PushCandidate(candidate)
	case reservoirMode:
		reservoirCandidates.PushCandidate(candidate)
	default:
		unsortedCandidates.PushCandidate(candidate)
	}
	w.PushCandidate(candidate)
	visited[entryID] = visitMark

	for w.Len() > 0 {
		if done != nil {
			select {
			case <-done:
				return nil, ctx.Err()
			default:
			}
		}

		current := w.PopCandidate()

		// Early termination condition - optimized for large datasets
		if heapMode {
			if len(heapCandidates.items) >= ef && current.Distance > heapCandidates.items[0].Distance {
				break
			}
		} else if reservoirMode {
			if reservoirCandidates.Full() && candidateWorse(current, reservoirCandidates.Threshold()) {
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
					if i+7 < len(scratch.prefetchedIDs) {
						d0, d1, d2, d3, d4, d5, d6, d7 := simd.L2Distance8PtrNEON(
							query,
							scratch.prefetchPtrs[i],
							scratch.prefetchPtrs[i+1],
							scratch.prefetchPtrs[i+2],
							scratch.prefetchPtrs[i+3],
							scratch.prefetchPtrs[i+4],
							scratch.prefetchPtrs[i+5],
							scratch.prefetchPtrs[i+6],
							scratch.prefetchPtrs[i+7],
						)
						batchIDs := scratch.prefetchedIDs[i : i+8]
						if heapMode {
							admitBatch4MaxHeap(&heapCandidates, w, ef, batchIDs[:4], d0, d1, d2, d3)
							admitBatch4MaxHeap(&heapCandidates, w, ef, batchIDs[4:], d4, d5, d6, d7)
						} else if reservoirMode {
							admitBatch4Reservoir(&reservoirCandidates, w, batchIDs[:4], d0, d1, d2, d3)
							admitBatch4Reservoir(&reservoirCandidates, w, batchIDs[4:], d4, d5, d6, d7)
						} else {
							admitBatch4Unsorted(&unsortedCandidates, w, ef, batchIDs[:4], d0, d1, d2, d3)
							admitBatch4Unsorted(&unsortedCandidates, w, ef, batchIDs[4:], d4, d5, d6, d7)
						}
						i += 8
						continue
					}
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
						} else if reservoirMode {
							admitBatch4Reservoir(&reservoirCandidates, w, batchIDs, d0, d1, d2, d3)
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
					} else if reservoirMode {
						admitCandidateReservoir(&reservoirCandidates, w, neighborID, neighborDistance)
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
						} else if reservoirMode {
							admitBatch4Reservoir(&reservoirCandidates, w, batchIDs, d0, d1, d2, d3)
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
				} else if reservoirMode {
					admitCandidateReservoir(&reservoirCandidates, w, neighborID, neighborDistance)
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
	if reservoirMode {
		return reservoirCandidates.Items(), nil
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
