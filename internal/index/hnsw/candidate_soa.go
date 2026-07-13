package hnsw

import "github.com/xDarkicex/libravdb/internal/util"

const soaCandidateVisited uint32 = 1 << 31
const soaCandidateIDMask uint32 = soaCandidateVisited - 1

// Compile-time proof that every valid registry ordinal leaves the visited bit
// available for queue-local expansion state.
const _ uint32 = soaCandidateVisited - maxNodeCapacity

// soaCandidateQueue is a bounded, distance-sorted beam. IDs and expansion
// state are kept apart from distances so lower-bound scans touch only floats.
// The high ID bit is available because the node registry is bounded below 2^28.
type soaCandidateQueue struct {
	ids       []uint32
	distances []float32
	size      int
	cursor    int
}

func newSOACandidateQueue(ids []uint32, distances []float32, capacity int) soaCandidateQueue {
	if capacity > len(ids) {
		capacity = len(ids)
	}
	if capacity > len(distances) {
		capacity = len(distances)
	}
	return soaCandidateQueue{
		ids:       ids[:capacity],
		distances: distances[:capacity],
	}
}

func (q *soaCandidateQueue) Len() int { return q.size }

func (q *soaCandidateQueue) HasUnexpanded() bool { return q.cursor < q.size }

func (q *soaCandidateQueue) Worst() util.Candidate {
	if q.size == 0 {
		return util.Candidate{}
	}
	i := q.size - 1
	return util.Candidate{ID: q.ids[i] & soaCandidateIDMask, Distance: q.distances[i]}
}

func (q *soaCandidateQueue) Insert(candidate util.Candidate) bool {
	if len(q.ids) == 0 || candidate.ID&soaCandidateVisited != 0 || candidate.Distance != candidate.Distance {
		return false
	}

	if q.size == len(q.ids) && !candidateBetter(candidate, q.Worst()) {
		return false
	}

	insertAt := q.lowerBound(candidate)
	oldSize := q.size
	if oldSize == len(q.ids) {
		oldSize--
		q.size--
		if q.cursor > q.size {
			q.cursor = q.size
		}
	}

	copy(q.ids[insertAt+1:oldSize+1], q.ids[insertAt:oldSize])
	copy(q.distances[insertAt+1:oldSize+1], q.distances[insertAt:oldSize])
	q.ids[insertAt] = candidate.ID
	q.distances[insertAt] = candidate.Distance
	q.size++
	if insertAt < q.cursor {
		q.cursor = insertAt
	}
	return true
}

func (q *soaCandidateQueue) PopClosestUnexpanded() (util.Candidate, bool) {
	if !q.HasUnexpanded() {
		return util.Candidate{}, false
	}

	i := q.cursor
	id := q.ids[i] & soaCandidateIDMask
	q.ids[i] |= soaCandidateVisited
	q.cursor++
	for q.cursor < q.size && q.ids[q.cursor]&soaCandidateVisited != 0 {
		q.cursor++
	}
	return util.Candidate{ID: id, Distance: q.distances[i]}, true
}

func (q *soaCandidateQueue) AppendCandidates(dst []util.Candidate) []util.Candidate {
	if cap(dst) < q.size {
		panic("hnsw: SoA result buffer is smaller than the candidate queue")
	}
	dst = dst[:q.size]
	for i := 0; i < q.size; i++ {
		dst[i] = util.Candidate{
			ID:       q.ids[i] & soaCandidateIDMask,
			Distance: q.distances[i],
		}
	}
	return dst
}

func (q *soaCandidateQueue) lowerBound(candidate util.Candidate) int {
	if q.size >= 96 {
		return q.lowerBoundHybrid(candidate)
	}
	return q.lowerBoundBinary(candidate)
}

func (q *soaCandidateQueue) lowerBoundBinary(candidate util.Candidate) int {
	lo, hi := 0, q.size
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		current := util.Candidate{
			ID:       q.ids[mid] & soaCandidateIDMask,
			Distance: q.distances[mid],
		}
		// Equal distances are ordered by ascending ID through candidateBetter.
		if candidateBetter(current, candidate) {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

func (q *soaCandidateQueue) lowerBoundHybrid(candidate util.Candidate) int {
	lo, hi := 0, q.size
	for hi-lo > 16 {
		mid := int(uint(lo+hi) >> 1)
		current := util.Candidate{
			ID:       q.ids[mid] & soaCandidateIDMask,
			Distance: q.distances[mid],
		}
		if candidateBetter(current, candidate) {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	for lo < hi {
		current := util.Candidate{
			ID:       q.ids[lo] & soaCandidateIDMask,
			Distance: q.distances[lo],
		}
		if !candidateBetter(current, candidate) {
			return lo
		}
		lo++
	}
	return hi
}

func admitBatch4SOA(queue *soaCandidateQueue, ids []uint32, d0, d1, d2, d3 float32) {
	if queue.Len() == len(queue.ids) {
		worst := queue.Worst().Distance
		if d0 >= worst && d1 >= worst && d2 >= worst && d3 >= worst {
			return
		}
	}
	queue.Insert(util.Candidate{ID: ids[0], Distance: d0})
	queue.Insert(util.Candidate{ID: ids[1], Distance: d1})
	queue.Insert(util.Candidate{ID: ids[2], Distance: d2})
	queue.Insert(util.Candidate{ID: ids[3], Distance: d3})
}
