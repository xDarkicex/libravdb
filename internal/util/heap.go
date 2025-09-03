package util

import "container/heap"

// Candidate represents a search candidate with distance
type Candidate struct {
	ID       uint32
	Distance float32
}

// MinHeap implements a min-heap for candidates
type MinHeap struct {
	candidates []*Candidate
	maxSize    int
}

// NewMinHeap creates a new min-heap
func NewMinHeap(maxSize int) *MinHeap {
	return &MinHeap{
		candidates: make([]*Candidate, 0, maxSize),
		maxSize:    maxSize,
	}
}

func (h *MinHeap) Len() int { return len(h.candidates) }

func (h *MinHeap) Less(i, j int) bool {
	return h.candidates[i].Distance < h.candidates[j].Distance
}

func (h *MinHeap) Swap(i, j int) {
	h.candidates[i], h.candidates[j] = h.candidates[j], h.candidates[i]
}

func (h *MinHeap) Push(x interface{}) {
	h.candidates = append(h.candidates, x.(*Candidate))
}

func (h *MinHeap) Pop() interface{} {
	old := h.candidates
	n := len(old)
	item := old[n-1]
	h.candidates = old[0 : n-1]
	return item
}

// PushCandidate adds a candidate to the heap
func (h *MinHeap) PushCandidate(c *Candidate) {
	heap.Push(h, c)
}

// PopCandidate removes and returns the minimum candidate
func (h *MinHeap) PopCandidate() *Candidate {
	if h.Len() == 0 {
		return nil
	}
	return heap.Pop(h).(*Candidate)
}

// MaxHeap implements a max-heap for candidates
type MaxHeap struct {
	candidates []*Candidate
	maxSize    int
}

// NewMaxHeap creates a new max-heap
func NewMaxHeap(maxSize int) *MaxHeap {
	return &MaxHeap{
		candidates: make([]*Candidate, 0, maxSize),
		maxSize:    maxSize,
	}
}

func (h *MaxHeap) Len() int { return len(h.candidates) }

func (h *MaxHeap) Less(i, j int) bool {
	return h.candidates[i].Distance > h.candidates[j].Distance // Reverse for max-heap
}

func (h *MaxHeap) Swap(i, j int) {
	h.candidates[i], h.candidates[j] = h.candidates[j], h.candidates[i]
}

func (h *MaxHeap) Push(x interface{}) {
	h.candidates = append(h.candidates, x.(*Candidate))
}

func (h *MaxHeap) Pop() interface{} {
	old := h.candidates
	n := len(old)
	item := old[n-1]
	h.candidates = old[0 : n-1]
	return item
}

// PushCandidate adds a candidate to the heap
func (h *MaxHeap) PushCandidate(c *Candidate) {
	heap.Push(h, c)
}

// PopCandidate removes and returns the maximum candidate
func (h *MaxHeap) PopCandidate() *Candidate {
	if h.Len() == 0 {
		return nil
	}
	return heap.Pop(h).(*Candidate)
}

// Top returns the maximum candidate without removing it
func (h *MaxHeap) Top() *Candidate {
	if h.Len() == 0 {
		return nil
	}
	return h.candidates[0]
}
