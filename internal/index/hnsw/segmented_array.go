package hnsw

import (
	"sync/atomic"
)

const (
	chunkSizeBits = 16
	chunkSize     = 1 << chunkSizeBits // 65536
	chunkMask     = chunkSize - 1
	maxChunks     = 4096 // Supports up to 268,435,456 nodes
)

// segmentedNodeArray is a lock-free, wait-free append-only array for HNSW nodes.
// It avoids the global write lock required to resize a standard slice.
type segmentedNodeArray struct {
	chunks [maxChunks]atomic.Pointer[[chunkSize]*Node]
	length atomic.Uint32
}

// newSegmentedNodeArray creates a new lock-free node array.
func newSegmentedNodeArray() *segmentedNodeArray {
	return &segmentedNodeArray{}
}

// Get returns the node at the given ordinal index.
// It is completely wait-free for readers.
func (s *segmentedNodeArray) Get(id uint32) *Node {
	chunkIdx := id >> chunkSizeBits
	if chunkIdx >= maxChunks {
		return nil
	}
	chunk := s.chunks[chunkIdx].Load()
	if chunk == nil {
		return nil
	}
	return chunk[id&chunkMask]
}

// Set stores the node at the given ordinal index.
// It is lock-free and allocates new chunks dynamically via CAS.
func (s *segmentedNodeArray) Set(id uint32, node *Node) {
	chunkIdx := id >> chunkSizeBits
	if chunkIdx >= maxChunks {
		panic("segmentedNodeArray: maximum capacity exceeded")
	}

	chunk := s.chunks[chunkIdx].Load()
	if chunk == nil {
		// Provision a new chunk
		newChunk := new([chunkSize]*Node)
		if s.chunks[chunkIdx].CompareAndSwap(nil, newChunk) {
			chunk = newChunk
		} else {
			// Another thread won the CAS, use their chunk
			chunk = s.chunks[chunkIdx].Load()
		}
	}

	chunk[id&chunkMask] = node

	// Update length if this ID expands it
	for {
		oldLen := s.length.Load()
		if id < oldLen {
			break
		}
		if s.length.CompareAndSwap(oldLen, id+1) {
			break
		}
	}
}

// Len returns the current length (maximum initialized ID + 1).
func (s *segmentedNodeArray) Len() int {
	return int(s.length.Load())
}

// segmentedStringArray is a lock-free, wait-free append-only array for strings.
// Used for mapping ordinals to string IDs.
type segmentedStringArray struct {
	chunks [maxChunks]atomic.Pointer[[chunkSize]string]
}

func newSegmentedStringArray() *segmentedStringArray {
	return &segmentedStringArray{}
}

func (s *segmentedStringArray) Get(id uint32) string {
	chunkIdx := id >> chunkSizeBits
	if chunkIdx >= maxChunks {
		return ""
	}
	chunk := s.chunks[chunkIdx].Load()
	if chunk == nil {
		return ""
	}
	return chunk[id&chunkMask]
}

func (s *segmentedStringArray) Set(id uint32, str string) {
	chunkIdx := id >> chunkSizeBits
	if chunkIdx >= maxChunks {
		panic("segmentedStringArray: maximum capacity exceeded")
	}

	chunk := s.chunks[chunkIdx].Load()
	if chunk == nil {
		newChunk := new([chunkSize]string)
		if s.chunks[chunkIdx].CompareAndSwap(nil, newChunk) {
			chunk = newChunk
		} else {
			chunk = s.chunks[chunkIdx].Load()
		}
	}

	chunk[id&chunkMask] = str
}
