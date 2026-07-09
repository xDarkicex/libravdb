package hnsw

import "unsafe"

const MaxLevel = 16

// Node represents a single node in the HNSW graph.
// Canonical vectors and metadata are owned outside the graph.
type Node struct {
	Links            [MaxLevel]*uint32
	Backlinks        [MaxLevel]*uint32
	LinkCounts       [MaxLevel]uint32
	BacklinkCounts   [MaxLevel]uint32
	LinkHeuristic    [MaxLevel]uint32
	CompressedVector []byte
	Vector           []float32
	VectorPtr        unsafe.Pointer
	Level            int
	Ordinal          uint32
	Slot             uint32
	InFlight         uint32 // Atomic boolean (1=in flight, 0=committed)
	PruneLock        uint32 // 1-byte micro-spinlock padded to uint32 for atomics
}

func (n *Node) setVector(vec []float32) {
	n.Vector = vec
	if len(vec) == 0 {
		n.VectorPtr = nil
		return
	}
	n.VectorPtr = unsafe.Pointer(&vec[0])
}
