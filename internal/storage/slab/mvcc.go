package slab

import (
	"sync/atomic"
	"unsafe"
)

// Arena simulates the off-heap mmap region.
// In the full daemon, this manages OS page boundaries and mmap expansions.
type Arena struct {
	data []byte
	tail uint64
}

// NewArena allocates a fixed byte slice to simulate mmap.
func NewArena(size uint64) *Arena {
	return &Arena{
		data: make([]byte, size),
		tail: 0, // Starts at 0
	}
}

// WriteSlab atomically claims space and writes the slab.
// The payload must include the 32-byte SlabHeader followed by the adjacency list.
func (a *Arena) WriteSlab(payload []byte) uint64 {
	length := uint64(len(payload))
	offset := atomic.AddUint64(&a.tail, length) - length
	
	// Fast zero-copy write directly into the pre-allocated slice.
	copy(a.data[offset:], payload)
	return offset
}

// GetSlab dereferences an offset into a strongly-typed SlabHeader pointer.
func (a *Arena) GetSlab(offset uint64) *SlabHeader {
	return (*SlabHeader)(unsafe.Pointer(&a.data[offset]))
}

// Node represents a single entity in the HNSW graph.
// It is anchored by a single 64-bit atomic RoutingPointer.
type Node struct {
	Ptr RoutingPointer // Accessed exclusively via atomic ops
}

// Read routes to the current slab via wait-free atomic load.
func (n *Node) Read() RoutingPointer {
	return RoutingPointer(atomic.LoadUint64((*uint64)(&n.Ptr)))
}

// Update performs the Lock-Free Compare-And-Swap (CAS) on the CoW slab.
// It acts as the core primitive for Lock-Free MVCC.
// If another transaction modified the node, this function will return false,
// allowing the writer to rebuild the slab (CAS-retry loop).
func (n *Node) Update(oldPtr RoutingPointer, newPtr RoutingPointer) bool {
	return atomic.CompareAndSwapUint64((*uint64)(&n.Ptr), uint64(oldPtr), uint64(newPtr))
}
