package slab

import (
	"math"
)

const (
	OffsetMask = uint64(0x0000FFFFFFFFFFFF) // 48 bits
	DegreeMask = uint64(0xFFFF000000000000) // 16 bits
	DegreeShift = 48
)

// RoutingPointer encodes both the physical memory offset to the slab and the degree (edge count).
// - Bits 48-63: Degree (up to 65,535 edges)
// - Bits 0-47: Offset (up to 256TB arena)
//
// Access must be strictly lock-free via atomic.LoadUint64 and atomic.CompareAndSwapUint64.
type RoutingPointer uint64

// Pack builds a single RoutingPointer.
func Pack(offset uint64, degree uint16) RoutingPointer {
	return RoutingPointer((uint64(degree) << DegreeShift) | (offset & OffsetMask))
}

// Offset extracts the 48-bit physical byte offset.
func (r RoutingPointer) Offset() uint64 {
	return uint64(r) & OffsetMask
}

// Degree extracts the 16-bit active edge count.
func (r RoutingPointer) Degree() uint16 {
	return uint16((uint64(r) & DegreeMask) >> DegreeShift)
}

// SlabHeader defines the cache-line-aligned prefix of a CoW Slab.
// Implementing the Bridge Protocol: 
// The tombstone is represented by setting Xmax to the active transaction. 
// Tombstoned nodes remain fully routable during HNSW traversal but are filtered out 
// of the final result set via visibility checks.
// 
// Size: 32 bytes
type SlabHeader struct {
	Xmin         uint64 // Transaction ID that created this slab
	Xmax         uint64 // Transaction ID that deleted this slab (tombstone). math.MaxUint64 = active.
	VectorOffset uint64 // 48-bit offset to physical vector data
	LayerCount   uint32 // Number of HNSW layers contained in this slab
	Padding      uint32 // Padding to align header perfectly to 32 bytes
}

// NewSlabHeader initializes a slab header.
func NewSlabHeader(xmin uint64, vectorOffset uint64, layerCount uint32) SlabHeader {
	return SlabHeader{
		Xmin:         xmin,
		Xmax:         math.MaxUint64, // Active by default
		VectorOffset: vectorOffset,
		LayerCount:   layerCount,
		Padding:      0,
	}
}

// IsTombstoned checks if the node was logically deleted prior to the snapshot xid.
func (s *SlabHeader) IsTombstoned(snapshotXid uint64) bool {
	// If Xmax is less than the current snapshot, it means the deletion committed 
	// before the reader started.
	return s.Xmax < snapshotXid
}

// IsVisible evaluates the standard MVCC Snapshot Isolation visibility rules.
func (s *SlabHeader) IsVisible(snapshotXid uint64) bool {
	// The node must have been created before the snapshot...
	if s.Xmin >= snapshotXid {
		return false
	}
	// ...and must not have been deleted before the snapshot.
	if s.Xmax < snapshotXid {
		return false
	}
	return true
}
