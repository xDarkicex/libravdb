package libravdb

import "errors"

// ---------------------------------------------------------------------------
// GraphNodeID — generic database-scoped fixed-width graph/record handle
//
// GraphNodeID is a nonzero uint64 that uniquely identifies a graph node or
// record within one database file. It is stable, collision-free, and never
// reused. It carries no causal semantics — it is a generic storage guarantee
// that prevents ordinal aliasing across collections and shards.
//
// Zero is invalid. A valid GraphNodeID is always > 0.
//
// GraphNodeID has no heap-owning fields (no string, map, slice, interface, or
// pointer). Any future runtime lookup table must use off-heap storage via
// xDarkicex/memory or persistent on-disk structures.
//
// LibraVDB owns GraphNodeID allocation and persistence; the durable allocator
// is implemented in a later leaf.
// ---------------------------------------------------------------------------

// GraphNodeID is a database-scoped, never-reused graph or record handle.
// The zero value is invalid. Valid identifiers are always nonzero.
type GraphNodeID uint64

// ErrInvalidGraphNodeID is returned when a GraphNodeID is zero.
var ErrInvalidGraphNodeID = errors.New("GraphNodeID: zero is invalid")

// IsValid reports whether this GraphNodeID is nonzero.
func (id GraphNodeID) IsValid() bool {
	return id > 0
}

// Validate returns a sentinel error if the GraphNodeID is zero, or nil.
// The invalid path does not allocate.
func (id GraphNodeID) Validate() error {
	if id == 0 {
		return ErrInvalidGraphNodeID
	}
	return nil
}
