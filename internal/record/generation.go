package record

import (
	"sync/atomic"
	"unsafe"

	"github.com/xDarkicex/memory"
)

const ordinalBits = 8
const ordinalFanout = 1 << ordinalBits
const ordinalLevels = 4

// ordinalNode is one off-heap radix page. owner marks nodes created by the
// current unpublished generation, allowing many mutations to share the same
// copied path without a Go map that tracks visited nodes.
type ordinalNode struct {
	owner uint64
	slots [ordinalFanout]unsafe.Pointer
}

var nextGenerationID atomic.Uint64

// Generation is an immutable published-state candidate. The Go object is a
// fixed-size control root only; every capacity-scaled record, key, and ordinal
// page is owned by its arena or IDMap mapping.
type Generation struct {
	parent  *Generation
	arena   *memory.Arena
	ids     *Directory
	root    *ordinalNode
	segment []unsafe.Pointer
	id      uint64
	refs    atomic.Int64
}

func cloneOrdinalNode(arena *memory.Arena, owner uint64, source *ordinalNode) (*ordinalNode, error) {
	node, err := memory.ArenaAlloc[ordinalNode](arena)
	if err != nil {
		return nil, err
	}
	if source != nil {
		*node = *source
	}
	node.owner = owner
	return node, nil
}

// NewGeneration consumes delta on success. It copies only the radix pages
// traversed by touched ordinals, while untouched paths remain shared with the
// immutable parent generation.
func NewGeneration(parent *Generation, delta *Delta) (*Generation, error) {
	if delta == nil || delta.arena == nil {
		return nil, ErrDeltaFull
	}
	capacity := delta.idCapacity
	if capacity < uint64(delta.count)*2 {
		capacity = uint64(delta.count) * 2
	}
	ids, err := NewDirectory(capacity, delta.idKeyBytes)
	if err != nil {
		return nil, err
	}
	generation := &Generation{arena: delta.arena, ids: ids, id: nextGenerationID.Add(1)}
	generation.refs.Store(1)
	if parent != nil {
		generation.parent = parent.Acquire()
	}
	var parentRoot *ordinalNode
	if parent != nil {
		parentRoot = parent.root
	}
	generation.root, err = cloneOrdinalNode(generation.arena, generation.id, parentRoot)
	if err != nil {
		_ = ids.Free()
		if generation.parent != nil {
			generation.parent.Release()
		}
		return nil, err
	}
	segment, err := memory.ArenaSlice[unsafe.Pointer](generation.arena, delta.Len())
	if err != nil {
		_ = ids.Free()
		if generation.parent != nil {
			generation.parent.Release()
		}
		return nil, err
	}
	generation.segment = segment
	for i := 0; i < delta.Len(); i++ {
		mutation := delta.At(i)
		after := mutation.After()
		if !after.Valid() {
			// A brand-new insert followed by delete does not alter a parent.
			continue
		}
		generation.segment = generation.segment[:len(generation.segment)+1]
		generation.segment[len(generation.segment)-1] = after.ptr
		if err := generation.ids.Put(mutation.ID(), after); err != nil {
			_ = ids.Free()
			if generation.parent != nil {
				generation.parent.Release()
			}
			return nil, err
		}
		if err := generation.setOrdinal(after.Ordinal(), after); err != nil {
			_ = ids.Free()
			if generation.parent != nil {
				generation.parent.Release()
			}
			return nil, err
		}
	}
	// The published generation owns the arena. The staging IDMap is no longer
	// needed because ids now maps directly to immutable RecordRefs.
	if err := delta.ids.Free(); err != nil {
		_ = ids.Free()
		if generation.parent != nil {
			generation.parent.Release()
		}
		return nil, err
	}
	delta.ids = nil
	delta.arena = nil
	delta.entries = nil
	delta.count = 0
	return generation, nil
}

func (g *Generation) setOrdinal(ordinal uint32, ref RecordRef) error {
	node := g.root
	for level := ordinalLevels - 1; level > 0; level-- {
		index := (ordinal >> uint(level*ordinalBits)) & (ordinalFanout - 1)
		child := (*ordinalNode)(node.slots[index])
		if child == nil || child.owner != g.id {
			cloned, err := cloneOrdinalNode(g.arena, g.id, child)
			if err != nil {
				return err
			}
			node.slots[index] = unsafe.Pointer(cloned)
			child = cloned
		}
		node = child
	}
	node.slots[ordinal&(ordinalFanout-1)] = ref.ptr
	return nil
}

func (g *Generation) getOrdinal(ordinal uint32) RecordRef {
	if g == nil || g.root == nil {
		return RecordRef{}
	}
	node := g.root
	for level := ordinalLevels - 1; level > 0; level-- {
		index := (ordinal >> uint(level*ordinalBits)) & (ordinalFanout - 1)
		node = (*ordinalNode)(node.slots[index])
		if node == nil {
			return RecordRef{}
		}
	}
	return RecordRef{ptr: node.slots[ordinal&(ordinalFanout-1)]}
}

// Lookup resolves an ID through this generation's delta chain. A tombstone is
// reported as found with a tombstone ref so callers can distinguish it from an
// ID absent from every generation.
func (g *Generation) Lookup(id BytesView) (RecordRef, bool) {
	for current := g; current != nil; current = current.parent {
		if ref, found := current.ids.Get(id); found {
			return ref, true
		}
	}
	return RecordRef{}, false
}

func (g *Generation) ByOrdinal(ordinal uint32) RecordRef { return g.getOrdinal(ordinal) }

// Visible reports whether ref is the exact live record selected by this
// generation. Flat can apply this in its scan loop without an ID-map probe.
func (g *Generation) Visible(ref RecordRef) bool {
	if !ref.Valid() || ref.Tombstone() {
		return false
	}
	return g.getOrdinal(ref.Ordinal()).ptr == ref.ptr
}

// Acquire returns a read lease. Every caller must pair it with Release before
// dereferencing a RecordRef obtained from the generation.
func (g *Generation) Acquire() *Generation {
	if g != nil {
		g.refs.Add(1)
	}
	return g
}

// Release destroys a generation only after all reader leases and child
// generation references are gone. Publication integration will retire this
// root through the collection's Hyaline path; this local reference count keeps
// un-published construction and unit tests safe in the meantime.
func (g *Generation) Release() {
	if g == nil || g.refs.Add(-1) != 0 {
		return
	}
	parent := g.parent
	if g.ids != nil {
		_ = g.ids.Free()
		g.ids = nil
	}
	if g.arena != nil {
		_ = g.arena.Free()
		g.arena = nil
	}
	g.root = nil
	g.segment = nil
	g.parent = nil
	if parent != nil {
		parent.Release()
	}
}

// Parent returns the older immutable generation. It is exposed for index
// segment scans; callers must retain the child generation lease throughout
// the traversal, which also retains every parent generation.
func (g *Generation) Parent() *Generation {
	if g == nil {
		return nil
	}
	return g.parent
}

func (g *Generation) SegmentLen() int {
	if g == nil {
		return 0
	}
	return len(g.segment)
}

func (g *Generation) SegmentAt(i int) RecordRef {
	if g == nil || i < 0 || i >= len(g.segment) {
		panic("record: segment index out of range")
	}
	return RecordRef{ptr: g.segment[i]}
}
