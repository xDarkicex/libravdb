package graph

import (
	"unsafe"

	"github.com/xDarkicex/memory"
)

// EdgeTableIndex is a zero-allocation, wait-free concurrent map
// backed by mmap'd off-heap memory.
type EdgeTableIndex struct {
	m *memory.HashMap
}

// NewEdgeTableIndex creates a new lock-free EdgeTableIndex.
func NewEdgeTableIndex(capacity uint64) *EdgeTableIndex {
	m, err := memory.NewHashMap(memory.HashMapConfig{
		Capacity: capacity,
	})
	if err != nil {
		panic(err)
	}
	return &EdgeTableIndex{
		m: m,
	}
}

// InsertIfAbsent adds a node's page slot in the index concurrently safely.
// Returns the actual page (existing or newly inserted) and a boolean indicating
// if an existing page was loaded (true) or the new page was inserted (false).
//
//go:nocheckptr
func (idx *EdgeTableIndex) InsertIfAbsent(nodeID uint64, page *EdgeTablePage) (*EdgeTablePage, bool) {
	existing, inserted := idx.m.PutIfAbsent(nodeID, unsafe.Pointer(page))
	if !inserted {
		return (*EdgeTablePage)(existing), true
	}
	return page, false
}

// Insert adds or updates a node's page slot in the index.
//
//go:nocheckptr
func (idx *EdgeTableIndex) Insert(nodeID uint64, page *EdgeTablePage) {
	idx.m.Put(nodeID, unsafe.Pointer(page))
}

// Lookup finds the page for a node ID.
// Safe for concurrent lock-free reads.
//
//go:nocheckptr
func (idx *EdgeTableIndex) Lookup(nodeID uint64) *EdgeTablePage {
	ptr, ok := idx.m.Get(nodeID)
	if !ok {
		return nil
	}
	return (*EdgeTablePage)(ptr)
}

// Iterate visits every non-empty node in the table.
func (idx *EdgeTableIndex) Iterate(fn func(nodeID uint64)) {
	if idx == nil || idx.m == nil || fn == nil {
		return
	}
	idx.m.Range(func(k uint64, v unsafe.Pointer) bool {
		fn(k)
		return true
	})
}

// Delete removes a node's page slot from the index.
func (idx *EdgeTableIndex) Delete(nodeID uint64) {
	idx.m.Delete(nodeID)
}

// Close unmaps the underlying memory map.
func (idx *EdgeTableIndex) Close() error {
	if idx == nil || idx.m == nil {
		return nil
	}
	if err := idx.m.Free(); err != nil {
		return err
	}
	idx.m = nil
	return nil
}
