package graph

import (
	"sync/atomic"
)

// Tombstone represents a deleted slot
const Tombstone uint32 = 0xFFFFFFFF

// hashUint64 provides a fast integer hash using the SplitMix64 algorithm.
func hashUint64(x uint64) uint64 {
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

// Insert adds or updates a node's page slot in the index.
// Not safe for concurrent writes; writers must synchronize.
func (idx *EdgeTableIndex) Insert(nodeID uint64, pageSlot uint32) {
	if float64(idx.size+1) >= float64(idx.capacity)*idx.loadFactor {
		idx.resize()
	}

	mask := idx.capacity - 1
	start := hashUint64(nodeID) & mask
	
	for i := uint64(0); i < idx.capacity; i++ {
		pos := (start + i) & mask
		entry := &idx.table[pos]
		
		if slot := atomic.LoadUint32(&entry.PageSlot); slot == 0 || slot == Tombstone {
			// Found empty slot or tombstone
			entry.NodeID = nodeID
			atomic.StoreUint32(&entry.PageSlot, pageSlot)
			idx.size++
			return
		}
		
		if entry.NodeID == nodeID {
			// Update existing
			atomic.StoreUint32(&entry.PageSlot, pageSlot)
			return
		}
	}
}

// Lookup finds the page slot for a node ID.
// Safe for concurrent lock-free reads.
func (idx *EdgeTableIndex) Lookup(nodeID uint64) uint32 {
	if idx.size == 0 {
		return 0
	}
	
	mask := idx.capacity - 1
	start := hashUint64(nodeID) & mask
	
	for i := uint64(0); i < idx.capacity; i++ {
		pos := (start + i) & mask
		entry := &idx.table[pos]
		
		slot := atomic.LoadUint32(&entry.PageSlot)
		if slot == 0 {
			// Hit empty slot, node not in table
			return 0
		}
		if slot == Tombstone {
			continue
		}
		
		if entry.NodeID == nodeID {
			return slot
		}
	}
	
	return 0
}

func (idx *EdgeTableIndex) resize() {
	oldTable := idx.table
	newCap := idx.capacity * 2
	idx.table = make([]EdgeTableLocator, newCap)
	idx.capacity = newCap
	idx.size = 0 // Recalculated during insertion
	
	for i := range oldTable {
		slot := oldTable[i].PageSlot
		if slot != 0 && slot != Tombstone {
			idx.Insert(oldTable[i].NodeID, slot)
		}
	}
}

// Delete marks a node's page slot as deleted using a tombstone.
func (idx *EdgeTableIndex) Delete(nodeID uint64) {
	if idx.size == 0 {
		return
	}
	
	mask := idx.capacity - 1
	start := hashUint64(nodeID) & mask
	
	for i := uint64(0); i < idx.capacity; i++ {
		pos := (start + i) & mask
		entry := &idx.table[pos]
		
		slot := atomic.LoadUint32(&entry.PageSlot)
		if slot == 0 {
			return // Not found
		}
		
		if slot != Tombstone && entry.NodeID == nodeID {
			atomic.StoreUint32(&entry.PageSlot, Tombstone)
			idx.size--
			return
		}
	}
}
