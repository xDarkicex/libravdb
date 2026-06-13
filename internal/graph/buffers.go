package graph

import (
	"unsafe"
)

// Bitset is an off-heap dense bitset for tracking visited nodes.
type Bitset struct {
	data []uint64
	slot []byte
}

func newBitset(slot []byte) *Bitset {
	// user data starts at offset 64
	userData := slot[64:]
	// length of user data in uint64s
	ptr := (*uint64)(unsafe.Pointer(&userData[0]))
	data := unsafe.Slice(ptr, len(userData)/8)
	return &Bitset{data: data, slot: slot}
}

// Test returns true if the bit for nodeID is set.
func (b *Bitset) Test(nodeID uint64) bool {
	word := nodeID / 64
	bit := nodeID % 64
	if int(word) >= len(b.data) {
		return false
	}
	return (b.data[word] & (1 << bit)) != 0
}

// Set marks the bit for nodeID as true.
func (b *Bitset) Set(nodeID uint64) {
	word := nodeID / 64
	bit := nodeID % 64
	if int(word) >= len(b.data) {
		return
	}
	b.data[word] |= (1 << bit)
}

// Clear zeroes out the entire bitset.
func (b *Bitset) Clear() {
	for i := range b.data {
		b.data[i] = 0
	}
}

// ClearBit clears a specific bit.
func (b *Bitset) ClearBit(nodeID uint64) {
	word := nodeID / 64
	bit := nodeID % 64
	if int(word) >= len(b.data) {
		return
	}
	b.data[word] &^= (1 << bit)
}

// NodeDepth represents an entry in the BFS frontier queue.
type NodeDepth struct {
	NodeID uint64
	Depth  int
}

// FrontierBuf is an off-heap queue for BFS traversal.
type FrontierBuf struct {
	data []NodeDepth
	slot []byte
	head int
	tail int
}

func newFrontierBuf(slot []byte) *FrontierBuf {
	userData := slot[64:]
	ptr := (*NodeDepth)(unsafe.Pointer(&userData[0]))
	// Calculate how many NodeDepth structs fit into userData
	capacity := len(userData) / int(unsafe.Sizeof(NodeDepth{}))
	data := unsafe.Slice(ptr, capacity)
	return &FrontierBuf{data: data, slot: slot}
}

// Push adds an item to the frontier queue.
func (f *FrontierBuf) Push(nodeID uint64, depth int) bool {
	if f.tail == len(f.data) {
		// Shift items to the front if we have space
		if f.head > 0 {
			n := copy(f.data, f.data[f.head:f.tail])
			f.tail = n
			f.head = 0
		}
		if f.tail == len(f.data) {
			return false // Queue full
		}
	}
	f.data[f.tail] = NodeDepth{NodeID: nodeID, Depth: depth}
	f.tail++
	return true
}

// Pop removes and returns the first item in the frontier queue.
func (f *FrontierBuf) Pop() (uint64, int) {
	if f.head == f.tail {
		return 0, 0
	}
	item := f.data[f.head]
	f.head++
	return item.NodeID, item.Depth
}

// Empty returns true if the queue is empty.
func (f *FrontierBuf) Empty() bool {
	return f.head == f.tail
}

// Clear resets the frontier queue without allocating.
func (f *FrontierBuf) Clear() {
	f.head = 0
	f.tail = 0
}
