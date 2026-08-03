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

// VisitedKey computes the bitset index for a (node, band) pair.
// numBands is the total number of edge bands in the pattern.
// Index = nodeID * numBands + band.  Callers must ensure the result
// fits within the bitset capacity (1M bits for default slot).
func VisitedKey(nodeID uint64, band int, numBands int) uint64 {
	return nodeID*uint64(numBands) + uint64(band)
}

// NodeDepth represents an entry in the BFS frontier queue.
// Kept at 16 bytes so pool sizing (4096 entries per 65536 B slot) is unchanged.
type NodeDepth struct {
	NodeID uint64
	Band   int32
	Step   int32
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
func (f *FrontierBuf) Push(nodeID uint64, band, step int) bool {
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
	f.data[f.tail] = NodeDepth{NodeID: nodeID, Band: int32(band), Step: int32(step)}
	f.tail++
	return true
}

// Pop removes and returns the first item in the frontier queue.
func (f *FrontierBuf) Pop() (uint64, int, int) {
	if f.head == f.tail {
		return 0, 0, 0
	}
	item := f.data[f.head]
	f.head++
	return item.NodeID, int(item.Band), int(item.Step)
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
