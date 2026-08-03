package btree

import (
	"bytes"
	"unsafe"
)

// Page layout constants.
const (
	PageSize       = 4096  // total slot size (matches EdgeTablePage)
	UserDataOffset = 64    // ShardedFreeList metadata header
	UserDataSize   = PageSize - UserDataOffset // 4032 bytes usable
	MaxKeyLen      = 2048  // keys must fit in half a page

	// Page flags (match LMDB conventions where possible).
	P_BRANCH uint16 = 0x01
	P_LEAF   uint16 = 0x02
)

// BTreeHeader is 32 bytes at the start of the user data area.
// Same size as EdgeTableHeader for consistency with the page pool.
type BTreeHeader struct {
	Generation   uint32 // MVCC version counter (atomic increment on mutation)
	PageSlot     uint32 // own PageRegistry slot ID
	RightSibling uint32 // B-link: right sibling PageRegistry ID (0 = none)
	LeftSibling  uint32 // B-link: left sibling PageRegistry ID (0 = none)
	FirstChild   uint32 // branch pages only: child for keys < first key (0 for leaf)
	Count        uint16 // number of keys in this page
	Flags        uint16 // P_LEAF | P_BRANCH
	Lower        uint16 // byte offset: end of ptr array (grows downward from UserDataSize)
	Upper        uint16 // byte offset: end of node data (grows upward from sizeof(Header))
	HyalineSlot  uint16 // shard index for Hyaline SMR
	_            [2]byte // pad to 32 bytes
}

// BTreePage occupies 4032 bytes of user data within a 4096-byte slot.
// Cast directly over mmap'd []byte via unsafe.Pointer, same pattern as EdgeTablePage.
//
// Layout (offsets within 4032-byte user data area):
//
//	[0..31]         Header (32 bytes)
//	[32..Upper)     Node data (grows upward as keys are inserted)
//	(Lower..4032]   Pointer array (grows downward, 2 bytes per entry)
//	[Upper..Lower)  Free space
type BTreePage struct {
	Header BTreeHeader
	_      [UserDataSize - 32]byte // 4000 bytes
}

// Ptrs returns the pointer array as a uint16 slice in key order.
// ptrs[0] = byte offset of the smallest key's node, up to ptrs[Count-1].
func (p *BTreePage) Ptrs() []uint16 {
	n := p.Header.Count
	if n == 0 {
		return nil
	}
	base := uintptr(unsafe.Pointer(p)) + uintptr(p.Header.Lower)
	return unsafe.Slice((*uint16)(unsafe.Pointer(base)), int(n))
}

// NodeAt returns a pointer to the i-th BTreeNode in key order.
func (p *BTreePage) NodeAt(i int) *BTreeNode {
	ptrs := p.Ptrs()
	if i < 0 || i >= len(ptrs) {
		return nil
	}
	return (*BTreeNode)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrs[i])))
}

// FreeSpace returns the number of bytes available between Upper and Lower.
func (p *BTreePage) FreeSpace() int {
	return int(p.Header.Lower) - int(p.Header.Upper)
}

// SpaceNeeded returns the bytes needed to insert a key-value pair (node + ptr).
func (p *BTreePage) SpaceNeeded(keyLen, valLen int) int {
	return 2 + nodeSize(keyLen, valLen)
}

// HighKey returns the first (smallest) key in this page.
func (p *BTreePage) HighKey() []byte {
	if p.Header.Count == 0 {
		return nil
	}
	return p.NodeAt(0).Key()
}

// MaxKey returns the last (largest) key in this page.
func (p *BTreePage) MaxKey() []byte {
	if p.Header.Count == 0 {
		return nil
	}
	return p.NodeAt(int(p.Header.Count) - 1).Key()
}

// IsLeaf returns true if this is a leaf page.
func (p *BTreePage) IsLeaf() bool { return p.Header.Flags&P_LEAF != 0 }

// IsBranch returns true if this is a branch (internal) page.
func (p *BTreePage) IsBranch() bool { return p.Header.Flags&P_BRANCH != 0 }

// findKey performs binary search for a key within this page.
// Returns the index where the key was found (found=true) or the insertion
// index (found=false). Uses bytes.Compare for comparison.
func (p *BTreePage) findKey(key []byte) (idx int, found bool) {
	lo, hi := 0, int(p.Header.Count)-1
	for lo <= hi {
		mid := (lo + hi) / 2
		cmp := bytes.Compare(key, p.NodeAt(mid).Key())
		if cmp < 0 {
			hi = mid - 1
		} else if cmp > 0 {
			lo = mid + 1
		} else {
			return mid, true
		}
	}
	return lo, false
}

// BTreeNode is a 12-byte header followed by key and optional value data.
// Equivalent to LMDB's MDB_node.
type BTreeNode struct {
	KeyLen uint16 // key length in bytes
	ValLen uint16 // value length in bytes (0 for branch nodes)
	Child  uint32 // child page slot ID (branch only; 0 for leaf)
	_      [4]byte // pad to 12 bytes
	// Followed by: key[KeyLen] + value[ValLen] (leaf) or key[KeyLen] only (branch)
}

const nodeHeaderSize = 12

func nodeSize(keyLen, valLen int) int {
	return nodeHeaderSize + keyLen + valLen
}

// Key returns the key bytes for this node (zero-copy).
func (n *BTreeNode) Key() []byte {
	if n.KeyLen == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(n))+nodeHeaderSize)), n.KeyLen)
}

// Value returns the value bytes for this node (zero-copy, leaf only).
func (n *BTreeNode) Value() []byte {
	if n.ValLen == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(n))+nodeHeaderSize+uintptr(n.KeyLen))), n.ValLen)
}

// Size returns the total byte size of this node.
func (n *BTreeNode) Size() int {
	return nodeSize(int(n.KeyLen), int(n.ValLen))
}

// initPage initializes a freshly allocated page.
func (p *BTreePage) initPage(flags uint16, pageSlot uint32, hyalineShard int) {
	p.Header.Flags = flags
	p.Header.PageSlot = pageSlot
	p.Header.HyalineSlot = uint16(hyalineShard)
	p.Header.Count = 0
	p.Header.Lower = UserDataSize
	p.Header.Upper = uint16(unsafe.Sizeof(BTreeHeader{}))
	p.Header.Generation = 0
	p.Header.RightSibling = 0
	p.Header.LeftSibling = 0
}

// resetPage reinitializes a recycled page, bumping its generation.
func (p *BTreePage) resetPage(flags uint16) {
	p.Header.Flags = flags
	p.Header.Count = 0
	p.Header.Lower = UserDataSize
	p.Header.Upper = uint16(unsafe.Sizeof(BTreeHeader{}))
	p.Header.RightSibling = 0
	p.Header.LeftSibling = 0
	p.Header.Generation++
}

// pageSlotBytes returns the full 4096-byte slot for pool return.
func pageSlotBytes(p *BTreePage) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(p))-UserDataOffset)), PageSize)
}

// pageData returns a []byte view of the user data area.
func (p *BTreePage) pageData() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(p)), UserDataSize)
}
