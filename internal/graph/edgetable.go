package graph

// EdgeTablePage stores edges for a single node with inline-first-8 layout.
//
// ShardedFreeList slot layout: each slot has 64 bytes of allocator metadata
// at offsets 0-63 (next ptr, batch_link, refs/batch_next, structIdx,
// homeShard, Hyaline reclaim chain). The user data area starts at offset 64,
// so this struct occupies 4032 bytes of user data. The full slot is 4096
// bytes (SlotSize). Page pointers returned to the graph layer point to the
// user data area, not the slot start.
type EdgeTablePage struct {
	Header  EdgeTableHeader // 24 bytes
	Inline  [8]Edge         // 128 bytes (8 × 16)
	Padding [3880]byte      // 3880 = 4032 - 24 - 128; remaining for overflow or future use
}

// EdgeTableHeader contains page metadata
type EdgeTableHeader struct {
	Count       uint16  // Total edge count (inline + overflow)
	InlineCap   uint16  // Always 8 for inline-first-8 layout
	Overflow    uint32  // Offset to overflow chain (0 if none)
	Generation  uint32  // MVCC version counter
	Mutex       uint64  // Per-page spin lock word
	HyalineSlot uint16  // Shard index for Hyaline SMR
	LayoutTag   uint8   // Layout version tag (0 for backwards compat, 1=V1, 2=V2)
	_           uint8   // Padding to 24 bytes
}

const (
	LayoutV1 uint8 = 1
	LayoutV2 uint8 = 2
)

// EdgeTableLocator maps node IDs to their EdgeTable pages
type EdgeTableLocator struct {
	NodeID   uint64 // Node identifier
	PageSlot uint32 // Slot index in ShardedFreeList
	_        uint32 // Padding to 16 bytes
}

// EdgeTableIndex is an open-addressing hash table with linear probing
type EdgeTableIndex struct {
	table      []EdgeTableLocator // Power-of-2 size for fast modulo
	size       uint64             // Number of entries
	capacity   uint64             // Table capacity (power of 2)
	loadFactor float64            // Trigger resize at 0.75
}

// NewEdgeTableIndex creates a new index with initial capacity
func NewEdgeTableIndex(capacity uint64) *EdgeTableIndex {
	if capacity == 0 {
		capacity = 16
	}
	// Round up to next power of 2
	capacity = nextPowerOf2(capacity)
	
	// Initialize with empty locators (NodeID 0, PageSlot 0)
	// We need a way to distinguish empty slots, 
	// assuming NodeID 0 is valid or we use a separate marker.
	// For simplicity, we can reserve a specific state for empty slots.
	// Actually, an open addressing table typically needs a way to mark empty/deleted.
	
	return &EdgeTableIndex{
		table:      make([]EdgeTableLocator, capacity),
		capacity:   capacity,
		loadFactor: 0.75,
	}
}

func nextPowerOf2(n uint64) uint64 {
	if n&(n-1) == 0 {
		return n
	}
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n |= n >> 32
	n++
	return n
}
