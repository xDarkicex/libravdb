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
	Header  EdgeTableHeader // 32 bytes
	Inline  [8]Edge         // 128 bytes (8 × 16)
	Padding [3872]byte      // 3872 = 4032 - 32 - 128; remaining for overflow or future use
}

// EdgeTableHeader contains page metadata
type EdgeTableHeader struct {
	Mutex       uint64 // Per-page spin lock word (aligned to 8 bytes)
	Overflow    uint32 // Offset to overflow chain (0 if none)
	Generation  uint32 // MVCC version counter
	PageSlot    uint32 // The ID registered in PageRegistry
	Count       uint16 // Total edge count (inline + overflow)
	InlineCap   uint16 // Always 8 for inline-first-8 layout
	HyalineSlot uint16 // Shard index for Hyaline SMR
	LayoutTag   uint8  // Layout version tag (0 for backwards compat, 1=V1, 2=V2)
	_           uint8  // Padding to 32 bytes
	_           uint32 // Padding to 32 bytes
}

const (
	LayoutV1 uint8 = 1
	LayoutV2 uint8 = 2
)
