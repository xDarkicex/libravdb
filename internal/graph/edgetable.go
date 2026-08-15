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
	Inline  [8]Edge         // 192 bytes (8 × 24)
	Padding [3808]byte      // 3808 = 4032 - 32 - 192; overflow edge records
}

// EdgeTableHeader contains page metadata
type EdgeTableHeader struct {
	Mutex        uint64 // Per-page spin lock word (aligned to 8 bytes)
	Overflow     uint32 // Offset to overflow chain (0 if none)
	PropertyRoot uint32 // Root of the node-owned edge-property byte chain
	Generation   uint32 // MVCC version counter
	PageSlot     uint32 // The ID registered in PageRegistry
	Count        uint16 // Total edge count (inline + overflow)
	InlineCap    uint16 // Always 8 for inline-first-8 layout
	HyalineSlot  uint16 // Shard index for Hyaline SMR
	LayoutTag    uint8  // Layout version tag (0 for backwards compat, 1=V1, 2=V2, 3=properties)
	_            uint8  // Padding to 32 bytes
}

const (
	LayoutV1 uint8 = 1
	LayoutV2 uint8 = 2
	LayoutV3 uint8 = 3
)

// EdgePropertyPage stores the versioned property bytes for one node's edge
// table. It uses the same 4032-byte user area and allocator as edge pages, but
// has its own registry identity and is linked from EdgeTableHeader.PropertyRoot.
// Data is an append-only logical byte stream; an edge reference points at a
// four-byte length prefix in that stream.
type EdgePropertyPage struct {
	Next uint32
	Used uint32
	_    [8]byte
	Data [4016]byte
}

const (
	EdgePageInlineCapacity   = 8
	EdgePageOverflowCapacity = 158
	EdgePageCapacity         = EdgePageInlineCapacity + EdgePageOverflowCapacity
	EdgePropertyPageDataSize = len(EdgePropertyPage{}.Data)
)
