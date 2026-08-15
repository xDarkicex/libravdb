package graph

// Edge represents a directed relationship between two nodes.
//
// PropertyRef is an opaque reference into the property chain owned by the
// EdgeTablePage that contains this edge. It is zero when the edge has no
// arbitrary properties. The reference is deliberately part of the physical
// edge record so properties follow the existing node-owned page/WAL/epoch
// machinery rather than living in a sidecar map.
type Edge struct {
	Target uint64  // 8 bytes - destination node ID
	Weight float32 // 4 bytes - edge weight for ranking/scoring
	Stamp  uint32  // 4 bytes: bits [31:24]=Kind, [23:0]=timestamp
	// high 32 bits: property page-chain root registry ID
	// low 32 bits: logical byte offset of the length-prefixed property value
	PropertyRef uint64
}

func (e *Edge) GetKind() uint8    { return uint8(e.Stamp >> 24) }
func (e *Edge) SetKind(k uint8)   { e.Stamp = (e.Stamp & 0x00FFFFFF) | (uint32(k) << 24) }
func (e *Edge) GetStamp() uint32  { return e.Stamp & 0x00FFFFFF }
func (e *Edge) SetStamp(s uint32) { e.Stamp = (e.Stamp & 0xFF000000) | (s & 0x00FFFFFF) }
