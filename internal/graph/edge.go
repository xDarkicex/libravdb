package graph

// Edge represents a directed relationship between two nodes.
// It is explicitly 16 bytes for cache-line alignment and predictable memory layout.
type Edge struct {
	Target uint64  // 8 bytes - destination node ID
	Weight float32 // 4 bytes - edge weight for ranking/scoring
	Stamp  uint32  // 4 bytes: bits [31:24]=Kind, [23:0]=timestamp
}

func (e *Edge) GetKind() uint8    { return uint8(e.Stamp >> 24) }
func (e *Edge) SetKind(k uint8)   { e.Stamp = (e.Stamp & 0x00FFFFFF) | (uint32(k) << 24) }
func (e *Edge) GetStamp() uint32  { return e.Stamp & 0x00FFFFFF }
func (e *Edge) SetStamp(s uint32) { e.Stamp = (e.Stamp & 0xFF000000) | (s & 0x00FFFFFF) }
