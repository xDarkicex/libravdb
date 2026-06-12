package graph

// Edge represents a directed relationship between two nodes.
// It is explicitly 16 bytes for cache-line alignment and predictable memory layout.
type Edge struct {
	Target uint64  // 8 bytes - destination node ID
	Weight float32 // 4 bytes - edge weight for ranking/scoring
	Stamp  uint32  // 4 bytes - engine-assigned timestamp for MVCC
	Kind   uint8   // 1 byte - consumer-defined edge type namespace
	_      [3]byte // 3 bytes - explicit padding for 16-byte alignment
}
