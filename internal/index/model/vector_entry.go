package model

// VectorEntry is the shared ingress record for vector indexes.
//
// It is caller-owned and may contain Go pointers. Index implementations must
// not retain the record itself. Any state retained after Insert or BatchInsert
// returns must first be copied into allocator-owned storage.
type VectorEntry struct {
	Metadata    map[string]interface{}
	ID          string
	Vector      []float32
	Version     uint64
	Ordinal     uint32
	GraphNodeID uint64
}
