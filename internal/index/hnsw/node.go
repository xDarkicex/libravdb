package hnsw

// Node represents a single node in the HNSW graph.
// Canonical vectors and metadata are owned outside the graph.
type Node struct {
	Links            [][]uint32
	Backlinks        [][]uint32
	CompressedVector []byte
	Level            int
	Ordinal          uint32
	Slot             uint32
}
