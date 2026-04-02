package hnsw

// Node represents a single node in the HNSW graph.
// Canonical vectors and metadata are owned outside the graph.
type Node struct {
	Ordinal          uint32
	Level            int
	Links            [][]uint32
	CompressedVector []byte
}
