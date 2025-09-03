package hnsw

// Node represents a single node in the HNSW graph
type Node struct {
	ID       string                 // User-provided ID
	Vector   []float32              // The vector data
	Level    int                    // Maximum level this node exists in
	Links    [][]uint32             // Adjacency lists for each level
	Metadata map[string]interface{} // User metadata
}
