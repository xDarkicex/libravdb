package hnsw

import (
	"github.com/xDarkicex/libravdb/internal/quant"
)

// Node represents a single node in the HNSW graph
type Node struct {
	ID               string                 // User-provided ID
	Vector           []float32              // The original vector data (may be nil if quantized)
	CompressedVector []byte                 // Quantized vector data (nil if not quantized)
	Level            int                    // Maximum level this node exists in
	Links            [][]uint32             // Adjacency lists for each level
	Metadata         map[string]interface{} // User metadata
}

// GetVector returns the vector for distance computation
// If quantized, it decompresses the vector; otherwise returns the original
func (n *Node) GetVector(quantizer quant.Quantizer) ([]float32, error) {
	if n.CompressedVector != nil && quantizer != nil {
		return quantizer.Decompress(n.CompressedVector)
	}
	return n.Vector, nil
}
