package hnsw

import (
	"github.com/xDarkicex/libravdb/internal/util"
)

// searchLevel performs optimized search at a specific level
func (h *Index) searchLevel(query []float32, entryPoint *Node, ef int, level int) []*util.Candidate {
	// Use slice-based visited tracking for better performance with large datasets
	visited := make([]bool, len(h.nodes))
	candidates := util.NewMaxHeap(ef * 2) // Larger working set
	w := util.NewMinHeap(ef)              // Dynamic list

	// Initialize with entry point
	entryID := h.findNodeID(entryPoint)
	if entryID == ^uint32(0) || entryID >= uint32(len(visited)) {
		return []*util.Candidate{}
	}

	// Compute distance handling quantization
	distance := h.computeDistanceOptimized(query, entryPoint)
	if distance < 0 {
		return []*util.Candidate{} // Error in distance computation
	}

	candidate := &util.Candidate{ID: entryID, Distance: distance}

	candidates.PushCandidate(candidate)
	w.PushCandidate(candidate)
	visited[entryID] = true

	for w.Len() > 0 {
		current := w.PopCandidate()

		// Early termination condition - optimized for large datasets
		if candidates.Len() >= ef && current.Distance > candidates.Top().Distance {
			break
		}

		// Explore neighbors
		currentNode := h.nodes[current.ID]
		if level < len(currentNode.Links) {
			// Process neighbors in batches for better cache locality
			neighbors := currentNode.Links[level]
			for _, neighborID := range neighbors {
				if neighborID < uint32(len(visited)) && !visited[neighborID] {
					visited[neighborID] = true

					// Compute distance with optimized method
					neighborNode := h.nodes[neighborID]
					neighborDistance := h.computeDistanceOptimized(query, neighborNode)
					if neighborDistance < 0 {
						continue // Skip if distance computation failed
					}

					neighborCandidate := &util.Candidate{
						ID:       neighborID,
						Distance: neighborDistance,
					}

					// Add to candidates if it's one of the ef closest
					if candidates.Len() < ef || neighborDistance < candidates.Top().Distance {
						candidates.PushCandidate(neighborCandidate)
						w.PushCandidate(neighborCandidate)

						// Remove furthest if we exceed ef
						if candidates.Len() > ef {
							candidates.PopCandidate()
						}
					}
				}
			}
		}
	}

	// Convert to sorted slice (closest first) with pre-allocated capacity
	result := make([]*util.Candidate, 0, candidates.Len())
	for candidates.Len() > 0 {
		result = append([]*util.Candidate{candidates.PopCandidate()}, result...)
	}

	return result
}

// computeDistanceOptimized provides optimized distance computation with error handling
func (h *Index) computeDistanceOptimized(query []float32, node *Node) float32 {
	if node.CompressedVector != nil && h.quantizer != nil {
		distance, err := h.quantizer.DistanceToQuery(node.CompressedVector, query)
		if err != nil {
			// Fall back to decompressed vector
			vec, decompErr := h.quantizer.Decompress(node.CompressedVector)
			if decompErr != nil {
				return -1 // Signal error
			}
			return h.distance(query, vec)
		}
		return distance
	} else if node.Vector != nil {
		return h.distance(query, node.Vector)
	}
	return -1 // Signal error - no vector available
}
