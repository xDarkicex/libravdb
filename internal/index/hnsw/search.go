package hnsw

import (
	"github.com/xDarkicex/libravdb/internal/util"
)

// searchLevel performs optimized search at a specific level
func (h *Index) searchLevel(query []float32, entryPoint *Node, ef int, level int) []*util.Candidate {
	visited := make(map[uint32]bool)
	candidates := util.NewMaxHeap(ef * 2) // Larger working set
	w := util.NewMinHeap(ef)              // Dynamic list

	// Initialize with entry point
	entryID := h.findNodeID(entryPoint)
	if entryID == ^uint32(0) {
		return []*util.Candidate{}
	}

	distance := h.distance(query, entryPoint.Vector)
	candidate := &util.Candidate{ID: entryID, Distance: distance}

	candidates.PushCandidate(candidate)
	w.PushCandidate(candidate)
	visited[entryID] = true

	for w.Len() > 0 {
		current := w.PopCandidate()

		// Early termination condition
		if candidates.Len() >= ef && current.Distance > candidates.Top().Distance {
			break
		}

		// Explore neighbors
		currentNode := h.nodes[current.ID]
		if level < len(currentNode.Links) {
			for _, neighborID := range currentNode.Links[level] {
				if !visited[neighborID] {
					visited[neighborID] = true

					neighborDistance := h.distance(query, h.nodes[neighborID].Vector)
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

	// Convert to sorted slice (closest first)
	result := make([]*util.Candidate, 0, candidates.Len())
	for candidates.Len() > 0 {
		result = append([]*util.Candidate{candidates.PopCandidate()}, result...)
	}

	return result
}
