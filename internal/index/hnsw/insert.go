package hnsw

import (
	"context"

	"github.com/xDarkicex/libravdb/internal/util"
)

// insertNode implements the optimized HNSW insertion algorithm
func (h *Index) insertNode(ctx context.Context, node *Node, nodeID uint32, searchVector []float32) error {
	// Handle the second node (simple connection to entry point)
	if h.size == 1 {
		entryID := h.findNodeID(h.entryPoint)
		if entryID != ^uint32(0) && node.Level >= 0 {
			appendUniqueLink(node, levelMaxLinks(h.config.M, 0), 0, entryID)
			appendUniqueLink(h.entryPoint, levelMaxLinks(h.config.M, 0), 0, nodeID)
		}
		return nil
	}

	// Initialize neighbor selector if not already done
	if h.neighborSelector == nil {
		h.neighborSelector = NewNeighborSelector(h.config.M, 2.0)
	}

	// Phase 1: Search from top level down to node.Level + 1 with ef=1 (greedy search)
	var singleEntry [1]util.Candidate
	entryPoints := h.appendFallbackEntryPoint(nil, searchVector, h.entryPoint, &singleEntry)

	for level := h.maxLevel; level > node.Level; level-- {
		currentNode := h.pickEntryNodeValues(entryPoints)
		if currentNode == nil {
			currentNode = h.entryPoint
		}
		greedy, ok := h.greedySearchLevelValue(searchVector, currentNode, level)
		if ok {
			entryPoints = singleEntry[:1]
			entryPoints[0] = greedy
		} else {
			entryPoints = h.appendFallbackEntryPoint(entryPoints[:0], searchVector, currentNode, &singleEntry)
		}
	}

	// Phase 2: From node.Level down to 0, search with efConstruction and connect.
	// Keep one scratch context for the whole insertion so we can reuse the
	// working-set buffers across levels.
	scratch := h.acquireSearchScratch()
	defer h.releaseSearchScratch(scratch)

	currentNode := h.pickEntryNodeValues(entryPoints)
	for level := node.Level; level >= 0; level-- {
		// Search for efConstruction candidates
		if currentNode == nil {
			currentNode = h.entryPoint
		}
		selected := h.searchAndSelectForConstructionWithScratch(
			searchVector,
			currentNode,
			h.config.EfConstruction,
			level,
			levelMaxLinks(h.config.M, level),
			scratch,
		)
		if len(selected) == 0 {
			selected = h.appendFallbackEntryPoint(selected[:0], searchVector, currentNode, &singleEntry)
		}

		// Connect bidirectionally
		h.connectBidirectionalOptimizedValues(nodeID, selected, level)

		// Prune connections of neighbors if they exceed maxM
		h.pruneNeighborConnectionsOptimizedValues(selected, level)

		if len(selected) > 0 {
			currentNode = h.nodes[selected[0].ID]
		}
	}

	return nil
}

func (h *Index) pickEntryNodeValues(entryPoints []util.Candidate) *Node {
	if len(entryPoints) == 0 {
		return nil
	}

	entryID := entryPoints[0].ID
	if int(entryID) >= len(h.nodes) {
		return nil
	}
	return h.nodes[entryID]
}

func (h *Index) fallbackEntryPoints(searchVector []float32, node *Node) []*util.Candidate {
	if node == nil {
		return nil
	}

	entryID := h.findNodeID(node)
	if entryID == ^uint32(0) || int(entryID) >= len(h.nodes) {
		return nil
	}

	distance := h.computeDistanceOptimized(searchVector, node)
	if distance < 0 {
		distance = 0
	}

	return []*util.Candidate{{
		ID:       entryID,
		Distance: distance,
	}}
}

func (h *Index) fallbackEntryPointsValues(searchVector []float32, node *Node) []util.Candidate {
	var singleEntry [1]util.Candidate
	return h.appendFallbackEntryPoint(nil, searchVector, node, &singleEntry)
}

func (h *Index) appendFallbackEntryPoint(dst []util.Candidate, searchVector []float32, node *Node, single *[1]util.Candidate) []util.Candidate {
	if node == nil {
		return nil
	}

	entryID := h.findNodeID(node)
	if entryID == ^uint32(0) || int(entryID) >= len(h.nodes) {
		return nil
	}

	distance := h.computeDistanceOptimized(searchVector, node)
	if distance < 0 {
		distance = 0
	}

	if cap(dst) == 0 {
		dst = single[:0]
	} else {
		dst = dst[:0]
	}
	dst = append(dst, util.Candidate{
		ID:       entryID,
		Distance: distance,
	})
	return dst
}

// Legacy method for backward compatibility - delegates to optimized version
func (h *Index) selectNeighborsHeuristic(queryVector []float32, candidates []*util.Candidate, level int) []*util.Candidate {
	if h.neighborSelector == nil {
		h.neighborSelector = NewNeighborSelector(h.config.M, 2.0)
	}
	return h.neighborSelector.SelectNeighborsOptimized(queryVector, candidates, level, h)
}

// Legacy method for backward compatibility - delegates to optimized version
func (h *Index) connectBidirectional(nodeID uint32, neighbors []*util.Candidate, level int) {
	h.connectBidirectionalOptimized(nodeID, neighbors, level)
}

// connectBidirectionalOptimized creates bidirectional connections with better memory management
func (h *Index) connectBidirectionalOptimized(nodeID uint32, neighbors []*util.Candidate, level int) {
	node := h.nodes[nodeID]
	maxLinks := levelMaxLinks(h.config.M, level)
	nodeLinks := ensureLinkCapacity(node.Links[level], len(node.Links[level])+len(neighbors), maxLinks)
	for _, neighbor := range neighbors {
		nodeLinks = append(nodeLinks, neighbor.ID)
	}
	node.Links[level] = nodeLinks

	for _, neighbor := range neighbors {
		neighborNode := h.nodes[neighbor.ID]
		if level >= len(neighborNode.Links) {
			continue
		}
		neighborLinks := ensureLinkCapacity(neighborNode.Links[level], len(neighborNode.Links[level])+1, maxLinks)
		neighborNode.Links[level] = append(neighborLinks, nodeID)
	}
}

func (h *Index) connectBidirectionalOptimizedValues(nodeID uint32, neighbors []util.Candidate, level int) {
	node := h.nodes[nodeID]
	maxLinks := levelMaxLinks(h.config.M, level)
	nodeLinks := ensureLinkCapacity(node.Links[level], len(node.Links[level])+len(neighbors), maxLinks)
	for _, neighbor := range neighbors {
		nodeLinks = append(nodeLinks, neighbor.ID)
	}
	node.Links[level] = nodeLinks

	for _, neighbor := range neighbors {
		neighborNode := h.nodes[neighbor.ID]
		if level >= len(neighborNode.Links) {
			continue
		}
		neighborLinks := ensureLinkCapacity(neighborNode.Links[level], len(neighborNode.Links[level])+1, maxLinks)
		neighborNode.Links[level] = append(neighborLinks, nodeID)
	}
}

// pruneNeighborConnectionsOptimized ensures neighbors don't exceed maxM connections using optimized algorithm
func (h *Index) pruneNeighborConnectionsOptimized(neighbors []*util.Candidate, level int) {
	if h.neighborSelector == nil {
		h.neighborSelector = NewNeighborSelector(h.config.M, 2.0)
	}

	maxLinks := levelMaxLinks(h.config.M, level)
	pruneThreshold := maxLinks + levelOverflowSlack(maxLinks)
	for _, neighbor := range neighbors {
		if int(neighbor.ID) >= len(h.nodes) {
			continue
		}
		neighborNode := h.nodes[neighbor.ID]
		if neighborNode == nil || level >= len(neighborNode.Links) || len(neighborNode.Links[level]) <= pruneThreshold {
			continue
		}
		if err := h.neighborSelector.PruneConnections(neighbor.ID, level, h); err != nil {
			// Log error but continue - this is not critical for correctness
			continue
		}
	}
}

func (h *Index) pruneNeighborConnectionsOptimizedValues(neighbors []util.Candidate, level int) {
	if h.neighborSelector == nil {
		h.neighborSelector = NewNeighborSelector(h.config.M, 2.0)
	}

	maxLinks := levelMaxLinks(h.config.M, level)
	pruneThreshold := maxLinks + levelOverflowSlack(maxLinks)
	for _, neighbor := range neighbors {
		if int(neighbor.ID) >= len(h.nodes) {
			continue
		}
		neighborNode := h.nodes[neighbor.ID]
		if neighborNode == nil || level >= len(neighborNode.Links) || len(neighborNode.Links[level]) <= pruneThreshold {
			continue
		}
		if err := h.neighborSelector.PruneConnections(neighbor.ID, level, h); err != nil {
			continue
		}
	}
}

// Legacy methods for backward compatibility
// pruneNeighborConnections ensures neighbors don't exceed maxM connections
func (h *Index) pruneNeighborConnections(neighbors []*util.Candidate, level int) {
	h.pruneNeighborConnectionsOptimized(neighbors, level)
}

func levelMaxLinks(base int, level int) int {
	if level == 0 {
		return base * 2
	}
	return base
}

func levelOverflowSlack(maxLinks int) int {
	return max(4, maxLinks/4)
}

func appendUniqueLink(node *Node, maxLinks int, level int, linkID uint32) bool {
	if node == nil || level >= len(node.Links) {
		return false
	}

	links := node.Links[level]
	for _, existing := range links {
		if existing == linkID {
			return false
		}
	}

	if cap(links) < len(links)+1 {
		newCap := len(links) + max(maxLinks, 1)
		newLinks := make([]uint32, len(links), newCap)
		copy(newLinks, links)
		links = newLinks
	}

	links = append(links, linkID)
	node.Links[level] = links
	return true
}

func ensureLinkCapacity(links []uint32, needed int, maxLinks int) []uint32 {
	if cap(links) >= needed {
		return links
	}
	newCap := needed + levelOverflowSlack(maxLinks)
	if grown := cap(links) * 2; grown > newCap {
		newCap = grown
	}
	if newCap < len(links)+max(maxLinks, 1) {
		newCap = len(links) + max(maxLinks, 1)
	}
	newLinks := make([]uint32, len(links), newCap)
	copy(newLinks, links)
	return newLinks
}
