package hnsw

import (
	"context"
	"sync/atomic"

	"github.com/xDarkicex/libravdb/internal/util"
)

// insertNode implements the optimized HNSW insertion algorithm
func (h *Index) insertNode(ctx context.Context, node *Node, nodeID uint32, searchVector []float32) error {
	scratch := h.acquireSearchScratchWithEF(h.config.EfConstruction)
	defer h.releaseSearchScratch(scratch)

	// Handle the second node (simple connection to entry point)
	if h.size.Load() == 1 {
		entryID := h.findNodeID(h.getEntryPoint())
		entryNode := h.getEntryPoint()
		if entryID != ^uint32(0) && node.Level >= 0 {
			if h.appendWithSpinlock(node, node.Links[0], entryNode.Ordinal, h.config.M, 0) {
				h.appendWithSpinlock(entryNode, entryNode.Backlinks[0], nodeID, h.config.M, 0)
			}
			if h.appendWithSpinlock(entryNode, entryNode.Links[0], nodeID, h.config.M, 0) {
				h.appendWithSpinlock(node, node.Backlinks[0], entryNode.Ordinal, h.config.M, 0)
			}
		}
		return nil
	}

	// Initialize neighbor selector if not already done
	if h.neighborSelector == nil {
		h.neighborSelector = NewNeighborSelector(h.config.M, h.config.level0LinkMultiplier())
	}

	// Phase 1: Search from top level down to node.Level + 1 with ef=1 (greedy search)
	var singleEntry [1]util.Candidate

	var queryState any
	if h.quantizer != nil {
		queryState = h.quantizer.PrepareQuery(searchVector)
	}

	maxLevel := h.getMaxLevel()
	entryPoints := h.appendFallbackEntryPoint(nil, searchVector, h.getEntryPoint(), &singleEntry)

	for level := maxLevel; level > node.Level; level-- {
		currentNode := h.pickEntryNodeValues(entryPoints)
		if currentNode == nil {
			currentNode = h.getEntryPoint()
		}
		greedy, ok, err := h.greedySearchLevelValue(ctx, searchVector, currentNode, level, queryState)
		if err != nil {
			return err
		}
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
	currentNode := h.pickEntryNodeValues(entryPoints)
	startLevel := min(node.Level, maxLevel)
	for level := startLevel; level >= 0; level-- {
		// Search for efConstruction candidates
		if currentNode == nil {
			currentNode = h.getEntryPoint()
		}
		selected, err := h.searchAndSelectForConstructionWithScratch(
			searchVector,
			currentNode,
			h.config.EfConstruction,
			level,
			levelConstructionMaxLinks(h.config.M, level),
			scratch,
			queryState,
		)
		if err != nil {
			return err
		}
		if len(selected) == 0 {
			selected = h.appendFallbackEntryPoint(selected[:0], searchVector, currentNode, &singleEntry)
		}

		// Connect bidirectionally. The selected links already passed the
		// diversity heuristic; existing neighbor link lists repair themselves
		// at the point of overflow instead of via a separate full prune pass.
		h.connectBidirectionalOptimizedValues(nodeID, selected, level)

		if len(selected) > 0 {
			currentNode = h.nodes.Get(selected[0].ID)
		}
	}

	return nil
}

func (h *Index) pickEntryNodeValues(entryPoints []util.Candidate) *Node {
	if len(entryPoints) == 0 {
		return nil
	}

	entryID := entryPoints[0].ID
	if int(entryID) >= h.nodes.Len() {
		return nil
	}
	return h.nodes.Get(entryID)
}

func (h *Index) fallbackEntryPoints(searchVector []float32, node *Node) []*util.Candidate {
	if node == nil {
		return nil
	}

	entryID := h.findNodeID(node)
	if entryID == ^uint32(0) || int(entryID) >= h.nodes.Len() {
		return nil
	}

	distance, err := h.computeDistanceOptimized(searchVector, node, nil)
	if err != nil {
		return nil
	}
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
	if entryID == ^uint32(0) || int(entryID) >= h.nodes.Len() {
		return nil
	}

	distance, err := h.computeDistanceOptimized(searchVector, node, nil)
	if err != nil {
		return nil
	}
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
		h.neighborSelector = NewNeighborSelector(h.config.M, h.config.level0LinkMultiplier())
	}
	return h.neighborSelector.SelectNeighborsOptimized(queryVector, candidates, level, h)
}

// Legacy method for backward compatibility - delegates to optimized version
func (h *Index) connectBidirectional(nodeID uint32, neighbors []*util.Candidate, level int) {
	node := h.nodes.Get(nodeID)

	for _, neighbor := range neighbors {
		if int(neighbor.ID) >= h.nodes.Len() {
			continue
		}
		h.appendWithSpinlock(node, node.Links[level], neighbor.ID, h.config.M, level)

		// Add backlink to neighbor
		neighborNode := h.nodes.Get(neighbor.ID)
		if neighborNode != nil && level < (neighborNode.Level+1) {
			h.appendWithSpinlock(neighborNode, neighborNode.Backlinks[level], nodeID, h.config.M, level)
		}
	}

	for _, neighbor := range neighbors {
		if int(neighbor.ID) >= h.nodes.Len() {
			continue
		}
		neighborNode := h.nodes.Get(neighbor.ID)
		if neighborNode == nil || level >= (neighborNode.Level+1) {
			continue
		}

		if !h.appendWithSpinlock(neighborNode, neighborNode.Links[level], nodeID, h.config.M, level) {
			h.pruneNeighborConnections([]*util.Candidate{neighbor}, level)
			h.appendWithSpinlock(neighborNode, neighborNode.Links[level], nodeID, h.config.M, level)
		}

		// Add backlink to node
		if node != nil && level < (node.Level+1) {
			h.appendWithSpinlock(node, node.Backlinks[level], neighbor.ID, h.config.M, level)
		}
	}
}

func (h *Index) connectBidirectionalOptimizedValues(nodeID uint32, neighbors []util.Candidate, level int) {
	node := h.nodes.Get(nodeID)

	for _, neighbor := range neighbors {
		if int(neighbor.ID) >= h.nodes.Len() {
			continue
		}
		h.appendWithSpinlock(node, node.Links[level], neighbor.ID, h.config.M, level)

		// Add backlink to neighbor
		neighborNode := h.nodes.Get(neighbor.ID)
		if neighborNode != nil && level < (neighborNode.Level+1) {
			h.appendWithSpinlock(neighborNode, neighborNode.Backlinks[level], nodeID, h.config.M, level)
		}
	}
	if node != nil && level < (node.Level+1) {
		atomic.StoreUint32(&node.LinkHeuristic[level], atomic.LoadUint32(&node.LinkCounts[level]))
	}

	for _, neighbor := range neighbors {
		if int(neighbor.ID) >= h.nodes.Len() {
			continue
		}
		neighborNode := h.nodes.Get(neighbor.ID)
		if neighborNode == nil || level >= (neighborNode.Level+1) {
			continue
		}

		accepted := h.neighborSelector.connectLinkWithHeuristic(neighbor.ID, nodeID, level, h)

		// Add backlink to node
		if accepted && node != nil && level < (node.Level+1) {
			h.appendWithSpinlock(node, node.Backlinks[level], neighbor.ID, h.config.M, level)
		}
	}
}

// pruneNeighborConnectionsOptimized ensures neighbors don't exceed maxM connections using optimized algorithm
func (h *Index) pruneNeighborConnectionsOptimized(neighbors []*util.Candidate, level int) {
	if h.neighborSelector == nil {
		h.neighborSelector = NewNeighborSelector(h.config.M, h.config.level0LinkMultiplier())
	}

	pruneThreshold := linkArrayCapacity(h.config.M, level) - 1
	for _, neighbor := range neighbors {
		if int(neighbor.ID) >= h.nodes.Len() {
			continue
		}
		neighborNode := h.nodes.Get(neighbor.ID)
		if neighborNode == nil || level >= (neighborNode.Level+1) || len(h.getNodeLinks(neighborNode, level)) <= pruneThreshold {
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
		h.neighborSelector = NewNeighborSelector(h.config.M, h.config.level0LinkMultiplier())
	}

	pruneThreshold := linkArrayCapacity(h.config.M, level) - 1
	for _, neighbor := range neighbors {
		if int(neighbor.ID) >= h.nodes.Len() {
			continue
		}
		neighborNode := h.nodes.Get(neighbor.ID)
		if neighborNode == nil || level >= (neighborNode.Level+1) || len(h.getNodeLinks(neighborNode, level)) <= pruneThreshold {
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

func levelConstructionMaxLinks(base int, level int) int {
	if level == 0 {
		return linkArrayCapacity(base, level)
	}
	return levelMaxLinks(base, level)
}

func levelOverflowSlack(maxLinks int) int {
	return max(4, maxLinks/4)
}

func (h *Index) appendUniqueLink(node *Node, maxLinks int, level int, linkID uint32) bool {
	if node == nil || level >= (node.Level+1) {
		return false
	}
	return h.appendWithSpinlock(node, node.Links[level], linkID, h.config.M, level)
}

func (h *Index) manualConnect(nodeID uint32, linkID uint32, level int) bool {
	node := h.nodes.Get(nodeID)
	return h.appendWithSpinlock(node, node.Links[level], linkID, h.config.M, level)
}
