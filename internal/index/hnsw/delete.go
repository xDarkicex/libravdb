package hnsw

import (
	"context"
	"fmt"

	"github.com/xDarkicex/libravdb/internal/util"
)

// deleteNode removes a vector from the HNSW index (internal implementation)
func (h *Index) deleteNode(ctx context.Context, id string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.size == 0 {
		return fmt.Errorf("cannot delete from empty index")
	}
	nodeID, node := h.findNodeByID(id)
	if nodeID == ^uint32(0) {
		return fmt.Errorf("node with ID '%s' not found", id)
	}
	return h.deleteNodeLocked(ctx, nodeID, node, id)
}

func (h *Index) deleteNodeByOrdinal(ctx context.Context, ordinal uint32) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.size == 0 {
		return fmt.Errorf("cannot delete from empty index")
	}
	if ordinal >= uint32(len(h.nodes)) || h.nodes[ordinal] == nil {
		return fmt.Errorf("node with ordinal %d not found", ordinal)
	}
	return h.deleteNodeLocked(ctx, ordinal, h.nodes[ordinal], h.ordinalToID[ordinal])
}

func (h *Index) deleteNodeLocked(ctx context.Context, nodeID uint32, node *Node, id string) error {
	// Handle special case: deleting the only node
	if h.size == 1 {
		h.deleteStoredVector(node)
		h.nodes = h.nodes[:0]
		h.entryPoint = nil
		h.maxLevel = 0
		h.size = 0
		delete(h.idToIndex, id)
		delete(h.ordinalToID, nodeID)
		h.entryPointCandidates = h.entryPointCandidates[:0]
		return nil
	}

	// Remove all connections to this node from other nodes
	if err := h.removeAllConnections(ctx, nodeID, node); err != nil {
		return fmt.Errorf("failed to remove connections: %w", err)
	}

	// Handle entry point replacement if necessary
	if err := h.handleEntryPointReplacement(nodeID, node); err != nil {
		return fmt.Errorf("failed to handle entry point replacement: %w", err)
	}

	h.deleteStoredVector(node)

	// Remove the node from data structures
	h.removeNodeFromIndex(nodeID, id)

	h.size--
	return nil
}

// findNodeByID finds a node by its ID using O(1) map lookup
func (h *Index) findNodeByID(id string) (uint32, *Node) {
	if idx, exists := h.idToIndex[id]; exists {
		if idx < uint32(len(h.nodes)) && h.nodes[idx] != nil {
			return idx, h.nodes[idx]
		}
		delete(h.idToIndex, id)
	}
	return ^uint32(0), nil
}

// removeAllConnections removes all bidirectional connections to the target node
func (h *Index) removeAllConnections(ctx context.Context, targetID uint32, targetNode *Node) error {
	// For each level where the target node exists
	for level := 0; level <= targetNode.Level; level++ {
		// Get all neighbors of the target node at this level
		neighbors := make([]uint32, len(targetNode.Links[level]))
		copy(neighbors, targetNode.Links[level])

		// Remove target from every node that still references it. This repairs
		// older asymmetric graph state where incoming edges may exist without a
		// matching outgoing edge on the deleted node.
		neighbors = appendUniqueIDs(neighbors, h.removeIncomingConnections(targetID, level)...)

		// Reconnect the neighbors to maintain graph connectivity
		if err := h.reconnectNeighborsOptimized(ctx, neighbors, level, targetNode); err != nil {
			return fmt.Errorf("failed to reconnect neighbors at level %d: %w", level, err)
		}
	}

	return nil
}

func (h *Index) removeIncomingConnections(targetID uint32, level int) []uint32 {
	affected := make([]uint32, 0)
	for nodeID, node := range h.nodes {
		if node == nil || uint32(nodeID) == targetID || level >= len(node.Links) {
			continue
		}

		links := node.Links[level]
		if len(links) == 0 {
			continue
		}

		newLinks := links[:0]
		removed := false
		for _, linkID := range links {
			if linkID == targetID {
				removed = true
				continue
			}
			newLinks = append(newLinks, linkID)
		}
		if removed {
			node.Links[level] = newLinks
			affected = append(affected, uint32(nodeID))
		}
	}
	return affected
}

// removeConnection removes a specific connection between two nodes at a given level
func (h *Index) removeConnection(fromID, toID uint32, level int) {
	fromNode := h.nodes[fromID]
	if fromNode == nil || level >= len(fromNode.Links) {
		return
	}

	// Find and remove the connection efficiently
	links := fromNode.Links[level]
	for i, linkID := range links {
		if linkID == toID {
			// Remove by swapping with last element and truncating
			links[i] = links[len(links)-1]
			fromNode.Links[level] = links[:len(links)-1]
			break
		}
	}
}

// reconnectNeighborsOptimized attempts to reconnect neighbors with precomputed distances
func (h *Index) reconnectNeighborsOptimized(ctx context.Context, neighbors []uint32, level int, deletedNode *Node) error {
	if len(neighbors) < 2 {
		return nil // Nothing to reconnect
	}

	// Check for cancellation before expensive reconnection
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	maxM := h.config.M
	if level == 0 {
		maxM = maxM * 2 // Level 0 can have more connections
	}

	// Precompute all distances between valid neighbors
	validNeighbors := make([]uint32, 0, len(neighbors))
	for _, neighborID := range neighbors {
		if neighborID < uint32(len(h.nodes)) && h.nodes[neighborID] != nil {
			validNeighbors = append(validNeighbors, neighborID)
		}
	}

	if len(validNeighbors) < 2 {
		return nil
	}

	// Precompute distance matrix
	distanceCache := make(map[[2]uint32]float32)
	for i, id1 := range validNeighbors {
		// Check for cancellation during distance computation
		if i%10 == 0 { // Check every 10 iterations to avoid too much overhead
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
		}

		for j, id2 := range validNeighbors {
			if i >= j {
				continue // Only compute upper triangle
			}

			node1 := h.nodes[id1]
			node2 := h.nodes[id2]
			if node1 == nil || node2 == nil {
				continue
			}

			dist, err := h.computeDistance(nil, nil, node1, node2)
			if err != nil {
				continue
			}
			distanceCache[[2]uint32{id1, id2}] = dist
			distanceCache[[2]uint32{id2, id1}] = dist
		}
	}

	// For each neighbor, try to connect it to other neighbors
	for _, neighborID := range validNeighbors {
		// Check for cancellation between neighbors
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		neighborNode := h.nodes[neighborID]
		if neighborNode == nil || level >= len(neighborNode.Links) {
			continue
		}

		// Current number of connections for this neighbor
		currentConnections := len(neighborNode.Links[level])
		if currentConnections >= maxM {
			continue // Already at capacity
		}

		// Find potential new connections from remaining neighbors
		candidates := make([]*util.Candidate, 0)
		for _, otherID := range validNeighbors {
			if neighborID == otherID || h.nodes[otherID] == nil {
				continue
			}

			// Check if already connected
			if h.hasConnection(neighborID, otherID, level) {
				continue
			}

			// Use precomputed distance
			dist := distanceCache[[2]uint32{neighborID, otherID}]
			candidates = append(candidates, &util.Candidate{
				ID:       otherID,
				Distance: dist,
			})
		}

		if len(candidates) == 0 {
			continue
		}

		// Select best candidates to connect
		availableSlots := maxM - currentConnections
		numToSelect := min(len(candidates), availableSlots)

		// Use distance-based selection
		selected := h.selectBestCandidatesByDistance(candidates, numToSelect)

		// Create bidirectional connections
		for _, candidate := range selected {
			h.createBidirectionalConnection(neighborID, candidate.ID, level)
		}
	}

	return nil
}

// hasConnection checks if two nodes are connected at a given level
func (h *Index) hasConnection(nodeID1, nodeID2 uint32, level int) bool {
	if nodeID1 >= uint32(len(h.nodes)) || nodeID2 >= uint32(len(h.nodes)) {
		return false
	}

	node1 := h.nodes[nodeID1]
	if node1 == nil || level >= len(node1.Links) {
		return false
	}

	for _, linkID := range node1.Links[level] {
		if linkID == nodeID2 {
			return true
		}
	}
	return false
}

// selectBestCandidatesByDistance selects candidates with smallest distances
func (h *Index) selectBestCandidatesByDistance(candidates []*util.Candidate, numToSelect int) []*util.Candidate {
	if len(candidates) <= numToSelect {
		return candidates
	}

	// Sort candidates by distance (bubble sort for small arrays, or use heap for larger ones)
	for i := 0; i < len(candidates)-1; i++ {
		for j := 0; j < len(candidates)-i-1; j++ {
			if candidates[j].Distance > candidates[j+1].Distance {
				candidates[j], candidates[j+1] = candidates[j+1], candidates[j]
			}
		}
	}

	return candidates[:numToSelect]
}

// createBidirectionalConnection creates a bidirectional connection between two nodes
func (h *Index) createBidirectionalConnection(nodeID1, nodeID2 uint32, level int) {
	// Add nodeID2 to nodeID1's connections
	node1 := h.nodes[nodeID1]
	if node1 != nil && level < len(node1.Links) {
		appendUniqueLink(node1, levelMaxLinks(h.config.M, level), level, nodeID2)
	}

	// Add nodeID1 to nodeID2's connections
	node2 := h.nodes[nodeID2]
	if node2 != nil && level < len(node2.Links) {
		appendUniqueLink(node2, levelMaxLinks(h.config.M, level), level, nodeID1)
	}
}

// handleEntryPointReplacement handles the case where the deleted node is the entry point
func (h *Index) handleEntryPointReplacement(deletedID uint32, deletedNode *Node) error {
	// Only need to replace if the deleted node is the entry point
	if h.entryPoint != deletedNode {
		// Remove from entry point candidates if present
		h.removeFromEntryPointCandidates(deletedID)
		return nil
	}

	// Try to find replacement from entry point candidates first
	newEntryPoint := h.findBestEntryPointCandidate(deletedID)
	if newEntryPoint != nil {
		h.entryPoint = newEntryPoint
		h.maxLevel = newEntryPoint.Level
		return nil
	}

	// Fallback: scan all nodes for highest level
	var fallbackEntryPoint *Node
	newMaxLevel := -1

	for i, node := range h.nodes {
		if node == nil || uint32(i) == deletedID {
			continue
		}

		if node.Level > newMaxLevel {
			newMaxLevel = node.Level
			fallbackEntryPoint = node
		}
	}

	if fallbackEntryPoint == nil {
		return fmt.Errorf("could not find replacement entry point")
	}

	h.entryPoint = fallbackEntryPoint
	h.maxLevel = newMaxLevel

	// Rebuild entry point candidates list
	h.rebuildEntryPointCandidates()

	return nil
}

// findBestEntryPointCandidate finds the best entry point from candidates list
func (h *Index) findBestEntryPointCandidate(excludeID uint32) *Node {
	var bestNode *Node
	bestLevel := -1

	for _, candidateID := range h.entryPointCandidates {
		if candidateID == excludeID || candidateID >= uint32(len(h.nodes)) {
			continue
		}

		node := h.nodes[candidateID]
		if node != nil && node.Level > bestLevel {
			bestLevel = node.Level
			bestNode = node
		}
	}

	return bestNode
}

// removeFromEntryPointCandidates removes a node from the entry point candidates list
func (h *Index) removeFromEntryPointCandidates(nodeID uint32) {
	for i, candidateID := range h.entryPointCandidates {
		if candidateID == nodeID {
			// Remove by swapping with last element
			h.entryPointCandidates[i] = h.entryPointCandidates[len(h.entryPointCandidates)-1]
			h.entryPointCandidates = h.entryPointCandidates[:len(h.entryPointCandidates)-1]
			break
		}
	}
}

// rebuildEntryPointCandidates rebuilds the entry point candidates list
func (h *Index) rebuildEntryPointCandidates() {
	h.entryPointCandidates = h.entryPointCandidates[:0] // Clear existing

	// Add nodes with level >= threshold (e.g., level 2+) as candidates
	levelThreshold := 2
	for i, node := range h.nodes {
		if node != nil && node.Level >= levelThreshold {
			h.entryPointCandidates = append(h.entryPointCandidates, uint32(i))
		}
	}
}

// removeNodeFromIndex removes a node from all index data structures
func (h *Index) removeNodeFromIndex(nodeID uint32, id string) {
	delete(h.idToIndex, id)
	delete(h.ordinalToID, nodeID)

	h.removeFromEntryPointCandidates(nodeID)

	if nodeID < uint32(len(h.nodes)) {
		h.nodes[nodeID].Links = nil
		h.nodes[nodeID].CompressedVector = nil
		h.nodes[nodeID] = nil
	}
}

func (h *Index) deleteStoredVector(node *Node) {
	if node == nil || h.provider != nil || h.rawVectorStore == nil {
		return
	}
	_ = h.rawVectorStore.Delete(VectorRef{
		Kind:  VectorEncodingRaw,
		Slot:  node.Ordinal,
		Bytes: uint32(h.config.Dimension * 4),
		Valid: true,
	})
}

func appendUniqueIDs(dst []uint32, ids ...uint32) []uint32 {
	for _, id := range ids {
		seen := false
		for _, existing := range dst {
			if existing == id {
				seen = true
				break
			}
		}
		if !seen {
			dst = append(dst, id)
		}
	}
	return dst
}
