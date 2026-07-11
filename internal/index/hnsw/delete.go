package hnsw

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"slices"
	"sync/atomic"

	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/memory"
)

// deleteNode removes a vector from the HNSW index (internal implementation)
func (h *Index) deleteNode(ctx context.Context, id string) error {
	if h.size.Load() == 0 {
		return fmt.Errorf("cannot delete from empty index")
	}
	nodeID, node := h.findNodeByID(id)
	if nodeID == ^uint32(0) {
		return fmt.Errorf("node with ID '%s': %w", id, util.ErrNotFound)
	}
	return h.deleteNodeInternal(ctx, nodeID, node, id)
}

func (h *Index) deleteNodeByOrdinal(ctx context.Context, ordinal uint32) error {
	if h.size.Load() == 0 {
		return fmt.Errorf("cannot delete from empty index")
	}
	if ordinal >= uint32(h.nodes.Len()) || h.nodes.Get(ordinal) == nil {
		return fmt.Errorf("node with ordinal %d: %w", ordinal, util.ErrNotFound)
	}
	return h.deleteNodeInternal(ctx, ordinal, h.nodes.Get(ordinal), h.ordinalToID.Get(ordinal))
}

func (h *Index) deleteNodeInternal(ctx context.Context, nodeID uint32, node *Node, id string) error {
	// Handle special case: deleting the only node
	if h.size.Load() == 1 {
		h.deleteStoredVector(node)
		h.retireNodeStorage(nodeID, node)
		h.globalState.Store(0)
		if id != "" {
			h.idToIndex.Delete(hashID(id))
		}
		h.ordinalToID.Set(nodeID, "")
		h.size.Store(0)
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

	h.size.Add(-1)
	return nil
}

// findNodeByID finds a node by its ID using O(1) map lookup
func (h *Index) findNodeByID(id string) (uint32, *Node) {
	if node, exists := h.idToIndex.Get(hashID(id)); exists {
		// Wait! deleteNode requires an ordinal.
		idx := node.Ordinal
		if idx < uint32(h.nodes.Len()) && h.nodes.Get(idx) != nil {
			return idx, h.nodes.Get(idx)
		}
		h.idToIndex.Delete(hashID(id))
	}
	return ^uint32(0), nil
}

// removeAllConnections removes all bidirectional connections to the target node
func (h *Index) removeAllConnections(ctx context.Context, targetID uint32, targetNode *Node) error {
	// For each level where the target node exists
	for level := 0; level <= targetNode.Level; level++ {
		// Get all neighbors of the target node at this level
		targetLinks := h.getNodeLinks(targetNode, level)
		neighbors := make([]uint32, len(targetLinks))
		copy(neighbors, targetLinks)

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
	targetNode := h.nodes.Get(targetID)
	if targetNode == nil || level >= (targetNode.Level+1) {
		return affected
	}

	targetBacklinks := h.getNodeBacklinks(targetNode, level)
	for _, incomingID := range targetBacklinks {
		if incomingID >= uint32(h.nodes.Len()) {
			continue
		}
		node := h.nodes.Get(incomingID)
		if node == nil || level >= (node.Level+1) {
			continue
		}

		for !h.acquirePruneLock(node) {
			runtime.Gosched()
		}

		links := h.getNodeLinks(node, level)
		lastIdx := len(links) - 1
		if lastIdx < 0 {
			h.releasePruneLock(node)
			continue
		}

		for i, linkID := range links {
			if linkID == targetID {
				lastVal := links[lastIdx]
				atomic.StoreUint32(&links[i], lastVal)
				atomic.StoreUint32(&links[lastIdx], SentinelNodeID)
				atomic.StoreUint32(&node.LinkCounts[level], uint32(lastIdx))
				affected = append(affected, incomingID)
				break
			}
		}

		h.releasePruneLock(node)
	}
	return affected
}

// removeConnection removes a specific connection between two nodes at a given level,
// using sorted lock acquisition to strictly prevent circular deadlocks.
func (h *Index) removeConnection(fromID, toID uint32, level int) {
	fromNode := h.nodes.Get(fromID)
	toNode := h.nodes.Get(toID)

	if fromNode == nil || toNode == nil {
		return
	}
	if level >= (fromNode.Level+1) && level >= (toNode.Level+1) {
		return
	}

	// Sorted Lock Acquisition to prevent deadlocks
	var first, second *Node
	if fromID < toID {
		first, second = fromNode, toNode
	} else if fromID > toID {
		first, second = toNode, fromNode
	} else {
		// Cannot remove connection to self in a meaningful way
		return
	}

	for !h.acquirePruneLock(first) {
		runtime.Gosched()
	}
	for !h.acquirePruneLock(second) {
		runtime.Gosched()
	}

	defer func() {
		h.releasePruneLock(second)
		h.releasePruneLock(first)
	}()

	// Remove from links (forward direction)
	if level < (fromNode.Level + 1) {
		links := h.getNodeLinks(fromNode, level)
		lastIdx := len(links) - 1
		for i, linkID := range links {
			if linkID == toID {
				lastVal := links[lastIdx]
				atomic.StoreUint32(&links[i], lastVal)
				atomic.StoreUint32(&links[lastIdx], SentinelNodeID)
				atomic.StoreUint32(&fromNode.LinkCounts[level], uint32(lastIdx))
				break
			}
		}
	}

	// Remove from backlinks (reverse direction)
	if level < (toNode.Level + 1) {
		backlinks := h.getNodeBacklinks(toNode, level)
		lastIdx := len(backlinks) - 1
		for i, id := range backlinks {
			if id == fromID {
				lastVal := backlinks[lastIdx]
				atomic.StoreUint32(&backlinks[i], lastVal)
				atomic.StoreUint32(&backlinks[lastIdx], SentinelNodeID)
				atomic.StoreUint32(&toNode.BacklinkCounts[level], uint32(lastIdx))
				break
			}
		}
	}
}

// deleteBacklink is now integrated into removeConnection,
// keeping it as a no-op just in case it's called elsewhere,
// though it shouldn't be needed if bidirectional removal is always synchronized.
func (h *Index) deleteBacklink(fromID, toID uint32, level int) {
	// Handled by removeConnection directly now to avoid deadlocks.
}

// reconnectNeighborsOptimized attempts to reconnect neighbors with precomputed distances
func (h *Index) reconnectNeighborsOptimized(ctx context.Context, neighbors []uint32, level int, deletedNode *Node) error {
	if len(neighbors) < 2 {
		return nil // Nothing to reconnect
	}

	arena := h.scratchPool.Get().(*memory.Arena)
	defer func() {
		arena.Reset()
		h.scratchPool.Put(arena)
	}()

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
	validNeighbors, err := memory.ArenaSlice[uint32](arena, len(neighbors))
	if err != nil {
		return fmt.Errorf("arena allocate validNeighbors: %w", err)
	}
	for _, neighborID := range neighbors {
		if neighborID < uint32(h.nodes.Len()) && h.nodes.Get(neighborID) != nil {
			validNeighbors = append(validNeighbors, neighborID)
		}
	}

	if len(validNeighbors) < 2 {
		return nil
	}
	D := len(validNeighbors)

	// Precompute a D×D distance matrix so each node's distance to every
	// other node is computed once, not once per ordered pair.
	distMat, err := memory.ArenaSlice[float32](arena, D*D)
	if err != nil {
		return fmt.Errorf("arena allocate distance matrix: %w", err)
	}
	distMat = distMat[:D*D]
	for i := 0; i < D; i++ {
		ni := validNeighbors[i]
		nodeI := h.nodes.Get(ni)
		for j := i + 1; j < D; j++ {
			nj := validNeighbors[j]
			nodeJ := h.nodes.Get(nj)
			d, err := h.computeDistance(nil, nil, nodeI, nodeJ)
			if err != nil {
				d = float32(math.Inf(1))
			}
			distMat[i*D+j] = d
			distMat[j*D+i] = d
		}
	}

	// For each neighbor, try to connect it to other neighbors
	for ni, neighborID := range validNeighbors {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		neighborNode := h.nodes.Get(neighborID)
		if neighborNode == nil || level >= (neighborNode.Level+1) {
			continue
		}

		currentConnections := len(h.getNodeLinks(neighborNode, level))

		minConnections := maxM / 2
		if minConnections < 1 {
			minConnections = 1
		}

		if currentConnections >= minConnections {
			continue
		}

		// Find potential new connections from remaining neighbors.
		// Distances come from the precomputed matrix — no per-pair recompute.
		candidatesSlice, _ := memory.ArenaSlice[*util.Candidate](arena, len(validNeighbors))
		candidates := candidatesSlice[:0]
		for nj, otherID := range validNeighbors {
			if ni == nj || h.nodes.Get(otherID) == nil {
				continue
			}

			if h.hasConnection(neighborID, otherID, level) {
				continue
			}

			d := distMat[ni*D+nj]
			if d == float32(math.Inf(1)) {
				continue
			}

			candidates = append(candidates, &util.Candidate{
				ID:       otherID,
				Distance: d,
			})
		}

		if len(candidates) == 0 {
			continue
		}

		// Select best candidates to connect (aim to restore to maxM)
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
	if nodeID1 >= uint32(h.nodes.Len()) || nodeID2 >= uint32(h.nodes.Len()) {
		return false
	}

	node1 := h.nodes.Get(nodeID1)
	if node1 == nil || level >= (node1.Level+1) {
		return false
	}

	links := h.getNodeLinks(node1, level)
	for _, linkID := range links {
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

	// Sort candidates by distance using standard library slices.SortFunc
	slices.SortFunc(candidates, func(a, b *util.Candidate) int {
		if a.Distance < b.Distance {
			return -1
		}
		if a.Distance > b.Distance {
			return 1
		}
		return 0
	})

	return candidates[:numToSelect]
}

// createBidirectionalConnection creates a bidirectional connection between two nodes
func (h *Index) createBidirectionalConnection(nodeID1, nodeID2 uint32, level int) {
	node1 := h.nodes.Get(nodeID1)
	node2 := h.nodes.Get(nodeID2)

	if node1 != nil && level < (node1.Level+1) {
		if h.appendUniqueLink(node1, levelMaxLinks(h.config.M, level), level, nodeID2) {
			if node2 != nil && level < (node2.Level+1) {
				h.appendWithSpinlock(node2, node2.Backlinks[level], nodeID1, h.config.M, level)
			}
		}
	}

	if node2 != nil && level < (node2.Level+1) {
		if h.appendUniqueLink(node2, levelMaxLinks(h.config.M, level), level, nodeID1) {
			if node1 != nil && level < (node1.Level+1) {
				h.appendWithSpinlock(node1, node1.Backlinks[level], nodeID2, h.config.M, level)
			}
		}
	}
}

// handleEntryPointReplacement handles the case where the deleted node is the entry point
func (h *Index) handleEntryPointReplacement(deletedID uint32, deletedNode *Node) error {
	// Only need to replace
	if h.getEntryPoint() == nil {
		return nil
	}

	// Fallback to scan all nodes for highest level
	var fallbackEntryPoint *Node
	newMaxLevel := -1

	for i := 0; i < h.nodes.Len(); i++ {
		node := h.nodes.Get(uint32(i))
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

	h.setEntryPoint(fallbackEntryPoint)
	// maxLevel is handled atomically
	return nil
}

// removeNodeFromIndex removes a node from all index data structures
func (h *Index) removeNodeFromIndex(nodeID uint32, id string) {
	if id != "" {
		h.idToIndex.Delete(hashID(id))
	}
	h.ordinalToID.Set(nodeID, "")

	if nodeID < uint32(h.nodes.Len()) {
		node := h.nodes.Get(nodeID)
		if node == nil {
			return
		}
		h.retireNodeStorage(nodeID, node)
	}
}

// retireNodeStorage removes a node from the registry before reclaiming any of
// its off-heap link storage. Writers that captured the old pointer serialize
// on PruneLock and revalidate the registry after acquiring it.
func (h *Index) retireNodeStorage(nodeID uint32, node *Node) {
	if node == nil {
		return
	}

	h.nodes.Set(nodeID, nil)
	for !h.acquirePruneLock(node) {
		runtime.Gosched()
	}
	h.freeNodeLinks(node)
	node.CompressedVector = nil
	node.setVector(nil)
	h.releasePruneLock(node)
}

func (h *Index) deleteStoredVector(node *Node) {
	if node == nil || h.provider != nil || h.rawVectorStore == nil || node.Slot == SentinelNodeID {
		return
	}
	_ = h.rawVectorStore.Delete(VectorRef{
		Kind:  VectorEncodingRaw,
		Slot:  node.Slot,
		Bytes: uint32(h.config.Dimension * 4),
		Valid: true,
	})
	node.setVector(nil)
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
