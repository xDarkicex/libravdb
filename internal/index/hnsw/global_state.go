package hnsw

func packGlobalState(node *Node) uint64 {
	if node == nil {
		return 0
	}
	return (uint64(node.Ordinal) << 32) | uint64(uint32(node.Level)+1)
}

func unpackGlobalLevel(state uint64) int {
	if state == 0 {
		return 0
	}
	return int(uint32(state) - 1)
}

// getEntryPoint returns the current entry point node by unpacking the global state
func (h *Index) getEntryPoint() *Node {
	state := h.globalState.Load()
	if state == 0 {
		return nil
	}
	epID := uint32(state >> 32)
	return h.nodes.Get(epID)
}

// getMaxLevel returns the current maximum level by unpacking the global state
func (h *Index) getMaxLevel() int {
	state := h.globalState.Load()
	return unpackGlobalLevel(state)
}

// setEntryPoint updates the global entry point and its level atomically
func (h *Index) setEntryPoint(node *Node) {
	if node == nil {
		h.globalState.Store(0)
		return
	}
	h.globalState.Store(packGlobalState(node))
}

// initializeEntryPointCAS publishes node only when the graph is still empty.
// Unlike updateEntryPointCAS, it never replaces an existing lower-level entry
// point, so only the actual first node may skip graph insertion.
func (h *Index) initializeEntryPointCAS(node *Node) bool {
	if node == nil {
		return false
	}
	return h.globalState.CompareAndSwap(0, packGlobalState(node))
}

// updateEntryPointCAS attempts to update the global entry point using CompareAndSwap.
// It succeeds only if the node's level is strictly greater than the current max level.
// If the global state is empty (0), it will also succeed to set the first entry point.
func (h *Index) updateEntryPointCAS(node *Node) bool {
	if node == nil {
		return false
	}

	for {
		currentState := h.globalState.Load()

		// If there is no entry point, we can just attempt to set it.
		if currentState == 0 {
			newState := packGlobalState(node)
			if h.globalState.CompareAndSwap(0, newState) {
				return true
			}
			continue
		}

		currentMaxLevel := unpackGlobalLevel(currentState)
		if node.Level <= currentMaxLevel {
			return false // Another node with a higher or equal level beat us to it
		}

		newState := packGlobalState(node)
		if h.globalState.CompareAndSwap(currentState, newState) {
			return true
		}
		// CAS failed due to concurrent update, retry loop
	}
}
