package graph

import (
	"sync/atomic"
	"unsafe"
)

// VisitAction is invoked for each node during traversal.
// Return false to stop traversal early.
type VisitAction func(nodeID uint64, depth int) bool

// BFS performs a lock-free breadth-first search starting from 'start'.
// It uses caller-provided off-heap bitset and frontier buffers to ensure zero heap allocations on the hot path.
func (g *graphStore) BFS(start uint64, maxDepth int, visit VisitAction, bitset *Bitset, frontier *FrontierBuf) error {
	if maxDepth <= 0 {
		maxDepth = 1 << 20 // Enforce a 1M limit
	}
	bitset.Clear()
	frontier.Clear()
	
	frontier.Push(start, 0)
	bitset.Set(start)
	g.metrics.bfsCalls.Add(1)

	for !frontier.Empty() {
		node, depth := frontier.Pop()
		
		// Invoke callback
		if !visit(node, depth) {
			return nil // Early termination
		}
		g.metrics.bfsNodesVisited.Add(1)
		
		if depth > maxDepth {
			break
		}
		
		// Get neighbors (lock-free read inline to avoid allocations)
		shard := node % uint64(g.cfg.PageShards)
	retry:
		oldTail := frontier.tail
		g.pagePool.HyalineEnter(int(shard))
		
		pageSlot := g.index.Lookup(node)
		if pageSlot == 0 {
			g.pagePool.HyalineLeave(int(shard))
			continue
		}
		
		page := g.pageReg.Get(pageSlot)
		gen := atomic.LoadUint32(&page.Header.Generation)
		totalCount := page.Header.Count
		
		currPage := page
		remaining := totalCount
		
		for currPage != nil && remaining > 0 {
			pageCount := remaining
			if pageCount > 250 {
				pageCount = 250
			}
			
			if pageCount <= 8 {
				for i := uint16(0); i < pageCount; i++ {
					target := currPage.Inline[i].Target
					if !bitset.Test(target) {
						bitset.Set(target)
						frontier.Push(target, depth+1)
					}
				}
			} else {
				// Process inline first
				for i := uint16(0); i < 8; i++ {
					target := currPage.Inline[i].Target
					if !bitset.Test(target) {
						bitset.Set(target)
						frontier.Push(target, depth+1)
					}
				}
				extra := unsafe.Slice((*Edge)(unsafe.Pointer(&currPage.Padding[0])), 242)
				extraCount := pageCount - 8
				for i := uint16(0); i < extraCount; i++ {
					target := extra[i].Target
					if !bitset.Test(target) {
						bitset.Set(target)
						frontier.Push(target, depth+1)
					}
				}
			}
			
			remaining -= pageCount
			if currPage.Header.Overflow != 0 {
				currPage = g.pageReg.Get(currPage.Header.Overflow)
				g.metrics.chainedPageReads.Add(1)
			} else {
				currPage = nil
			}
		}
		
		// Validate generation (detect concurrent writes)
		if atomic.LoadUint32(&page.Header.Generation) != gen {
			g.pagePool.HyalineLeave(int(shard))
			// Rollback frontier and bitset to prevent torn data pollution
			for i := oldTail; i < frontier.tail; i++ {
				bitset.ClearBit(frontier.data[i].NodeID)
			}
			frontier.tail = oldTail
			goto retry
		}
		g.pagePool.HyalineLeave(int(shard))
	}
	
	return nil
}
