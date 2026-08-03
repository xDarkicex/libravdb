package graph

import (
	"sync/atomic"
	"unsafe"
)

// VisitAction is invoked for each node during traversal.
// Return false to stop traversal early.
type VisitAction func(nodeID uint64, depth int) bool

// EdgePlan describes a single edge in a path pattern: its direction,
// how many consecutive hops it applies, and the minimum hop requirement.
// A MATCH path (a)-[e1]->{1,3}(b)-[e2]->(c) produces two EdgePlans:
// {Dir:1, Min:1, Max:3}, {Dir:1, Min:0, Max:1}.
type EdgePlan struct {
	Dir     int8    // 1=outbound, -1=inbound, 0=both
	_       [7]byte // align following fields to 8-byte boundary
	KindSet KindSet // edge kind filter; zero value means no filtering
	Min     int     // minimum hops (0 for ->*, 1 for ->+ or default)
	Max     int     // maximum hops (repeat count: 1 for default, large for ->+/->*)
}

// edgeDirAt returns the direction for the given depth by scanning
// the cumulative edge bands. Caller guarantees depth < sum of Max.
func edgeDirAt(edges []EdgePlan, depth int) int8 {
	offset := 0
	for i := range edges {
		end := offset + edges[i].Max
		if depth < end {
			return edges[i].Dir
		}
		offset = end
	}
	return 1 // fallback: outbound (should not be reached)
}

// kindSetAt returns the KindSet filter for the given flattened depth.
// A zero-value KindSet means no filtering is needed (match all kinds).
func kindSetAt(edges []EdgePlan, depth int) KindSet {
	offset := 0
	for i := range edges {
		end := offset + edges[i].Max
		if depth < end {
			return edges[i].KindSet
		}
		offset = end
	}
	return KindSet{}
}

// BFS performs a lock-free breadth-first search starting from 'start'.
// It uses caller-provided off-heap bitset and frontier buffers to ensure zero heap allocations on the hot path.
func (g *graphStore) BFS(start uint64, maxDepth int, visit VisitAction, bitset *Bitset, frontier *FrontierBuf) error {
	if maxDepth <= 0 {
		maxDepth = 1 << 20
	}
	bitset.Clear()
	frontier.Clear()

	if frontier.Push(start, 0) {
		bitset.Set(start)
	}
	g.metrics.bfsCalls.Add(1)

	for !frontier.Empty() {
		node, depth := frontier.Pop()

		if !visit(node, depth) {
			return nil
		}
		g.metrics.bfsNodesVisited.Add(1)

		if depth == maxDepth {
			continue
		}

		shard := node % uint64(g.cfg.PageShards)
	retry:
		oldTail := frontier.tail
		g.pagePool.HyalineEnter(int(shard))

		page := g.index.Lookup(node)
		if page == nil {
			g.pagePool.HyalineLeave(int(shard))
			continue
		}

		gen := g.enumerateTargets(page, depth, bitset, frontier, KindSet{})

		if atomic.LoadUint32(&page.Header.Generation) != gen {
			g.pagePool.HyalineLeave(int(shard))
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

// BFSPattern performs a direction-aware breadth-first search.
// Edges describe per-depth-band direction; maxDepth is the sum of edge Max values.
// Direction is resolved on-the-fly from edges[depth] without materializing a flat slice.
func (g *graphStore) BFSPattern(start uint64, edges []EdgePlan, maxDepth int, visit VisitAction, bitset *Bitset, frontier *FrontierBuf) error {
	bitset.Clear()
	frontier.Clear()

	if frontier.Push(start, 0) {
		bitset.Set(start)
	}
	g.metrics.bfsCalls.Add(1)

	for !frontier.Empty() {
		node, depth := frontier.Pop()

		if !visit(node, depth) {
			return nil
		}
		g.metrics.bfsNodesVisited.Add(1)

		if depth >= maxDepth {
			continue
		}

		dir := edgeDirAt(edges, depth)
		ks := kindSetAt(edges, depth)
		shard := node % uint64(g.cfg.PageShards)

		// Forward direction (outbound)
		if dir >= 0 {
			g.pagePool.HyalineEnter(int(shard))
			oldTail := frontier.tail
			page := g.index.Lookup(node)
			if page != nil {
				gen := g.enumerateTargets(page, depth, bitset, frontier, ks)
				if atomic.LoadUint32(&page.Header.Generation) != gen {
					for i := oldTail; i < frontier.tail; i++ {
						bitset.ClearBit(frontier.data[i].NodeID)
					}
					frontier.tail = oldTail
				}
			}
			g.pagePool.HyalineLeave(int(shard))
		}

		// Reverse direction (inbound)
		if dir <= 0 {
			g.reverse.pool.HyalineEnter(int(shard))
			oldTail := frontier.tail
			page := g.reverse.locator.Lookup(node)
			if page != nil {
				gen := g.enumerateTargets(page, depth, bitset, frontier, ks)
				if atomic.LoadUint32(&page.Header.Generation) != gen {
					for i := oldTail; i < frontier.tail; i++ {
						bitset.ClearBit(frontier.data[i].NodeID)
					}
					frontier.tail = oldTail
				}
			}
			g.reverse.pool.HyalineLeave(int(shard))
		}
	}

	return nil
}

// enumerateTargets walks a node's edge page chain and pushes unvisited targets
// into the frontier. The caller must hold HyalineEnter on the appropriate pool.
// Returns the generation snapshot taken at entry; the caller compares with a
// re-read to detect concurrent writes.
func (g *graphStore) enumerateTargets(page *EdgeTablePage, depth int, bitset *Bitset, frontier *FrontierBuf, kindFilter KindSet) uint32 {
	gen := atomic.LoadUint32(&page.Header.Generation)
	totalCount := page.Header.Count

	currPage := page
	remaining := totalCount

	for currPage != nil && remaining > 0 {
		pageCount := remaining
		if pageCount > 250 {
			pageCount = 250
		}

		filterActive := kindFilter != (KindSet{})
		if pageCount <= 8 {
			for i := uint16(0); i < pageCount; i++ {
				if filterActive && !kindFilter.Has(currPage.Inline[i].GetKind()) {
					continue
				}
				target := currPage.Inline[i].Target
				if !bitset.Test(target) {
					if frontier.Push(target, depth+1) {
						bitset.Set(target)
					}
				}
			}
		} else {
			for i := uint16(0); i < 8; i++ {
				target := currPage.Inline[i].Target
				if !bitset.Test(target) {
					if frontier.Push(target, depth+1) {
						bitset.Set(target)
					}
				}
			}
			extra := unsafe.Slice((*Edge)(unsafe.Pointer(&currPage.Padding[0])), 242)
			extraCount := pageCount - 8
			for i := uint16(0); i < extraCount; i++ {
				if filterActive && !kindFilter.Has(extra[i].GetKind()) {
					continue
				}
				target := extra[i].Target
				if !bitset.Test(target) {
					if frontier.Push(target, depth+1) {
						bitset.Set(target)
					}
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

	return gen
}
