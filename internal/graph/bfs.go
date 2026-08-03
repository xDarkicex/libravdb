package graph

import (
	"sync/atomic"
	"unsafe"
)

// VisitAction is invoked for each node during traversal.
// band is the edge band index (0..len(edges)-1), step is the hop count
// within that band. Return false to stop traversal early.
// Callers can check band == numBands-1 to identify final-band matches.
type VisitAction func(nodeID uint64, band int, step int) bool

// EdgePlan describes a single edge in a path pattern: its direction,
// how many consecutive hops it applies, and the minimum hop requirement.
// A MATCH path (a)-[e1]->{1,3}(b)-[e2]->(c) produces two EdgePlans:
// {Dir:1, Min:1, Max:3, KindSet:{1}}, {Dir:1, Min:1, Max:1, KindSet:{2}}.
type EdgePlan struct {
	Dir     int8    // 1=outbound, -1=inbound, 0=both
	_       [7]byte // align following fields to 8-byte boundary
	KindSet KindSet // edge kind filter; zero value means no filtering
	Min     int     // minimum hops (0 for ->*, 1 for ->+ or default)
	Max     int     // maximum hops (repeat count: 1 for default, large for ->+/->*)
}

// BFS performs a lock-free breadth-first search starting from 'start'.
// It uses caller-provided off-heap bitset and frontier buffers to ensure
// zero heap allocations on the hot path.  Single-band (no pattern), so
// band is always 0 and step equals depth.
func (g *graphStore) BFS(start uint64, maxDepth int, visit VisitAction, bitset *Bitset, frontier *FrontierBuf) error {
	if maxDepth <= 0 {
		maxDepth = 1 << 20
	}
	bitset.Clear()
	frontier.Clear()

	if frontier.Push(start, 0, 0) {
		bitset.Set(start)
	}
	g.metrics.bfsCalls.Add(1)

	for !frontier.Empty() {
		node, band, step := frontier.Pop()

		if !visit(node, band, step) {
			return nil
		}
		g.metrics.bfsNodesVisited.Add(1)

		if step >= maxDepth {
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

		gen := g.enumerateTargets(page, 0, step, bitset, frontier, KindSet{}, 1)

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

// BFSPattern performs a band-stateful breadth-first search over a
// multi-edge path pattern. Each edge band has its own direction, kind
// filter, and min/max hop bounds.  Frontier entries carry (node, band,
// step) so variable-length bands chain correctly — a node at step
// edges[b].Max in band b transitions to band b+1 at step 0.
//
// Visited dedup uses per-(node, band) keying via VisitedKey so a node
// reached in band 0 can be expanded as a transition into band 1.
func (g *graphStore) BFSPattern(start uint64, edges []EdgePlan, maxDepth int, visit VisitAction, bitset *Bitset, frontier *FrontierBuf) error {
	numBands := len(edges)
	if numBands == 0 {
		return nil
	}
	lastBand := numBands - 1

	bitset.Clear()
	frontier.Clear()

	// Seed: start vertex at band 0, step 0.
	key := VisitedKey(start, 0, numBands)
	if frontier.Push(start, 0, 0) {
		bitset.Set(key)
	}
	g.metrics.bfsCalls.Add(1)

	for !frontier.Empty() {
		node, band, step := frontier.Pop()

		if !visit(node, band, step) {
			return nil
		}
		g.metrics.bfsNodesVisited.Add(1)

		// Within-band expansion: step < Max means we can take another hop
		// in the current band.
		if step < edges[band].Max {
			dir := edges[band].Dir
			ks := edges[band].KindSet
			shard := node % uint64(g.cfg.PageShards)

			// Forward direction (outbound)
			if dir >= 0 {
				g.pagePool.HyalineEnter(int(shard))
				oldTail := frontier.tail
				page := g.index.Lookup(node)
				if page != nil {
					gen := g.enumerateTargets(page, band, step, bitset, frontier, ks, numBands)
					if atomic.LoadUint32(&page.Header.Generation) != gen {
						for i := oldTail; i < frontier.tail; i++ {
							nd := frontier.data[i]
							bitset.ClearBit(VisitedKey(nd.NodeID, int(nd.Band), numBands))
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
					gen := g.enumerateTargets(page, band, step, bitset, frontier, ks, numBands)
					if atomic.LoadUint32(&page.Header.Generation) != gen {
						for i := oldTail; i < frontier.tail; i++ {
							nd := frontier.data[i]
							bitset.ClearBit(VisitedKey(nd.NodeID, int(nd.Band), numBands))
						}
						frontier.tail = oldTail
					}
				}
				g.reverse.pool.HyalineLeave(int(shard))
			}
		}

		// Band transition: if step >= Min (node satisfies the minimum-hop
		// requirement for this band) and there is a next band, push the
		// node itself as the entry point of band+1 at step 0.  The next
		// iteration will expand it with band+1's direction/kind filter.
		if band < lastBand && step >= edges[band].Min {
			nextKey := VisitedKey(node, band+1, numBands)
			if !bitset.Test(nextKey) {
				if frontier.Push(node, band+1, 0) {
					bitset.Set(nextKey)
				}
			}
		}
	}

	return nil
}

// enumerateTargets walks a node's edge page chain and pushes unvisited targets
// into the frontier. Targets are pushed at (band, step+1) — same band, one
// step deeper.  The caller must hold HyalineEnter on the appropriate pool.
// Returns the generation snapshot taken at entry; the caller compares with a
// re-read to detect concurrent writes.
func (g *graphStore) enumerateTargets(page *EdgeTablePage, band int, step int, bitset *Bitset, frontier *FrontierBuf, kindFilter KindSet, numBands int) uint32 {
	gen := atomic.LoadUint32(&page.Header.Generation)
	totalCount := page.Header.Count

	currPage := page
	remaining := totalCount
	nextStep := step + 1
	filterActive := kindFilter != (KindSet{})

	for currPage != nil && remaining > 0 {
		pageCount := remaining
		if pageCount > 250 {
			pageCount = 250
		}

		if pageCount <= 8 {
			for i := uint16(0); i < pageCount; i++ {
				if filterActive && !kindFilter.Has(currPage.Inline[i].GetKind()) {
					continue
				}
				target := currPage.Inline[i].Target
				key := VisitedKey(target, band, numBands)
				if !bitset.Test(key) {
					if frontier.Push(target, band, nextStep) {
						bitset.Set(key)
					}
				}
			}
		} else {
			for i := uint16(0); i < 8; i++ {
				if filterActive && !kindFilter.Has(currPage.Inline[i].GetKind()) {
					continue
				}
				target := currPage.Inline[i].Target
				key := VisitedKey(target, band, numBands)
				if !bitset.Test(key) {
					if frontier.Push(target, band, nextStep) {
						bitset.Set(key)
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
				key := VisitedKey(target, band, numBands)
				if !bitset.Test(key) {
					if frontier.Push(target, band, nextStep) {
						bitset.Set(key)
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
