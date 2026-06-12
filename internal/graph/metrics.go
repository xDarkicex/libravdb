package graph

import (
	"sync/atomic"
	"time"
)

// GraphStats exposes metrics for monitoring graph layer operations.
type GraphStats struct {
	EdgesAdded        uint64
	EdgesRemoved      uint64
	PagesAllocated    uint64
	OverfullPages     uint64
	ChainedPageReads  uint64
	BFSCalls          uint64
	BFSNodesVisited   uint64
	WALReplayDuration time.Duration
}

// storeMetrics represents the internal atomic counters.
type storeMetrics struct {
	edgesAdded        atomic.Uint64
	edgesRemoved      atomic.Uint64
	pagesAllocated    atomic.Uint64
	overfullPages     atomic.Uint64
	chainedPageReads  atomic.Uint64
	bfsCalls          atomic.Uint64
	bfsNodesVisited   atomic.Uint64
	walReplayDuration atomic.Int64 // nanoseconds
}

func (m *storeMetrics) get() GraphStats {
	return GraphStats{
		EdgesAdded:        m.edgesAdded.Load(),
		EdgesRemoved:      m.edgesRemoved.Load(),
		PagesAllocated:    m.pagesAllocated.Load(),
		OverfullPages:     m.overfullPages.Load(),
		ChainedPageReads:  m.chainedPageReads.Load(),
		BFSCalls:          m.bfsCalls.Load(),
		BFSNodesVisited:   m.bfsNodesVisited.Load(),
		WALReplayDuration: time.Duration(m.walReplayDuration.Load()),
	}
}
