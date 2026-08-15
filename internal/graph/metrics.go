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
	OffHeapMemory     uint64
	// MutationGeneration increments after each successfully published graph
	// mutation batch. Controllers use it to detect derived-metric staleness.
	MutationGeneration uint64
	// PageRank publication metadata is intentionally independent of the
	// algorithm implementation; a future maintenance job records these values
	// once it atomically publishes a rank vector.
	LastPageRankGeneration uint64
	LastPageRankLSN        uint64
	PageRankDuration       time.Duration
	PageRankAvailable      bool
	PageRankStale          bool
}

// storeMetrics represents the internal atomic counters.
type storeMetrics struct {
	edgesAdded             atomic.Uint64
	edgesRemoved           atomic.Uint64
	pagesAllocated         atomic.Uint64
	overfullPages          atomic.Uint64
	chainedPageReads       atomic.Uint64
	bfsCalls               atomic.Uint64
	bfsNodesVisited        atomic.Uint64
	walReplayDuration      atomic.Int64 // nanoseconds
	offHeapMemory          atomic.Uint64
	mutationGeneration     atomic.Uint64
	lastPageRankGeneration atomic.Uint64
	lastPageRankLSN        atomic.Uint64
	pageRankDuration       atomic.Int64
	pageRankAvailable      atomic.Bool
}

func (m *storeMetrics) get() GraphStats {
	mutationGeneration := m.mutationGeneration.Load()
	pageRankGeneration := m.lastPageRankGeneration.Load()
	pageRankAvailable := m.pageRankAvailable.Load()
	return GraphStats{
		EdgesAdded:             m.edgesAdded.Load(),
		EdgesRemoved:           m.edgesRemoved.Load(),
		PagesAllocated:         m.pagesAllocated.Load(),
		OverfullPages:          m.overfullPages.Load(),
		ChainedPageReads:       m.chainedPageReads.Load(),
		BFSCalls:               m.bfsCalls.Load(),
		BFSNodesVisited:        m.bfsNodesVisited.Load(),
		WALReplayDuration:      time.Duration(m.walReplayDuration.Load()),
		OffHeapMemory:          m.offHeapMemory.Load(),
		MutationGeneration:     mutationGeneration,
		LastPageRankGeneration: pageRankGeneration,
		LastPageRankLSN:        m.lastPageRankLSN.Load(),
		PageRankDuration:       time.Duration(m.pageRankDuration.Load()),
		PageRankAvailable:      pageRankAvailable,
		PageRankStale:          pageRankAvailable && pageRankGeneration != mutationGeneration,
	}
}
