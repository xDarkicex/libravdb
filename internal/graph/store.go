package graph

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/memory"
)

// StagedGraphOpKind classifies the mutation type in the ordered operation log.
// Ordered replay must process operations in original append order; grouping by
// kind is not semantically correct because edge add/remove sequences are observable.
type StagedGraphOpKind uint8

const (
	StagedGraphEdgeAdd    StagedGraphOpKind = iota
	StagedGraphEdgeRemove
	StagedGraphNodeDrop
)

// StagedGraphOp records a single graph mutation with sufficient detail for
// ordered savepoint rollback replay. Every AddEdge / RemoveEdge / DropNodeEdges
// call appends exactly one ordered op so the sequence is losslessly restorable.
type StagedGraphOp struct {
	Kind       StagedGraphOpKind
	Collection string
	Src        uint64
	Tgt        uint64
	EdgeKind   uint8
	Weight     float32
	NodeID     uint64
}

// Txn is a minimal transaction context for graph operations.
type Txn struct {
	ID        uint64
	walWriter storage.GraphWALWriter
	store     *graphStore
	adds      []storage.GraphEdgeOp
	removes   []storage.GraphEdgeOp
	nodeDrops []storage.GraphNodeDropOp
	closed    bool

	// orderedOps records every mutation in append order for savepoint rollback
	// replay. It is the canonical ordered history; the add/remove/nodeDrop slices
	// are derived from it and must remain consistent.
	orderedOps []StagedGraphOp

	// epochLSN is the snapshot LSN for read isolation. When >0, NeighborsOverlay
	// and InboundNeighborsOverlay read from NeighborsAtLSN(nodeID, epochLSN)
	// instead of the live graph. Set by epoch transactions at BeginEpochTx time.
	epochLSN uint64

	// collection is the collection name owning this graph transaction.
	// Set by EpochTx.GraphTxn for epoch transactions. Used to include
	// collection identity in WAL graph edge frames for per-collection recovery.
	collection string
}

// Commit flushes all accumulated edge mutations through the unified batch
// system and waits for the durable WAL commit LSN before publishing topology
// and stamping temporal edge intervals.
//
// When epochLSN > 0 (epoch snapshot isolation), Commit uses batch-clone: each
// affected node's page chain is cloned once, all staged edges for that node
// are applied to the clone, and the index is atomically swapped via HashMap.Put.
// This eliminates spinlock contention and provides write-write safety for
// concurrent epoch sessions.
func (t *Txn) Commit(ctx context.Context) error {
	if t == nil || t.closed {
		return fmt.Errorf("graph transaction is closed")
	}
	t.closed = true

	// ── Direct path (all transactions) ──
	// NOTE: CoW batch-clone (commitCow) is disabled until it routes
	// through the unified WAL transaction path for durable replay.
	// Epoch graph commits currently use the standard applyCommittedOps
	// path, which is WAL-durable and recovery-safe.
	if t.walWriter == nil {
		return t.store.applyCommittedOps(t.adds, t.removes, t.nodeDrops)
	}
	store := t.store
	adds := t.adds
	removes := t.removes
	nodeDrops := t.nodeDrops
	_, err := t.walWriter.AppendGraphEdges(ctx, adds, removes, nodeDrops, func(lsn uint64) {
		if err := store.applyCommittedOps(adds, removes, nodeDrops); err == nil {
			store.recordEdgeCommitLSN(lsn, adds, removes, nodeDrops)
		}
	})
	return err
}

// commitCow performs batch-clone commit for epoch transactions.
// Staged edges are grouped by affected node, each node's page chain is
// cloned once, all edges are applied to the clone, and the index is
// atomically swapped.
func (t *Txn) commitCow() error {
	g := t.store
	if len(t.nodeDrops) > 0 {
		// Handle node drops through the existing path for now.
		return g.applyCommittedOps(t.adds, t.removes, t.nodeDrops)
	}

	// Phase A: Group staged adds/removes by affected node.
	// forwardAdds: edges to add to forward index (keyed by src node)
	// reverseAdds: edges to add to reverse index (keyed by tgt node)
	type edgeWithKind struct {
		tgt   uint64
		kind  uint8
		weight float32
	}
	forwardAdds := make(map[uint64][]Edge)
	reverseAdds := make(map[uint64][]Edge)
	forwardRemoves := make(map[uint64][]edgeWithKind)

	for _, op := range t.adds {
		fEdge := Edge{Target: op.Tgt, Weight: op.Weight}
		fEdge.SetKind(op.Kind)
		forwardAdds[op.Src] = append(forwardAdds[op.Src], fEdge)

		rEdge := Edge{Target: op.Src, Weight: op.Weight}
		rEdge.SetKind(op.Kind)
		reverseAdds[op.Tgt] = append(reverseAdds[op.Tgt], rEdge)
	}
	for _, op := range t.removes {
		forwardRemoves[op.Src] = append(forwardRemoves[op.Src], edgeWithKind{op.Tgt, op.Kind, 0})
	}

	// Phase B: Clone page chains and apply edges.
	cowPages := make(map[uint64]*EdgeTablePage)
	cowReverse := make(map[uint64]*EdgeTablePage)

	for nodeID, edges := range forwardAdds {
		shard := int(nodeID % uint64(g.cfg.PageShards))
		oldPage := g.index.Lookup(nodeID)
		var page *EdgeTablePage
		if oldPage != nil {
			var err error
			page, err = g.clonePageChain(oldPage, g.pagePools[0], shard)
			if err != nil {
				return fmt.Errorf("clone forward page for node %d: %w", nodeID, err)
			}
		} else {
			// First edge for this node: allocate a fresh page.
			page = g.newPage(g.pagePools[0], shard)
		}
		for _, e := range edges {
			g.writeEdgeToClonedPage(page, e)
		}
		cowPages[nodeID] = page
	}

	for nodeID, edges := range reverseAdds {
		shard := int(nodeID % uint64(g.cfg.PageShards))
		oldPage := g.reverse.locator.Lookup(nodeID)
		var page *EdgeTablePage
		if oldPage != nil {
			var err error
			page, err = g.clonePageChain(oldPage, g.reverse.pool, shard)
			if err != nil {
				return fmt.Errorf("clone reverse page for node %d: %w", nodeID, err)
			}
		} else {
			page = g.newPage(g.reverse.pool, shard)
		}
		for _, e := range edges {
			g.writeEdgeToClonedPage(page, e)
		}
		cowReverse[nodeID] = page
	}

	// Phase C: Atomic index swap.
	for nodeID, newPage := range cowPages {
		oldPage := g.index.Lookup(nodeID)
		g.index.Insert(nodeID, newPage)
		if oldPage != nil {
			g.retirePageChain(nodeID, g.index, g.pagePools[0])
			// Re-insert new page after retirement cleared it.
			g.index.Insert(nodeID, newPage)
		}
	}
	for nodeID, newPage := range cowReverse {
		g.reverse.locator.Insert(nodeID, newPage)
	}

	// Phase D: Record temporal edge intervals for LSN-based queries.
	if g.walWriter != nil {
		commitLSN := g.globalStamp.Load() // approximate; real LSN would come from WAL
		g.recordEdgeCommitLSN(uint64(commitLSN), t.adds, t.removes, t.nodeDrops)
	}

	t.adds = nil
	t.removes = nil
	t.nodeDrops = nil
	return nil
}

// writeEdgeToClonedPage appends an edge to a cloned page chain.
// The clone is a full deep copy registered in g.pageReg, so overflow
// chain traversal works via g.pageReg.Get(cloned.Header.Overflow).
// No spinlock — the clone is private to this goroutine.
func (g *graphStore) writeEdgeToClonedPage(page *EdgeTablePage, edge Edge) {
	totalCount := page.Header.Count
	if totalCount < 8 {
		page.Inline[totalCount] = edge
		page.Header.Count++
		return
	}

	currPage := page
	edgesToSkip := totalCount
	for edgesToSkip >= 250 {
		if currPage.Header.Overflow == 0 {
			return
		}
		currPage = g.pageReg.Get(currPage.Header.Overflow)
		if currPage == nil {
			return
		}
		edgesToSkip -= 250
	}

	if edgesToSkip < 8 {
		currPage.Inline[edgesToSkip] = edge
	} else {
		idx := edgesToSkip - 8
		if idx < 242 {
			extra := unsafe.Slice((*Edge)(unsafe.Pointer(&currPage.Padding[0])), 242)
			extra[idx] = edge
		}
	}
	page.Header.Count++
}

// Rollback discards all staged graph mutations. Since graph operations are
// published only by Commit, rollback never appends WAL frames and never
// changes the live topology.
func (t *Txn) Rollback() error {
	if t == nil || t.closed {
		return fmt.Errorf("graph transaction is closed")
	}
	t.adds = nil
	t.removes = nil
	t.nodeDrops = nil
	t.closed = true
	return nil
}

// SetEpochLSN pins the read snapshot for this transaction. When >0,
// NeighborsOverlay and InboundNeighborsOverlay read from the LSN-filtered
// temporal view instead of the live graph. Zero means live reads.
func (t *Txn) SetEpochLSN(lsn uint64) {
	if t != nil {
		t.epochLSN = lsn
	}
}

// SetCollection sets the owning collection name for this graph transaction.
func (t *Txn) SetCollection(name string) {
	if t != nil {
		t.collection = name
	}
}

// StagedOps returns the accumulated edge operations for combined atomic
// commit with record operations through the storage engine.
func (t *Txn) StagedOps() (adds, removes []storage.GraphEdgeOp, nodeDrops []storage.GraphNodeDropOp) {
	if t == nil {
		return nil, nil, nil
	}
	return t.adds, t.removes, t.nodeDrops
}

// OrderedStagedOps returns an immutable copy of the canonical ordered mutation
// log. Savepoint rollback replays the prefix up to the saved position, preserving
// original append order — edge add/remove/drop sequences are semantically
// observable and must never be replayed grouped by kind.
func (t *Txn) OrderedStagedOps() []StagedGraphOp {
	if t == nil {
		return nil
	}
	out := make([]StagedGraphOp, len(t.orderedOps))
	copy(out, t.orderedOps)
	return out
}

// RemapNodeIDs replaces provisional node IDs in staged operations with
// their committed counterparts. Called after record commit assigns permanent
// node IDs to provisionally-staged records within an epoch transaction.
func (t *Txn) RemapNodeIDs(mapping map[uint64]uint64) {
	if t == nil {
		return
	}
	for i := range t.adds {
		if repl, ok := mapping[t.adds[i].Src]; ok {
			t.adds[i].Src = repl
		}
		if repl, ok := mapping[t.adds[i].Tgt]; ok {
			t.adds[i].Tgt = repl
		}
	}
	for i := range t.removes {
		if repl, ok := mapping[t.removes[i].Src]; ok {
			t.removes[i].Src = repl
		}
		if repl, ok := mapping[t.removes[i].Tgt]; ok {
			t.removes[i].Tgt = repl
		}
	}
	// nodeDrops don't reference other nodes, just the dropped node itself.
	for i := range t.nodeDrops {
		drop := &t.nodeDrops[i]
		if repl, ok := mapping[drop.NodeID]; ok {
			drop.NodeID = repl
		}
	}
}

// ApplyInMemory publishes staged graph operations to the in-memory
// topology. Called after the storage engine has durably committed the
// combined record+graph WAL transaction.
func (t *Txn) ApplyInMemory() error {
	if t == nil || t.closed {
		return fmt.Errorf("graph transaction is closed")
	}
	t.closed = true
	return t.store.applyCommittedOps(t.adds, t.removes, t.nodeDrops)
}

// AddEdge adds a directed edge to the graph within this transaction.
func (t *Txn) AddEdge(src, tgt uint64, weight float32, kind uint8) error {
	if t == nil || t.closed {
		return fmt.Errorf("graph transaction is closed")
	}
	t.adds = append(t.adds, storage.GraphEdgeOp{Collection: t.collection, Src: src, Tgt: tgt, Weight: weight, Kind: kind})
	t.orderedOps = append(t.orderedOps, StagedGraphOp{
		Kind: StagedGraphEdgeAdd, Collection: t.collection,
		Src: src, Tgt: tgt, EdgeKind: kind, Weight: weight,
	})
	return nil
}

// RemoveEdge removes a directed edge from the graph within this transaction.
//
// Ordered log invariant: every call appends exactly one entry so savepoint
// positions remain stable. When the edge exists in staged adds, it is removed
// from the adds slice (cancelling the staged add) but a StagedGraphEdgeRemove
// is still appended to orderedOps. During replay, the fresh Txn's RemoveEdge
// will find the replayed AddEdge in its own staged adds and cancel it identically.
func (t *Txn) RemoveEdge(src, tgt uint64, kind uint8) error {
	if t == nil || t.closed {
		return fmt.Errorf("graph transaction is closed")
	}
	// Always record the ordered op first — append-only invariant.
	t.orderedOps = append(t.orderedOps, StagedGraphOp{
		Kind: StagedGraphEdgeRemove, Collection: t.collection,
		Src: src, Tgt: tgt, EdgeKind: kind,
	})

	// If the edge was staged as an add in this same transaction, cancel it
	// by removing from adds. The overlay will then fall through to the base
	// (or remaining staged ops) for this edge.
	for i := range t.adds {
		if t.adds[i].Src == src && t.adds[i].Tgt == tgt && t.adds[i].Kind == kind {
			t.adds = append(t.adds[:i], t.adds[i+1:]...)
			return nil
		}
	}

	// Edge is not in staged adds — must exist in the base graph.
	if _, err := t.store.edge(src, tgt, kind); err != nil {
		return err
	}
	t.removes = append(t.removes, storage.GraphEdgeOp{Collection: t.collection, Src: src, Tgt: tgt, Kind: kind})
	return nil
}

// DropNodeEdges removes all edges incident to a node.
func (t *Txn) DropNodeEdges(nodeID uint64) error {
	if t == nil || t.closed {
		return fmt.Errorf("graph transaction is closed")
	}
	t.nodeDrops = append(t.nodeDrops, storage.GraphNodeDropOp{Collection: t.collection, NodeID: nodeID})
	t.orderedOps = append(t.orderedOps, StagedGraphOp{
		Kind: StagedGraphNodeDrop, Collection: t.collection, NodeID: nodeID,
	})
	return nil
}

// NeighborsOverlay returns the live neighbors plus this transaction's staged
// edge changes. It is the primitive used by epoch read-your-writes traversal;
// it never publishes or WAL-logs the staged operations.
//
// When epochLSN > 0, the base neighbor set is read from the LSN-filtered
// temporal view (NeighborsAtLSN) instead of the live graph. This provides
// snapshot isolation: concurrent commits from other sessions at higher LSNs
// are invisible within this transaction.
func (t *Txn) NeighborsOverlay(nodeID uint64) ([]Edge, error) {
	if t == nil || t.closed {
		return nil, fmt.Errorf("graph transaction is closed")
	}
	var base []Edge
	var err error
	if t.epochLSN > 0 {
		base, err = t.store.NeighborsAtLSN(nodeID, t.epochLSN)
	} else {
		base, err = t.store.Neighbors(nodeID)
	}
	if err != nil {
		return nil, err
	}
	for _, op := range t.removes {
		if op.Src != nodeID {
			continue
		}
		for i := range base {
			if base[i].Target == op.Tgt && base[i].GetKind() == op.Kind {
				base = append(base[:i], base[i+1:]...)
				break
			}
		}
	}
	for _, op := range t.adds {
		if op.Src != nodeID {
			continue
		}
		base = append(base, Edge{Target: op.Tgt, Weight: op.Weight, Stamp: uint32(op.Kind) << 24})
	}
	return base, nil
}

// InboundNeighborsAtLSN returns inbound edges (v→nodeID) visible at snapshotLSN.
// Combines live inbound edges with temporal-only edges visible at the snapshot.
func (g *graphStore) InboundNeighborsAtLSN(nodeID uint64, snapshotLSN uint64) ([]Edge, error) {
	liveEdges, err := g.InboundNeighbors(nodeID)
	g.temporalMu.Lock()
	defer g.temporalMu.Unlock()
	if err != nil {
		return nil, err
	}

	// Build result: live edges visible at snapshot + temporal-only inbound edges.
	seen := make(map[edgeTemporalKey]bool)
	var result []Edge

	// Pass 1: include live inbound edges visible at snapshot.
	for _, e := range liveEdges {
		// Inbound: edge goes from e.Target → nodeID.
		key := edgeTemporalKey{Src: e.Target, Tgt: nodeID, Kind: e.GetKind()}
		state, ok := g.temporalEdges[key]
		if !ok {
			result = append(result, e)
			seen[key] = true
			continue
		}
		visible := false
		for i := len(state.Versions) - 1; i >= 0; i-- {
			if state.Versions[i].BeginLSN <= snapshotLSN &&
				(state.Versions[i].EndLSN == 0 || snapshotLSN < state.Versions[i].EndLSN) {
				visible = true
				break
			}
		}
		if visible {
			result = append(result, e)
			seen[key] = true
		}
	}

	// Pass 2: add temporal-only inbound edges visible at snapshot.
	if g.temporalEdges != nil {
		for key, state := range g.temporalEdges {
			if key.Tgt != nodeID || seen[key] {
				continue
			}
			for i := len(state.Versions) - 1; i >= 0; i-- {
				v := state.Versions[i]
				if v.BeginLSN <= snapshotLSN && (v.EndLSN == 0 || snapshotLSN < v.EndLSN) {
					e := Edge{Target: key.Src, Weight: v.Weight}
					e.SetKind(key.Kind)
					result = append(result, e)
					break
				}
			}
		}
	}
	return result, nil
}

// InboundNeighborsOverlay is the inbound counterpart of NeighborsOverlay.
func (t *Txn) InboundNeighborsOverlay(nodeID uint64) ([]Edge, error) {
	if t == nil || t.closed {
		return nil, fmt.Errorf("graph transaction is closed")
	}
	var base []Edge
	var err error
	if t.epochLSN > 0 {
		base, err = t.store.InboundNeighborsAtLSN(nodeID, t.epochLSN)
	} else {
		base, err = t.store.InboundNeighbors(nodeID)
	}
	if err != nil {
		return nil, err
	}
	for _, op := range t.removes {
		if op.Tgt != nodeID {
			continue
		}
		for i := range base {
			if base[i].Target == op.Src && base[i].GetKind() == op.Kind {
				base = append(base[:i], base[i+1:]...)
				break
			}
		}
	}
	for _, op := range t.adds {
		if op.Tgt == nodeID {
			base = append(base, Edge{Target: op.Src, Weight: op.Weight, Stamp: uint32(op.Kind) << 24})
		}
	}
	return base, nil
}

// InboundNeighborsAtLSN is now on the Graph interface for epoch inbound queries.
// It delegates to the store-level implementation above.

// Graph provides edge storage and traversal operations
type Graph interface {
	BeginTxn() *Txn
	AddEdge(txn *Txn, src, tgt uint64, weight float32, kind uint8) error
	RemoveEdge(txn *Txn, src, tgt uint64, kind uint8) error
	DropNodeEdges(txn *Txn, nodeID uint64) error
	Neighbors(nodeID uint64) ([]Edge, error)
	NeighborsAtLSN(nodeID uint64, snapshotLSN uint64) ([]Edge, error)
	Degree(nodeID uint64) (int, error)
	InboundNeighbors(nodeID uint64) ([]Edge, error)
	InboundDegree(nodeID uint64) (int, error)
	NeighborsAny(nodeID uint64, kindSet KindSet) ([]Edge, error)
	ForEachEdge(fn func(src, tgt uint64, edge Edge) bool)

	BFS(start uint64, maxDepth int, visit VisitAction, bitset *Bitset, frontier *FrontierBuf) error
	BFSPattern(start uint64, edges []EdgePlan, maxDepth int, visit VisitAction, bitset *Bitset, frontier *FrontierBuf) error
	GetBitset() (*Bitset, error)
	PutBitset(b *Bitset)
	GetFrontierBuf() (*FrontierBuf, error)
	PutFrontierBuf(f *FrontierBuf)
	Stats() GraphStats
	GraphCentrality(nodeID uint64) float64
	CentralityAtLSN(nodeID uint64, snapshotLSN uint64) float64
	RecordPageRankPublication(snapshotLSN uint64, duration time.Duration)

	// Vertex label registry (MVP: in-memory only, not persisted).
	RegisterVertexLabel(nodeID uint64, label string)
	GetLabelNodes(label string) []uint64

	Close() error
}

type graphStore struct {
	cfg            GraphConfig
	edgePool       *memory.ShardedFreeList
	pagePools      []*memory.ShardedFreeList // segmented
	pageSegments   map[*EdgeTablePage]int
	bitsetPool     *memory.ShardedFreeList
	frontierPool   *memory.ShardedFreeList
	pageReg        *PageRegistry
	index          *EdgeTableIndex
	reverse        *ReverseIndex
	manifest       *DBManifest
	globalStamp    atomic.Uint32
	metrics        storeMetrics
	lastFlushedGen uint32
	nextTxnID      atomic.Uint64
	walWriter      storage.GraphWALWriter

	// MVP node label registry: in-memory only, not persisted.
	// Used for label-scan seeding in graph queries.
	labelToNodes map[string][]uint64 // label → node IDs

	// Temporal edge index: tracks LSN-based visibility for edges.
	// Protected by temporalMu for concurrent access from WAL callbacks
	// and graph queries.
	temporalMu    sync.Mutex
	temporalEdges map[edgeTemporalKey]*edgeTemporalState

	// collectionName is the owning collection for WAL frame identity.
	collectionName string
}

// edgeTemporalKey uniquely identifies a directed edge for temporal tracking.
type edgeTemporalKey struct {
	Src  uint64
	Tgt  uint64
	Kind uint8
}

// edgeTemporalVersion is one visibility interval for an edge.
// The version is visible for snapshot LSNs S where BeginLSN <= S < EndLSN
// (EndLSN == 0 means still live). Versions in a chain are ordered by BeginLSN
// and never overlap.
type edgeTemporalVersion struct {
	BeginLSN uint64
	EndLSN   uint64 // 0 = currently live
	Weight   float32
}

// edgeTemporalState holds the ordered version chain for one edge identity.
type edgeTemporalState struct {
	Versions []edgeTemporalVersion // ascending BeginLSN, non-overlapping
}

// NewGraph initializes the Graph store with off-heap allocators.
func NewGraph(cfg GraphConfig) (Graph, error) {
	edgePool, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  uint64(cfg.EdgeSlots * cfg.EdgeSlotSize),
		SlotSize:  uint64(cfg.EdgeSlotSize),
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 32,
		Prealloc:  false,
	}, 64, cfg.EdgeShards)
	if err != nil {
		return nil, err
	}

	pagePool0, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  uint64(cfg.PageSlots * 4096),
		SlotSize:  4096,
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 32,
		Prealloc:  false,
	}, 64, cfg.PageShards)
	if err != nil {
		edgePool.Free()
		return nil, err
	}

	bitsetPool, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  uint64(cfg.BitsetPoolSize * 131072),
		SlotSize:  131072,
		SlabSize:  uint64(cfg.BitsetPoolSize * 131072),
		SlabCount: 2,
		Prealloc:  false,
	}, 64, 64)
	if err != nil {
		edgePool.Free()
		pagePool0.Free()
		return nil, err
	}

	frontierPool, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  uint64(cfg.FrontierPoolSize * 65536),
		SlotSize:  65536,
		SlabSize:  uint64(cfg.FrontierPoolSize * 65536),
		SlabCount: 2,
		Prealloc:  false,
	}, 64, 64)
	if err != nil {
		edgePool.Free()
		pagePool0.Free()
		bitsetPool.Free()
		return nil, err
	}

	revIdx, err := newReverseIndex(cfg)
	if err != nil {
		edgePool.Free()
		pagePool0.Free()
		bitsetPool.Free()
		frontierPool.Free()
		return nil, err
	}

	return &graphStore{
		cfg:          cfg,
		edgePool:     edgePool,
		pagePools:    []*memory.ShardedFreeList{pagePool0},
		pageSegments: make(map[*EdgeTablePage]int),
		bitsetPool:   bitsetPool,
		frontierPool: frontierPool,
		pageReg:      NewPageRegistry(),
		index:        NewEdgeTableIndex(1024),
		reverse:      revIdx,
		manifest:     NewDBManifest(),
		labelToNodes: make(map[string][]uint64),
	}, nil
}

func tryLockPage(m *uint64) bool {
	for i := 0; i < 100; i++ {
		if atomic.CompareAndSwapUint64(m, 0, 1) {
			return true
		}
		runtime.Gosched()
	}
	return false
}

func retryOp(op func() error) error {
	backoff := 1 * time.Millisecond
	maxBackoff := 64 * time.Millisecond
	for i := 0; i < 100; i++ {
		err := op()
		if err == nil {
			return nil
		}
		if err == ErrConcurrentModification {
			jitter := time.Duration(rand.Intn(1<<min(i, 10))) * time.Microsecond
			time.Sleep(backoff + jitter)
			backoff *= 2
			if backoff > maxBackoff {
				backoff = maxBackoff
			}
			continue
		}
		return err
	}
	return ErrConcurrentModification
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func unlockPage(m *uint64) {
	atomic.StoreUint64(m, 0)
}

// allocatePageSlot is a segmented allocator for the forward page pools.
// For non-segmented pools (reverse index), it falls through to pool.Allocate().
func (g *graphStore) allocatePageSlot(pool *memory.ShardedFreeList, shard int) (*memory.ShardedFreeList, []byte, error) {
	// Only segment the forward page pool; reverse pool is standalone.
	if pool != g.pagePools[0] {
		slot, err := pool.Allocate()
		return pool, slot, err
	}
	for i := len(g.pagePools) - 1; i >= 0; i-- {
		g.pagePools[i].HyalineEnter(shard)
		slot, err := g.pagePools[i].Allocate()
		g.pagePools[i].HyalineLeave(shard)
		if err == nil {
			return g.pagePools[i], slot, nil
		}
		if err != memory.ErrFreelistExhausted {
			return nil, nil, err
		}
	}
	newPool, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  uint64(g.cfg.PageSlots * 4096),
		SlotSize:  4096,
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 1,
		Prealloc:  false,
	}, 64, g.cfg.PageShards)
	if err != nil {
		return nil, nil, err
	}
	g.pagePools = append(g.pagePools, newPool)
	newPool.HyalineEnter(shard)
	slot, err := newPool.Allocate()
	newPool.HyalineLeave(shard)
	if err != nil {
		return nil, nil, err
	}
	return newPool, slot, nil
}

func (g *graphStore) appendEdgeToTable(nodeID uint64, edge Edge, index *EdgeTableIndex, pool *memory.ShardedFreeList) error {
	shard := nodeID % uint64(g.cfg.PageShards)
	pool.HyalineEnter(int(shard))
	defer pool.HyalineLeave(int(shard))

	page := index.Lookup(nodeID)

	if page == nil {
		_, slotBytes, err := g.allocatePageSlot(pool, int(shard))
		if err != nil {
			// Memory exhaustion handling: attempt GC and retry
			runtime.GC()
			_, slotBytes, err = g.allocatePageSlot(pool, int(shard))
			if err != nil {
				return err
			}
		}
		g.metrics.pagesAllocated.Add(1)
		// The user data area starts at offset 64.
		page = (*EdgeTablePage)(unsafe.Pointer(&slotBytes[64]))

		page.Header.Count = 0
		page.Header.InlineCap = 8
		page.Header.Overflow = 0
		page.Header.Generation = 0
		page.Header.Mutex = 0
		page.Header.HyalineSlot = uint16(shard)
		page.Header.LayoutTag = LayoutV2

		page.Header.PageSlot = g.pageReg.Register(page)

		actualPage, loaded := index.InsertIfAbsent(nodeID, page)
		if loaded {
			// Another thread concurrently created the page.
			g.pageReg.Unregister(page.Header.PageSlot)
			pool.Deallocate(slotBytes)
			if pool == g.pagePools[0] {
				// Decrement by 1 using two's complement for atomic Add
				g.metrics.pagesAllocated.Add(^uint64(0))
			}
			page = actualPage
		}
	}

	if !tryLockPage(&page.Header.Mutex) {
		return ErrConcurrentModification
	}

	totalCount := page.Header.Count
	if totalCount < 8 {
		page.Inline[totalCount] = edge
	} else {
		currPage := page
		edgesToSkip := totalCount

		for edgesToSkip >= 250 {
			if currPage.Header.Overflow == 0 {
				_, slotBytes, err := g.allocatePageSlot(pool, int(shard))
				if err != nil {
					// Memory exhaustion handling: attempt GC and retry
					runtime.GC()
					slotBytes, err = pool.Allocate()
					if err != nil {
						unlockPage(&page.Header.Mutex)
						return err
					}
				}
				g.metrics.pagesAllocated.Add(1)
				g.metrics.overfullPages.Add(1)
				newPage := (*EdgeTablePage)(unsafe.Pointer(&slotBytes[64]))
				newPage.Header.Overflow = 0
				newPage.Header.LayoutTag = LayoutV2

				newSlot := g.pageReg.Register(newPage)
				currPage.Header.Overflow = newSlot
			}
			currPage = g.pageReg.Get(currPage.Header.Overflow)
			edgesToSkip -= 250
		}

		if edgesToSkip < 8 {
			currPage.Inline[edgesToSkip] = edge
		} else {
			idx := edgesToSkip - 8
			extra := unsafe.Slice((*Edge)(unsafe.Pointer(&currPage.Padding[0])), 242)
			extra[idx] = edge
		}
	}

	page.Header.Count++

	atomic.AddUint32(&page.Header.Generation, 1)
	unlockPage(&page.Header.Mutex)

	return nil
}

func (g *graphStore) removeEdgeFromTable(nodeID uint64, targetToRemove uint64, kindToRemove uint8, index *EdgeTableIndex, pool *memory.ShardedFreeList) error {
	shard := nodeID % uint64(g.cfg.PageShards)
	pool.HyalineEnter(int(shard))
	defer pool.HyalineLeave(int(shard))

	page := index.Lookup(nodeID)
	if page == nil {
		return ErrEdgeNotFound
	}

	// page already set

	if !tryLockPage(&page.Header.Mutex) {
		return ErrConcurrentModification
	}
	defer unlockPage(&page.Header.Mutex)

	totalCount := page.Header.Count
	if totalCount == 0 {
		return ErrEdgeNotFound
	}

	var targetEdgePtr *Edge
	var lastEdgePtr *Edge
	var prevToLastPage *EdgeTablePage
	var lastPage *EdgeTablePage = page

	currPage := page
	remaining := totalCount

	for currPage != nil && remaining > 0 {
		pageCount := remaining
		if pageCount > 250 {
			pageCount = 250
		}

		inlineLimit := pageCount
		if inlineLimit > 8 {
			inlineLimit = 8
		}
		for i := uint16(0); i < inlineLimit; i++ {
			edge := &currPage.Inline[i]
			if targetEdgePtr == nil && edge.Target == targetToRemove && edge.GetKind() == kindToRemove {
				targetEdgePtr = edge
			}
			if remaining == 1 {
				lastEdgePtr = edge
			}
			remaining--
		}

		if pageCount > 8 {
			extraCount := pageCount - 8
			extra := unsafe.Slice((*Edge)(unsafe.Pointer(&currPage.Padding[0])), 242)
			for i := uint16(0); i < extraCount; i++ {
				edge := &extra[i]
				if targetEdgePtr == nil && edge.Target == targetToRemove && edge.GetKind() == kindToRemove {
					targetEdgePtr = edge
				}
				if remaining == 1 {
					lastEdgePtr = edge
				}
				remaining--
			}
		}

		if remaining > 0 {
			if currPage.Header.Overflow != 0 {
				prevToLastPage = currPage
				currPage = g.pageReg.Get(currPage.Header.Overflow)
				lastPage = currPage
			} else {
				currPage = nil
			}
		} else {
			currPage = nil
		}
	}

	if targetEdgePtr == nil {
		return ErrEdgeNotFound
	}

	*targetEdgePtr = *lastEdgePtr

	page.Header.Count--
	atomic.AddUint32(&page.Header.Generation, 1)

	if totalCount > 250 && (totalCount-1)%250 == 0 {
		if prevToLastPage != nil {
			prevToLastPage.Header.Overflow = 0
			slotBytes := unsafe.Slice((*byte)(unsafe.Add(unsafe.Pointer(lastPage), -64)), 4096)
			pool.Retire(slotBytes)
		}
	}
	return nil
}

func (g *graphStore) neighborsFromTable(nodeID uint64, index *EdgeTableIndex, pool *memory.ShardedFreeList, numShards int) ([]Edge, error) {
	shard := nodeID % uint64(numShards)

retry:
	pool.HyalineEnter(int(shard))

	page := index.Lookup(nodeID)
	if page == nil {
		pool.HyalineLeave(int(shard))
		return []Edge{}, nil
	}

	// page already set
	gen := atomic.LoadUint32(&page.Header.Generation)
	totalCount := page.Header.Count

	edges := make([]Edge, 0, totalCount)

	currPage := page
	remaining := totalCount

	for currPage != nil && remaining > 0 {
		pageCount := remaining
		if pageCount > 250 {
			pageCount = 250
		}

		if pageCount <= 8 {
			edges = append(edges, currPage.Inline[:pageCount]...)
			remaining -= pageCount
		} else {
			edges = append(edges, currPage.Inline[:8]...)
			extra := unsafe.Slice((*Edge)(unsafe.Pointer(&currPage.Padding[0])), 242)
			extraCount := pageCount - 8
			edges = append(edges, extra[:extraCount]...)
			remaining -= pageCount
		}

		if currPage.Header.Overflow != 0 {
			currPage = g.pageReg.Get(currPage.Header.Overflow)
			g.metrics.chainedPageReads.Add(1)
		} else {
			currPage = nil
		}
	}

	if atomic.LoadUint32(&page.Header.Generation) != gen {
		pool.HyalineLeave(int(shard))
		goto retry
	}

	pool.HyalineLeave(int(shard))
	return edges, nil
}

// BeginTxn starts a new graph transaction.
func (g *graphStore) BeginTxn() *Txn {
	return &Txn{
		ID:         g.nextTxnID.Add(1),
		walWriter:  g.walWriter,
		store:      g,
		collection: g.collectionName,
	}
}

// SetWALWriter wires the storage engine's WAL writer to the graph so
// Txn.Commit() durably records edge mutations.
func (g *graphStore) SetWALWriter(w storage.GraphWALWriter) {
	g.walWriter = w
}

// SetCollectionName stores the owning collection name. Called by
// Collection.SetGraph so that Txn.AddEdge/Txn.RemoveEdge propagate
// the collection identity into WAL frames for per-collection recovery.
func (g *graphStore) SetCollectionName(name string) {
	g.collectionName = name
}

// ── Temporal edge visibility ──────────────────────────────────────────

// recordEdgeCommitLSN stamps one committed graph transaction. The operation
// lists are deliberately transaction-local: a storage flush can coalesce
// independently submitted graph transactions, so a graph-wide pending queue
// cannot safely be associated with one commit LSN.
func (g *graphStore) recordEdgeCommitLSN(lsn uint64, adds, removes []storage.GraphEdgeOp, nodeDrops []storage.GraphNodeDropOp) {
	g.temporalMu.Lock()
	defer g.temporalMu.Unlock()
	if g.temporalEdges == nil {
		g.temporalEdges = make(map[edgeTemporalKey]*edgeTemporalState)
	}
	for _, add := range adds {
		key := edgeTemporalKey{Src: add.Src, Tgt: add.Tgt, Kind: add.Kind}
		state, ok := g.temporalEdges[key]
		if !ok {
			state = &edgeTemporalState{}
			g.temporalEdges[key] = state
		}
		// Close any currently live version at this LSN, then append a new one.
		for i := range state.Versions {
			if state.Versions[i].EndLSN == 0 {
				state.Versions[i].EndLSN = lsn
			}
		}
		state.Versions = append(state.Versions, edgeTemporalVersion{
			BeginLSN: lsn, EndLSN: 0, Weight: add.Weight,
		})
	}
	for _, remove := range removes {
		key := edgeTemporalKey{Src: remove.Src, Tgt: remove.Tgt, Kind: remove.Kind}
		if state, ok := g.temporalEdges[key]; ok {
			for i := range state.Versions {
				if state.Versions[i].EndLSN == 0 {
					state.Versions[i].EndLSN = lsn
				}
			}
		}
	}
	for _, drop := range nodeDrops {
		nid := drop.NodeID
		for key, state := range g.temporalEdges {
			if key.Src != nid && key.Tgt != nid {
				continue
			}
			for i := range state.Versions {
				if state.Versions[i].EndLSN == 0 {
					state.Versions[i].EndLSN = lsn
				}
			}
		}
	}
}

// RecordEdgeAddLSN is used during WAL replay to directly register an edge
// add at a known LSN without going through the pending queue.
func (g *graphStore) RecordEdgeAddLSN(src, tgt uint64, weight float32, kind uint8, lsn uint64) {
	g.temporalMu.Lock()
	defer g.temporalMu.Unlock()
	if g.temporalEdges == nil {
		g.temporalEdges = make(map[edgeTemporalKey]*edgeTemporalState)
	}
	key := edgeTemporalKey{Src: src, Tgt: tgt, Kind: kind}
	state, ok := g.temporalEdges[key]
	if !ok {
		state = &edgeTemporalState{}
		g.temporalEdges[key] = state
	}
	// Close any live version, append new one.
	for i := range state.Versions {
		if state.Versions[i].EndLSN == 0 {
			state.Versions[i].EndLSN = lsn
		}
	}
	state.Versions = append(state.Versions, edgeTemporalVersion{
		BeginLSN: lsn, EndLSN: 0, Weight: weight,
	})
}

// RecordEdgeRemoveLSN is the replay counterpart of RecordEdgeAddLSN.
func (g *graphStore) RecordEdgeRemoveLSN(src, tgt uint64, kind uint8, lsn uint64) {
	g.temporalMu.Lock()
	defer g.temporalMu.Unlock()
	if g.temporalEdges == nil {
		g.temporalEdges = make(map[edgeTemporalKey]*edgeTemporalState)
	}
	key := edgeTemporalKey{Src: src, Tgt: tgt, Kind: kind}
	if state, ok := g.temporalEdges[key]; ok {
		for i := range state.Versions {
			if state.Versions[i].EndLSN == 0 {
				state.Versions[i].EndLSN = lsn
			}
		}
	}
}

// NeighborsAtLSN returns edges from nodeID that are visible at snapshotLSN.
// It combines live edges with temporal-only edges (no longer live) that were
// visible at the snapshot. Pre-temporal edges (no temporal state) are always
// included. Temporal edges are filtered by version chain visibility.
func (g *graphStore) NeighborsAtLSN(nodeID uint64, snapshotLSN uint64) ([]Edge, error) {
	liveEdges, err := g.Neighbors(nodeID)
	g.temporalMu.Lock()
	defer g.temporalMu.Unlock()
	if err != nil {
		return nil, err
	}

	// Build result set: start with live edges, remove those not visible
	// at snapshot, add temporal-only edges that were visible.
	seen := make(map[edgeTemporalKey]bool)
	var result []Edge

	// Pass 1: include live edges visible at snapshot.
	for _, e := range liveEdges {
		key := edgeTemporalKey{Src: nodeID, Tgt: e.Target, Kind: e.GetKind()}
		state, ok := g.temporalEdges[key]
		if !ok {
			result = append(result, e)
			seen[key] = true
			continue
		}
		visible := false
		for i := len(state.Versions) - 1; i >= 0; i-- {
			if state.Versions[i].BeginLSN <= snapshotLSN &&
				(state.Versions[i].EndLSN == 0 || snapshotLSN < state.Versions[i].EndLSN) {
				visible = true
				break
			}
		}
		if visible {
			result = append(result, e)
			seen[key] = true
		}
	}

	// Pass 2: add temporal-only edges (no longer live but visible at snapshot).
	if g.temporalEdges != nil {
		for key, state := range g.temporalEdges {
			if key.Src != nodeID || seen[key] {
				continue
			}
			for i := len(state.Versions) - 1; i >= 0; i-- {
				v := state.Versions[i]
				if v.BeginLSN <= snapshotLSN && (v.EndLSN == 0 || snapshotLSN < v.EndLSN) {
					e := Edge{Target: key.Tgt, Weight: v.Weight}
					e.SetKind(key.Kind)
					result = append(result, e)
					break
				}
			}
		}
	}
	return result, nil
}

// clonePageChain deep-copies a page and its entire overflow chain.
// The clone is allocated from the given pool and registered in the page
// registry. The spinlock is reset (private page, no concurrent writers).
// Shallow copy of the page body is safe: Inline[8]Edge is value-typed and
// Padding[3872]byte has no pointers. Only Header.Overflow (uint32 slot ID)
// needs updating for cloned overflow pages.
func (g *graphStore) clonePageChain(head *EdgeTablePage, pool *memory.ShardedFreeList, shard int) (*EdgeTablePage, error) {
	if head == nil {
		return nil, nil
	}

	_, slotBytes, err := g.allocatePageSlot(pool, shard)
	if err != nil {
		return nil, err
	}
	g.metrics.pagesAllocated.Add(1)

	cloned := (*EdgeTablePage)(unsafe.Pointer(&slotBytes[64]))
	*cloned = *head // shallow copy entire 4096 bytes
	cloned.Header.Mutex = 0
	cloned.Header.PageSlot = g.pageReg.Register(cloned)

	// Clone overflow chain recursively.
	if head.Header.Overflow != 0 {
		overflow := g.pageReg.Get(head.Header.Overflow)
		clonedOverflow, err := g.clonePageChain(overflow, pool, shard)
		if err != nil {
			g.pageReg.Unregister(cloned.Header.PageSlot)
			pool.Deallocate(slotBytes)
			g.metrics.pagesAllocated.Add(^uint64(0)) // decrement
			return nil, err
		}
		cloned.Header.Overflow = clonedOverflow.Header.PageSlot
	}

	return cloned, nil
}

// newPage allocates and initializes a fresh edge table page.
func (g *graphStore) newPage(pool *memory.ShardedFreeList, shard int) *EdgeTablePage {
	_, slotBytes, err := g.allocatePageSlot(pool, shard)
	if err != nil {
		return nil
	}
	g.metrics.pagesAllocated.Add(1)
	page := (*EdgeTablePage)(unsafe.Pointer(&slotBytes[64]))
	page.Header.Count = 0
	page.Header.InlineCap = 8
	page.Header.Overflow = 0
	page.Header.Generation = 0
	page.Header.Mutex = 0
	page.Header.HyalineSlot = uint16(shard)
	page.Header.LayoutTag = LayoutV2
	page.Header.PageSlot = g.pageReg.Register(page)
	return page
}

func (g *graphStore) retirePageChain(nodeID uint64, index *EdgeTableIndex, pool *memory.ShardedFreeList) {
	page := index.Lookup(nodeID)
	if page == nil {
		return
	}

	index.Delete(nodeID)

	// page already set

	if !tryLockPage(&page.Header.Mutex) {
		// retirePageChain is internal cleanup, if we fail to lock, we just skip it or retry
		// But let's just spin in a fallback loop for cleanup since it must complete
		for !tryLockPage(&page.Header.Mutex) {
			time.Sleep(time.Millisecond)
		}
	}
	defer unlockPage(&page.Header.Mutex)

	currPage := page
	for currPage != nil {
		nextSlot := currPage.Header.Overflow
		slotBytes := unsafe.Slice((*byte)(unsafe.Add(unsafe.Pointer(currPage), -64)), 4096)
		pool.Retire(slotBytes)

		if nextSlot != 0 {
			currPage = g.pageReg.Get(nextSlot)
		} else {
			currPage = nil
		}
	}
}

func (g *graphStore) AddEdge(txn *Txn, src, tgt uint64, weight float32, kind uint8) error {
	if txn == nil {
		return ErrNoTransaction
	}
	stamp := g.globalStamp.Add(1)
	return g.AddEdgeWithStamp(txn, src, tgt, weight, kind, stamp)
}

func (g *graphStore) AddEdgeWithStamp(txn *Txn, src, tgt uint64, weight float32, kind uint8, stamp uint32) error {
	fEdge := Edge{Target: tgt, Weight: weight}
	fEdge.SetStamp(stamp)
	fEdge.SetKind(kind)
	err := retryOp(func() error {
		return g.appendEdgeToTable(src, fEdge, g.index, g.pagePools[0])
	})
	if err != nil {
		return err
	}

	rEdge := Edge{Target: src, Weight: weight}
	rEdge.SetStamp(stamp)
	rEdge.SetKind(kind)
	err = retryOp(func() error {
		return g.appendEdgeToTable(tgt, rEdge, g.reverse.locator, g.reverse.pool)
	})
	if err != nil {
		_ = retryOp(func() error {
			return g.removeEdgeFromTable(src, tgt, kind, g.index, g.pagePools[0])
		})
		return err
	}

	g.metrics.edgesAdded.Add(1)
	return nil
}

// addEdgeInternal performs the in-memory edge insertion without Txn validation
// or WAL recording. Used by ReplayEdgeAdd during recovery when the WAL frame
// is already committed.
func (g *graphStore) addEdgeInternal(txn *Txn, src, tgt uint64, weight float32, kind uint8) error {
	stamp := g.globalStamp.Add(1)
	return g.AddEdgeWithStamp(txn, src, tgt, weight, kind, stamp)
}

// removeEdgeInternal performs the in-memory edge removal without Txn validation
// or WAL recording. Used by ReplayEdgeRemove during recovery.
func (g *graphStore) removeEdgeInternal(txn *Txn, src, tgt uint64, kind uint8) error {
	return g.RemoveEdge(txn, src, tgt, kind)
}

// dropNodeEdgesInternal performs the in-memory node drop without Txn validation
// or WAL recording. Used by ReplayNodeEdgeDrop during recovery.
func (g *graphStore) dropNodeEdgesInternal(txn *Txn, nodeID uint64) error {
	return g.DropNodeEdges(txn, nodeID)
}

// ReplayEdgeAdd replays a committed edge-add from the WAL during recovery.
func (g *graphStore) ReplayEdgeAdd(src, tgt uint64, weight float32, kind uint8, commitLSN uint64) error {
	if err := g.addEdgeInternal(nil, src, tgt, weight, kind); err != nil {
		return err
	}
	g.RecordEdgeAddLSN(src, tgt, weight, kind, commitLSN)
	return nil
}

// ReplayEdgeRemove replays a committed edge-remove from the WAL during recovery.
func (g *graphStore) ReplayEdgeRemove(src, tgt uint64, kind uint8, commitLSN uint64) error {
	if err := g.removeEdgeInternal(nil, src, tgt, kind); err != nil {
		return err
	}
	g.RecordEdgeRemoveLSN(src, tgt, kind, commitLSN)
	return nil
}

// ReplayNodeEdgeDrop replays a committed node-edge-drop from the WAL during recovery.
func (g *graphStore) ReplayNodeEdgeDrop(nodeID uint64, commitLSN uint64) error {
	if err := g.dropNodeEdgesInternal(nil, nodeID); err != nil {
		return err
	}
	g.recordEdgeCommitLSN(commitLSN, nil, nil, []storage.GraphNodeDropOp{{NodeID: nodeID}})
	return nil
}

func (g *graphStore) ReplayVertexLabel(nodeID uint64, label string, _ uint64) error {
	g.registerVertexLabel(nodeID, label)
	return nil
}

func (g *graphStore) applyCommittedOps(adds, removes []storage.GraphEdgeOp, nodeDrops []storage.GraphNodeDropOp) error {
	for _, add := range adds {
		if err := g.addEdgeInternal(nil, add.Src, add.Tgt, add.Weight, add.Kind); err != nil {
			return err
		}
	}
	for _, remove := range removes {
		if err := g.removeEdgeInternal(nil, remove.Src, remove.Tgt, remove.Kind); err != nil {
			return err
		}
	}
	for _, drop := range nodeDrops {
		if err := g.dropNodeEdgesInternal(nil, drop.NodeID); err != nil {
			return err
		}
	}
	if len(adds) > 0 || len(removes) > 0 || len(nodeDrops) > 0 {
		g.metrics.mutationGeneration.Add(1)
	}
	return nil
}

// RecordPageRankPublication records an atomically published derived PageRank
// vector. The computation itself is intentionally separate from graph writes;
// controllers can use Stats to decide when to run it.
func (g *graphStore) RecordPageRankPublication(snapshotLSN uint64, duration time.Duration) {
	g.metrics.lastPageRankGeneration.Store(g.metrics.mutationGeneration.Load())
	g.metrics.lastPageRankLSN.Store(snapshotLSN)
	g.metrics.pageRankDuration.Store(duration.Nanoseconds())
	g.metrics.pageRankAvailable.Store(true)
}

func (g *graphStore) edge(src, tgt uint64, kind uint8) (Edge, error) {
	edges, err := g.Neighbors(src)
	if err != nil {
		return Edge{}, err
	}
	for _, edge := range edges {
		if edge.Target == tgt && edge.GetKind() == kind {
			return edge, nil
		}
	}
	return Edge{}, ErrEdgeNotFound
}

func (g *graphStore) RemoveEdge(txn *Txn, src, tgt uint64, kind uint8) error {
	edges, _ := g.Neighbors(src)
	var weight float32
	var stamp uint32
	var found bool
	for _, e := range edges {
		if e.Target == tgt && e.GetKind() == kind {
			weight = e.Weight
			stamp = e.GetStamp()
			found = true
			break
		}
	}
	if !found {
		return ErrEdgeNotFound
	}

	err := retryOp(func() error {
		return g.removeEdgeFromTable(src, tgt, kind, g.index, g.pagePools[0])
	})
	if err != nil {
		return err
	}

	err = retryOp(func() error {
		return g.removeEdgeFromTable(tgt, src, kind, g.reverse.locator, g.reverse.pool)
	})
	if err != nil && err != ErrEdgeNotFound {
		// Rollback forward remove
		fEdge := Edge{Target: tgt, Weight: weight}
		fEdge.SetStamp(stamp)
		fEdge.SetKind(kind)
		_ = retryOp(func() error {
			return g.appendEdgeToTable(src, fEdge, g.index, g.pagePools[0])
		})
		return err
	}

	g.metrics.edgesRemoved.Add(1)
	return nil
}

func (g *graphStore) DropNodeEdges(txn *Txn, nodeID uint64) error {
	var firstErr error
	inboundEdges, _ := g.neighborsFromTable(nodeID, g.reverse.locator, g.reverse.pool, g.cfg.PageShards)
	for _, edge := range inboundEdges {
		err := retryOp(func() error {
			return g.removeEdgeFromTable(edge.Target, nodeID, edge.GetKind(), g.index, g.pagePools[0])
		})
		if err != nil && err != ErrEdgeNotFound && firstErr == nil {
			firstErr = err
		}
	}

	outboundEdges, _ := g.neighborsFromTable(nodeID, g.index, g.pagePools[0], g.cfg.PageShards)
	for _, edge := range outboundEdges {
		err := retryOp(func() error {
			return g.removeEdgeFromTable(edge.Target, nodeID, edge.GetKind(), g.reverse.locator, g.reverse.pool)
		})
		if err != nil && err != ErrEdgeNotFound && firstErr == nil {
			firstErr = err
		}
	}

	g.retirePageChain(nodeID, g.index, g.pagePools[0])
	g.retirePageChain(nodeID, g.reverse.locator, g.reverse.pool)

	return firstErr
}

func (g *graphStore) Neighbors(nodeID uint64) ([]Edge, error) {
	return g.neighborsFromTable(nodeID, g.index, g.pagePools[0], g.cfg.PageShards)
}

func (g *graphStore) InboundNeighbors(nodeID uint64) ([]Edge, error) {
	return g.neighborsFromTable(nodeID, g.reverse.locator, g.reverse.pool, g.cfg.PageShards)
}

func (g *graphStore) Degree(nodeID uint64) (int, error) {
	return g.degreeFromTable(nodeID, g.index, g.pagePools[0], g.cfg.PageShards)
}

func (g *graphStore) InboundDegree(nodeID uint64) (int, error) {
	return g.degreeFromTable(nodeID, g.reverse.locator, g.reverse.pool, g.cfg.PageShards)
}

// GraphCentrality returns normalized inbound degree centrality for nodeID.
// Computed as inbound_degree(node) / max_inbound_degree in the graph.
// Returns 0.0 for zero-degree nodes and empty graphs.
func (g *graphStore) GraphCentrality(nodeID uint64) float64 {
	deg, err := g.InboundDegree(nodeID)
	if err != nil || deg == 0 {
		return 0.0
	}
	stats := g.Stats()
	if stats.EdgesAdded == 0 {
		return 0.0
	}
	// Use total edges as normalization factor (approximation of max degree).
	maxDeg := 0
	g.ForEachEdge(func(src, tgt uint64, e Edge) bool {
		if d, _ := g.InboundDegree(tgt); d > maxDeg {
			maxDeg = d
		}
		return true
	})
	if maxDeg == 0 {
		return 0.0
	}
	return float64(deg) / float64(maxDeg)
}

// CentralityAtLSN returns normalized inbound degree centrality for nodeID at
// the given snapshot LSN, using only edges visible at that LSN. Scans the
// temporal edge index to count inbound edges and find the maximum.
func (g *graphStore) CentralityAtLSN(nodeID uint64, snapshotLSN uint64) float64 {
	g.temporalMu.Lock()
	defer g.temporalMu.Unlock()

	if g.temporalEdges == nil {
		return 0.0
	}
	inbound := 0
	inboundByNode := make(map[uint64]int)
	for key, state := range g.temporalEdges {
		visible := false
		for i := len(state.Versions) - 1; i >= 0; i-- {
			v := state.Versions[i]
			if v.BeginLSN <= snapshotLSN && (v.EndLSN == 0 || snapshotLSN < v.EndLSN) {
				visible = true
				break
			}
		}
		if visible {
			inboundByNode[key.Tgt]++
			if key.Tgt == nodeID {
				inbound++
			}
		}
	}
	maxInbound := 0
	for _, c := range inboundByNode {
		if c > maxInbound {
			maxInbound = c
		}
	}
	if maxInbound == 0 {
		return 0.0
	}
	return float64(inbound) / float64(maxInbound)
}

func (g *graphStore) ForEachEdge(fn func(src, tgt uint64, edge Edge) bool) {
	stop := false
	g.index.Iterate(func(nodeID uint64) {
		if stop {
			return
		}
		edges, err := g.Neighbors(nodeID)
		if err != nil {
			return
		}
		for _, e := range edges {
			if !fn(nodeID, e.Target, e) {
				stop = true
				return
			}
		}
	})
}

func (g *graphStore) degreeFromTable(nodeID uint64, index *EdgeTableIndex, pool *memory.ShardedFreeList, shards int) (int, error) {
	shard := nodeID % uint64(shards)

retry:
	pool.HyalineEnter(int(shard))

	page := index.Lookup(nodeID)
	if page == nil {
		pool.HyalineLeave(int(shard))
		return 0, nil
	}

	// page already set
	gen := atomic.LoadUint32(&page.Header.Generation)
	count := int(page.Header.Count)

	if atomic.LoadUint32(&page.Header.Generation) != gen {
		pool.HyalineLeave(int(shard))
		goto retry
	}

	pool.HyalineLeave(int(shard))
	return count, nil
}

func (g *graphStore) NeighborsAny(nodeID uint64, kindSet KindSet) ([]Edge, error) {
	edges, err := g.Neighbors(nodeID)
	if err != nil {
		return nil, err
	}

	filtered := make([]Edge, 0, len(edges))
	for _, e := range edges {
		if kindSet.Has(e.GetKind()) {
			filtered = append(filtered, e)
		}
	}
	return filtered, nil
}

func (g *graphStore) Stats() GraphStats {
	stats := g.metrics.get()
	stats.OffHeapMemory = g.edgePool.Stats().Allocated +
		g.pagePools[0].Stats().Allocated +
		g.bitsetPool.Stats().Allocated +
		g.frontierPool.Stats().Allocated
	return stats
}

func (g *graphStore) GetBitset() (*Bitset, error) {
	slot, err := g.bitsetPool.Allocate()
	if err != nil {
		return nil, fmt.Errorf("Bitset Allocate failed: %w", err)
	}
	if slot == nil {
		return nil, fmt.Errorf("Bitset Allocate returned nil slot")
	}
	return newBitset(slot), nil
}

func (g *graphStore) PutBitset(b *Bitset) {
	if b == nil || b.slot == nil {
		return
	}
	// We can Retire the slot when we're done
	// Normally if the buffer was used in a read-only way, we can Deallocate,
	// but Retire is safer if it's managed via Hyaline.
	// Actually, these pools are purely for temporary caller buffers,
	// so we can use Retire or just Deallocate if it's strictly local to the caller.
	// For ShardedFreeList, we just Retire it.
	g.bitsetPool.Retire(b.slot)
}

func (g *graphStore) GetFrontierBuf() (*FrontierBuf, error) {
	slot, err := g.frontierPool.Allocate()
	if err != nil {
		return nil, fmt.Errorf("FrontierBuf Allocate failed: %w", err)
	}
	if slot == nil {
		return nil, fmt.Errorf("FrontierBuf Allocate returned nil slot")
	}
	return newFrontierBuf(slot), nil
}

func (g *graphStore) PutFrontierBuf(f *FrontierBuf) {
	if f == nil || f.slot == nil {
		return
	}
	g.frontierPool.Retire(f.slot)
}

func (g *graphStore) Close() error {
	var indexErr, err1, err2, err3, err4 error
	if g.index != nil {
		indexErr = g.index.Close()
		if indexErr == nil {
			g.index = nil
		}
	}
	if g.edgePool != nil {
		err1 = g.edgePool.Free()
	}
	if len(g.pagePools) > 0 {
		for _, p := range g.pagePools {
			p.Free()
		}
	}
	if g.bitsetPool != nil {
		err3 = g.bitsetPool.Free()
	}
	if g.frontierPool != nil {
		err4 = g.frontierPool.Free()
	}
	if g.reverse != nil {
		if err := g.reverse.Close(); err != nil && err4 == nil {
			err4 = err
		} else if err == nil {
			g.reverse = nil
		}
	}
	if indexErr != nil {
		return indexErr
	}
	if err1 != nil {
		return err1
	}
	if err2 != nil {
		return err2
	}
	if err3 != nil {
		return err3
	}
	return err4
}

func (g *graphStore) rebuildReverseIndex() {
	g.index.Iterate(func(nodeID uint64) {
		edges, _ := g.Neighbors(nodeID)
		for _, e := range edges {
			rEdge := Edge{Target: nodeID, Weight: e.Weight}
			rEdge.SetStamp(e.GetStamp())
			rEdge.SetKind(e.GetKind())
			_ = g.appendEdgeToTable(e.Target, rEdge, g.reverse.locator, g.reverse.pool)
		}
	})
}

// RegisterVertexLabel assigns a label to a graph node. This mapping is
// in-memory only and is not persisted. It is an MVP feature for label-scan
// seeding in graph queries.
func (g *graphStore) RegisterVertexLabel(nodeID uint64, label string) {
	g.registerVertexLabel(nodeID, label)
	if writer, ok := g.walWriter.(storage.GraphLabelWALWriter); ok {
		_ = writer.AppendGraphLabel(context.Background(), nodeID, label, nil)
	}
}

func (g *graphStore) registerVertexLabel(nodeID uint64, label string) {
	if g.labelToNodes == nil {
		g.labelToNodes = make(map[string][]uint64)
	}
	// Simple append — duplicates are possible if called twice for same node+label.
	// The caller is responsible for deduplication if needed.
	g.labelToNodes[label] = append(g.labelToNodes[label], nodeID)
}

// GetLabelNodes returns all node IDs registered under the given label.
// Returns nil if no nodes have the label.
func (g *graphStore) GetLabelNodes(label string) []uint64 {
	return g.labelToNodes[label]
}
