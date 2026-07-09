package graph

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/storage/wal"
	"github.com/xDarkicex/memory"
)

// Txn is a minimal transaction context for graph operations.
type Txn struct {
	ID      uint64
	wal     *wal.WAL
	records []*wal.Entry
	store   *graphStore
}

// AppendRecord accumulates a WAL record in the transaction.
func (t *Txn) AppendRecord(entry *wal.Entry) {
	t.records = append(t.records, entry)
}

// Commit flushes all accumulated records to the WAL, including a commit marker.
func (t *Txn) Commit(ctx context.Context) error {
	if t.wal == nil {
		return nil
	}
	commitRec := &WALTxnCommitRecord{TxnID: t.ID}
	entry := &wal.Entry{
		Operation: wal.OpTxnCommit,
		Data:      SerializeWALTxnCommitRecord(commitRec),
	}
	t.records = append(t.records, entry)
	return t.wal.AppendBatch(ctx, t.records)
}

// AddEdge adds a directed edge to the graph within this transaction.
func (t *Txn) AddEdge(src, tgt uint64, weight float32, kind uint8) error {
	return t.store.AddEdge(t, src, tgt, weight, kind)
}

// RemoveEdge removes a directed edge from the graph within this transaction.
func (t *Txn) RemoveEdge(src, tgt uint64, kind uint8) error {
	return t.store.RemoveEdge(t, src, tgt, kind)
}

// Graph provides edge storage and traversal operations
type Graph interface {
	BeginTxn() *Txn
	AddEdge(txn *Txn, src, tgt uint64, weight float32, kind uint8) error
	RemoveEdge(txn *Txn, src, tgt uint64, kind uint8) error
	DropNodeEdges(txn *Txn, nodeID uint64) error
	Neighbors(nodeID uint64) ([]Edge, error)
	Degree(nodeID uint64) (int, error)
	InboundNeighbors(nodeID uint64) ([]Edge, error)
	InboundDegree(nodeID uint64) (int, error)
	NeighborsAny(nodeID uint64, kindSet KindSet) ([]Edge, error)
	ForEachEdge(fn func(src, tgt uint64, edge Edge) bool)

	BFS(start uint64, maxDepth int, visit VisitAction, bitset *Bitset, frontier *FrontierBuf) error
	GetBitset() (*Bitset, error)
	PutBitset(b *Bitset)
	GetFrontierBuf() (*FrontierBuf, error)
	PutFrontierBuf(f *FrontierBuf)
	Stats() GraphStats

	Close() error
}

type graphStore struct {
	cfg            GraphConfig
	edgePool       *memory.ShardedFreeList
	pagePool       *memory.ShardedFreeList
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
	wal            *wal.WAL
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

	pagePool, err := memory.NewShardedFreeList(memory.FreeListConfig{
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
		pagePool.Free()
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
		pagePool.Free()
		bitsetPool.Free()
		return nil, err
	}

	revIdx, err := newReverseIndex(cfg)
	if err != nil {
		edgePool.Free()
		pagePool.Free()
		bitsetPool.Free()
		frontierPool.Free()
		return nil, err
	}

	return &graphStore{
		cfg:          cfg,
		edgePool:     edgePool,
		pagePool:     pagePool,
		bitsetPool:   bitsetPool,
		frontierPool: frontierPool,
		pageReg:      NewPageRegistry(),
		index:        NewEdgeTableIndex(1024),
		reverse:      revIdx,
		manifest:     NewDBManifest(),
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

func (g *graphStore) appendEdgeToTable(nodeID uint64, edge Edge, index *EdgeTableIndex, pool *memory.ShardedFreeList) error {
	shard := nodeID % uint64(g.cfg.PageShards)
	pool.HyalineEnter(int(shard))
	defer pool.HyalineLeave(int(shard))

	pageSlot := index.Lookup(nodeID)
	var page *EdgeTablePage

	if pageSlot == 0 {
		slotBytes, err := pool.Allocate()
		if err != nil {
			// Memory exhaustion handling: attempt GC and retry
			runtime.GC()
			slotBytes, err = pool.Allocate()
			if err != nil {
				return err
			}
		}
		if pool == g.pagePool {
			g.metrics.pagesAllocated.Add(1)
		}
		// The user data area starts at offset 64.
		page = (*EdgeTablePage)(unsafe.Pointer(&slotBytes[64]))

		page.Header.Count = 0
		page.Header.InlineCap = 8
		page.Header.Overflow = 0
		page.Header.Generation = 0
		page.Header.Mutex = 0
		page.Header.HyalineSlot = uint16(shard)
		page.Header.LayoutTag = LayoutV2

		pageSlot = g.pageReg.Register(page)
		index.Insert(nodeID, pageSlot)
	} else {
		page = g.pageReg.Get(pageSlot)
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
				slotBytes, err := pool.Allocate()
				if err != nil {
					// Memory exhaustion handling: attempt GC and retry
					runtime.GC()
					slotBytes, err = pool.Allocate()
					if err != nil {
						unlockPage(&page.Header.Mutex)
						return err
					}
				}
				if pool == g.pagePool {
					g.metrics.pagesAllocated.Add(1)
					g.metrics.overfullPages.Add(1)
				}
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

	pageSlot := index.Lookup(nodeID)
	if pageSlot == 0 {
		return ErrEdgeNotFound
	}

	page := g.pageReg.Get(pageSlot)

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

	pageSlot := index.Lookup(nodeID)
	if pageSlot == 0 {
		pool.HyalineLeave(int(shard))
		return []Edge{}, nil
	}

	page := g.pageReg.Get(pageSlot)
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
			if pool == g.pagePool {
				g.metrics.chainedPageReads.Add(1)
			}
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
		ID:    g.nextTxnID.Add(1),
		wal:   g.wal,
		store: g,
	}
}

func (g *graphStore) retirePageChain(nodeID uint64, index *EdgeTableIndex, pool *memory.ShardedFreeList) {
	pageSlot := index.Lookup(nodeID)
	if pageSlot == 0 {
		return
	}

	index.Delete(nodeID)

	page := g.pageReg.Get(pageSlot)

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
		return g.appendEdgeToTable(src, fEdge, g.index, g.pagePool)
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
			return g.removeEdgeFromTable(src, tgt, kind, g.index, g.pagePool)
		})
		return err
	}

	if txn != nil {
		record := &WALEdgeAddRecord{
			TxnID:  txn.ID,
			From:   src,
			To:     tgt,
			Weight: weight,
			Stamp:  stamp,
			Kind:   kind,
		}
		txn.AppendRecord(&wal.Entry{
			Operation: wal.OpEdgeAdd,
			Timestamp: uint64(time.Now().UnixNano()),
			Data:      SerializeWALEdgeAddRecord(record),
		})
	}

	g.metrics.edgesAdded.Add(1)
	return nil
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
		return g.removeEdgeFromTable(src, tgt, kind, g.index, g.pagePool)
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
			return g.appendEdgeToTable(src, fEdge, g.index, g.pagePool)
		})
		return err
	}

	if txn != nil {
		record := &WALEdgeRemoveRecord{
			TxnID: txn.ID,
			From:  src,
			To:    tgt,
			Kind:  kind,
		}
		txn.AppendRecord(&wal.Entry{
			Operation: wal.OpEdgeRemove,
			Timestamp: uint64(time.Now().UnixNano()),
			Data:      SerializeWALEdgeRemoveRecord(record),
		})
	}

	g.metrics.edgesRemoved.Add(1)
	return nil
}

func (g *graphStore) DropNodeEdges(txn *Txn, nodeID uint64) error {
	var firstErr error
	inboundEdges, _ := g.neighborsFromTable(nodeID, g.reverse.locator, g.reverse.pool, g.cfg.PageShards)
	for _, edge := range inboundEdges {
		err := retryOp(func() error {
			return g.removeEdgeFromTable(edge.Target, nodeID, edge.GetKind(), g.index, g.pagePool)
		})
		if err != nil && err != ErrEdgeNotFound && firstErr == nil {
			firstErr = err
		}
	}

	outboundEdges, _ := g.neighborsFromTable(nodeID, g.index, g.pagePool, g.cfg.PageShards)
	for _, edge := range outboundEdges {
		err := retryOp(func() error {
			return g.removeEdgeFromTable(edge.Target, nodeID, edge.GetKind(), g.reverse.locator, g.reverse.pool)
		})
		if err != nil && err != ErrEdgeNotFound && firstErr == nil {
			firstErr = err
		}
	}

	g.retirePageChain(nodeID, g.index, g.pagePool)
	g.retirePageChain(nodeID, g.reverse.locator, g.reverse.pool)

	if txn != nil {
		record := &WALNodeEdgeDropRecord{
			TxnID:  txn.ID,
			NodeID: nodeID,
		}
		txn.AppendRecord(&wal.Entry{
			Operation: wal.OpNodeEdgeDrop,
			Timestamp: uint64(time.Now().UnixNano()),
			Data:      SerializeWALNodeEdgeDropRecord(record),
		})
	}

	return firstErr
}

func (g *graphStore) Neighbors(nodeID uint64) ([]Edge, error) {
	return g.neighborsFromTable(nodeID, g.index, g.pagePool, g.cfg.PageShards)
}

func (g *graphStore) InboundNeighbors(nodeID uint64) ([]Edge, error) {
	return g.neighborsFromTable(nodeID, g.reverse.locator, g.reverse.pool, g.cfg.PageShards)
}

func (g *graphStore) Degree(nodeID uint64) (int, error) {
	return g.degreeFromTable(nodeID, g.index, g.pagePool, g.cfg.PageShards)
}

func (g *graphStore) InboundDegree(nodeID uint64) (int, error) {
	return g.degreeFromTable(nodeID, g.reverse.locator, g.reverse.pool, g.cfg.PageShards)
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

	pageSlot := index.Lookup(nodeID)
	if pageSlot == 0 {
		pool.HyalineLeave(int(shard))
		return 0, nil
	}

	page := g.pageReg.Get(pageSlot)
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
		g.pagePool.Stats().Allocated +
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
	var err1, err2, err3, err4 error
	if g.edgePool != nil {
		err1 = g.edgePool.Free()
	}
	if g.pagePool != nil {
		err2 = g.pagePool.Free()
	}
	if g.bitsetPool != nil {
		err3 = g.bitsetPool.Free()
	}
	if g.frontierPool != nil {
		err4 = g.frontierPool.Free()
	}
	if g.reverse != nil {
		_ = g.reverse.Close()
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
	for i := uint64(0); i < g.index.capacity; i++ {
		slot := atomic.LoadUint32(&g.index.table[i].PageSlot)
		if slot != 0 && slot != Tombstone {
			nodeID := g.index.table[i].NodeID
			edges, _ := g.Neighbors(nodeID)
			for _, e := range edges {
				rEdge := Edge{Target: nodeID, Weight: e.Weight}
				rEdge.SetStamp(e.GetStamp())
				rEdge.SetKind(e.GetKind())
				_ = g.appendEdgeToTable(e.Target, rEdge, g.reverse.locator, g.reverse.pool)
			}
		}
	}
}
