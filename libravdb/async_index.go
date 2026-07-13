package libravdb

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
	offheap "github.com/xDarkicex/memory"
)

var errAsyncIndexerClosed = errors.New("asynchronous indexer is closed")

// IndexingStats reports the durable-storage to derived-index gap. AppliedLSN is
// the exact contiguous transaction frontier: no known index mutation at or
// below that LSN remains queued or in flight.
type IndexingStats struct {
	DurableLSN uint64
	AppliedLSN uint64
	LSNLag     uint64
	Pending    uint64
	Reserved   uint64
	Capacity   int
	Failed     bool
}

type asyncIndexTask struct {
	firstLSN  uint64
	commitLSN uint64
	ordinal   uint32
	_         uint32
}

// asyncIndexSlot is one cache line. Producers publish task data by advancing
// sequence; consumers return the slot to its next ring generation the same way.
// No Go pointer is stored in the off-heap slot.
type asyncIndexSlot struct {
	sequence uint64
	active   uint64
	task     asyncIndexTask
	_        [24]byte
}

type asyncIndexFailure struct {
	err error
}

type asyncIndexStorage interface {
	storage.DurableRangeCollection
	GetByOrdinal(uint32) ([]float32, error)
	DurableFrontier() uint64
}

type asyncIndexQueue struct {
	collection *Collection
	storage    asyncIndexStorage
	arena      *offheap.Arena
	slots      []asyncIndexSlot
	workReady  chan struct{}
	capacity   uint64
	workers    int
	wg         sync.WaitGroup
	applyGate  sync.RWMutex

	enqueuePos  atomic.Uint64
	_           [56]byte
	dequeuePos  atomic.Uint64
	_           [56]byte
	outstanding atomic.Uint64
	_           [56]byte

	accepting atomic.Bool
	closing   atomic.Bool
	closed    atomic.Bool
	pending   atomic.Uint64
	reserved  atomic.Uint64
	durable   atomic.Uint64
	applied   atomic.Uint64
	published atomic.Uint64
	failure   atomic.Pointer[asyncIndexFailure]
}

func newAsyncIndexQueue(collection *Collection, depth, workers int) (*asyncIndexQueue, error) {
	if collection == nil || collection.index == nil {
		return nil, fmt.Errorf("asynchronous indexer requires an initialized index")
	}
	store, ok := collection.storage.(asyncIndexStorage)
	if !ok {
		return nil, fmt.Errorf("collection storage does not expose durable LSN and ordinal vector access")
	}
	if depth < 32 {
		return nil, fmt.Errorf("asynchronous index queue depth must be at least 32")
	}
	if workers <= 0 {
		return nil, fmt.Errorf("asynchronous index worker count must be positive")
	}

	slotBytes := uint64(unsafe.Sizeof(asyncIndexSlot{})) * uint64(depth)
	arena, err := offheap.NewArena(slotBytes, 64)
	if err != nil {
		return nil, fmt.Errorf("allocate asynchronous index queue: %w", err)
	}
	slots, err := offheap.ArenaSlice[asyncIndexSlot](arena, depth)
	if err != nil {
		_ = arena.Free()
		return nil, fmt.Errorf("allocate asynchronous index slot ring: %w", err)
	}
	slots = slots[:depth]
	for i := range slots {
		atomic.StoreUint64(&slots[i].sequence, uint64(i))
	}

	q := &asyncIndexQueue{
		collection: collection,
		storage:    store,
		arena:      arena,
		slots:      slots,
		workReady:  make(chan struct{}, workers),
		capacity:   uint64(depth),
		workers:    workers,
	}
	frontier := store.DurableFrontier()
	q.durable.Store(frontier)
	q.applied.Store(frontier)
	q.accepting.Store(true)
	q.wg.Add(workers)
	for i := 0; i < workers; i++ {
		go q.worker()
	}
	return q, nil
}

func (q *asyncIndexQueue) reserve(ctx context.Context, count int) error {
	if count <= 0 {
		return nil
	}
	if uint64(count) > q.capacity {
		return fmt.Errorf("asynchronous index batch of %d exceeds queue capacity %d", count, q.capacity)
	}
	if !q.accepting.Load() {
		return q.currentError(errAsyncIndexerClosed)
	}

	wanted := uint64(count)
	var backoff lockFreeBackoff
	for {
		if !q.accepting.Load() {
			return q.currentError(errAsyncIndexerClosed)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		used := q.outstanding.Load()
		if used <= q.capacity-wanted && q.outstanding.CompareAndSwap(used, used+wanted) {
			q.reserved.Add(wanted)
			if !q.accepting.Load() {
				q.cancelReservation(count)
				return q.currentError(errAsyncIndexerClosed)
			}
			return nil
		}
		backoff.wait()
	}
}

func (q *asyncIndexQueue) cancelReservation(count int) {
	if count <= 0 {
		return
	}
	q.reserved.Add(^uint64(count - 1))
	q.outstanding.Add(^uint64(count - 1))
	q.advanceAppliedIfDrained()
}

func (q *asyncIndexQueue) commit(entries []*index.VectorEntry, durable storage.DurableRange) {
	if len(entries) == 0 {
		return
	}
	q.pending.Add(uint64(len(entries)))
	for _, entry := range entries {
		q.enqueue(asyncIndexTask{firstLSN: durable.FirstLSN, commitLSN: durable.CommitLSN, ordinal: entry.Ordinal})
	}
	atomicMax(&q.durable, durable.CommitLSN)
	q.published.Add(1)
	q.reserved.Add(^uint64(len(entries) - 1))
	q.signalWorkers(min(len(entries), q.workers))
}

func (q *asyncIndexQueue) commitOne(entry *index.VectorEntry, durable storage.DurableRange) {
	q.pending.Add(1)
	q.enqueue(asyncIndexTask{firstLSN: durable.FirstLSN, commitLSN: durable.CommitLSN, ordinal: entry.Ordinal})
	atomicMax(&q.durable, durable.CommitLSN)
	q.published.Add(1)
	q.reserved.Add(^uint64(0))
	q.signalWorkers(1)
}

func (q *asyncIndexQueue) worker() {
	defer q.wg.Done()
	for {
		if task, pos, ok := q.pop(); ok {
			q.apply(task, pos)
			continue
		}
		if q.closing.Load() && q.outstanding.Load() == 0 {
			return
		}
		<-q.workReady
	}
}

func (q *asyncIndexQueue) pop() (asyncIndexTask, uint64, bool) {
	for {
		pos := q.dequeuePos.Load()
		slot := &q.slots[pos%q.capacity]
		sequence := atomic.LoadUint64(&slot.sequence)
		delta := int64(sequence) - int64(pos+1)
		switch {
		case delta == 0:
			if !q.dequeuePos.CompareAndSwap(pos, pos+1) {
				continue
			}
			return slot.task, pos, true
		case delta < 0:
			return asyncIndexTask{}, 0, false
		default:
			runtime.Gosched()
		}
	}
}

func (q *asyncIndexQueue) enqueue(task asyncIndexTask) {
	pos := q.enqueuePos.Add(1) - 1
	slot := &q.slots[pos%q.capacity]
	var backoff lockFreeBackoff
	for atomic.LoadUint64(&slot.sequence) != pos {
		backoff.wait()
	}
	slot.task = task
	atomic.StoreUint64(&slot.active, 1)
	atomic.StoreUint64(&slot.sequence, pos+1)
}

func (q *asyncIndexQueue) apply(task asyncIndexTask, pos uint64) {
	id, err := q.storage.GetIDByOrdinal(context.Background(), task.ordinal)
	var vector []float32
	if err == nil {
		vector, err = q.storage.GetByOrdinal(task.ordinal)
	}

	// Keep publication, active-state retirement, and slot reuse within the same
	// read-side gate so a precise frontier scan cannot observe a recycled task.
	q.applyGate.RLock()
	if err == nil {
		entry := index.VectorEntry{ID: id, Vector: vector, Ordinal: task.ordinal}
		err = q.collection.index.Insert(context.Background(), entryForIndex(q.collection.config.Metric, &entry))
	}
	if err != nil {
		q.recordFailure(fmt.Errorf("apply durable transaction %d ordinal %d: %w", task.commitLSN, task.ordinal, err))
	}

	slot := &q.slots[pos%q.capacity]
	atomic.StoreUint64(&slot.active, 0)
	atomic.StoreUint64(&slot.sequence, pos+q.capacity)
	q.pending.Add(^uint64(0))
	q.outstanding.Add(^uint64(0))
	q.applyGate.RUnlock()
	q.advanceAppliedIfDrained()
}

func (q *asyncIndexQueue) flush(ctx context.Context) error {
	var backoff lockFreeBackoff
	for {
		if q.outstanding.Load() == 0 {
			return q.failureValue()
		}
		if err := ctx.Err(); err != nil {
			return ctx.Err()
		}
		backoff.wait()
	}
}

func (q *asyncIndexQueue) close() error {
	owner := q.closing.CompareAndSwap(false, true)
	if owner {
		q.accepting.Store(false)
		_ = q.flush(context.Background())
		q.signalWorkers(q.workers)
		q.wg.Wait()
		q.closed.Store(true)
	} else {
		for !q.closed.Load() {
			runtime.Gosched()
		}
	}
	var err error
	if owner && q.arena != nil {
		err = q.arena.Free()
		q.arena = nil
		q.slots = nil
	}
	if failure := q.failureValue(); failure != nil {
		if err != nil {
			return errors.Join(failure, err)
		}
		return failure
	}
	return err
}

func (q *asyncIndexQueue) stats() IndexingStats {
	applied := q.preciseApplied()
	// Read durable after the frontier scan so a concurrently published WAL
	// transaction cannot make AppliedLSN appear newer than a stale DurableLSN.
	durable := q.durable.Load()
	lag := uint64(0)
	if durable > applied {
		lag = durable - applied
	}
	return IndexingStats{
		DurableLSN: durable,
		AppliedLSN: applied,
		LSNLag:     lag,
		Pending:    q.pending.Load(),
		Reserved:   q.reserved.Load(),
		Capacity:   int(q.capacity),
		Failed:     q.failure.Load() != nil,
	}
}

// preciseApplied returns the highest WAL frontier for which this index has no
// known missing mutation. The short exclusive gate freezes worker publication
// and slot reuse while the off-heap ring is inspected; producers may keep
// admitting work, with reserved/published counters detecting an unstable scan.
func (q *asyncIndexQueue) preciseApplied() uint64 {
	q.applyGate.Lock()
	defer q.applyGate.Unlock()
	return q.preciseAppliedLocked()
}

func (q *asyncIndexQueue) preciseAppliedLocked() uint64 {
	if q.failure.Load() != nil {
		return q.applied.Load()
	}
	for {
		if q.reserved.Load() != 0 {
			return q.applied.Load()
		}
		version := q.published.Load()
		durable := q.durable.Load()
		minPending := uint64(0)
		for i := range q.slots {
			slot := &q.slots[i]
			if atomic.LoadUint64(&slot.active) == 0 {
				continue
			}
			lsn := slot.task.firstLSN
			if lsn != 0 && (minPending == 0 || lsn < minPending) {
				minPending = lsn
			}
		}
		if q.reserved.Load() != 0 || q.published.Load() != version {
			continue
		}

		candidate := durable
		if minPending != 0 && minPending <= candidate {
			candidate = minPending - 1
		}
		atomicMax(&q.applied, candidate)
		return q.applied.Load()
	}
}

func (q *asyncIndexQueue) recordFailure(err error) {
	if err == nil {
		return
	}
	failure := &asyncIndexFailure{err: err}
	if q.failure.CompareAndSwap(nil, failure) {
		q.accepting.Store(false)
	}
}

func (q *asyncIndexQueue) failureValue() error {
	failure := q.failure.Load()
	if failure == nil {
		return nil
	}
	return failure.err
}

func (q *asyncIndexQueue) currentError(fallback error) error {
	if err := q.failureValue(); err != nil {
		return err
	}
	return fallback
}

func (q *asyncIndexQueue) advanceAppliedIfDrained() {
	if q.outstanding.Load() == 0 && q.failure.Load() == nil {
		q.applied.Store(q.durable.Load())
	}
}

func (q *asyncIndexQueue) signalWorkers(count int) {
	for i := 0; i < count; i++ {
		select {
		case q.workReady <- struct{}{}:
		default:
			return
		}
	}
}

type lockFreeBackoff uint32

func (b *lockFreeBackoff) wait() {
	step := uint32(*b)
	if step < 32 {
		*b++
		runtime.Gosched()
		return
	}
	time.Sleep(25 * time.Microsecond)
}

func atomicMax(dst *atomic.Uint64, value uint64) {
	for current := dst.Load(); value > current; current = dst.Load() {
		if dst.CompareAndSwap(current, value) {
			return
		}
	}
}

// FlushIndex waits until every durable asynchronous insert has been applied to
// the derived index. It is a no-op for synchronous collections.
func (c *Collection) FlushIndex(ctx context.Context) error {
	if c == nil || c.asyncIndex == nil {
		return nil
	}
	return c.asyncIndex.flush(ctx)
}

func (c *Collection) lockAsyncMutation(ctx context.Context) (func(), error) {
	if c == nil || c.asyncIndex == nil {
		return func() {}, nil
	}
	c.asyncMutation.Lock()
	if err := c.asyncIndex.flush(ctx); err != nil {
		c.asyncMutation.Unlock()
		return nil, err
	}
	return c.asyncMutation.Unlock, nil
}

// IndexingStats returns the current durable-to-index lag for this collection.
func (c *Collection) IndexingStats() IndexingStats {
	if c == nil || c.asyncIndex == nil {
		return IndexingStats{}
	}
	return c.asyncIndex.stats()
}
