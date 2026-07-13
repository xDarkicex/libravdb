package libravdb

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
	offheap "github.com/xDarkicex/memory"
)

var errAsyncIndexerClosed = errors.New("asynchronous indexer is closed")

// IndexingStats reports the durable-storage to derived-index gap. AppliedLSN
// is conservative: it advances to DurableLSN whenever the bounded queue fully
// drains, never ahead of work that may still be in flight.
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
	durableLSN uint64
	ordinal    uint32
	_          uint32
}

type asyncIndexStorage interface {
	storage.DurableCollection
	GetByOrdinal(uint32) ([]float32, error)
}

type asyncIndexQueue struct {
	collection *Collection
	storage    asyncIndexStorage
	arena      *offheap.Arena
	tasks      []asyncIndexTask
	tokens     chan struct{}
	workReady  chan struct{}
	changed    chan struct{}
	stop       chan struct{}
	closed     chan struct{}
	failed     chan struct{}

	mu       sync.Mutex
	errMu    sync.Mutex
	head     int
	tail     int
	count    int
	workers  int
	closeOne sync.Once
	failOne  sync.Once
	wg       sync.WaitGroup

	accepting  atomic.Bool
	pending    atomic.Uint64
	reserved   atomic.Uint64
	durable    atomic.Uint64
	applied    atomic.Uint64
	failure    error
	failedFlag atomic.Bool
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

	taskBytes := uint64(unsafe.Sizeof(asyncIndexTask{})) * uint64(depth)
	arena, err := offheap.NewArena(taskBytes, 64)
	if err != nil {
		return nil, fmt.Errorf("allocate asynchronous index queue: %w", err)
	}
	tasks, err := offheap.ArenaSlice[asyncIndexTask](arena, depth)
	if err != nil {
		_ = arena.Free()
		return nil, fmt.Errorf("allocate asynchronous index task ring: %w", err)
	}
	tasks = tasks[:depth]

	q := &asyncIndexQueue{
		collection: collection,
		storage:    store,
		arena:      arena,
		tasks:      tasks,
		tokens:     make(chan struct{}, depth),
		workReady:  make(chan struct{}, workers),
		changed:    make(chan struct{}, 1),
		stop:       make(chan struct{}),
		closed:     make(chan struct{}),
		failed:     make(chan struct{}),
		workers:    workers,
	}
	for i := 0; i < depth; i++ {
		q.tokens <- struct{}{}
	}
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
	if count > cap(q.tokens) {
		return fmt.Errorf("asynchronous index batch of %d exceeds queue capacity %d", count, cap(q.tokens))
	}
	if !q.accepting.Load() {
		return q.currentError(errAsyncIndexerClosed)
	}

	acquired := 0
	for acquired < count {
		select {
		case <-ctx.Done():
			q.releaseTokens(acquired)
			return ctx.Err()
		case <-q.failed:
			q.releaseTokens(acquired)
			return q.currentError(errAsyncIndexerClosed)
		case <-q.closed:
			q.releaseTokens(acquired)
			return errAsyncIndexerClosed
		case <-q.tokens:
			acquired++
		}
	}
	if !q.accepting.Load() {
		q.releaseTokens(acquired)
		return q.currentError(errAsyncIndexerClosed)
	}
	q.reserved.Add(uint64(count))
	q.signalChanged()
	return nil
}

func (q *asyncIndexQueue) cancelReservation(count int) {
	if count <= 0 {
		return
	}
	q.reserved.Add(^uint64(count - 1))
	q.releaseTokens(count)
	q.advanceAppliedIfDrained()
	q.signalChanged()
}

func (q *asyncIndexQueue) commit(entries []*index.VectorEntry, durableLSN uint64) {
	if len(entries) == 0 {
		return
	}
	q.mu.Lock()
	for _, entry := range entries {
		q.tasks[q.tail] = asyncIndexTask{durableLSN: durableLSN, ordinal: entry.Ordinal}
		q.tail++
		if q.tail == len(q.tasks) {
			q.tail = 0
		}
		q.count++
	}
	q.reserved.Add(^uint64(len(entries) - 1))
	q.pending.Add(uint64(len(entries)))
	atomicMax(&q.durable, durableLSN)
	q.mu.Unlock()
	for i := 0; i < min(len(entries), q.workers); i++ {
		select {
		case q.workReady <- struct{}{}:
		default:
		}
	}
	q.signalChanged()
}

func (q *asyncIndexQueue) commitOne(entry *index.VectorEntry, durableLSN uint64) {
	q.mu.Lock()
	q.tasks[q.tail] = asyncIndexTask{durableLSN: durableLSN, ordinal: entry.Ordinal}
	q.tail++
	if q.tail == len(q.tasks) {
		q.tail = 0
	}
	q.count++
	q.reserved.Add(^uint64(0))
	q.pending.Add(1)
	atomicMax(&q.durable, durableLSN)
	q.mu.Unlock()
	select {
	case q.workReady <- struct{}{}:
	default:
	}
	q.signalChanged()
}

func (q *asyncIndexQueue) worker() {
	defer q.wg.Done()
	for {
		if task, ok := q.pop(); ok {
			q.apply(task)
			continue
		}
		select {
		case <-q.workReady:
		case <-q.stop:
			return
		}
	}
}

func (q *asyncIndexQueue) pop() (asyncIndexTask, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.count == 0 {
		return asyncIndexTask{}, false
	}
	task := q.tasks[q.head]
	q.head++
	if q.head == len(q.tasks) {
		q.head = 0
	}
	q.count--
	return task, true
}

func (q *asyncIndexQueue) apply(task asyncIndexTask) {
	id, err := q.storage.GetIDByOrdinal(context.Background(), task.ordinal)
	if err == nil {
		var vector []float32
		vector, err = q.storage.GetByOrdinal(task.ordinal)
		if err == nil {
			entry := index.VectorEntry{ID: id, Vector: vector, Ordinal: task.ordinal}
			err = q.collection.index.Insert(context.Background(), entryForIndex(q.collection.config.Metric, &entry))
		}
	}
	if err != nil {
		q.recordFailure(fmt.Errorf("apply durable LSN %d ordinal %d: %w", task.durableLSN, task.ordinal, err))
	}

	q.pending.Add(^uint64(0))
	q.tokens <- struct{}{}
	q.advanceAppliedIfDrained()
	q.signalChanged()
}

func (q *asyncIndexQueue) flush(ctx context.Context) error {
	for {
		if q.pending.Load() == 0 && q.reserved.Load() == 0 {
			return q.failureValue()
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-q.changed:
		}
	}
}

func (q *asyncIndexQueue) close() error {
	q.closeOne.Do(func() {
		q.accepting.Store(false)
		close(q.closed)
		_ = q.flush(context.Background())
		close(q.stop)
		q.wg.Wait()
	})
	var err error
	if q.arena != nil {
		err = q.arena.Free()
		q.arena = nil
		q.tasks = nil
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
	durable := q.durable.Load()
	applied := q.applied.Load()
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
		Capacity:   cap(q.tokens),
		Failed:     q.failedFlag.Load(),
	}
}

func (q *asyncIndexQueue) recordFailure(err error) {
	if err == nil {
		return
	}
	q.failOne.Do(func() {
		q.errMu.Lock()
		q.failure = err
		q.errMu.Unlock()
		q.failedFlag.Store(true)
		q.accepting.Store(false)
		close(q.failed)
	})
}

func (q *asyncIndexQueue) failureValue() error {
	q.errMu.Lock()
	defer q.errMu.Unlock()
	return q.failure
}

func (q *asyncIndexQueue) currentError(fallback error) error {
	if err := q.failureValue(); err != nil {
		return err
	}
	return fallback
}

func (q *asyncIndexQueue) releaseTokens(count int) {
	for i := 0; i < count; i++ {
		q.tokens <- struct{}{}
	}
}

func (q *asyncIndexQueue) signalChanged() {
	select {
	case q.changed <- struct{}{}:
	default:
	}
}

func (q *asyncIndexQueue) advanceAppliedIfDrained() {
	if q.pending.Load() == 0 && q.reserved.Load() == 0 && !q.failedFlag.Load() {
		q.applied.Store(q.durable.Load())
	}
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
