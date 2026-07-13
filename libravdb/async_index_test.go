package libravdb

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

func TestAsyncIndexQueueDurableLagAndDrain(t *testing.T) {
	db, err := Open(
		WithStoragePath(testDBPath(t)),
		WithAsyncIndexing(64, 2),
		WithMaxConcurrentWrites(32),
		WithMaxWriteQueueDepth(64),
	)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(
		context.Background(),
		"async",
		WithDimension(8),
		WithMetric(L2Distance),
		WithHNSW(4, 32, 32),
	)
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if collection.asyncIndex == nil {
		t.Fatal("asynchronous index queue was not configured")
	}
	if got := unsafe.Sizeof(asyncIndexTask{}); got != 24 {
		t.Fatalf("task size = %d, want 24", got)
	}
	if got := unsafe.Sizeof(asyncIndexSlot{}); got != 64 {
		t.Fatalf("slot size = %d, want 64", got)
	}
	if ptr := uintptr(unsafe.Pointer(unsafe.SliceData(collection.asyncIndex.slots))); ptr&63 != 0 {
		t.Fatalf("off-heap task ring is not 64-byte aligned: %#x", ptr)
	}
	statsDone := make(chan struct{})
	statsErr := make(chan error, 1)
	var statsWG sync.WaitGroup
	statsWG.Add(1)
	go func() {
		defer statsWG.Done()
		for {
			select {
			case <-statsDone:
				return
			default:
				stats := collection.IndexingStats()
				if stats.AppliedLSN > stats.DurableLSN {
					statsErr <- fmt.Errorf("applied frontier exceeds durable frontier: %+v", stats)
					return
				}
			}
		}
	}()

	const total = 64
	start := make(chan struct{})
	errCh := make(chan error, total)
	var wg sync.WaitGroup
	for i := 0; i < total; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			vector := make([]float32, 8)
			vector[i%len(vector)] = 1
			errCh <- collection.Insert(context.Background(), fmt.Sprintf("v-%03d", i), vector, nil)
		}(i)
	}
	close(start)
	wg.Wait()
	close(statsDone)
	statsWG.Wait()
	select {
	case err := <-statsErr:
		t.Fatal(err)
	default:
	}
	close(errCh)
	for err := range errCh {
		if err != nil {
			t.Fatalf("insert: %v", err)
		}
	}

	if err := collection.FlushIndex(context.Background()); err != nil {
		t.Fatalf("flush index: %v", err)
	}
	stats := collection.IndexingStats()
	if stats.DurableLSN == 0 {
		t.Fatal("durable LSN did not advance")
	}
	if stats.AppliedLSN != stats.DurableLSN || stats.LSNLag != 0 {
		t.Fatalf("index did not reach durable frontier: %+v", stats)
	}
	if stats.Pending != 0 || stats.Reserved != 0 {
		t.Fatalf("queue did not drain: %+v", stats)
	}
	if got := collection.index.Size(); got != total {
		t.Fatalf("index size = %d, want %d", got, total)
	}
	if got, err := collection.storage.Count(context.Background()); err != nil || got != total {
		t.Fatalf("storage count = %d, err=%v, want %d", got, err, total)
	}
}

func TestAsyncIndexPreciseAppliedFrontier(t *testing.T) {
	q := &asyncIndexQueue{
		slots:    make([]asyncIndexSlot, 4),
		capacity: 4,
	}
	q.applied.Store(90)
	q.durable.Store(110)
	q.slots[0].task.firstLSN = 100
	q.slots[0].task.commitLSN = 104
	q.slots[1].task.firstLSN = 110
	q.slots[1].task.commitLSN = 110
	atomic.StoreUint64(&q.slots[0].active, 1)
	atomic.StoreUint64(&q.slots[1].active, 1)

	if got := q.preciseApplied(); got != 99 {
		t.Fatalf("applied frontier with commit 100 pending = %d, want 99", got)
	}
	atomic.StoreUint64(&q.slots[0].active, 0)
	if got := q.preciseApplied(); got != 109 {
		t.Fatalf("applied frontier with commit 110 pending = %d, want 109", got)
	}

	q.reserved.Store(1)
	atomic.StoreUint64(&q.slots[1].active, 0)
	if got := q.preciseApplied(); got != 109 {
		t.Fatalf("unknown reservation advanced frontier to %d, want 109", got)
	}
	q.reserved.Store(0)
	if got := q.preciseApplied(); got != 110 {
		t.Fatalf("fully applied frontier = %d, want 110", got)
	}
}

func TestAsyncIndexingConfiguresWALAdmissionDefaults(t *testing.T) {
	db, err := Open(
		WithStoragePath(testDBPath(t)),
		WithAsyncIndexing(64, 2),
		WithDurability(DurabilityUnsafeNoSync),
	)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	if db.config.MaxConcurrentWrites != 32 {
		t.Fatalf("async max concurrent writes = %d, want 32", db.config.MaxConcurrentWrites)
	}
	if db.config.MaxWriteQueueDepth != 64 {
		t.Fatalf("async write queue depth = %d, want 64", db.config.MaxWriteQueueDepth)
	}

	explicit, err := Open(
		WithStoragePath(testDBPath(t)),
		WithMaxConcurrentWrites(8),
		WithMaxWriteQueueDepth(12),
		WithAsyncIndexing(64, 2),
		WithDurability(DurabilityUnsafeNoSync),
	)
	if err != nil {
		t.Fatalf("open explicit config: %v", err)
	}
	defer explicit.Close()
	if explicit.config.MaxConcurrentWrites != 8 || explicit.config.MaxWriteQueueDepth != 12 {
		t.Fatalf("explicit admission limits were overwritten: writes=%d queue=%d",
			explicit.config.MaxConcurrentWrites, explicit.config.MaxWriteQueueDepth)
	}
}

func TestAsyncIndexBatchLargerThanQueueRejectedBeforeWAL(t *testing.T) {
	db, err := Open(
		WithStoragePath(testDBPath(t)),
		WithAsyncIndexing(32, 1),
		WithDurability(DurabilityUnsafeNoSync),
	)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(
		context.Background(),
		"bounded",
		WithDimension(4),
		WithMetric(L2Distance),
		WithHNSW(4, 16, 16),
	)
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	entries := make([]VectorEntry, 33)
	for i := range entries {
		entries[i] = VectorEntry{ID: fmt.Sprintf("v-%03d", i), Vector: []float32{float32(i), 0, 0, 0}}
	}
	if err := collection.InsertBatch(context.Background(), entries); err == nil {
		t.Fatal("batch larger than bounded queue unexpectedly succeeded")
	}
	if got, err := collection.storage.Count(context.Background()); err != nil || got != 0 {
		t.Fatalf("rejected batch reached durable storage: count=%d err=%v", got, err)
	}
}

func TestAsyncIndexMutationBarrierDrainsBeforeUpdateAndDelete(t *testing.T) {
	db, err := Open(
		WithStoragePath(testDBPath(t)),
		WithAsyncIndexing(32, 1),
		WithDurability(DurabilityUnsafeNoSync),
	)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	collection, err := db.CreateCollection(
		context.Background(),
		"mutations",
		WithDimension(4),
		WithMetric(L2Distance),
		WithHNSW(4, 16, 16),
	)
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.Insert(context.Background(), "v", []float32{1, 0, 0, 0}, nil); err != nil {
		t.Fatalf("insert: %v", err)
	}
	if err := collection.Update(context.Background(), "v", []float32{0, 1, 0, 0}, nil); err != nil {
		t.Fatalf("update after async insert: %v", err)
	}
	results, err := collection.Search(context.Background(), []float32{0, 1, 0, 0}, 1)
	if err != nil {
		t.Fatalf("search updated vector: %v", err)
	}
	if len(results.Results) != 1 || results.Results[0].ID != "v" {
		t.Fatalf("updated vector missing from index: %+v", results)
	}
	if err := collection.Delete(context.Background(), "v"); err != nil {
		t.Fatalf("delete after async insert: %v", err)
	}
	if got := collection.index.Size(); got != 0 {
		t.Fatalf("index size after delete = %d, want 0", got)
	}
}

func TestAsyncIndexLockFreeRingMultipleWraps(t *testing.T) {
	db, err := Open(
		WithStoragePath(testDBPath(t)),
		WithAsyncIndexing(32, 4),
		WithMaxConcurrentWrites(8),
		WithMaxWriteQueueDepth(64),
		WithDurability(DurabilityUnsafeNoSync),
	)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	collection, err := db.CreateCollection(
		context.Background(),
		"wraps",
		WithDimension(4),
		WithMetric(L2Distance),
		WithHNSW(4, 16, 16),
	)
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	const total = 512
	jobs := make(chan int, total)
	errCh := make(chan error, 8)
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				err := collection.Insert(context.Background(), fmt.Sprintf("v-%04d", i), []float32{float32(i), 1, 0, 0}, nil)
				if err != nil {
					errCh <- err
					return
				}
			}
		}()
	}
	for i := 0; i < total; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatalf("insert: %v", err)
	}
	if err := collection.FlushIndex(context.Background()); err != nil {
		t.Fatalf("flush: %v", err)
	}
	if got := collection.index.Size(); got != total {
		t.Fatalf("index size after ring wraps = %d, want %d", got, total)
	}
	stats := collection.IndexingStats()
	if stats.Pending != 0 || stats.Reserved != 0 || stats.LSNLag != 0 {
		t.Fatalf("ring did not fully drain: %+v", stats)
	}
}

func TestAsyncIndexCloseAndRecoveryRebuild(t *testing.T) {
	path := testDBPath(t)
	db, err := Open(
		WithStoragePath(path),
		WithAsyncIndexing(32, 2),
		WithDurability(DurabilityUnsafeNoSync),
	)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	collection, err := db.CreateCollection(
		context.Background(),
		"recover",
		WithDimension(4),
		WithMetric(L2Distance),
		WithHNSW(4, 16, 16),
	)
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	for i := 0; i < 20; i++ {
		if err := collection.Insert(context.Background(), fmt.Sprintf("v-%03d", i), []float32{float32(i), 1, 0, 0}, nil); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	if err := collection.FlushIndex(context.Background()); err != nil {
		t.Fatalf("flush before checkpoint: %v", err)
	}
	compactor, ok := db.storage.(interface{ Compact() error })
	if !ok {
		t.Fatal("single-file storage does not expose Compact")
	}
	if err := compactor.Compact(); err != nil {
		t.Fatalf("compact async index checkpoint: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	reopened, err := Open(
		WithStoragePath(path),
		WithAsyncIndexing(32, 2),
		WithDurability(DurabilityUnsafeNoSync),
	)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer reopened.Close()
	recovered, err := reopened.GetCollection("recover")
	if err != nil {
		t.Fatalf("get recovered collection: %v", err)
	}
	if got := recovered.index.Size(); got != 20 {
		t.Fatalf("recovered index size = %d, want 20", got)
	}
}
