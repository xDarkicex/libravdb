package singlefile

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
)

func TestNewInitializesSingleFileDatabase(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.libravdb")

	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)
	defer engine.Close()

	stat, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat database file: %v", err)
	}
	if stat.Size() < 3*pageSize {
		t.Fatalf("expected initialized database file, size=%d", stat.Size())
	}

	header, err := engine.readHeader()
	if err != nil {
		t.Fatalf("read header: %v", err)
	}
	if got := string(header.Magic[:]); got != fileMagic {
		t.Fatalf("unexpected file magic: %q", got)
	}
	if header.PageSize != pageSize {
		t.Fatalf("unexpected page size: %d", header.PageSize)
	}
}

func TestNewRejectsMalformedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.libravdb")
	if err := os.WriteFile(path, []byte("not-a-valid-database"), 0644); err != nil {
		t.Fatalf("write malformed file: %v", err)
	}

	if _, err := New(path); err == nil {
		t.Fatal("expected malformed file to fail")
	}
}

func TestMetapageFailoverOnCorruption(t *testing.T) {
	path := filepath.Join(t.TempDir(), "failover.libravdb")

	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)

	collection, err := engine.CreateCollection("global", &storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.Insert(context.Background(), &index.VectorEntry{
		ID:       "r1",
		Vector:   []float32{1, 0, 0},
		Metadata: map[string]interface{}{"source": "spec"},
	}); err != nil {
		t.Fatalf("insert record: %v", err)
	}
	if err := engine.Close(); err != nil {
		t.Fatalf("close engine: %v", err)
	}

	page := make([]byte, pageSize)
	file, err := os.OpenFile(path, os.O_RDWR, 0644)
	if err != nil {
		t.Fatalf("open file: %v", err)
	}
	if _, err := file.ReadAt(page, pageSize); err != nil {
		file.Close()
		t.Fatalf("read metapage: %v", err)
	}
	binary.LittleEndian.PutUint32(page[68:72], 0)
	if _, err := file.WriteAt(page, pageSize); err != nil {
		file.Close()
		t.Fatalf("corrupt metapage: %v", err)
	}
	file.Close()

	reopenedIface, err := New(path)
	if err != nil {
		t.Fatalf("reopen with one corrupted metapage: %v", err)
	}
	reopened := reopenedIface.(*Engine)
	defer reopened.Close()

	names, err := reopened.ListCollections()
	if err != nil {
		t.Fatalf("list collections: %v", err)
	}
	if len(names) != 1 || names[0] != "global" {
		t.Fatalf("expected recovered collection, got %v", names)
	}
}

func TestReplayIgnoresUncommittedTransactions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "replay.libravdb")

	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)

	if err := engine.createCollection("global", storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	}); err != nil {
		t.Fatalf("create collection: %v", err)
	}

	engine.mu.Lock()
	txID := engine.nextTxIDLocked()
	beginLSN := engine.nextLSNLocked()
	putLSN := engine.nextLSNLocked()
	begin := newFrame(recordTypeTxBegin, beginLSN, txID, 0, emptyPayload())
	payload, err := encodeRecordPutPayloadBinary(recordPutPayload{
		Collection: "global",
		ID:         "dangling",
		Vector:     []float32{1, 0, 0},
		Metadata:   map[string]interface{}{"source": "dangling"},
	})
	if err != nil {
		engine.mu.Unlock()
		t.Fatalf("marshal payload: %v", err)
	}
	put := newFrame(recordTypeRecordPut, putLSN, txID, beginLSN, payload)
	if _, err := engine.appendTransactionLocked([]walRecord{begin, put}); err != nil {
		engine.mu.Unlock()
		t.Fatalf("append uncommitted transaction: %v", err)
	}
	engine.mu.Unlock()

	if err := engine.Close(); err != nil {
		t.Fatalf("close engine: %v", err)
	}

	reopenedIface, err := New(path)
	if err != nil {
		t.Fatalf("reopen engine: %v", err)
	}
	reopened := reopenedIface.(*Engine)
	defer reopened.Close()

	collectionIface, err := reopened.GetCollection("global")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}
	count, err := collectionIface.Count(context.Background())
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if count != 0 {
		t.Fatalf("expected uncommitted record to be ignored, got count=%d", count)
	}

	stats := reopened.RecoveryStats()
	if stats.DiscardedTransactions == 0 {
		t.Fatalf("expected discarded transaction to be tracked, got %+v", stats)
	}
}

func TestRecoveryStatsTrackReplayedAndDiscardedTransactions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "recovery-stats.libravdb")

	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)

	if err := engine.createCollection("global", storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	}); err != nil {
		t.Fatalf("create collection: %v", err)
	}

	collectionIface, err := engine.GetCollection("global")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}
	if err := collectionIface.Insert(context.Background(), &index.VectorEntry{
		ID:       "committed",
		Vector:   []float32{1, 0, 0},
		Metadata: map[string]interface{}{"source": "committed"},
	}); err != nil {
		t.Fatalf("insert committed record: %v", err)
	}

	engine.mu.Lock()
	txID := engine.nextTxIDLocked()
	beginLSN := engine.nextLSNLocked()
	putLSN := engine.nextLSNLocked()
	begin := newFrame(recordTypeTxBegin, beginLSN, txID, 0, emptyPayload())
	payload, err := encodeRecordPutPayloadBinary(recordPutPayload{
		Collection: "global",
		ID:         "dangling",
		Vector:     []float32{0, 1, 0},
		Metadata:   map[string]interface{}{"source": "dangling"},
	})
	if err != nil {
		engine.mu.Unlock()
		t.Fatalf("encode payload: %v", err)
	}
	put := newFrame(recordTypeRecordPut, putLSN, txID, beginLSN, payload)
	if _, err := engine.appendTransactionLocked([]walRecord{begin, put}); err != nil {
		engine.mu.Unlock()
		t.Fatalf("append dangling transaction: %v", err)
	}
	engine.mu.Unlock()

	if err := engine.Close(); err != nil {
		t.Fatalf("close engine: %v", err)
	}

	reopenedIface, err := New(path)
	if err != nil {
		t.Fatalf("reopen engine: %v", err)
	}
	reopened := reopenedIface.(*Engine)
	defer reopened.Close()

	stats := reopened.RecoveryStats()
	if stats.ReplayedTransactions == 0 {
		t.Fatalf("expected replayed transaction count > 0, got %+v", stats)
	}
	if stats.DiscardedTransactions == 0 {
		t.Fatalf("expected discarded transaction count > 0, got %+v", stats)
	}
}

func TestBatchWALCommitMultipleRecordsInOneTransaction(t *testing.T) {
	path := filepath.Join(t.TempDir(), "batch_commit.libravdb")

	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)
	defer engine.Close()

	// Create collection
	_, err = engine.CreateCollection("test", &storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	// Insert multiple records in one batch (via InsertBatch)
	entries := []*index.VectorEntry{
		{ID: "r1", Vector: []float32{1, 0, 0}, Metadata: map[string]interface{}{"idx": 1}},
		{ID: "r2", Vector: []float32{0, 1, 0}, Metadata: map[string]interface{}{"idx": 2}},
		{ID: "r3", Vector: []float32{0, 0, 1}, Metadata: map[string]interface{}{"idx": 3}},
	}
	if err := engine.collections["test"].InsertBatch(context.Background(), entries); err != nil {
		t.Fatalf("batch insert: %v", err)
	}

	// Verify all records are visible
	col := engine.collections["test"]
	for _, want := range entries {
		got, err := col.Get(context.Background(), want.ID)
		if err != nil {
			t.Fatalf("get %s: %v", want.ID, err)
		}
		if got.ID != want.ID {
			t.Errorf("got ID %s, want %s", got.ID, want.ID)
		}
	}
}

func TestConcurrentSingleInsertsCoalesceIntoFewerTransactions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "group_commit.libravdb")

	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)
	defer engine.Close()

	_, err = engine.CreateCollection("test", &storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	origWindow := groupCommitWindow.Load()
	groupCommitWindow.Store(int64(20 * time.Millisecond))
	defer func() { groupCommitWindow.Store(origWindow) }()

	before := engine.WriteStats()

	const numInserts = 32
	start := make(chan struct{})
	errCh := make(chan error, numInserts)
	var wg sync.WaitGroup
	for i := 0; i < numInserts; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			col, err := engine.GetCollection("test")
			if err != nil {
				errCh <- err
				return
			}
			if err := col.Insert(context.Background(), &index.VectorEntry{
				ID:     fmt.Sprintf("vec_%d", i),
				Vector: []float32{float32(i), 0, 0},
			}); err != nil {
				errCh <- err
			}
		}(i)
	}

	close(start)
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			t.Fatalf("insert failed: %v", err)
		}
	}

	after := engine.WriteStats()
	txDelta := after.WALTransactions - before.WALTransactions
	if txDelta == 0 {
		t.Fatal("expected at least one WAL transaction for concurrent inserts")
	}
	if txDelta >= numInserts {
		t.Fatalf("expected concurrent inserts to coalesce, got %d WAL transactions for %d inserts", txDelta, numInserts)
	}
	if txDelta > 4 {
		t.Fatalf("expected group commit to keep WAL transactions small, got %d for %d inserts", txDelta, numInserts)
	}
}

func TestConcurrentSingleInsertsGroupCommitWindowSweep(t *testing.T) {
	cases := []struct {
		name   string
		window time.Duration
	}{
		{name: "disabled", window: 0},
		{name: "default", window: 1 * time.Millisecond},
		{name: "larger", window: 5 * time.Millisecond},
	}

	const numInserts = 256

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "group_commit_sweep.libravdb")

			engineIface, err := New(path)
			if err != nil {
				t.Fatalf("new engine: %v", err)
			}
			engine := engineIface.(*Engine)
			defer engine.Close()

			_, err = engine.CreateCollection("test", &storage.CollectionConfig{
				Dimension:      3,
				Metric:         2,
				IndexType:      0,
				M:              16,
				EfConstruction: 100,
				EfSearch:       50,
				ML:             1.0,
				Version:        1,
				RawVectorStore: "memory",
				RawStoreCap:    1024,
			})
			if err != nil {
				t.Fatalf("create collection: %v", err)
			}

			origWindow := groupCommitWindow.Load()
			groupCommitWindow.Store(int64(tc.window))
			defer func() { groupCommitWindow.Store(origWindow) }()

			before := engine.WriteStats()

			start := make(chan struct{})
			errCh := make(chan error, numInserts)
			var wg sync.WaitGroup
			begin := time.Now()
			for i := 0; i < numInserts; i++ {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					<-start
					col, err := engine.GetCollection("test")
					if err != nil {
						errCh <- err
						return
					}
					if err := col.Insert(context.Background(), &index.VectorEntry{
						ID:     fmt.Sprintf("vec_%d", i),
						Vector: []float32{float32(i), 0, 0},
					}); err != nil {
						errCh <- err
					}
				}(i)
			}

			close(start)
			wg.Wait()
			close(errCh)
			for err := range errCh {
				if err != nil {
					t.Fatalf("insert failed: %v", err)
				}
			}

			elapsed := time.Since(begin)
			after := engine.WriteStats()
			txDelta := after.WALTransactions - before.WALTransactions
			flushDelta := after.BatchFlushes - before.BatchFlushes
			entryDelta := after.BufferedVectorEntries - before.BufferedVectorEntries
			t.Logf("window=%s tx_delta=%d flush_delta=%d entry_delta=%d wal_bytes=%d elapsed=%s",
				tc.window,
				txDelta,
				flushDelta,
				entryDelta,
				after.WALBytes-before.WALBytes,
				elapsed,
			)

			if txDelta == 0 {
				t.Fatal("expected at least one WAL transaction")
			}
			if txDelta > numInserts {
				t.Fatalf("expected coalescing, got %d WAL transactions for %d inserts", txDelta, numInserts)
			}
		})
	}
}

func TestBatchWALCommitRecoveryReplaysAllRecords(t *testing.T) {
	path := filepath.Join(t.TempDir(), "batch_recovery.libravdb")

	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)

	// Create collection
	_, err = engine.CreateCollection("test", &storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	})
	if err != nil {
		engine.Close()
		t.Fatalf("create collection: %v", err)
	}

	// Insert 5 records in one batch
	entries := make([]*index.VectorEntry, 5)
	for i := 0; i < 5; i++ {
		entries[i] = &index.VectorEntry{
			ID:       string(rune('a' + i)),
			Vector:   []float32{float32(i), 0, 0},
			Metadata: map[string]interface{}{"idx": i},
		}
	}
	col := engine.collections["test"]
	if err := col.InsertBatch(context.Background(), entries); err != nil {
		engine.Close()
		t.Fatalf("batch insert: %v", err)
	}

	if err := engine.Close(); err != nil {
		t.Fatalf("close engine: %v", err)
	}

	// Reopen and verify all records are recovered
	reopenedIface, err := New(path)
	if err != nil {
		t.Fatalf("reopen engine: %v", err)
	}
	reopened := reopenedIface.(*Engine)
	defer reopened.Close()

	reopenedCol, err := reopened.GetCollection("test")
	if err != nil {
		t.Fatalf("get collection after reopen: %v", err)
	}
	for _, want := range entries {
		got, err := reopenedCol.Get(context.Background(), want.ID)
		if err != nil {
			t.Errorf("recovery get %s: %v", want.ID, err)
			continue
		}
		if got.ID != want.ID {
			t.Errorf("got ID %s, want %s", got.ID, want.ID)
		}
	}
}

// Note: Duplicate rejection is handled at the collection layer (libravdb),
// not at the storage layer. The storage layer's InsertBatch just persists
// records without checking for duplicates. Collection-level tests verify
// duplicate rejection behavior.

// TestA: InsertBatch returns error when collection is deleted
func TestInsertBatchReturnsErrorOnDeletedCollection(t *testing.T) {
	path := filepath.Join(t.TempDir(), "deleted_col.libravdb")

	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)
	defer engine.Close()

	// Create a collection
	_, err = engine.CreateCollection("test", &storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	// Get a handle to the collection
	col, err := engine.GetCollection("test")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}

	// Insert a record to make the collection valid
	if err := col.Insert(context.Background(), &index.VectorEntry{
		ID:     "r1",
		Vector: []float32{1, 0, 0},
	}); err != nil {
		t.Fatalf("initial insert: %v", err)
	}

	// Delete the collection
	if err := engine.DeleteCollection("test"); err != nil {
		t.Fatalf("delete collection: %v", err)
	}

	// Try to InsertBatch on the deleted collection handle - should return error
	if err := col.InsertBatch(context.Background(), []*index.VectorEntry{
		{ID: "r2", Vector: []float32{0, 1, 0}},
	}); err == nil {
		t.Errorf("expected error for InsertBatch on deleted collection, got nil")
	}
}

// TestB: Insert returns error when collection is deleted
func TestInsertReturnsErrorOnDeletedCollection(t *testing.T) {
	path := filepath.Join(t.TempDir(), "deleted_col2.libravdb")

	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)
	defer engine.Close()

	// Create a collection
	_, err = engine.CreateCollection("test", &storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	// Get a handle to the collection
	col, err := engine.GetCollection("test")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}

	// Insert a record to make the collection valid
	if err := col.Insert(context.Background(), &index.VectorEntry{
		ID:     "r1",
		Vector: []float32{1, 0, 0},
	}); err != nil {
		t.Fatalf("initial insert: %v", err)
	}

	// Delete the collection
	if err := engine.DeleteCollection("test"); err != nil {
		t.Fatalf("delete collection: %v", err)
	}

	// Try to Insert on the deleted collection handle - should return error
	if err := col.Insert(context.Background(), &index.VectorEntry{
		ID:     "r2",
		Vector: []float32{0, 1, 0},
	}); err == nil {
		t.Errorf("expected error for Insert on deleted collection, got nil")
	}
}

// TestC: Verify recovery still works after error propagation changes
func TestBatchWALRecoveryStillWorks(t *testing.T) {
	path := filepath.Join(t.TempDir(), "batch_recovery2.libravdb")

	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)

	// Create collection
	_, err = engine.CreateCollection("test", &storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	})
	if err != nil {
		engine.Close()
		t.Fatalf("create collection: %v", err)
	}

	col := engine.collections["test"]
	entries := make([]*index.VectorEntry, 5)
	for i := 0; i < 5; i++ {
		entries[i] = &index.VectorEntry{
			ID:       string(rune('a' + i)),
			Vector:   []float32{float32(i), 0, 0},
			Metadata: map[string]interface{}{"idx": i},
		}
	}
	if err := col.InsertBatch(context.Background(), entries); err != nil {
		engine.Close()
		t.Fatalf("batch insert: %v", err)
	}

	if err := engine.Close(); err != nil {
		t.Fatalf("close engine: %v", err)
	}

	// Reopen and verify all records are recovered
	reopenedIface, err := New(path)
	if err != nil {
		t.Fatalf("reopen engine: %v", err)
	}
	reopened := reopenedIface.(*Engine)
	defer reopened.Close()

	reopenedCol, err := reopened.GetCollection("test")
	if err != nil {
		t.Fatalf("get collection after reopen: %v", err)
	}
	for _, want := range entries {
		got, err := reopenedCol.Get(context.Background(), want.ID)
		if err != nil {
			t.Errorf("recovery get %s: %v", want.ID, err)
			continue
		}
		if got.ID != want.ID {
			t.Errorf("got ID %s, want %s", got.ID, want.ID)
		}
	}
}

// TestD: Verify New does not leak goroutine on initialization failure
func TestNewNoGoroutineLeakOnFailure(t *testing.T) {
	// Use the startBatchFlusher hook to detect if the flusher was started
	startCount := 0
	origHook := startBatchFlusher
	startBatchFlusher = func(e *Engine) {
		startCount++
	}
	defer func() { startBatchFlusher = origHook }()

	path := filepath.Join(t.TempDir(), "bad_init.libravdb")

	// Write a malformed file
	if err := os.WriteFile(path, []byte("not a valid database"), 0644); err != nil {
		t.Fatalf("write malformed file: %v", err)
	}

	// This should fail cleanly
	if _, err := New(path); err == nil {
		t.Fatalf("expected error for malformed database, got nil")
	}

	// The hook should NOT have been called - flusher never started
	if startCount != 0 {
		t.Errorf("expected flusher hook to not be called on init failure, was called %d times", startCount)
	}
}

// TestF: Verify flushBatch re-queues only the failed suffix, not the committed prefix
func TestFlushBatchReQueuesOnlyFailedSuffix(t *testing.T) {
	path := filepath.Join(t.TempDir(), "suffix_requeue.libravdb")

	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)
	defer engine.Close()

	// Create collection "good" but NOT collection "bad"
	_, err = engine.CreateCollection("good", &storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	})
	if err != nil {
		t.Fatalf("create collection good: %v", err)
	}

	// Manually populate batchBuffer with two batches:
	// batch 0: collection "good" with valid entry (will succeed)
	// batch 1: collection "bad" with valid entry (will fail - collection doesn't exist)
	engine.batchBuffer.mu.Lock()
	engine.batchBuffer.entries = []batchEntry{
		{
			collection: "good",
			entries: []*index.VectorEntry{
				{ID: "good_entry", Vector: []float32{1, 0, 0}},
			},
		},
		{
			collection: "bad",
			entries: []*index.VectorEntry{
				{ID: "bad_entry", Vector: []float32{0, 1, 0}},
			},
		},
	}
	engine.batchBuffer.mu.Unlock()

	// Call flushBatch - should fail on the "bad" collection
	flushErr := engine.flushBatch()
	if flushErr == nil {
		t.Fatalf("expected error when flushing bad collection, got nil")
	}

	// Verify the "good" entry was committed
	goodCol, err := engine.GetCollection("good")
	if err != nil {
		t.Fatalf("get good collection: %v", err)
	}
	got, err := goodCol.Get(context.Background(), "good_entry")
	if err != nil {
		t.Errorf("good entry should be committed but Get failed: %v", err)
	}
	if got == nil || got.ID != "good_entry" {
		t.Errorf("good_entry not found in good collection")
	}

	// Verify the buffer contains ONLY the failed suffix (batch for "bad"), not the full list
	engine.batchBuffer.mu.Lock()
	remainingEntries := engine.batchBuffer.entries
	engine.batchBuffer.mu.Unlock()

	if len(remainingEntries) != 1 {
		t.Fatalf("expected 1 remaining batch, got %d", len(remainingEntries))
	}
	if remainingEntries[0].collection != "bad" {
		t.Errorf("expected remaining batch to be for 'bad' collection, got '%s'", remainingEntries[0].collection)
	}
	if len(remainingEntries[0].entries) != 1 {
		t.Errorf("expected 1 entry in remaining batch, got %d", len(remainingEntries[0].entries))
	}
	if remainingEntries[0].entries[0].ID != "bad_entry" {
		t.Errorf("expected remaining entry to be 'bad_entry', got '%s'", remainingEntries[0].entries[0].ID)
	}
}

// TestE: Verify that re-queued entries are preserved on flush error
func TestFlushErrorReQueuesRemainingBatches(t *testing.T) {
	path := filepath.Join(t.TempDir(), "flush_requeue.libravdb")

	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)
	defer engine.Close()

	// Create collection
	_, err = engine.CreateCollection("test", &storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	// Insert some valid entries first
	col := engine.collections["test"]
	for i := 0; i < 3; i++ {
		if err := col.Insert(context.Background(), &index.VectorEntry{
			ID:     fmt.Sprintf("valid_%d", i),
			Vector: []float32{float32(i), 0, 0},
		}); err != nil {
			t.Fatalf("insert valid entry: %v", err)
		}
	}

	// Get a stale handle
	staleCol, err := engine.GetCollection("test")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}

	// Delete the collection to make the handle stale
	if err := engine.DeleteCollection("test"); err != nil {
		t.Fatalf("delete collection: %v", err)
	}

	// Now try to insert - should fail but not panic
	err = staleCol.Insert(context.Background(), &index.VectorEntry{
		ID:     "should_fail",
		Vector: []float32{1, 0, 0},
	})
	if err == nil {
		t.Errorf("expected error on stale collection insert, got nil")
	}
}

func TestCompactEmptyEngine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compact-empty.libravdb")
	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)
	defer engine.Close()

	if err := engine.Compact(); err != nil {
		t.Fatalf("Compact() on empty engine should succeed: %v", err)
	}
	if engine.CompactionErrors() != 0 {
		t.Errorf("expected 0 compaction errors on empty engine, got %d", engine.CompactionErrors())
	}
}

func TestCompactPreservesDataAndReducesFileSize(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compact-data.libravdb")
	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)

	_, err = engine.CreateCollection("test", &storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	col := engine.collections["test"]

	for i := 0; i < 50; i++ {
		if err := col.Insert(context.Background(), &index.VectorEntry{
			ID:     fmt.Sprintf("r%d", i),
			Vector: []float32{float32(i), 0, 0},
		}); err != nil {
			t.Fatalf("insert record %d: %v", i, err)
		}
	}

	preCompact, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat pre-compact: %v", err)
	}
	preSize := preCompact.Size()

	if err := engine.Compact(); err != nil {
		t.Fatalf("Compact() failed: %v", err)
	}

	postCompact, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat post-compact: %v", err)
	}
	postSize := postCompact.Size()

	for i := 50; i < 100; i++ {
		if err := col.Insert(context.Background(), &index.VectorEntry{
			ID:     fmt.Sprintf("r%d", i),
			Vector: []float32{float32(i), 0, 0},
		}); err != nil {
			t.Fatalf("insert record %d: %v", i, err)
		}
	}

	if err := engine.Close(); err != nil {
		t.Fatalf("close engine: %v", err)
	}

	engineIface2, err := New(path)
	if err != nil {
		t.Fatalf("reopen engine: %v", err)
	}
	engine2 := engineIface2.(*Engine)
	defer engine2.Close()

	col2, err := engine2.GetCollection("test")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}

	for i := 0; i < 100; i++ {
		entry, err := col2.Get(context.Background(), fmt.Sprintf("r%d", i))
		if err != nil {
			t.Fatalf("get record r%d: %v", i, err)
		}
		if entry == nil {
			t.Fatalf("record r%d not found after reopen", i)
		}
	}

	if postSize >= preSize {
		t.Errorf("expected Compact() to reduce file size: pre=%d post=%d", preSize, postSize)
	}
}

func TestOpenExistingIgnoresOrphanCompactSidecar(t *testing.T) {
	path := filepath.Join(t.TempDir(), "orphan-compact.libravdb")
	sidecarPath := path + ".compact"

	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)

	_, err = engine.CreateCollection("test", &storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	col := engine.collections["test"]
	if err := col.Insert(context.Background(), &index.VectorEntry{
		ID: "survivor", Vector: []float32{1, 2, 3},
	}); err != nil {
		t.Fatalf("insert: %v", err)
	}
	if err := engine.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read original: %v", err)
	}
	if err := os.WriteFile(sidecarPath, original, 0644); err != nil {
		t.Fatalf("write sidecar: %v", err)
	}

	engineIface2, err := New(path)
	if err != nil {
		t.Fatalf("open with orphan sidecar should succeed: %v", err)
	}
	engine2 := engineIface2.(*Engine)
	defer engine2.Close()

	col2, err := engine2.GetCollection("test")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}
	entry, err := col2.Get(context.Background(), "survivor")
	if err != nil {
		t.Fatalf("get survivor: %v", err)
	}
	if entry == nil {
		t.Fatal("survivor record not found after recovery with orphan sidecar")
	}
}

func TestCompactPreventsUnboundedFileGrowth(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compact-growth.libravdb")
	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)

	_, err = engine.CreateCollection("test", &storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	col := engine.collections["test"]

	var sizes []int64

	for cycle := 0; cycle < 5; cycle++ {
		for i := 0; i < 20; i++ {
			id := fmt.Sprintf("c%d_r%d", cycle, i)
			if err := col.Insert(context.Background(), &index.VectorEntry{
				ID: id, Vector: []float32{float32(cycle), float32(i), 0},
			}); err != nil {
				t.Fatalf("insert %s: %v", id, err)
			}
		}

		if err := engine.Compact(); err != nil {
			t.Fatalf("Compact() cycle %d: %v", cycle, err)
		}

		stat, err := os.Stat(path)
		if err != nil {
			t.Fatalf("stat cycle %d: %v", cycle, err)
		}
		sizes = append(sizes, stat.Size())
	}

	if err := engine.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	first := sizes[0]
	last := sizes[len(sizes)-1]
	if last > first*5 {
		t.Errorf("file size grew unboundedly across cycles: first=%d last=%d sizes=%v",
			first, last, sizes)
	}

	engineIface2, err := New(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	engine2 := engineIface2.(*Engine)
	defer engine2.Close()

	col2, err := engine2.GetCollection("test")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}
	for cycle := 0; cycle < 5; cycle++ {
		for i := 0; i < 20; i++ {
			id := fmt.Sprintf("c%d_r%d", cycle, i)
			entry, err := col2.Get(context.Background(), id)
			if err != nil {
				t.Fatalf("get %s: %v", id, err)
			}
			if entry == nil {
				t.Fatalf("record %s not found after cycles", id)
			}
		}
	}
}

func TestDeleteReleasesOrdinalSlot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "delete-ordinal.libravdb")
	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)

	_, err = engine.CreateCollection("test", &storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	col := engine.collections["test"]

	if err := col.Insert(context.Background(), &index.VectorEntry{
		ID: "doomed", Vector: []float32{1, 2, 3},
	}); err != nil {
		t.Fatalf("insert: %v", err)
	}

	// Capture the assigned ordinal.
	ordinal, err := col.GetIDByOrdinal(context.Background(), 0)
	if err != nil {
		t.Fatalf("GetIDByOrdinal(0): %v", err)
	}
	if ordinal != "doomed" {
		t.Fatalf("expected ordinal 0 -> doomed, got %q", ordinal)
	}

	// Delete the record.
	if err := col.Delete(context.Background(), "doomed"); err != nil {
		t.Fatalf("delete: %v", err)
	}

	// Immediately after delete, GetByOrdinal must return not-found.
	_, err = col.GetByOrdinal(0)
	if err == nil {
		t.Fatal("expected GetByOrdinal(0) to fail after delete")
	}

	// GetIDByOrdinal must also return not-found.
	_, err = col.GetIDByOrdinal(context.Background(), 0)
	if err == nil {
		t.Fatal("expected GetIDByOrdinal(0) to fail after delete")
	}

	// Close and reopen — ordinalToID is reconstructed from Records.
	// Deleted records must not re-populate the ordinal slot.
	if err := engine.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	engineIface2, err := New(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	engine2 := engineIface2.(*Engine)
	defer engine2.Close()

	// Populate the handle cache so we can access the concrete *Collection.
	if _, err := engine2.GetCollection("test"); err != nil {
		t.Fatalf("get collection: %v", err)
	}
	col2 := engine2.collections["test"]

	_, err = col2.GetByOrdinal(0)
	if err == nil {
		t.Fatal("expected GetByOrdinal(0) to fail after reopen")
	}

	_, err = col2.GetIDByOrdinal(context.Background(), 0)
	if err == nil {
		t.Fatal("expected GetIDByOrdinal(0) to fail after reopen")
	}
}

// TestReplayWALSkipsCorruptNonWALHeader verifies that replayWAL skips
// non-WAL chunks before reading their payload. Corrupting the PayloadLen
// in an index chunk header should not cause a hard failure — replayWAL
// skips it (kind != chunkTypeWAL), then the next iteration's chunkMagic
// check fails and breaks the loop cleanly.
func TestReplayWALSkipsCorruptNonWALHeader(t *testing.T) {
	path := filepath.Join(t.TempDir(), "replay-corrupt-header.libravdb")

	// Create and populate an engine with a compacted index chunk.
	eng, err := New(path)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	engine := eng.(*Engine)
	_, err = engine.CreateCollection("test", &storage.CollectionConfig{
		Dimension:      3,
		Metric:         2,
		IndexType:      0, // Flat
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		RawVectorStore: "memory",
		RawStoreCap:    1024,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	col := engine.collections["test"]
	if err := col.Insert(context.Background(), &index.VectorEntry{
		ID: "a", Vector: []float32{1, 2, 3},
	}); err != nil {
		t.Fatalf("insert: %v", err)
	}
	if err := engine.Compact(); err != nil {
		t.Fatalf("compact: %v", err)
	}
	engine.Close()

	// Corrupt the PayloadLen field in the index chunk header.
	// Chunk header layout: Magic(4) + Kind(2) + Version(2) + PayloadLen(4) + Checksum(4) = 16 bytes
	// PayloadLen is at offset 8 in the header.
	f, err := os.OpenFile(path, os.O_RDWR, 0644)
	if err != nil {
		t.Fatalf("open for corruption: %v", err)
	}
	// Read header to find active metapage.
	headerBuf := make([]byte, pageSize)
	if _, err := f.ReadAt(headerBuf, 0); err != nil {
		f.Close()
		t.Fatalf("read header: %v", err)
	}
	activeMeta := binary.LittleEndian.Uint64(headerBuf[40:48])

	// Read active metapage to find IndexOffset.
	metaBuf := make([]byte, pageSize)
	if _, err := f.ReadAt(metaBuf, int64(activeMeta)*pageSize); err != nil {
		f.Close()
		t.Fatalf("read metapage: %v", err)
	}
	indexOffset := int64(binary.LittleEndian.Uint64(metaBuf[68:76]))
	if indexOffset == 0 {
		f.Close()
		t.Fatal("no index chunk to corrupt")
	}

	// Corrupt PayloadLen (offset 8 in chunk header) to a wildly wrong value.
	corruptBytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(corruptBytes, 0xFFFFFFFF)
	if _, err := f.WriteAt(corruptBytes, indexOffset+8); err != nil {
		f.Close()
		t.Fatalf("corrupt PayloadLen: %v", err)
	}
	f.Close()

	// Reopen should succeed — replayWAL skips the non-WAL chunk before
	// reading its (now-corrupt) payload, then the next iteration fails
	// chunkMagic check and breaks cleanly.
	eng2, err := New(path)
	if err != nil {
		t.Fatalf("reopen after header corruption: %v", err)
	}
	defer eng2.Close()

	// Verify the collection exists and data was rebuilt from records.
	colIface, err := eng2.GetCollection("test")
	if err != nil {
		t.Fatalf("GetCollection: %v", err)
	}
	col2 := colIface.(*Collection)
	vec, err := col2.GetByOrdinal(0)
	if err != nil {
		t.Fatalf("GetByOrdinal(0): %v", err)
	}
	if len(vec) != 3 || vec[0] != 1 || vec[1] != 2 || vec[2] != 3 {
		t.Fatalf("unexpected vector: %v", vec)
	}
}

// TestOrdinalPreReservation_CrashRecovery verifies that ordinals reserved via the
// atomic counter but never committed do not leak into NextOrdinal after restart.
func TestOrdinalPreReservation_CrashRecovery(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ordinal_crash.libravdb")

	engIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	eng := engIface.(*Engine)
	defer eng.Close()

	_, err = eng.CreateCollection("test", &storage.CollectionConfig{
		Dimension: 3, Metric: 2, IndexType: 0, M: 16, EfConstruction: 100, EfSearch: 50,
		ML: 1.0, Version: 1, RawVectorStore: "memory", RawStoreCap: 1024,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	colIface, err := eng.GetCollection("test")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}
	col := colIface.(*Collection)

	// Assign ordinals via atomic counter but do NOT commit via putRecords/flushBatch.
	// These reservations must be treated as ephemeral after restart.
	uncommitted := []*index.VectorEntry{
		{ID: "u1", Vector: []float32{1, 0, 0}},
		{ID: "u2", Vector: []float32{0, 1, 0}},
	}
	if err := col.assignOrdinals(uncommitted); err != nil {
		t.Fatalf("assign ordinals: %v", err)
	}
	t.Logf("uncommitted ordinals: u1=%d u2=%d", uncommitted[0].Ordinal, uncommitted[1].Ordinal)

	eng.Close()

	// Reopen: recovery must recompute NextOrdinal from committed state only.
	engIface2, err := New(path)
	if err != nil {
		t.Fatalf("reopen engine: %v", err)
	}
	eng2 := engIface2.(*Engine)
	defer eng2.Close()

	colIface2, err := eng2.GetCollection("test")
	if err != nil {
		t.Fatalf("get collection after reopen: %v", err)
	}
	col2 := colIface2.(*Collection)

	next, err := col2.NextOrdinal(context.Background())
	if err != nil {
		t.Fatalf("NextOrdinal: %v", err)
	}
	if next != 0 {
		t.Fatalf("expected NextOrdinal=0 (no committed entries), got %d", next)
	}

	reserved := eng2.state.Collections["test"].reservedNextOrdinal.Load()
	if reserved != 0 {
		t.Fatalf("expected reservedNextOrdinal=0 after recovery, got %d", reserved)
	}
}

// TestOrdinalPreReservation_NextOrdinalAfterCommit verifies that NextOrdinal
// and reservedNextOrdinal reflect committed state after writes and after restart.
func TestOrdinalPreReservation_NextOrdinalAfterCommit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ordinal_commit.libravdb")

	engIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	eng := engIface.(*Engine)
	defer eng.Close()

	_, err = eng.CreateCollection("test", &storage.CollectionConfig{
		Dimension: 3, Metric: 2, IndexType: 0, M: 16, EfConstruction: 100, EfSearch: 50,
		ML: 1.0, Version: 1, RawVectorStore: "memory", RawStoreCap: 1024,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	colIface, err := eng.GetCollection("test")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}
	col := colIface.(*Collection)

	for i := 0; i < 5; i++ {
		entry := &index.VectorEntry{
			ID: fmt.Sprintf("e%d", i), Vector: []float32{float32(i), 0, 0},
		}
		if err := col.Insert(context.Background(), entry); err != nil {
			t.Fatalf("insert e%d: %v", i, err)
		}
	}

	next, err := col.NextOrdinal(context.Background())
	if err != nil {
		t.Fatalf("NextOrdinal: %v", err)
	}
	if next != 5 {
		t.Fatalf("expected NextOrdinal=5 after 5 inserts, got %d", next)
	}

	persisted := eng.state.Collections["test"]
	reserved := persisted.reservedNextOrdinal.Load()
	if reserved < 5 {
		t.Fatalf("reservedNextOrdinal=%d < NextOrdinal=%d after commits", reserved, next)
	}
	t.Logf("NextOrdinal=%d reservedNextOrdinal=%d", next, reserved)

	eng.Close()

	engIface2, err := New(path)
	if err != nil {
		t.Fatalf("reopen engine: %v", err)
	}
	eng2 := engIface2.(*Engine)
	defer eng2.Close()

	colIface2, err := eng2.GetCollection("test")
	if err != nil {
		t.Fatalf("get collection after reopen: %v", err)
	}
	col2 := colIface2.(*Collection)

	next2, err := col2.NextOrdinal(context.Background())
	if err != nil {
		t.Fatalf("NextOrdinal after reopen: %v", err)
	}
	if next2 != 5 {
		t.Fatalf("expected NextOrdinal=5 after reopen, got %d", next2)
	}

	reserved2 := eng2.state.Collections["test"].reservedNextOrdinal.Load()
	if reserved2 != 5 {
		t.Fatalf("expected reservedNextOrdinal=5 after reopen, got %d", reserved2)
	}
}

// TestOrdinalPreReservation_ConcurrentAssignments verifies that concurrent
// atomic ordinal assignments produce unique, contiguous ordinals.
func TestOrdinalPreReservation_ConcurrentAssignments(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ordinal_concurrent.libravdb")

	engIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	eng := engIface.(*Engine)
	defer eng.Close()

	_, err = eng.CreateCollection("test", &storage.CollectionConfig{
		Dimension: 3, Metric: 2, IndexType: 0, M: 16, EfConstruction: 100, EfSearch: 50,
		ML: 1.0, Version: 1, RawVectorStore: "memory", RawStoreCap: 1024,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	colIface, err := eng.GetCollection("test")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}
	col := colIface.(*Collection)

	const numG = 8
	const perG = 100
	var wg sync.WaitGroup
	ordinals := make([]uint32, numG*perG)
	var ordMu sync.Mutex

	for g := 0; g < numG; g++ {
		wg.Add(1)
		go func(gid int) {
			defer wg.Done()
			for i := 0; i < perG; i++ {
				entry := &index.VectorEntry{
					ID: fmt.Sprintf("g%d_i%d", gid, i), Vector: []float32{float32(gid), float32(i), 0},
				}
				if err := col.assignOrdinals([]*index.VectorEntry{entry}); err != nil {
					t.Errorf("assignOrdinals g=%d i=%d: %v", gid, i, err)
					return
				}
				ordMu.Lock()
				ordinals[gid*perG+i] = entry.Ordinal
				ordMu.Unlock()
			}
		}(g)
	}
	wg.Wait()

	seen := make(map[uint32]bool, len(ordinals))
	var minO, maxO uint32 = ^uint32(0), 0
	for _, o := range ordinals {
		if seen[o] {
			t.Fatalf("duplicate ordinal %d", o)
		}
		seen[o] = true
		if o < minO {
			minO = o
		}
		if o > maxO {
			maxO = o
		}
	}

	expected := uint32(numG * perG)
	if uint32(len(seen)) != expected {
		t.Fatalf("expected %d unique ordinals, got %d", expected, len(seen))
	}
	if minO != 0 || maxO != expected-1 {
		t.Fatalf("expected range [0,%d], got [%d,%d]", expected-1, minO, maxO)
	}
	t.Logf("concurrent: %d goroutines x %d = %d unique ordinals [%d,%d]",
		numG, perG, expected, minO, maxO)
}
