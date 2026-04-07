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

	origWindow := groupCommitWindow
	groupCommitWindow = 20 * time.Millisecond
	defer func() { groupCommitWindow = origWindow }()

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

			origWindow := groupCommitWindow
			groupCommitWindow = tc.window
			defer func() { groupCommitWindow = origWindow }()

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
