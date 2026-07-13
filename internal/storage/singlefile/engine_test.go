package singlefile

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
)

type recoveryIndexProvider struct {
	mu           sync.Mutex
	deserialized []string
	rebuilt      []string
}

func (p *recoveryIndexProvider) SerializeIndex(string) ([]byte, error) {
	return []byte("checkpoint-index"), nil
}

func (p *recoveryIndexProvider) DeserializeIndex(name string, _ []byte, _ *storage.CollectionConfig) error {
	p.mu.Lock()
	p.deserialized = append(p.deserialized, name)
	p.mu.Unlock()
	return nil
}

func (p *recoveryIndexProvider) RebuildIndex(name string, _ *storage.CollectionConfig) error {
	p.mu.Lock()
	p.rebuilt = append(p.rebuilt, name)
	p.mu.Unlock()
	return nil
}

func (*recoveryIndexProvider) IndexTypeVersion(string) (uint8, uint16) {
	return 0, formatVersion
}

func (*recoveryIndexProvider) SnapshotVectors(context.Context) error { return nil }

func TestWALWriteBufferUsesReusableOffHeapArena(t *testing.T) {
	path := filepath.Join(t.TempDir(), "wal_arena.libravdb")
	engineIface, err := New(path, WithWALSync(false))
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)
	defer engine.Close()

	engine.mu.Lock()
	first, temporary, err := engine.allocateWALWriteBufferLocked(4096)
	if err != nil {
		engine.mu.Unlock()
		t.Fatalf("allocate first buffer: %v", err)
	}
	if temporary != nil {
		engine.mu.Unlock()
		t.Fatal("normal WAL buffer unexpectedly used a temporary arena")
	}
	firstPtr := uintptr(unsafe.Pointer(unsafe.SliceData(first)))
	if firstPtr&63 != 0 {
		engine.mu.Unlock()
		t.Fatalf("WAL buffer is not 64-byte aligned: %#x", firstPtr)
	}

	second, temporary, err := engine.allocateWALWriteBufferLocked(4096)
	engine.mu.Unlock()
	if err != nil {
		t.Fatalf("allocate second buffer: %v", err)
	}
	if temporary != nil {
		t.Fatal("normal WAL buffer unexpectedly used a temporary arena")
	}
	secondPtr := uintptr(unsafe.Pointer(unsafe.SliceData(second)))
	if secondPtr != firstPtr {
		t.Fatalf("WAL arena was not reused: first=%#x second=%#x", firstPtr, secondPtr)
	}
}

func TestWALSyncDefaultsOnAndUnsafeModeIsExplicit(t *testing.T) {
	defaultIface, err := New(filepath.Join(t.TempDir(), "default_sync.libravdb"))
	if err != nil {
		t.Fatalf("new default engine: %v", err)
	}
	defaultEngine := defaultIface.(*Engine)
	if !defaultEngine.walSync {
		t.Fatal("WAL sync must be enabled by default")
	}
	if err := defaultEngine.Close(); err != nil {
		t.Fatalf("close default engine: %v", err)
	}

	unsafeIface, err := New(filepath.Join(t.TempDir(), "unsafe_sync.libravdb"), WithWALSync(false))
	if err != nil {
		t.Fatalf("new unsafe engine: %v", err)
	}
	unsafeEngine := unsafeIface.(*Engine)
	if unsafeEngine.walSync {
		t.Fatal("WithWALSync(false) did not disable sync")
	}
	if err := unsafeEngine.Close(); err != nil {
		t.Fatalf("close unsafe engine: %v", err)
	}
}

func TestGroupCommitSyncsOnceAndAcknowledgesAllWriters(t *testing.T) {
	path := filepath.Join(t.TempDir(), "group_sync.libravdb")
	engineIface, err := New(path, WithWALSync(false))
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)
	defer engine.Close()

	collection, err := engine.CreateCollection("vectors", &storage.CollectionConfig{Dimension: 3})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	engine.walSync = true
	var syncCalls atomic.Uint64
	engine.walSyncFn = func(*os.File) error {
		syncCalls.Add(1)
		return nil
	}

	origWindow := groupCommitWindow.Load()
	groupCommitWindow.Store(int64(20 * time.Millisecond))
	defer groupCommitWindow.Store(origWindow)

	const writers = 32
	start := make(chan struct{})
	errCh := make(chan error, writers)
	var wg sync.WaitGroup
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			errCh <- collection.Insert(context.Background(), &index.VectorEntry{
				ID:     fmt.Sprintf("vector-%d", i),
				Vector: []float32{float32(i), 1, 2},
			})
		}(i)
	}
	close(start)
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			t.Fatalf("insert: %v", err)
		}
	}

	calls := syncCalls.Load()
	if calls == 0 {
		t.Fatal("durable group did not invoke WAL sync")
	}
	if calls >= writers {
		t.Fatalf("group commit issued %d syncs for %d writers", calls, writers)
	}
	if got := engine.WriteStats().BufferedVectorEntries; got != writers {
		t.Fatalf("durable entries = %d, want %d", got, writers)
	}
}

func TestGroupCommitReturnsSyncFailureToWriter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sync_failure.libravdb")
	engineIface, err := New(path, WithWALSync(false))
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)
	defer engine.Close()

	collection, err := engine.CreateCollection("vectors", &storage.CollectionConfig{Dimension: 3})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	syncFailure := errors.New("injected sync failure")
	engine.walSync = true
	engine.walSyncFn = func(*os.File) error { return syncFailure }

	err = collection.Insert(context.Background(), &index.VectorEntry{
		ID:     "vector",
		Vector: []float32{1, 2, 3},
	})
	if !errors.Is(err, syncFailure) {
		t.Fatalf("insert error = %v, want injected sync failure", err)
	}
	if exists, existsErr := collection.Exists(context.Background(), "vector"); existsErr != nil || exists {
		t.Fatalf("failed durable insert became visible: exists=%v err=%v", exists, existsErr)
	}

	engine.walSyncFn = nil
}

func TestRecoveryRebuildsIndexTouchedAfterCheckpoint(t *testing.T) {
	path := filepath.Join(t.TempDir(), "replay_index_lag.libravdb")
	provider := &recoveryIndexProvider{}
	engineIface, err := New(path, WithIndexSnapshotProvider(provider))
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)
	collection, err := engine.CreateCollection("vectors", &storage.CollectionConfig{
		Dimension:      3,
		IndexType:      0,
		RawVectorStore: "memory",
		RawStoreCap:    16,
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.Insert(context.Background(), &index.VectorEntry{
		ID: "checkpointed", Vector: []float32{1, 0, 0},
	}); err != nil {
		t.Fatalf("insert checkpointed record: %v", err)
	}

	engine.mu.Lock()
	err = engine.checkpointLocked()
	engine.mu.Unlock()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}

	if err := collection.Insert(context.Background(), &index.VectorEntry{
		ID: "wal-only", Vector: []float32{0, 1, 0},
	}); err != nil {
		t.Fatalf("insert post-checkpoint record: %v", err)
	}

	// Simulate process loss: stop the flusher and close the descriptor without
	// running Engine.Close, which would checkpoint the current state.
	engine.cancel()
	if engine.walWriteArena != nil {
		_ = engine.walWriteArena.Free()
		engine.walWriteArena = nil
	}
	for _, persisted := range engine.state.Collections {
		if persisted.vectorSFL != nil {
			persisted.vectorSFL.Free()
			persisted.vectorSFL = nil
		}
	}
	if err := engine.file.Close(); err != nil {
		t.Fatalf("crash close: %v", err)
	}

	recoveryProvider := &recoveryIndexProvider{}
	reopenedIface, err := New(path, WithIndexSnapshotProvider(recoveryProvider))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	reopened := reopenedIface.(*Engine)
	defer reopened.Close()

	recoveryProvider.mu.Lock()
	deserialized := append([]string(nil), recoveryProvider.deserialized...)
	rebuilt := append([]string(nil), recoveryProvider.rebuilt...)
	recoveryProvider.mu.Unlock()
	if len(deserialized) != 1 || deserialized[0] != "vectors" {
		t.Fatalf("deserialized indexes = %v, want [vectors]", deserialized)
	}
	if len(rebuilt) != 1 || rebuilt[0] != "vectors" {
		t.Fatalf("rebuilt indexes = %v, want [vectors]", rebuilt)
	}

	recoveredCollection, err := reopened.GetCollection("vectors")
	if err != nil {
		t.Fatalf("get recovered collection: %v", err)
	}
	if exists, err := recoveredCollection.Exists(context.Background(), "wal-only"); err != nil || !exists {
		t.Fatalf("post-checkpoint record recovery: exists=%v err=%v", exists, err)
	}
}

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
	page[88] ^= 0xff // V2 metapage checksum.
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

func TestRecoveryFallsBackFromTruncatedNewestSnapshot(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "live-source.libravdb")
	copyPath := filepath.Join(dir, "live-copy.libravdb")

	engineIface, err := New(sourcePath)
	if err != nil {
		t.Fatal(err)
	}
	engine := engineIface.(*Engine)
	collection, err := engine.CreateCollection("records", &storage.CollectionConfig{Dimension: 2})
	if err != nil {
		engine.Close()
		t.Fatal(err)
	}
	if err := collection.Insert(context.Background(), &index.VectorEntry{
		ID: "before-checkpoint", Vector: []float32{1, 0},
	}); err != nil {
		engine.Close()
		t.Fatal(err)
	}
	engine.mu.Lock()
	err = engine.checkpointLocked()
	engine.mu.Unlock()
	if err != nil {
		engine.Close()
		t.Fatal(err)
	}

	if err := collection.Insert(context.Background(), &index.VectorEntry{
		ID: "wal-after-older-checkpoint", Vector: []float32{0, 1},
	}); err != nil {
		engine.Close()
		t.Fatal(err)
	}
	engine.mu.Lock()
	err = engine.checkpointLocked()
	engine.mu.Unlock()
	if err != nil {
		engine.Close()
		t.Fatal(err)
	}

	meta1, err1 := engine.readMetaPage(1)
	meta2, err2 := engine.readMetaPage(2)
	if err1 != nil || err2 != nil {
		engine.Close()
		t.Fatalf("read source metapages: meta1=%v meta2=%v", err1, err2)
	}
	newest, older := meta1, meta2
	if meta2.MetaEpoch > meta1.MetaEpoch {
		newest, older = meta2, meta1
	}
	if newest.MetaEpoch <= older.MetaEpoch || newest.SnapshotLength < 2 {
		engine.Close()
		t.Fatalf("checkpoint epochs/length unsuitable for fallback test: newest=%d older=%d length=%d", newest.MetaEpoch, older.MetaEpoch, newest.SnapshotLength)
	}

	contents, err := os.ReadFile(sourcePath)
	if err != nil {
		engine.Close()
		t.Fatal(err)
	}
	if err := os.WriteFile(copyPath, contents, 0o600); err != nil {
		engine.Close()
		t.Fatal(err)
	}
	truncatedSize := int64(newest.SnapshotOffset + 16 + newest.SnapshotLength/2)
	if err := os.Truncate(copyPath, truncatedSize); err != nil {
		engine.Close()
		t.Fatal(err)
	}
	if err := engine.Close(); err != nil {
		t.Fatal(err)
	}

	reopenedIface, err := New(copyPath)
	if err != nil {
		t.Fatalf("open live copy with truncated newest snapshot: %v", err)
	}
	reopened := reopenedIface.(*Engine)
	defer reopened.Close()
	if reopened.activeMetaPage != older.PageNumber {
		t.Fatalf("selected metapage %d, want older complete page %d", reopened.activeMetaPage, older.PageNumber)
	}

	recovered, err := reopened.GetCollection("records")
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"before-checkpoint", "wal-after-older-checkpoint"} {
		exists, err := recovered.Exists(context.Background(), id)
		if err != nil || !exists {
			t.Fatalf("recovered record %q: exists=%v err=%v", id, exists, err)
		}
	}
}

func TestRecoveryIgnoresTruncatedFinalWALFrame(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "wal-source.libravdb")
	copyPath := filepath.Join(dir, "wal-copy.libravdb")

	engineIface, err := New(sourcePath)
	if err != nil {
		t.Fatal(err)
	}
	engine := engineIface.(*Engine)
	collection, err := engine.CreateCollection("records", &storage.CollectionConfig{Dimension: 2})
	if err != nil {
		engine.Close()
		t.Fatal(err)
	}
	engine.mu.Lock()
	err = engine.checkpointLocked()
	engine.mu.Unlock()
	if err != nil {
		engine.Close()
		t.Fatal(err)
	}

	for _, entry := range []*index.VectorEntry{
		{ID: "committed-prefix", Vector: []float32{1, 0}},
		{ID: "commit-cut-by-copy", Vector: []float32{0, 1}},
	} {
		if err := collection.Insert(context.Background(), entry); err != nil {
			engine.Close()
			t.Fatal(err)
		}
	}

	contents, err := os.ReadFile(sourcePath)
	if err != nil {
		engine.Close()
		t.Fatal(err)
	}
	lastWALOffset := -1
	lastWALPayload := uint32(0)
	for offset := 3 * pageSize; offset+16 <= len(contents); {
		header := decodeChunkHeader(contents[offset : offset+16])
		if header.Magic != chunkMagic || uint64(header.PayloadLen) > uint64(len(contents)-offset-16) {
			break
		}
		if header.Kind == chunkTypeWAL {
			lastWALOffset = offset
			lastWALPayload = header.PayloadLen
		}
		offset += 16 + int(header.PayloadLen)
	}
	if lastWALOffset < 0 || lastWALPayload < 2 {
		engine.Close()
		t.Fatalf("could not locate final WAL frame: offset=%d payload=%d", lastWALOffset, lastWALPayload)
	}
	truncatedSize := lastWALOffset + 16 + int(lastWALPayload/2)
	if err := os.WriteFile(copyPath, contents[:truncatedSize], 0o600); err != nil {
		engine.Close()
		t.Fatal(err)
	}
	if err := engine.Close(); err != nil {
		t.Fatal(err)
	}

	reopenedIface, err := New(copyPath)
	if err != nil {
		t.Fatalf("open copy ending inside final WAL frame: %v", err)
	}
	reopened := reopenedIface.(*Engine)
	defer reopened.Close()
	recovered, err := reopened.GetCollection("records")
	if err != nil {
		t.Fatal(err)
	}
	if exists, err := recovered.Exists(context.Background(), "committed-prefix"); err != nil || !exists {
		t.Fatalf("complete transaction before torn tail was not recovered: exists=%v err=%v", exists, err)
	}
	if exists, err := recovered.Exists(context.Background(), "commit-cut-by-copy"); err != nil || exists {
		t.Fatalf("transaction without complete copied commit became visible: exists=%v err=%v", exists, err)
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

// TestF: a failed flush does not publish an unsynced prefix or retry an
// ambiguous on-disk transaction automatically.
func TestFlushBatchFailureDoesNotPublishOrRequeue(t *testing.T) {
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

	// The valid prefix was appended but the group never crossed its sync
	// boundary, so it must not be visible in the live state.
	goodCol, err := engine.GetCollection("good")
	if err != nil {
		t.Fatalf("get good collection: %v", err)
	}
	got, err := goodCol.Get(context.Background(), "good_entry")
	if err == nil || got != nil {
		t.Fatalf("unsynced prefix became visible: entry=%v err=%v", got, err)
	}

	// An append failure is ambiguous; automatic requeue could duplicate a
	// transaction whose complete bytes reached the kernel.
	engine.batchBuffer.mu.Lock()
	remainingEntries := engine.batchBuffer.entries
	engine.batchBuffer.mu.Unlock()

	if len(remainingEntries) != 0 {
		t.Fatalf("ambiguous failed group was requeued: %d batches", len(remainingEntries))
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

func TestNewSyncsParentDirectory(t *testing.T) {
	originalSync := syncDatabaseParent
	defer func() { syncDatabaseParent = originalSync }()

	var calls int
	syncDatabaseParent = func(string) error {
		calls++
		return nil
	}

	engineIface, err := New(filepath.Join(t.TempDir(), "new-publish.libravdb"))
	if err != nil {
		t.Fatal(err)
	}
	if err := engineIface.Close(); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("parent directory sync calls = %d, want 1", calls)
	}
}

func TestCompactParentSyncFailureKeepsEngineUsable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compact-sync-failure.libravdb")
	engineIface, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	engine := engineIface.(*Engine)

	originalSync := syncDatabaseParent
	defer func() { syncDatabaseParent = originalSync }()
	syncFailure := errors.New("injected parent sync failure")
	syncDatabaseParent = func(string) error { return syncFailure }
	err = engine.Compact()
	syncDatabaseParent = originalSync
	if !errors.Is(err, syncFailure) {
		engine.Close()
		t.Fatalf("Compact() error = %v, want injected sync failure", err)
	}
	if engine.file == nil {
		t.Fatal("engine file was left nil after completed rename and failed directory sync")
	}
	if engine.CompactionErrors() != 1 {
		t.Fatalf("compaction errors = %d, want 1", engine.CompactionErrors())
	}
	if _, err := engine.CreateCollection("still-usable", &storage.CollectionConfig{Dimension: 2}); err != nil {
		engine.Close()
		t.Fatalf("engine unusable after directory sync failure: %v", err)
	}
	if err := engine.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := New(path)
	if err != nil {
		t.Fatalf("reopen after directory sync failure: %v", err)
	}
	defer reopened.Close()
	if _, err := reopened.GetCollection("still-usable"); err != nil {
		t.Fatalf("post-failure write was not recoverable: %v", err)
	}
}

func TestBackupParentSyncFailureRemovesDestination(t *testing.T) {
	sourcePath := filepath.Join(t.TempDir(), "backup-sync-source.libravdb")
	destinationPath := filepath.Join(t.TempDir(), "backup-sync-destination.libravdb")
	engineIface, err := New(sourcePath)
	if err != nil {
		t.Fatal(err)
	}
	engine := engineIface.(*Engine)
	defer engine.Close()

	originalSync := syncDatabaseParent
	defer func() { syncDatabaseParent = originalSync }()
	syncFailure := errors.New("injected backup parent sync failure")
	syncDatabaseParent = func(string) error { return syncFailure }
	err = engine.Backup(context.Background(), destinationPath)
	syncDatabaseParent = originalSync
	if !errors.Is(err, syncFailure) {
		t.Fatalf("Backup() error = %v, want injected sync failure", err)
	}
	if _, err := os.Stat(destinationPath); !os.IsNotExist(err) {
		t.Fatalf("failed backup destination was retained: %v", err)
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
	activeMeta := binary.LittleEndian.Uint64(headerBuf[64:72])

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

func TestEngine_Vacuum(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vacuum.libravdb")
	db, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	engine := db.(*Engine)

	coll, err := engine.CreateCollection("test_coll", &storage.CollectionConfig{Dimension: 2, Metric: 0})
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Insert 1000 records
	var entries []*index.VectorEntry
	for i := 0; i < 1000; i++ {
		entries = append(entries, &index.VectorEntry{
			ID:      fmt.Sprintf("id-%d", i),
			Ordinal: uint32(i + 1),
			Vector:  []float32{1.0, float32(i)},
		})
	}
	// Bypass missing InsertBatch and use loop
	for _, entry := range entries {
		if err := coll.Insert(context.Background(), entry); err != nil {
			t.Fatalf("Insert: %v", err)
		}
	}

	// Delete 500 records
	for i := 0; i < 500; i++ {
		if err := coll.Delete(context.Background(), fmt.Sprintf("id-%d", i)); err != nil {
			t.Fatalf("Delete: %v", err)
		}
	}

	// Force checkpoint
	engine.mu.Lock()
	if err := engine.checkpointLocked(); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	engine.mu.Unlock()

	stat, _ := os.Stat(path)
	preVacuumSize := stat.Size()

	// Run Vacuum
	if err := engine.Vacuum(context.Background()); err != nil {
		t.Fatalf("Vacuum: %v", err)
	}

	stat, _ = os.Stat(path)
	postVacuumSize := stat.Size()

	if postVacuumSize > preVacuumSize {
		t.Fatalf("Vacuum did not shrink or maintain file size: pre=%d, post=%d", preVacuumSize, postVacuumSize)
	}

	// Verify remaining records are accessible
	for i := 500; i < 1000; i++ {
		exists, err := coll.Exists(context.Background(), fmt.Sprintf("id-%d", i))
		if err != nil || !exists {
			t.Fatalf("Expected record id-%d to exist", i)
		}
	}

	// Verify deleted records are gone
	for i := 0; i < 500; i++ {
		exists, err := coll.Exists(context.Background(), fmt.Sprintf("id-%d", i))
		if err != nil || exists {
			t.Fatalf("Expected record id-%d to be deleted", i)
		}
	}

	if err := engine.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestEngine_Backup(t *testing.T) {
	path := filepath.Join(t.TempDir(), "backup_source.libravdb")
	backupPath := filepath.Join(t.TempDir(), "backup_dest.libravdb")

	db, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	engine := db.(*Engine)

	coll, err := engine.CreateCollection("test_coll", &storage.CollectionConfig{Dimension: 2, Metric: 0})
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Insert 10 records
	for i := 0; i < 10; i++ {
		entry := &index.VectorEntry{
			ID:      fmt.Sprintf("id-%d", i),
			Ordinal: uint32(i + 1),
			Vector:  []float32{1.0, float32(i)},
		}
		if err := coll.Insert(context.Background(), entry); err != nil {
			t.Fatalf("Insert: %v", err)
		}
	}

	// Create backup
	if err := engine.Backup(context.Background(), backupPath); err != nil {
		t.Fatalf("Backup failed: %v", err)
	}

	// Verify original engine still works
	exists, _ := coll.Exists(context.Background(), "id-5")
	if !exists {
		t.Fatalf("Original engine lost record after backup")
	}

	if err := engine.Close(); err != nil {
		t.Fatalf("Close original: %v", err)
	}

	// Verify backup engine works
	backupDB, err := New(backupPath)
	if err != nil {
		t.Fatalf("Failed to open backup: %v", err)
	}
	backupEngine := backupDB.(*Engine)
	defer backupEngine.Close()

	backupColl, err := backupEngine.GetCollection("test_coll")
	if err != nil {
		t.Fatalf("Failed to get collection from backup: %v", err)
	}

	for i := 0; i < 10; i++ {
		exists, err := backupColl.Exists(context.Background(), fmt.Sprintf("id-%d", i))
		if err != nil || !exists {
			t.Fatalf("Backup missing record id-%d", i)
		}
	}
}

func TestEngine_Drop(t *testing.T) {
	path := filepath.Join(t.TempDir(), "drop.libravdb")

	db, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	engine := db.(*Engine)

	// Create some data
	coll, err := engine.CreateCollection("test_coll", &storage.CollectionConfig{Dimension: 2, Metric: 0})
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Drop engine
	if err := engine.Drop(context.Background()); err != nil {
		t.Fatalf("Drop failed: %v", err)
	}

	// Verify file is gone
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("Database file still exists after Drop")
	}

	// Verify ops fail
	if err := coll.Insert(context.Background(), &index.VectorEntry{ID: "test"}); err == nil {
		t.Fatalf("Expected Insert to fail on dropped engine")
	}
}
