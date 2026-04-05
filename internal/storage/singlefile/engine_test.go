package singlefile

import (
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

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
