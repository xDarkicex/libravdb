package singlefile

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/xDarkicex/libravdb/internal/storage"
)

// appendExternally writes frames the way another process would.
//
// The instance's own view of the file stays where it was: the watermark
// does not move and the counters do not advance. That is what a write by
// another process looks like from here, and it is the case the refresh
// exists for.
//
// It returns the LSN and tx id the frames were written with, so the caller
// can write the rest of the transaction later.
func appendExternally(t *testing.T, engine *Engine, frames []walRecord) {
	t.Helper()
	engine.mu.Lock()
	defer engine.mu.Unlock()
	lsn := engine.lastLSN.Load()
	txID := engine.lastTxID.Load()
	watermark := engine.walAppliedOffset
	if _, err := engine.appendTransactionLocked(frames); err != nil {
		t.Fatalf("append: %v", err)
	}
	engine.lastLSN.Store(lsn)
	engine.lastTxID.Store(txID)
	engine.walAppliedOffset = watermark
}

// externalTx is a transaction written by another process.
type externalTx struct {
	// txID identifies the transaction.
	txID uint64
	// beginLSN is the LSN of its begin frame.
	beginLSN uint64
	// putLSN is the LSN of its record frame.
	putLSN uint64
}

// beginExternalTx writes the begin and record frames of a transaction, and
// leaves the commit for later.
func beginExternalTx(t *testing.T, engine *Engine, collection, id string) externalTx {
	t.Helper()
	engine.mu.Lock()
	tx := externalTx{
		txID:     engine.lastTxID.Load() + 100,
		beginLSN: engine.lastLSN.Load() + 1,
	}
	tx.putLSN = tx.beginLSN + 1
	engine.mu.Unlock()

	payload, err := encodeRecordPutPayloadBinary(recordPutPayload{
		Collection: collection,
		ID:         id,
		Vector:     []float32{1, 0, 0},
		Metadata:   map[string]interface{}{"source": id},
	})
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	begin := newFrame(recordTypeTxBegin, tx.beginLSN, tx.txID, 0, emptyPayload())
	put := newFrame(recordTypeRecordPut, tx.putLSN, tx.txID, tx.beginLSN, payload)
	appendExternally(t, engine, []walRecord{begin, put})
	return tx
}

// commitExternalTx writes the commit frame of a transaction.
func commitExternalTx(t *testing.T, engine *Engine, tx externalTx) {
	t.Helper()
	engine.mu.Lock()
	commit := engine.makeTxCommitFrame(tx.putLSN+1, tx.txID, tx.putLSN)
	engine.mu.Unlock()
	appendExternally(t, engine, []walRecord{commit})
}

// newTestEngine makes an engine with one collection, ready for the tests.
func newTestEngine(t *testing.T, name string) *Engine {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	engineIface, err := New(path)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine := engineIface.(*Engine)
	t.Cleanup(func() { _ = engine.Close() })
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
	return engine
}

// refresh runs the scan the way a commit would.
func refresh(t *testing.T, engine *Engine) {
	t.Helper()
	engine.mu.Lock()
	defer engine.mu.Unlock()
	if err := engine.refreshFromDiskLocked(); err != nil {
		t.Fatalf("refresh: %v", err)
	}
}

// TestRefreshAppliesATransactionSplitAcrossTwoScans checks that a
// transaction whose frames are read before its commit frame is written is
// still applied.
//
// The scan can see a transaction's begin and its records while another
// process is still writing it. Those frames are not applied — a
// transaction is applied when its commit arrives — and they must not be
// skipped either. Pending frames survive the first scan, so the next scan
// can apply them together with the commit.
//
// Without retaining pending frames, the second scan finds only the commit and
// applies a transaction with no records in it, and the records are lost.
func TestRefreshAppliesATransactionSplitAcrossTwoScans(t *testing.T) {
	engine := newTestEngine(t, "split.libravdb")
	ctx := context.Background()
	collectionIface, err := engine.GetCollection("global")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}
	collection := collectionIface.(*Collection)

	tx := beginExternalTx(t, engine, "global", "split")

	// The first scan sees the begin and the record, and no commit.
	refresh(t, engine)

	if exists, err := collection.Exists(ctx, "split"); err != nil {
		t.Fatalf("exists: %v", err)
	} else if exists {
		t.Fatal("an uncommitted record became visible")
	}

	// Now the writer commits the transaction.
	commitExternalTx(t, engine, tx)

	// The second scan has to apply the whole transaction, not just its
	// commit frame.
	refresh(t, engine)

	exists, err := collection.Exists(ctx, "split")
	if err != nil {
		t.Fatalf("exists: %v", err)
	}
	if !exists {
		t.Fatal("the record of a transaction split across two scans was lost")
	}
}

// TestRefreshKeepsReadingAfterAnUncommittedTransaction checks that a
// transaction that has not committed does not stop the scan.
//
// The scan has to keep looking: a transaction that never commits must not
// hold up every commit that comes after it.
func TestRefreshKeepsReadingAfterAnUncommittedTransaction(t *testing.T) {
	engine := newTestEngine(t, "after.libravdb")
	ctx := context.Background()
	collectionIface, err := engine.GetCollection("global")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}
	collection := collectionIface.(*Collection)

	// A transaction that begins and never commits.
	engine.mu.Lock()
	openTxID := engine.lastTxID.Load() + 100
	openBeginLSN := engine.lastLSN.Load() + 1
	openBegin := newFrame(recordTypeTxBegin, openBeginLSN, openTxID, 0, emptyPayload())
	engine.mu.Unlock()
	appendExternally(t, engine, []walRecord{openBegin})

	// A second transaction, committed, after it.
	tx := beginExternalTx(t, engine, "global", "after")
	commitExternalTx(t, engine, tx)

	refresh(t, engine)

	exists, err := collection.Exists(ctx, "after")
	if err != nil {
		t.Fatalf("exists: %v", err)
	}
	if !exists {
		t.Fatal("a committed transaction after an open one was not applied")
	}
}
