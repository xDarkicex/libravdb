package singlefile

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"testing/quick"

	"github.com/xDarkicex/libravdb/internal/storage"
)

func feedEngine(t *testing.T) *Engine {
	t.Helper()
	e := coordinationEngine(t, filepath.Join(t.TempDir(), "feed.db"))
	if err := e.createCollection("rows", storage.CollectionConfig{Dimension: 1}); err != nil {
		t.Fatal(err)
	}
	return e
}
func feedWrite(t *testing.T, e *Engine, n int) {
	t.Helper()
	ops := make([]storage.TxOperation, n)
	for i := range ops {
		ops[i] = storage.TxOperation{Type: storage.TxOperationPut, Collection: "rows", ID: string(rune('a' + i)), Vector: []float32{1}}
	}
	if _, err := e.CommitTxDurable(context.Background(), ops); err != nil {
		t.Fatal(err)
	}
}
func feedCheckpoint(t *testing.T, e *Engine) string {
	t.Helper()
	cursor, err := e.ChangeCheckpoint(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return cursor
}
func feedRead(t *testing.T, e *Engine, cursor string, limit int) storage.ChangePage {
	t.Helper()
	page, err := e.ReadChanges(context.Background(), cursor, limit)
	if err != nil {
		t.Fatal(err)
	}
	return page
}
func TestChangesPaginationAndRecovery(t *testing.T) {
	e := feedEngine(t)
	cursor := feedCheckpoint(t, e)
	feedWrite(t, e, 3)
	first := feedRead(t, e, cursor, 2)
	if len(first.Rows.Operations) != 2 || !first.More || first.TransactionComplete {
		t.Fatalf("first page: %+v", first)
	}
	second := feedRead(t, e, first.Next, 2)
	if len(second.Rows.Operations) != 1 || second.More || !second.TransactionComplete {
		t.Fatalf("second page: %+v", second)
	}
	if first.Rows.CommitLSNs[0] != second.Rows.CommitLSNs[0] || second.Rows.Positions[0] != 2 {
		t.Fatal("lost transaction position")
	}
	checkFeedRecovery(t, e, cursor)
}
func checkFeedRecovery(t *testing.T, e *Engine, cursor string) {
	t.Helper()
	before := feedRead(t, e, cursor, 10)
	if err := e.Compact(); err != nil {
		t.Fatal(err)
	}
	path := e.path
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	reopened := coordinationEngine(t, path)
	after := feedRead(t, reopened, cursor, 10)
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("feed changed across compaction/reopen:\n%+v\n%+v", before, after)
	}
}
func TestChangesRejectOtherHistoryAndInvalidOffsets(t *testing.T) {
	e := feedEngine(t)
	feedWrite(t, e, 2)
	cursor := feedCheckpoint(t, e)
	other := feedEngine(t)
	_, err := other.ReadChanges(context.Background(), cursor, 1)
	if !errors.Is(err, storage.ErrChangeCursor) {
		t.Fatalf("other store: %v", err)
	}
	p, err := decodeChangePosition(cursor)
	if err != nil {
		t.Fatal(err)
	}
	p.Nonce[0] ^= 1
	_, err = e.ReadChanges(context.Background(), encodeChangePosition(p), 1)
	if !errors.Is(err, storage.ErrChangeCursor) {
		t.Fatalf("divergent history: %v", err)
	}
	p, _ = decodeChangePosition(cursor)
	p.Offset = ^uint32(0)
	_, err = e.ReadChanges(context.Background(), encodeChangePosition(p), 1)
	if !errors.Is(err, storage.ErrChangeCursor) {
		t.Fatalf("offset: %v", err)
	}
}
func TestChangesFailedTransactionsInvisible(t *testing.T) {
	e := feedEngine(t)
	cursor := feedCheckpoint(t, e)
	_, err := e.CommitTxDurable(context.Background(), []storage.TxOperation{{Type: storage.TxOperationPut, Collection: "missing", ID: "x"}})
	if err == nil {
		t.Fatal("invalid transaction succeeded")
	}
	page := feedRead(t, e, cursor, 10)
	if len(page.Rows.Operations) != 0 || page.Next != cursor {
		t.Fatalf("failed write appeared: %+v", page)
	}
}
func TestChangeCursorRoundTrip(t *testing.T) {
	property := func(file, start, lsn uint64, offset uint32, nonce [16]byte) bool {
		p := changePosition{File: file, Start: start, LSN: lsn, Offset: offset, Nonce: nonce}
		got, err := decodeChangePosition(encodeChangePosition(p))
		return err == nil && got == p
	}
	if err := quick.Check(property, nil); err != nil {
		t.Fatal(err)
	}
}
func TestChangesPaginationMatchesWholeFeed(t *testing.T) {
	e := feedEngine(t)
	cursor := feedCheckpoint(t, e)
	feedWrite(t, e, 7)
	whole := feedRead(t, e, cursor, 100)
	for limit := 1; limit <= 8; limit++ {
		checkFeedPages(t, e, cursor, whole, limit)
	}
}
func checkFeedPages(t *testing.T, e *Engine, cursor string, whole storage.ChangePage, limit int) {
	t.Helper()
	var ids []string
	for steps := 0; steps < 20; steps++ {
		page := feedRead(t, e, cursor, limit)
		ids = append(ids, page.Rows.RecordIDs...)
		cursor = page.Next
		if !page.More {
			if !reflect.DeepEqual(ids, whole.Rows.RecordIDs) {
				t.Fatal("pagination omitted/repeated records")
			}
			return
		}
	}
	t.Fatal("pagination did not terminate")
}
func BenchmarkReadChangesTail(b *testing.B) {
	raw, err := New(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	e := raw.(*Engine)
	b.Cleanup(func() {
		if err := e.Close(); err != nil {
			b.Error(err)
		}
	})
	if err := e.createCollection("rows", storage.CollectionConfig{Dimension: 1}); err != nil {
		b.Fatal(err)
	}
	seedChangeBenchmark(b, e)
	cursor, err := e.ChangeCheckpoint(context.Background())
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		page, err := e.ReadChanges(context.Background(), cursor, 20)
		if err != nil {
			b.Fatal(err)
		}
		if page.More {
			b.Fatal("unexpected changes")
		}
	}
}

func seedChangeBenchmark(b *testing.B, e *Engine) {
	b.Helper()
	for i := 0; i < 1000; i++ {
		if _, err := e.CommitTxDurable(context.Background(), []storage.TxOperation{{Type: storage.TxOperationPut, Collection: "rows", ID: "a", Vector: []float32{1}}}); err != nil {
			b.Fatal(err)
		}
	}
}

func TestChangesRefreshOtherWriter(t *testing.T) {
	e := feedEngine(t)
	cursor := feedCheckpoint(t, e)
	other := coordinationEngine(t, e.path)
	feedWrite(t, other, 2)
	page := feedRead(t, e, cursor, 10)
	if len(page.Rows.RecordIDs) != 2 {
		t.Fatalf("remote commits absent: %+v", page)
	}
	if page.Rows.BeforeLSNs[0] == 0 {
		t.Fatal("missing before-transaction reference")
	}
}
func TestChangesReplayUncheckpointedWAL(t *testing.T) {
	e := feedEngine(t)
	e.mu.Lock()
	err := e.checkpointLocked()
	e.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	cursor := feedCheckpoint(t, e)
	feedWrite(t, e, 3)
	expected := feedRead(t, e, cursor, 10)
	copied := copyChangeFile(t, e.path)
	reopened := coordinationEngine(t, copied)
	actual := feedRead(t, reopened, cursor, 10)
	if !reflect.DeepEqual(expected, actual) {
		t.Fatalf("WAL replay changed feed:\n%+v\n%+v", expected, actual)
	}
}
func copyChangeFile(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	copied := filepath.Join(t.TempDir(), "copy.db")
	if err := os.WriteFile(copied, data, 0600); err != nil {
		t.Fatal(err)
	}
	return copied
}
func TestChangesSyncFailureIsNotPublished(t *testing.T) {
	e := feedEngine(t)
	cursor := feedCheckpoint(t, e)
	e.walSyncFn = func(*os.File) error { return errors.New("sync probe") }
	_, err := e.CommitTxDurable(context.Background(), []storage.TxOperation{{Type: storage.TxOperationPut, Collection: "rows", ID: "ambiguous", Vector: []float32{1}}})
	if err == nil {
		t.Fatal("sync failure succeeded")
	}
	// This handle must not publish an ambiguous transaction. A reopened handle
	// can legitimately recover its complete commit marker from the copied WAL.
	e.mu.RLock()
	tail := encodeChangePosition(e.changeTail())
	e.mu.RUnlock()
	if tail != cursor {
		t.Fatal("ambiguous write published to live feed")
	}
	copied := copyChangeFile(t, e.path)
	e.walSyncFn = nil
	recovered := coordinationEngine(t, copied)
	page := feedRead(t, recovered, cursor, 10)
	if len(page.Rows.RecordIDs) != 1 || page.Rows.RecordIDs[0] != "ambiguous" {
		t.Fatalf("recovery did not enumerate durable marker: %+v", page)
	}
}

func TestChangesLegacyBoundaryIsExplicit(t *testing.T) {
	e := feedEngine(t)
	e.mu.Lock()
	frame := newFrame(recordTypeTxCommit, e.lastLSN.Load()+1, 1, 0, encodeTxCommitPayload(txCommitPayload{Timestamp: 1}))
	err := e.replayChanges([]walRecord{frame})
	releaseWALFramePayloads([]walRecord{frame})
	e.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	page := feedRead(t, e, "", 1)
	if page.StartLSN != frame.Header.LSN || len(page.Rows.Operations) != 0 {
		t.Fatalf("legacy history falsely enumerated: %+v", page)
	}
}
func TestChangesSnapshotCodecRejectsTruncatedFeed(t *testing.T) {
	e := feedEngine(t)
	feedWrite(t, e, 2)
	e.mu.RLock()
	data, err := encodeStateBinary(e.state)
	e.mu.RUnlock()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := decodeStateBinary(data[:len(data)-1]); err == nil {
		t.Fatal("truncated feed accepted")
	}
}
func BenchmarkReadChangesPage(b *testing.B) {
	raw, err := New(filepath.Join(b.TempDir(), "page.db"))
	if err != nil {
		b.Fatal(err)
	}
	e := raw.(*Engine)
	b.Cleanup(func() {
		if err := e.Close(); err != nil {
			b.Error(err)
		}
	})
	if err := e.createCollection("rows", storage.CollectionConfig{Dimension: 1}); err != nil {
		b.Fatal(err)
	}
	cursor, err := e.ChangeCheckpoint(context.Background())
	if err != nil {
		b.Fatal(err)
	}
	seedChangeBenchmark(b, e)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		page, err := e.ReadChanges(context.Background(), cursor, 20)
		if err != nil {
			b.Fatal(err)
		}
		if len(page.Rows.Operations) != 20 {
			b.Fatal("wrong page size")
		}
	}
}
