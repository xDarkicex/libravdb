package singlefile

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"testing"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
)

func coordinationEngine(t *testing.T, path string) *Engine {
	t.Helper()
	raw, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	e := raw.(*Engine)
	t.Cleanup(func() { _ = e.Close() })
	return e
}

func TestOrdinalReservationsAreSharedBeforeInsert(t *testing.T) {
	path := filepath.Join(t.TempDir(), "reserved.db")
	a := coordinationEngine(t, path)
	if err := a.createCollection("rows", storage.CollectionConfig{Dimension: 1}); err != nil {
		t.Fatal(err)
	}
	b := coordinationEngine(t, path)
	ca, _ := a.GetCollection("rows")
	cb, _ := b.GetCollection("rows")
	first := &index.VectorEntry{ID: "a", Vector: []float32{1}}
	second := &index.VectorEntry{ID: "b", Vector: []float32{2}}
	ctx := context.Background()
	if err := ca.AssignOrdinals(ctx, []*index.VectorEntry{first}); err != nil {
		t.Fatal(err)
	}
	if err := cb.AssignOrdinals(ctx, []*index.VectorEntry{second}); err != nil {
		t.Fatal(err)
	}
	if first.Ordinal == second.Ordinal {
		t.Fatalf("two engines reserved ordinal %d", first.Ordinal)
	}
	if err := cb.Insert(ctx, second); err != nil {
		t.Fatal(err)
	}
	if err := ca.Insert(ctx, first); err != nil {
		t.Fatal(err)
	}
	if err := a.refreshIfBehind(); err != nil {
		t.Fatal(err)
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	col := a.state.Collections["rows"]
	if len(col.Records) != 2 || col.ordinalToID[first.Ordinal] != "a" || col.ordinalToID[second.Ordinal] != "b" {
		t.Fatalf("records lost from ordinal slots: %v", col.ordinalToID)
	}
}

func TestCompactionLocksBothInodesAcrossReplacement(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compact.db")
	e := coordinationEngine(t, path)
	original := replaceDatabaseFile
	t.Cleanup(func() { replaceDatabaseFile = original })
	replaceDatabaseFile = func(src, dst string) error {
		for _, path := range []string{src, dst} {
			observer, err := os.OpenFile(path, os.O_RDWR, 0)
			if err != nil {
				return err
			}
			err = syscall.Flock(int(observer.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
			observer.Close()
			if err == nil {
				t.Errorf("inode %s was not locked during replacement", path)
			}
		}
		return original(src, dst)
	}
	if err := e.Compact(); err != nil {
		t.Fatal(err)
	}
}

func TestConcurrentOrdinalReservationsWithinEngine(t *testing.T) {
	e := coordinationEngine(t, filepath.Join(t.TempDir(), "goroutines.db"))
	if err := e.createCollection("rows", storage.CollectionConfig{Dimension: 1}); err != nil {
		t.Fatal(err)
	}
	col, _ := e.GetCollection("rows")
	var wg sync.WaitGroup
	ordinals := make(chan uint32, 32)
	for i := range 32 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			entry := &index.VectorEntry{ID: fmt.Sprint(i), Vector: []float32{1}}
			if err := col.AssignOrdinals(context.Background(), []*index.VectorEntry{entry}); err != nil {
				t.Error(err)
				return
			}
			ordinals <- entry.Ordinal
		}(i)
	}
	wg.Wait()
	close(ordinals)
	seen := map[uint32]bool{}
	for ordinal := range ordinals {
		if seen[ordinal] {
			t.Errorf("duplicate ordinal %d", ordinal)
		}
		seen[ordinal] = true
	}
	if len(seen) != 32 {
		t.Fatalf("reserved %d distinct ordinals", len(seen))
	}
}

func TestRefreshUpdatesCheckpointMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), "checkpoint.db")
	a := coordinationEngine(t, path)
	if err := a.createCollection("rows", storage.CollectionConfig{Dimension: 1}); err != nil {
		t.Fatal(err)
	}
	b := coordinationEngine(t, path)
	col, _ := b.GetCollection("rows")
	if err := col.Insert(context.Background(), &index.VectorEntry{ID: "remote", Vector: []float32{1}}); err != nil {
		t.Fatal(err)
	}
	b.mu.Lock()
	err := b.checkpointLocked()
	wantPage, wantEpoch := b.activeMetaPage, b.metaEpoch
	b.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if err := a.refreshIfBehind(); err != nil {
		t.Fatal(err)
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.activeMetaPage != wantPage || a.metaEpoch != wantEpoch {
		t.Fatalf("checkpoint metadata = (%d, %d), want (%d, %d)", a.activeMetaPage, a.metaEpoch, wantPage, wantEpoch)
	}
	if a.state.Collections["rows"].Records["remote"] == nil {
		t.Fatal("remote checkpoint hid committed WAL")
	}
}

func TestOrdinalReservationRecoveredBeforeRecordExists(t *testing.T) {
	path := filepath.Join(t.TempDir(), "recovery.db")
	a := coordinationEngine(t, path)
	if err := a.createCollection("rows", storage.CollectionConfig{Dimension: 1}); err != nil {
		t.Fatal(err)
	}
	ca, _ := a.GetCollection("rows")
	reserved := &index.VectorEntry{ID: "pending"}
	if err := ca.AssignOrdinals(context.Background(), []*index.VectorEntry{reserved}); err != nil {
		t.Fatal(err)
	}
	// Open from the on-disk snapshot and WAL, without checkpointing a first.
	b := coordinationEngine(t, path)
	cb, _ := b.GetCollection("rows")
	next := &index.VectorEntry{ID: "next"}
	if err := cb.AssignOrdinals(context.Background(), []*index.VectorEntry{next}); err != nil {
		t.Fatal(err)
	}
	if next.Ordinal <= reserved.Ordinal {
		t.Fatalf("recovery reused reserved ordinal %d", next.Ordinal)
	}
}

func TestStorageInsertReassignsRemoteOrdinalCollision(t *testing.T) {
	path := filepath.Join(t.TempDir(), "storage.db")
	a := coordinationEngine(t, path)
	if err := a.createCollection("rows", storage.CollectionConfig{Dimension: 1}); err != nil {
		t.Fatal(err)
	}
	b := coordinationEngine(t, path)
	ca, _ := a.GetCollection("rows")
	cb, _ := b.GetCollection("rows")
	first := &index.VectorEntry{ID: "a", Vector: []float32{1}}
	second := &index.VectorEntry{ID: "b", Vector: []float32{2}}
	// Private storage reservations are speculative and may initially collide.
	if err := ca.(*Collection).assignOrdinal(first); err != nil {
		t.Fatal(err)
	}
	if err := cb.(*Collection).assignOrdinal(second); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if err := ca.Insert(ctx, first); err != nil {
		t.Fatal(err)
	}
	if err := cb.Insert(ctx, second); err != nil {
		t.Fatal(err)
	}
	if first.Ordinal == second.Ordinal {
		t.Fatalf("WAL append kept colliding ordinal %d", first.Ordinal)
	}
}

func TestRefreshRetainsPendingFramesAcrossOtherCommit(t *testing.T) {
	e := newTestEngine(t, "interleaved.db")
	tx := beginExternalTx(t, e, "global", "pending")
	refresh(t, e)
	// Write a different committed transaction with higher LSNs, while tx is open.
	e.mu.Lock()
	beginLSN := tx.putLSN + 1
	otherID := tx.txID + 1
	begin := newFrame(recordTypeTxBegin, beginLSN, otherID, 0, emptyPayload())
	commit := e.makeTxCommitFrame(beginLSN+1, otherID, beginLSN)
	e.mu.Unlock()
	appendExternally(t, e, []walRecord{begin, commit})
	refresh(t, e)
	e.mu.Lock()
	lateCommit := e.makeTxCommitFrame(beginLSN+2, tx.txID, tx.putLSN)
	e.mu.Unlock()
	appendExternally(t, e, []walRecord{lateCommit})
	refresh(t, e)
	col, _ := e.GetCollection("global")
	exists, err := col.Exists(context.Background(), "pending")
	if err != nil || !exists {
		t.Fatalf("pending record lost after another commit: exists=%v err=%v", exists, err)
	}
}

func TestGraphIDReservationsAreSharedAndRecovered(t *testing.T) {
	path := filepath.Join(t.TempDir(), "graph-reservations.db")
	a := coordinationEngine(t, path)
	b := coordinationEngine(t, path)
	ctx := context.Background()
	first, err := a.ReserveGraphNodeIDs(ctx, 10)
	if err != nil {
		t.Fatal(err)
	}
	second, err := b.ReserveGraphNodeIDs(ctx, 5)
	if err != nil {
		t.Fatal(err)
	}
	if second < first+10 {
		t.Fatalf("overlapping graph ID ranges: first=%d second=%d", first, second)
	}
	c := coordinationEngine(t, path)
	third, err := c.ReserveGraphNodeIDs(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	if third < second+5 {
		t.Fatalf("recovery reused graph IDs: second=%d third=%d", second, third)
	}
}

func TestDatabaseCreatesNoCoordinationFiles(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "single.db")
	e := coordinationEngine(t, path)
	if err := e.createCollection("rows", storage.CollectionConfig{Dimension: 1}); err != nil {
		t.Fatal(err)
	}
	col, _ := e.GetCollection("rows")
	if err := col.Insert(context.Background(), &index.VectorEntry{ID: "one", Vector: []float32{1}}); err != nil {
		t.Fatal(err)
	}
	if err := e.Vacuum(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "single.db" {
		t.Fatalf("single-file database created extra files: %v", entries)
	}
}

func TestConsumerLockFileDoesNotBlockDatabase(t *testing.T) {
	path := filepath.Join(t.TempDir(), "consumer.db")
	consumer, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		t.Fatal(err)
	}
	defer consumer.Close()
	if err := syscall.Flock(int(consumer.Fd()), syscall.LOCK_EX); err != nil {
		t.Fatal(err)
	}
	e := coordinationEngine(t, path)
	if err := e.createCollection("rows", storage.CollectionConfig{Dimension: 1}); err != nil {
		t.Fatal(err)
	}
	col, _ := e.GetCollection("rows")
	if err := col.Insert(context.Background(), &index.VectorEntry{ID: "one", Vector: []float32{1}}); err != nil {
		t.Fatal(err)
	}
}

func TestOpenEngineFollowsCompactionReplacement(t *testing.T) {
	path := filepath.Join(t.TempDir(), "replacement.db")
	a := coordinationEngine(t, path)
	if err := a.createCollection("rows", storage.CollectionConfig{Dimension: 1}); err != nil {
		t.Fatal(err)
	}
	b := coordinationEngine(t, path)
	ca, _ := a.GetCollection("rows")
	cb, _ := b.GetCollection("rows")
	ctx := context.Background()
	// b is behind when a replaces the inode. Recovery must load the new snapshot,
	// including records that no longer have record-put WAL in the compacted file.
	if err := ca.Insert(ctx, &index.VectorEntry{ID: "before", Vector: []float32{1}}); err != nil {
		t.Fatal(err)
	}
	if err := a.Compact(); err != nil {
		t.Fatal(err)
	}
	if err := cb.Insert(ctx, &index.VectorEntry{ID: "after", Vector: []float32{2}}); err != nil {
		t.Fatal(err)
	}
	if err := b.Compact(); err != nil {
		t.Fatal(err)
	}
	if err := a.refreshIfBehind(); err != nil {
		t.Fatal(err)
	}
	for _, col := range []storage.Collection{ca, cb} {
		for _, id := range []string{"before", "after"} {
			exists, err := col.Exists(ctx, id)
			if err != nil || !exists {
				t.Fatalf("lost %s after replacement: exists=%v err=%v", id, exists, err)
			}
		}
	}
}
