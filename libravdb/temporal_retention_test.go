package libravdb

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestTemporalRetention_CompactRecordHistory prunes old record versions
// and verifies retained snapshots are still correct.
func TestTemporalRetention_CompactRecordHistory(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/retention_records.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// T1: Insert V1=[1,0,0].
	col.Insert(context.Background(), "r1", []float32{1, 0, 0}, nil)
	time.Sleep(20 * time.Millisecond)
	snap1, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	lsn1 := snap1.LSN
	snap1.Close()

	// T2: Update to V2=[2,0,0].
	col.Update(context.Background(), "r1", []float32{2, 0, 0}, nil)
	time.Sleep(20 * time.Millisecond)
	snap2, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	lsn2 := snap2.LSN
	snap2.Close()

	// Before compaction: both versions visible.
	rec1, _ := col.GetAtLSN(context.Background(), "r1", lsn1)
	if rec1 == nil || rec1.Vector[0] != 1 {
		t.Fatal("pre-compact T1: should see V1")
	}
	rec2, _ := col.GetAtLSN(context.Background(), "r1", lsn2)
	if rec2 == nil || rec2.Vector[0] != 2 {
		t.Fatal("pre-compact T2: should see V2")
	}

	// Compact up to lsn1 (keep T2 and later).
	newBoundary, err := db.CompactHistory(context.Background())
	if err != nil {
		t.Fatalf("CompactHistory: %v", err)
	}
	t.Logf("newBoundary=%d", newBoundary)

	// After compaction: T2 still works (retained), T1 may be expired.
	rec2b, _ := col.GetAtLSN(context.Background(), "r1", lsn2)
	if rec2b == nil || rec2b.Vector[0] != 2 {
		t.Error("post-compact T2: V2 should still be visible")
	}
}

// TestTemporalRetention_ExpiredSnapshot verifies ErrRetentionExpired.
func TestTemporalRetention_ExpiredSnapshot(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/retention_expired.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	col.Insert(context.Background(), "r1", []float32{1, 0, 0}, nil)
	time.Sleep(20 * time.Millisecond)

	// Query at a time before the first commit.
	_, err = db.SnapshotAt(context.Background(), time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC))
	if err == nil {
		t.Error("should get retention error for ancient timestamp")
	}
}

// TestTemporalRetention_SnapshotPin prevents compaction past active pins.
func TestTemporalRetention_SnapshotPin(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir()+"/retention_pin.libravdb"), WithTemporalRetention(10*time.Millisecond))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	col.Insert(context.Background(), "r1", []float32{1, 0, 0}, nil)
	time.Sleep(20 * time.Millisecond)

	// Pin a snapshot at T1.
	snap, err := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	if err != nil {
		t.Fatalf("SnapshotAt: %v", err)
	}
	pinnedLSN := snap.LSN

	// Update record so there is a newer version.
	time.Sleep(20 * time.Millisecond)
	col.Update(context.Background(), "r1", []float32{2, 0, 0}, nil)
	time.Sleep(20 * time.Millisecond)

	// Compact — should respect the pin and NOT compact past pinnedLSN.
	boundary, err := db.CompactHistory(context.Background())
	if err != nil {
		t.Fatalf("CompactHistory (with pin): %v", err)
	}
	if boundary > pinnedLSN {
		t.Fatalf("CompactHistory advanced boundary past pin: boundary=%d pinned=%d", boundary, pinnedLSN)
	}

	// Pinned snapshot should still work.
	rec, err := col.GetAtLSN(context.Background(), "r1", pinnedLSN)
	if err != nil {
		t.Fatalf("GetAtLSN pinned: %v", err)
	}
	if rec == nil || rec.Vector[0] != 1 {
		t.Errorf("pinned snapshot should still see V1 after compact, got %v", rec)
	}

	// Release pin and compact again — now boundary can advance.
	snap.Close()
	time.Sleep(20 * time.Millisecond)
	newBoundary, err := db.CompactHistory(context.Background())
	if err != nil {
		t.Fatalf("CompactHistory (after unpin): %v", err)
	}
	if newBoundary < pinnedLSN {
		t.Fatalf("CompactHistory after unpin did not advance past pin: newBoundary=%d pinned=%d", newBoundary, pinnedLSN)
	}
}

// TestTemporalRetention_ReopenAfterCompact verifies history survives restart.
func TestTemporalRetention_ReopenAfterCompact(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/retention_reopen.libravdb"

	db, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	col, _ := db.CreateCollection(context.Background(), "c", WithDimension(3))
	col.Insert(context.Background(), "r1", []float32{1, 0, 0}, nil)
	time.Sleep(20 * time.Millisecond)
	col.Update(context.Background(), "r1", []float32{2, 0, 0}, nil)
	time.Sleep(20 * time.Millisecond)
	snap, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	lsn := snap.LSN
	snap.Close()

	// Compact and close.
	db.CompactHistory(context.Background())
	db.Close()

	// Reopen and verify.
	db2, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatalf("Reopen: %v", err)
	}
	defer db2.Drop(context.Background())
	col2, _ := db2.GetCollection("c")
	rec, err := col2.GetAtLSN(context.Background(), "r1", lsn)
	if err != nil {
		t.Fatalf("GetAtLSN after reopen: %v", err)
	}
	if rec == nil || rec.Vector[0] != 2 {
		t.Error("V2 should survive compact+reopen")
	}
}

// TestTemporalRetention_DefaultConservative verifies no pruning by default.
func TestTemporalRetention_DefaultConservative(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/retention_default.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	// Default config: no retention pruning.
	if db.config.Temporal.RetainDuration != 0 {
		t.Error("default RetainDuration should be 0 (no pruning)")
	}
}

// TestTemporalRetention_ConcurrentCompactAndSnapshot verifies that concurrent
// compactions never prune history out from under actively pinned snapshots,
// and that any snapshot handle returned to a caller reads successfully.
func TestTemporalRetention_ConcurrentCompactAndSnapshot(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/retention_concurrent.libravdb"

	// Short retention so commits quickly become eligible for compaction.
	retention := 10 * time.Millisecond
	db, err := Open(WithStoragePath(path), WithTemporalRetention(retention))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "concurrent_test", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Insert and update across time so there are multiple historical versions and commit LSNs.
	for i := 0; i < 30; i++ {
		col.Insert(context.Background(), fmt.Sprintf("r_%d", i), []float32{float32(i), 0, 0}, nil)
		if i%5 == 0 {
			time.Sleep(2 * time.Millisecond)
		}
	}
	for i := 0; i < 30; i++ {
		col.Update(context.Background(), fmt.Sprintf("r_%d", i), []float32{float32(i * 10), 0, 0}, nil)
		if i%5 == 0 {
			time.Sleep(2 * time.Millisecond)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	var readErrors atomic.Uint64
	var expiredCount atomic.Uint64
	var successfulReads atomic.Uint64

	// Compactor goroutines: repeatedly run CompactHistory
	for c := 0; c < 2; c++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ctx.Err() == nil {
				_, _ = db.CompactHistory(ctx)
				time.Sleep(1 * time.Millisecond)
			}
		}()
	}

	// Reader goroutines: repeatedly attempt SnapshotAt
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for ctx.Err() == nil {
				target := time.Now().UTC().Add(-time.Duration(workerID*5) * time.Millisecond)
				snap, err := db.SnapshotAt(ctx, target)
				if err != nil {
					if errors.Is(err, ErrRetentionExpired) {
						expiredCount.Add(1)
						continue
					}
					if ctx.Err() != nil {
						return
					}
					readErrors.Add(1)
					t.Errorf("unexpected SnapshotAt error: %v", err)
					return
				}

				// The snapshot was pinned and returned. A pinned snapshot must NEVER fail
				// with ErrRetentionExpired when reading records visible at that snapshot.
				rec, err := col.GetAtLSN(ctx, "r_0", snap.LSN)
				if err != nil {
					if errors.Is(err, ErrRetentionExpired) {
						readErrors.Add(1)
						t.Errorf("CRITICAL BUG: pinned snapshot LSN %d failed with ErrRetentionExpired!", snap.LSN)
					} else {
						readErrors.Add(1)
						t.Errorf("GetAtLSN error: %v", err)
					}
				} else if rec != nil {
					successfulReads.Add(1)
				}
				snap.Close()
			}
		}(r)
	}

	wg.Wait()

	if errCount := readErrors.Load(); errCount > 0 {
		t.Fatalf("Concurrent compaction test failed with %d read errors", errCount)
	}
	t.Logf("Concurrent test passed: %d successful reads, %d clean expirations",
		successfulReads.Load(), expiredCount.Load())

	// Verify compaction works after all snapshots are closed (no stale pins).
	time.Sleep(20 * time.Millisecond)
	finalBoundary, err := db.CompactHistory(context.Background())
	if err != nil {
		t.Fatalf("final CompactHistory failed: %v", err)
	}
	if finalBoundary == 0 {
		t.Fatal("final CompactHistory should have advanced boundary, but was 0")
	}
}
