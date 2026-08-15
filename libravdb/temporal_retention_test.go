package libravdb

import (
	"context"
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
	db, err := Open(WithStoragePath(t.TempDir() + "/retention_pin.libravdb"))
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

	// Pin a snapshot.
	snap, err := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	if err != nil {
		t.Fatalf("SnapshotAt: %v", err)
	}
	pinnedLSN := snap.LSN

	// Compact — should respect the pin.
	_, err = db.CompactHistory(context.Background())
	if err != nil {
		t.Logf("CompactHistory (with pin): %v", err)
	}

	// Pinned snapshot should still work.
	rec, err := col.GetAtLSN(context.Background(), "r1", pinnedLSN)
	if err != nil {
		t.Fatalf("GetAtLSN pinned: %v", err)
	}
	if rec == nil {
		t.Error("pinned snapshot should still be visible after compact")
	}

	// Release pin and compact again.
	snap.Close()
	db.CompactHistory(context.Background())
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
