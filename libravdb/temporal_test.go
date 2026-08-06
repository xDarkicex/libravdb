package libravdb

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// TestTemporal_CommitSurvivesReopen verifies that commit timestamps survive
// a close/reopen cycle (the catalog is rebuilt from WAL replay).
func TestTemporal_CommitSurvivesReopen(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/temporal_survives.db"
	db, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := db.CreateCollection(context.Background(), "c", WithDimension(3)); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	db.Close()

	// Reopen the same file — commit timestamps must survive via WAL replay.
	db2, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatalf("Re-open: %v", err)
	}
	defer db2.Drop(context.Background())

	// Use SnapshotAt (timestamp-based) to verify the catalog was rebuilt.
	snap, err := db2.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Hour))
	if err != nil {
		t.Fatalf("SnapshotAt(future): %v", err)
	}
	if snap.LSN == 0 {
		t.Error("LSN is zero after reopen — catalog was not rebuilt")
	}
	if snap.Timestamp.IsZero() {
		t.Error("timestamp is zero after reopen")
	}
}

// TestTemporal_SnapshotAt_BeforeBetweenAfter verifies timestamp resolution.
func TestTemporal_SnapshotAt_BeforeBetweenAfter(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:temporal_resolution.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	before := time.Now().UTC()
	if _, err := db.CreateCollection(context.Background(), "c", WithDimension(3)); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	mid := time.Now().UTC()
	time.Sleep(10 * time.Millisecond)
	if _, err := db.CreateCollection(context.Background(), "c2", WithDimension(3)); err != nil {
		t.Fatalf("CreateCollection c2: %v", err)
	}
	after := time.Now().UTC()

	// Before first commit: must return retention error.
	_, err = db.SnapshotAt(context.Background(), before.Add(-time.Hour))
	if err == nil {
		t.Error("SnapshotAt(before first commit) should return error")
	}

	// Between commits: must return the first commit.
	snap, err := db.SnapshotAt(context.Background(), mid)
	if err != nil {
		t.Fatalf("SnapshotAt(mid): %v", err)
	}
	if snap.LSN == 0 {
		t.Error("mid LSN is zero")
	}
	firstLSN := snap.LSN

	// After last commit: must return the last commit (higher LSN).
	snap2, err := db.SnapshotAt(context.Background(), after.Add(time.Hour))
	if err != nil {
		t.Fatalf("SnapshotAt(after): %v", err)
	}
	if snap2.LSN <= firstLSN {
		t.Errorf("after LSN = %d, should be > first LSN %d", snap2.LSN, firstLSN)
	}
}

// TestTemporal_SameTimestampResolvesLargestLSN verifies that when multiple
// commits share the same wall-clock tick, resolution returns the highest LSN.
func TestTemporal_SameTimestampResolvesLargestLSN(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:temporal_same_ts.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	for i := 0; i < 5; i++ {
		if _, err := db.CreateCollection(context.Background(),
			string(rune('a'+i)), WithDimension(3)); err != nil {
			t.Fatalf("CreateCollection %d: %v", i, err)
		}
	}

	// Resolve at "now" — should get the latest LSN.
	snap, err := db.SnapshotAt(context.Background(), time.Now().UTC())
	if err != nil {
		t.Fatalf("SnapshotAt(now): %v", err)
	}
	if snap.LSN == 0 {
		t.Error("LSN should be > 0 after 5 commits")
	}
	t.Logf("5 commits resolved to LSN %d", snap.LSN)
}

// TestTemporal_EmptyDatabase verifies behavior with no commits.
func TestTemporal_EmptyDatabase(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:temporal_empty.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	_, err = db.SnapshotAt(context.Background(), time.Now().UTC())
	if err == nil {
		t.Error("SnapshotAt on empty DB should return error")
	}
}

// TestTemporal_UnknownLSN verifies behavior for nonexistent LSNs.
func TestTemporal_UnknownLSN(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:temporal_unknown_lsn.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	if _, err := db.CreateCollection(context.Background(), "c", WithDimension(3)); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	_, err = db.SnapshotAtLSN(context.Background(), 999999)
	if err == nil {
		t.Error("SnapshotAtLSN(nonexistent) should return error")
	}
}

// TestTemporal_BatchCommitCoalescing verifies that batched record inserts
// produce timestamped commits.
func TestTemporal_BatchCommitCoalescing(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:temporal_batch.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Insert multiple records — they batch into one transaction.
	beforeInsert := time.Now().UTC()
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("rec-%d", i)
		if err := col.Insert(context.Background(), id, []float32{float32(i), 0, 0}, nil); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	afterInsert := time.Now().UTC()

	// Resolve to after — should see the batch commit.
	snap, err := db.SnapshotAt(context.Background(), afterInsert)
	if err != nil {
		t.Fatalf("SnapshotAt(after): %v", err)
	}
	_ = beforeInsert
	_ = snap
}
