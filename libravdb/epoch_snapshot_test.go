package libravdb

import (
	"context"
	"path/filepath"
	"testing"
)

// TestEpochCommitReleasesSnapshotPin checks that a committed epoch releases the
// temporal snapshot it pinned when it began. The pin keeps history from being
// compacted, so an epoch that leaks it blocks compaction for the life of the
// process.
func TestEpochCommitReleasesSnapshotPin(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(filepath.Join(t.TempDir(), "epoch_pin.libravdb")))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = db.Close() }()

	if _, err := db.CreateCollection(ctx, "items", WithMetadataOnly()); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	before, err := db.TemporalStats(ctx)
	if err != nil {
		t.Fatalf("TemporalStats: %v", err)
	}

	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}
	during, err := db.TemporalStats(ctx)
	if err != nil {
		t.Fatalf("TemporalStats: %v", err)
	}
	if during.ActiveLeaseCount != before.ActiveLeaseCount+1 {
		t.Fatalf("active leases during epoch = %d, want %d", during.ActiveLeaseCount, before.ActiveLeaseCount+1)
	}

	if err := epoch.Commit(ctx); err != nil {
		t.Fatalf("Commit: %v", err)
	}

	after, err := db.TemporalStats(ctx)
	if err != nil {
		t.Fatalf("TemporalStats: %v", err)
	}
	if after.ActiveLeaseCount != before.ActiveLeaseCount {
		t.Fatalf("active leases after commit = %d, want %d", after.ActiveLeaseCount, before.ActiveLeaseCount)
	}
}

// TestEpochRollbackReleasesSnapshotPin checks the same for a rolled-back
// epoch.
func TestEpochRollbackReleasesSnapshotPin(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(filepath.Join(t.TempDir(), "epoch_rollback_pin.libravdb")))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = db.Close() }()

	if _, err := db.CreateCollection(ctx, "items", WithMetadataOnly()); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	before, err := db.TemporalStats(ctx)
	if err != nil {
		t.Fatalf("TemporalStats: %v", err)
	}

	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}
	if err := epoch.Rollback(ctx); err != nil {
		t.Fatalf("Rollback: %v", err)
	}

	after, err := db.TemporalStats(ctx)
	if err != nil {
		t.Fatalf("TemporalStats: %v", err)
	}
	if after.ActiveLeaseCount != before.ActiveLeaseCount {
		t.Fatalf("active leases after rollback = %d, want %d", after.ActiveLeaseCount, before.ActiveLeaseCount)
	}
}
