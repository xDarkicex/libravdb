package libravdb

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestTemporalStatsZeroHistory(t *testing.T) {
	ctx := context.Background()
	retention := 7 * time.Second
	db, err := Open(WithStoragePath(":memory:temporal-stats-empty"), WithTemporalRetention(retention))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	stats, err := db.TemporalStats(ctx)
	if err != nil {
		t.Fatalf("TemporalStats: %v", err)
	}
	if stats.ActiveLeaseCount != 0 || stats.DistinctPinnedLSNCount != 0 ||
		stats.MinimumPinnedLSN != 0 || stats.OldestPinAge != 0 ||
		stats.OldestRetainedLSN != 0 || !stats.OldestRetainedAt.IsZero() ||
		stats.LastCompactionBoundary != 0 {
		t.Fatalf("zero-history stats = %#v", stats)
	}
	if stats.ConfiguredRetention != retention {
		t.Fatalf("ConfiguredRetention = %v, want %v", stats.ConfiguredRetention, retention)
	}
}

func TestTemporalStatsOneLSNPinnedTwice(t *testing.T) {
	ctx := context.Background()
	db, lsn := temporalStatsDatabase(t, ":memory:temporal-stats-one")
	defer db.Close()

	first, err := db.SnapshotAtLSN(ctx, lsn)
	if err != nil {
		t.Fatalf("first SnapshotAtLSN: %v", err)
	}
	time.Sleep(2 * time.Millisecond)
	second, err := db.SnapshotAtLSN(ctx, lsn)
	if err != nil {
		t.Fatalf("second SnapshotAtLSN: %v", err)
	}

	stats, err := db.TemporalStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.ActiveLeaseCount != 2 || stats.DistinctPinnedLSNCount != 1 || stats.MinimumPinnedLSN != lsn {
		t.Fatalf("pinned-twice stats = %#v", stats)
	}
	if stats.OldestPinAge < 2*time.Millisecond {
		t.Fatalf("OldestPinAge = %v, want at least 2ms", stats.OldestPinAge)
	}
	ageBeforeRepin := stats.OldestPinAge

	first.Close()
	stats, err = db.TemporalStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.ActiveLeaseCount != 1 || stats.DistinctPinnedLSNCount != 1 || stats.MinimumPinnedLSN != lsn {
		t.Fatalf("after first close = %#v", stats)
	}
	if stats.OldestPinAge < ageBeforeRepin {
		t.Fatalf("age reset after repin: before=%v after=%v", ageBeforeRepin, stats.OldestPinAge)
	}

	second.Close()
	second.Close()
	stats, err = db.TemporalStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.ActiveLeaseCount != 0 || stats.DistinctPinnedLSNCount != 0 || stats.MinimumPinnedLSN != 0 || stats.OldestPinAge != 0 {
		t.Fatalf("after final/repeated close = %#v", stats)
	}
}

func TestTemporalStatsMultiplePinsCloseOutOfOrder(t *testing.T) {
	ctx := context.Background()
	db, firstLSN := temporalStatsDatabase(t, ":memory:temporal-stats-many")
	defer db.Close()

	time.Sleep(2 * time.Millisecond)
	if _, err := db.CreateCollection(ctx, "second", WithDimension(1)); err != nil {
		t.Fatal(err)
	}
	secondLSN, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatal(err)
	}

	second, err := db.SnapshotAtLSN(ctx, secondLSN)
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(3 * time.Millisecond)
	first, err := db.SnapshotAtLSN(ctx, firstLSN)
	if err != nil {
		t.Fatal(err)
	}

	stats, err := db.TemporalStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.ActiveLeaseCount != 2 || stats.DistinctPinnedLSNCount != 2 || stats.MinimumPinnedLSN != firstLSN {
		t.Fatalf("multiple-pin stats = %#v, first=%d second=%d", stats, firstLSN, secondLSN)
	}
	if stats.OldestPinAge < 3*time.Millisecond {
		t.Fatalf("OldestPinAge = %v, want at least 3ms", stats.OldestPinAge)
	}

	first.Close()
	stats, err = db.TemporalStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.ActiveLeaseCount != 1 || stats.DistinctPinnedLSNCount != 1 || stats.MinimumPinnedLSN != secondLSN {
		t.Fatalf("after closing minimum = %#v", stats)
	}

	second.Close()
}

func TestTemporalStatsRetentionCompactionAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "temporal-stats-retention.libravdb")
	retention := time.Millisecond
	db, err := Open(WithStoragePath(path), WithTemporalRetention(retention))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.CreateCollection(ctx, "docs", WithDimension(1)); err != nil {
		t.Fatal(err)
	}
	col, err := db.GetCollection("docs")
	if err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "one", []float32{1}, nil); err != nil {
		t.Fatal(err)
	}
	time.Sleep(5 * time.Millisecond)

	boundary, err := db.CompactHistory(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if boundary == 0 {
		t.Fatal("CompactHistory returned zero boundary after committed history")
	}
	stats, err := db.TemporalStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.OldestRetainedLSN == 0 || stats.OldestRetainedAt.IsZero() || stats.LastCompactionBoundary == 0 {
		t.Fatalf("post-compaction stats = %#v", stats)
	}
	if stats.LastCompactionBoundary != stats.OldestRetainedLSN || stats.OldestRetainedLSN != boundary {
		t.Fatalf("boundary mismatch stats=%#v returned=%d", stats, boundary)
	}

	secondBoundary, err := db.CompactHistory(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if secondBoundary < boundary {
		t.Fatalf("compaction boundary moved backward: first=%d second=%d", boundary, secondBoundary)
	}
	retainedAt := stats.OldestRetainedAt
	retainedLSN := stats.OldestRetainedLSN
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := Open(WithStoragePath(path), WithTemporalRetention(retention))
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	reopenedStats, err := reopened.TemporalStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if reopenedStats.ActiveLeaseCount != 0 || reopenedStats.DistinctPinnedLSNCount != 0 || reopenedStats.MinimumPinnedLSN != 0 || reopenedStats.OldestPinAge != 0 {
		t.Fatalf("reopened active pin stats = %#v", reopenedStats)
	}
	if reopenedStats.OldestRetainedLSN != retainedLSN || !reopenedStats.OldestRetainedAt.Equal(retainedAt) || reopenedStats.LastCompactionBoundary != retainedLSN {
		t.Fatalf("reopened retained stats = %#v, before=%#v", reopenedStats, stats)
	}
}

func TestTemporalStatsConcurrentReadersAndPins(t *testing.T) {
	ctx := context.Background()
	db, lsn := temporalStatsDatabase(t, ":memory:temporal-stats-concurrent")
	defer db.Close()

	const workers = 8
	const iterations = 100
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				snap, err := db.SnapshotAtLSN(ctx, lsn)
				if err != nil {
					errs <- err
					return
				}
				stats, err := db.TemporalStats(ctx)
				if err != nil {
					errs <- err
					snap.Close()
					return
				}
				if stats.DistinctPinnedLSNCount > stats.ActiveLeaseCount || stats.OldestPinAge < 0 ||
					(stats.DistinctPinnedLSNCount == 0 && stats.MinimumPinnedLSN != 0) {
					errs <- errors.New("impossible temporal statistics")
					snap.Close()
					return
				}
				snap.Close()
			}
		}()
	}

	for i := 0; i < iterations; i++ {
		stats, err := db.TemporalStats(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if stats.OldestPinAge < 0 || stats.DistinctPinnedLSNCount > stats.ActiveLeaseCount {
			t.Fatalf("impossible concurrent stats = %#v", stats)
		}
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestTemporalStatsContextAndClosedDatabase(t *testing.T) {
	ctx := context.Background()
	db, _ := temporalStatsDatabase(t, ":memory:temporal-stats-errors")

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := db.TemporalStats(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled TemporalStats error = %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := db.TemporalStats(ctx); !errors.Is(err, ErrDatabaseClosed) {
		t.Fatalf("closed TemporalStats error = %v", err)
	}
}

func temporalStatsDatabase(t *testing.T, path string) (*Database, uint64) {
	t.Helper()
	ctx := context.Background()
	db, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.CreateCollection(ctx, "first", WithDimension(1)); err != nil {
		db.Close()
		t.Fatal(err)
	}
	lsn, err := db.LatestCommitLSN(ctx)
	if err != nil {
		db.Close()
		t.Fatal(err)
	}
	return db, lsn
}
