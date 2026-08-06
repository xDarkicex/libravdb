package libravdb

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// TestTemporalANN_BasicBuild verifies cache entry construction from visible vectors.
func TestTemporalANN_BasicBuild(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir()+"/temporal_ann_basic.libravdb"),
		WithTemporalANNCache(128<<20, 16))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Insert vectors at T1.
	for i := 0; i < 100; i++ {
		id := fmt.Sprintf("v%d", i)
		col.Insert(context.Background(), id, []float32{float32(i), 0, 0}, nil)
	}
	time.Sleep(20 * time.Millisecond)
	snap, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	lsn := snap.LSN
	snap.Close()

	// The cache should be initialized.
	if db.temporalCache == nil {
		t.Fatal("temporalCache not initialized")
	}

	// Force a cache build via getOrBuild.
	cfg := col.Config()
	key := temporalIndexKey{
		collection: col.name,
		lsn:        lsn,
		dimension:  cfg.Dimension,
		metric:     int(cfg.Metric),
		m:          cfg.M,
		efConst:    cfg.EfConstruction,
	}
	entry, err := db.temporalCache.getOrBuild(context.Background(), key, col)
	if err != nil {
		t.Fatalf("getOrBuild: %v", err)
	}
	defer db.temporalCache.release(entry)

	// The index should contain all 100 vectors.
	if entry.index.Size() != 100 {
		t.Errorf("index size = %d, want 100", entry.index.Size())
	}
	if len(entry.ordinalToID) != 100 {
		t.Errorf("ordinalToID length = %d, want 100", len(entry.ordinalToID))
	}

	// Search the temporal index.
	results, err := entry.index.Search(context.Background(), []float32{50, 0, 0}, 5, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Error("search returned no results")
	}
}

// TestTemporalANN_ExactFallback verifies exact path when cache disabled.
func TestTemporalANN_ExactFallback(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/temporal_ann_exact.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	// No ANN config → exact only.
	if db.temporalANNEnabled() {
		t.Error("temporal ANN should be disabled by default")
	}

	col, _ := db.CreateCollection(context.Background(), "c", WithDimension(3))
	col.Insert(context.Background(), "r1", []float32{1, 0, 0}, nil)
	time.Sleep(20 * time.Millisecond)

	// Exact temporal query should still work.
	snap, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	rec, err := col.GetAtLSN(context.Background(), "r1", snap.LSN)
	snap.Close()
	if err != nil {
		t.Fatalf("GetAtLSN: %v", err)
	}
	if rec == nil || rec.Vector[0] != 1 {
		t.Error("exact temporal should work without ANN")
	}
}

// TestTemporalANN_NoLiveIndexLeak verifies cached index doesn't contain future data.
func TestTemporalANN_NoLiveIndexLeak(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir()+"/temporal_ann_leak.libravdb"),
		WithTemporalANNCache(128<<20, 16))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, _ := db.CreateCollection(context.Background(), "c", WithDimension(3))

	// T1: Insert V1.
	col.Insert(context.Background(), "r1", []float32{1, 0, 0}, nil)
	time.Sleep(20 * time.Millisecond)
	snap, _ := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	lsn1 := snap.LSN
	snap.Close()

	// T2: Insert V2 (after T1 snapshot).
	col.Insert(context.Background(), "r2", []float32{2, 0, 0}, nil)

	// Build cache at T1 — should only contain r1.
	cfg := col.Config()
	key := temporalIndexKey{collection: col.name, lsn: lsn1, dimension: cfg.Dimension, metric: int(cfg.Metric), m: cfg.M, efConst: cfg.EfConstruction}
	entry, err := db.temporalCache.getOrBuild(context.Background(), key, col)
	if err != nil {
		t.Fatalf("getOrBuild: %v", err)
	}
	defer db.temporalCache.release(entry)

	// Verify r2 is NOT in the T1 index.
	for _, oid := range entry.ordinalToID {
		if oid == "r2" {
			t.Error("T1 cache contains r2 inserted after T1")
		}
	}
}
