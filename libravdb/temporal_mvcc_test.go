package libravdb

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// TestMVCC_InsertUpdateDelete verifies visibility across the full lifecycle.
func TestMVCC_InsertUpdateDelete(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/mvcc_lifecycle.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// T1: Insert with vector [1,0,0].
	if err := col.Insert(context.Background(), "r1", []float32{1, 0, 0}, map[string]interface{}{"v": "one"}); err != nil {
		t.Fatalf("Insert T1: %v", err)
	}
	snap1, err := db.SnapshotAt(context.Background(), time.Now().UTC())
	if err != nil {
		t.Fatalf("SnapshotAt T1: %v", err)
	}
	lsn1 := snap1.LSN

	// T2: Update with vector [2,0,0].
	if err := col.Update(context.Background(), "r1", []float32{2, 0, 0}, map[string]interface{}{"v": "two"}); err != nil {
		t.Fatalf("Update T2: %v", err)
	}
	snap2, err := db.SnapshotAt(context.Background(), time.Now().UTC())
	if err != nil {
		t.Fatalf("SnapshotAt T2: %v", err)
	}
	lsn2 := snap2.LSN

	// T3: Delete.
	if err := col.Delete(context.Background(), "r1"); err != nil {
		t.Fatalf("Delete T3: %v", err)
	}
	snap3, err := db.SnapshotAt(context.Background(), time.Now().UTC())
	if err != nil {
		t.Fatalf("SnapshotAt T3: %v", err)
	}
	lsn3 := snap3.LSN

	// Read at T1: should see [1,0,0] with "one".
	rec1, err := col.GetAtLSN(context.Background(), "r1", lsn1)
	if err != nil {
		t.Fatalf("GetAtLSN T1: %v", err)
	}
	if rec1 == nil {
		t.Fatal("T1: record should exist")
	}
	if rec1.Vector[0] != 1 || rec1.Vector[1] != 0 || rec1.Vector[2] != 0 {
		t.Errorf("T1 vector = %v, want [1,0,0]", rec1.Vector)
	}
	if rec1.Metadata["v"] != "one" {
		t.Errorf("T1 metadata[v] = %v, want one", rec1.Metadata["v"])
	}

	// Read at T2: should see [2,0,0] with "two".
	rec2, err := col.GetAtLSN(context.Background(), "r1", lsn2)
	if err != nil {
		t.Fatalf("GetAtLSN T2: %v", err)
	}
	if rec2 == nil {
		t.Fatal("T2: record should exist")
	}
	if rec2.Vector[0] != 2 {
		t.Errorf("T2 vector[0] = %v, want 2", rec2.Vector[0])
	}
	if rec2.Metadata["v"] != "two" {
		t.Errorf("T2 metadata[v] = %v, want two", rec2.Metadata["v"])
	}

	// Read at T3: should be nil (deleted).
	rec3, err := col.GetAtLSN(context.Background(), "r1", lsn3)
	if err != nil {
		t.Fatalf("GetAtLSN T3: %v", err)
	}
	if rec3 != nil {
		t.Error("T3: record should be deleted (nil)")
	}

	// Current read: should still work (record exists if we re-insert, but
	// here it's deleted, so Get should return not-found).
	_, err = col.Get(context.Background(), "r1")
	if err == nil {
		t.Error("current Get should return not-found for deleted record")
	}
}

// TestMVCC_CurrentReadUnaffected verifies non-temporal reads still work.
func TestMVCC_CurrentReadUnaffected(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/mvcc_current.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	if err := col.Insert(context.Background(), "r1", []float32{5, 0, 0}, map[string]interface{}{"x": "y"}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	rec, err := col.Get(context.Background(), "r1")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if rec.Vector[0] != 5 {
		t.Errorf("vector[0] = %v, want 5", rec.Vector[0])
	}
}

// TestMVCC_SurvivesReopen verifies historical versions persist across restart.
func TestMVCC_SurvivesReopen(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/mvcc_reopen.libravdb"

	db, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	if err := col.Insert(context.Background(), "r1", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	snap1, _ := db.SnapshotAt(context.Background(), time.Now().UTC())

	if err := col.Update(context.Background(), "r1", []float32{2, 0, 0}, nil); err != nil {
		t.Fatalf("Update: %v", err)
	}
	db.Close()

	// Reopen and verify historical read.
	db2, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatalf("Reopen: %v", err)
	}
	defer db2.Drop(context.Background())

	col2, err := db2.GetCollection("c")
	if err != nil {
		t.Fatalf("GetCollection: %v", err)
	}
	rec1, err := col2.GetAtLSN(context.Background(), "r1", snap1.LSN)
	if err != nil {
		t.Fatalf("GetAtLSN after reopen: %v", err)
	}
	if rec1 == nil {
		t.Fatal("historical record should survive reopen")
	}
	if rec1.Vector[0] != 1 {
		t.Errorf("vector[0] = %v, want 1 (old version)", rec1.Vector[0])
	}
}

// TestMVCC_BeforeFirstCommit verifies pre-commit LSNs return nil.
func TestMVCC_BeforeFirstCommit(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/mvcc_before.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := col.Insert(context.Background(), "r1", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("Insert: %v", err)
	}

	// LSN 0 is before any commit — record should simply not exist.
	rec, err := col.GetAtLSN(context.Background(), "r1", 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rec != nil {
		t.Error("LSN 0: record should not exist (nil)")
	}
}

// TestMVCC_ListVisibleAtLSN verifies iteration at a snapshot.
func TestMVCC_ListVisibleAtLSN(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/mvcc_list.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("r%d", i)
		if err := col.Insert(context.Background(), id, []float32{float32(i), 0, 0}, nil); err != nil {
			t.Fatalf("Insert %s: %v", id, err)
		}
	}
	snap, err := db.SnapshotAt(context.Background(), time.Now().UTC())
	if err != nil {
		t.Fatalf("SnapshotAt: %v", err)
	}

	// Delete one record.
	if err := col.Delete(context.Background(), "r2"); err != nil {
		t.Fatalf("Delete r2: %v", err)
	}

	// At snapshot LSN, all 5 records should be visible.
	count := 0
	if err := col.ListVisibleAtLSN(context.Background(), snap.LSN, func(r *Record) bool {
		count++
		return true
	}); err != nil {
		t.Fatalf("ListVisibleAtLSN: %v", err)
	}
	if count != 5 {
		t.Errorf("got %d records, want 5", count)
	}
}
