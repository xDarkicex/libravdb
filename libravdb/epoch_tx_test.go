package libravdb

import (
	"context"
	"testing"
)

func TestEpochTxRollbackDiscardsRecords(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:epoch_tx"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "scratch", WithDimension(2))
	if err != nil {
		t.Fatal(err)
	}
	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := epoch.Insert(ctx, "scratch", "hypothesis", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}
	if err := epoch.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := col.Get(ctx, "hypothesis"); err == nil {
		t.Fatal("rolled-back record is visible")
	}
}

func TestEpochTxCommitPublishesRecords(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:epoch_tx_commit"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "scratch", WithDimension(2))
	if err != nil {
		t.Fatal(err)
	}
	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := epoch.Insert(ctx, "scratch", "hypothesis", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}
	if err := epoch.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := col.Get(ctx, "hypothesis"); err != nil {
		t.Fatalf("committed record missing: %v", err)
	}
}

func TestEpochTxListRecordsIncludesStagedWrites(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:epoch_tx_overlay"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.CreateCollection(ctx, "scratch", WithDimension(2)); err != nil {
		t.Fatal(err)
	}
	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := epoch.Insert(ctx, "scratch", "hypothesis", []float32{1, 0}, map[string]interface{}{"kind": "agent"}); err != nil {
		t.Fatal(err)
	}
	recs, err := epoch.ListRecords(ctx, "scratch")
	if err != nil {
		t.Fatal(err)
	}
	if len(recs) != 1 || recs[0].ID != "hypothesis" {
		t.Fatalf("staged record missing from overlay: %#v", recs)
	}
	if err := epoch.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
}
