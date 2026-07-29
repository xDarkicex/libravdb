package libravdb

import (
	"context"
	"testing"
)

func TestTransactionRoutesShardedMutationWithUnshardedMutation(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	sharded, err := db.CreateCollection(
		ctx, "logical", WithDimension(3), WithFlat(), WithSharding(true),
	)
	if err != nil {
		t.Fatalf("CreateCollection(logical): %v", err)
	}
	outbox, err := db.CreateCollection(ctx, "outbox", WithDimension(1), WithFlat())
	if err != nil {
		t.Fatalf("CreateCollection(outbox): %v", err)
	}

	if err := db.WithTx(ctx, func(tx Tx) error {
		if err := tx.Insert(ctx, "logical", "record", []float32{1, 0, 0}, nil); err != nil {
			return err
		}
		return tx.Insert(ctx, "outbox", "event", []float32{0}, nil)
	}); err != nil {
		t.Fatalf("WithTx: %v", err)
	}

	if _, err := sharded.Get(ctx, "record"); err != nil {
		t.Fatalf("sharded record missing after commit: %v", err)
	}
	if _, err := outbox.Get(ctx, "event"); err != nil {
		t.Fatalf("outbox record missing after commit: %v", err)
	}
	if graphNodeID, err := sharded.LookupNodeID(ctx, "record"); err != nil || graphNodeID == 0 {
		t.Fatalf("LookupNodeID=(%d,%v), want nonzero ID", graphNodeID, err)
	}

	if err := db.WithTx(ctx, func(tx Tx) error {
		if err := tx.Upsert(ctx, "logical", "record", []float32{0, 1, 0}, map[string]interface{}{"version": "updated"}); err != nil {
			return err
		}
		return tx.Insert(ctx, "outbox", "update-event", []float32{0}, nil)
	}); err != nil {
		t.Fatalf("WithTx upsert: %v", err)
	}
	updated, err := sharded.Get(ctx, "record")
	if err != nil {
		t.Fatalf("Get after upsert: %v", err)
	}
	if updated.Metadata["version"] != "updated" {
		t.Fatalf("updated metadata=%v, want updated", updated.Metadata["version"])
	}

	if err := db.WithTx(ctx, func(tx Tx) error {
		if err := tx.Delete(ctx, "logical", "record"); err != nil {
			return err
		}
		return tx.Insert(ctx, "outbox", "delete-event", []float32{0}, nil)
	}); err != nil {
		t.Fatalf("WithTx delete: %v", err)
	}
	if _, err := sharded.Get(ctx, "record"); err == nil {
		t.Fatal("sharded record remains after transactional delete")
	}
	if _, err := outbox.Get(ctx, "delete-event"); err != nil {
		t.Fatalf("delete outbox record missing: %v", err)
	}
}
