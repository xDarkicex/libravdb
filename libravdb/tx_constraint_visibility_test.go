package libravdb

import (
	"context"
	"testing"
)

func TestTxForeignKeySeesStagedParentAndRejectsStagedDelete(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "tx_fk_visibility")
	defer db.Close()

	exec(t, db, "CREATE TABLE tx_parents (id TEXT PRIMARY KEY)")
	exec(t, db, "CREATE TABLE tx_children (id TEXT PRIMARY KEY, parent_id TEXT REFERENCES tx_parents(id) ON DELETE CASCADE)")

	tx, err := db.BeginTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := tx.Insert(ctx, "tx_parents", "p1", nil, nil); err != nil {
		t.Fatalf("stage parent: %v", err)
	}
	if err := tx.Insert(ctx, "tx_children", "c1", nil, map[string]interface{}{"parent_id": "p1"}); err != nil {
		t.Fatalf("FK to staged parent rejected: %v", err)
	}
	if err := tx.Delete(ctx, "tx_parents", "p1"); err != nil {
		t.Fatalf("delete staged parent: %v", err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatalf("commit staged parent/child cascade: %v", err)
	}
	children, err := db.Query(ctx, "SELECT id FROM tx_children")
	if err != nil {
		t.Fatal(err)
	}
	if len(children.Results) != 0 {
		t.Fatalf("staged child survived parent cascade: %#v", children.Results)
	}
}

func TestTxInsertAndUpsertApplyDefaultsAndCheck(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "tx_defaults_checks")
	defer db.Close()

	exec(t, db, "CREATE TABLE tx_settings (id TEXT PRIMARY KEY, retries INTEGER DEFAULT 3, CHECK (retries >= 0))")

	tx, err := db.BeginTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	callerMetadata := map[string]interface{}{}
	if err := tx.Insert(ctx, "tx_settings", "s1", nil, callerMetadata); err != nil {
		t.Fatalf("transaction insert with default: %v", err)
	}
	if _, exists := callerMetadata["retries"]; exists {
		t.Fatal("transaction insert mutated caller metadata while applying default")
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	rec, err := getColl(t, db, "tx_settings").Get(ctx, "s1")
	if err != nil {
		t.Fatal(err)
	}
	if rec.Metadata["retries"] != int64(3) {
		t.Fatalf("transaction default = %#v, want int64(3)", rec.Metadata["retries"])
	}

	tx, err = db.BeginTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := tx.Upsert(ctx, "tx_settings", "s2", nil, map[string]interface{}{"retries": int64(-1)}); err == nil {
		t.Fatal("transaction upsert violating CHECK succeeded")
	}
	_ = tx.Rollback(ctx)
}
