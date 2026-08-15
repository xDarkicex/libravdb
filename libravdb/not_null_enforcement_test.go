package libravdb

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

func requireNotNullError(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("expected NOT NULL violation, got nil")
	}
	if !strings.Contains(strings.ToUpper(err.Error()), "NOT NULL") {
		t.Fatalf("expected NOT NULL error, got %v", err)
	}
}

func TestNotNull_DirectCollectionWrites(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "not_null_direct")
	defer db.Close()
	exec(t, db, "CREATE TABLE required_rows (id TEXT PRIMARY KEY, name TEXT NOT NULL, note TEXT)")
	coll := getColl(t, db, "required_rows")

	requireNotNullError(t, coll.Insert(ctx, "missing", nil, map[string]interface{}{"note": "x"}))
	requireNotNullError(t, coll.Insert(ctx, "nil", nil, map[string]interface{}{"name": nil}))
	if err := coll.Insert(ctx, "empty", nil, map[string]interface{}{"name": ""}); err != nil {
		t.Fatalf("empty string is a valid non-NULL value: %v", err)
	}
	requireNotNullError(t, coll.Update(ctx, "empty", nil, map[string]interface{}{"name": nil}))
	requireNotNullError(t, coll.Upsert(ctx, "upsert", nil, nil))

	if _, err := db.Query(ctx, "INSERT INTO required_rows (id, name) VALUES ('sql-null', NULL)"); err == nil {
		t.Fatal("explicit SQL NULL bypassed NOT NULL")
	}
	if _, err := db.Query(ctx, "INSERT INTO required_rows (id, name) VALUES ('sql-empty', '')"); err != nil {
		t.Fatalf("empty SQL string should satisfy NOT NULL: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO required_rows (id, name) VALUES ('sql-word', 'NULL')"); err != nil {
		t.Fatalf("quoted NULL text should satisfy NOT NULL: %v", err)
	}
	word, err := getColl(t, db, "required_rows").Get(ctx, "sql-word")
	if err != nil || word.Metadata["name"] != "NULL" {
		t.Fatalf("quoted NULL text was not preserved: %#v err=%v", word, err)
	}
	if _, err := db.Query(ctx, "UPDATE required_rows SET name = NULL WHERE id = 'sql-empty'"); err == nil {
		t.Fatal("explicit SQL NULL UPDATE bypassed NOT NULL")
	}
}

func TestNotNull_BatchAndDefaults(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "not_null_batch")
	defer db.Close()
	exec(t, db, "CREATE TABLE batch_required (id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'new')")
	coll := getColl(t, db, "batch_required")

	err := coll.InsertBatch(ctx, []VectorEntry{{ID: "bad", Metadata: map[string]interface{}{"name": "ok"}}, {ID: "missing", Metadata: nil}})
	requireNotNullError(t, err)
	if _, err := coll.Get(ctx, "bad"); err == nil {
		t.Fatal("failed batch partially published a row")
	}
	if err := coll.InsertBatch(ctx, []VectorEntry{{ID: "defaulted", Metadata: map[string]interface{}{"name": "ok"}}}); err != nil {
		t.Fatalf("DEFAULT should satisfy NOT NULL for omitted status: %v", err)
	}
	rec, err := coll.Get(ctx, "defaulted")
	if err != nil || rec.Metadata["status"] != "new" {
		t.Fatalf("defaulted row=%#v err=%v, want status=new", rec, err)
	}
}

func TestNotNull_TransactionalAndSavepointOverlay(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "not_null_tx")
	defer db.Close()
	exec(t, db, "CREATE TABLE tx_required (id TEXT PRIMARY KEY, name TEXT NOT NULL)")
	exec(t, db, "CREATE TABLE tx_default (id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT 'defaulted')")
	exec(t, db, "INSERT INTO tx_required (id, name) VALUES ('epoch-base', 'value')")

	tx, err := db.BeginTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	requireNotNullError(t, tx.Insert(ctx, "tx_required", "missing", nil, nil))
	if err := tx.Insert(ctx, "tx_required", "ok", nil, map[string]interface{}{"name": "value"}); err != nil {
		t.Fatal(err)
	}
	requireNotNullError(t, tx.Update(ctx, "tx_required", "ok", nil, map[string]interface{}{"name": nil}))
	requireNotNullError(t, tx.Upsert(ctx, "tx_required", "upsert-missing", nil, nil))
	if err := tx.Insert(ctx, "tx_default", "default-row", nil, nil); err != nil {
		t.Fatalf("transaction DEFAULT should satisfy NOT NULL: %v", err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	defaultRow, err := getColl(t, db, "tx_default").Get(ctx, "default-row")
	if err != nil || defaultRow.Metadata["name"] != "defaulted" {
		t.Fatalf("transaction default row=%#v err=%v, want name=defaulted", defaultRow, err)
	}

	session, err := db.NewSQLSession(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	if err := session.Exec("BEGIN EPOCH"); err != nil {
		t.Fatal(err)
	}
	if err := session.Exec("INSERT INTO tx_required (id, name) VALUES ('epoch-ok', 'value')"); err != nil {
		t.Fatal(err)
	}
	requireNotNullError(t, session.Exec("UPDATE tx_required SET name = NULL WHERE id = 'epoch-base'"))
	if err := session.Exec("SAVEPOINT branch"); err != nil {
		t.Fatal(err)
	}
	requireNotNullError(t, session.Exec("INSERT INTO tx_required (id) VALUES ('epoch-bad')"))
	if err := session.Exec("ROLLBACK TO SAVEPOINT branch"); err != nil {
		t.Fatal(err)
	}
	if err := session.Exec("COMMIT"); err != nil {
		t.Fatal(err)
	}
	if _, err := getColl(t, db, "tx_required").Get(ctx, "epoch-bad"); err == nil {
		t.Fatal("failed epoch insert became visible after savepoint rollback")
	}
}

func TestNotNull_CascadeUpdateAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "not_null_reopen.libravdb")
	db, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatal(err)
	}
	exec(t, db, "CREATE TABLE parents (id TEXT PRIMARY KEY, code TEXT UNIQUE NOT NULL)")
	exec(t, db, "CREATE TABLE children (id TEXT PRIMARY KEY, parent_code TEXT NOT NULL REFERENCES parents(code) ON UPDATE CASCADE)")
	exec(t, db, "INSERT INTO parents (id, code) VALUES ('p1', 'c1')")
	exec(t, db, "INSERT INTO children (id, parent_code) VALUES ('child', 'c1')")
	exec(t, db, "UPDATE parents SET code = 'c2' WHERE id = 'p1'")
	child := getColl(t, db, "children")
	rec, err := child.Get(ctx, "child")
	if err != nil || rec.Metadata["parent_code"] != "c2" {
		t.Fatalf("cascade update lost required child value: %#v err=%v", rec, err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	db, err = Open(WithStoragePath(path))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	requireNotNullError(t, getColl(t, db, "children").Insert(ctx, "missing", nil, nil))
	if err := getColl(t, db, "children").Insert(ctx, "empty", nil, map[string]interface{}{"parent_code": ""}); err == nil {
		// Empty is non-NULL, but it must still fail the FK in this schema.
		t.Fatal("empty FK value unexpectedly bypassed referential validation")
	}
}
