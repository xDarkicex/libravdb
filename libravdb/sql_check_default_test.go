package libravdb

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

func TestCheckConstraint_InsertReject(t *testing.T) {
	db := openTempDB(t, "check_insert")
	defer db.Close()

	exec(t, db, `CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, price FLOAT, CHECK (price > 0))`)

	// Valid: price > 0
	exec(t, db, `INSERT INTO products (id, name, price) VALUES ('p1', 'widget', 9.99)`)

	// Invalid: price <= 0
	_, err := db.Query(context.Background(),
		`INSERT INTO products (id, name, price) VALUES ('p2', 'gadget', -1)`)
	if err == nil {
		t.Fatal("expected CHECK constraint failure for price <= 0")
	}
	t.Logf("CHECK reject: %v", err)
}

func TestCheckConstraint_UpdateReject(t *testing.T) {
	db := openTempDB(t, "check_update")
	defer db.Close()

	exec(t, db, `CREATE TABLE items (id TEXT PRIMARY KEY, qty INTEGER, CHECK (qty >= 0))`)
	exec(t, db, `INSERT INTO items (id, qty) VALUES ('i1', 10)`)

	// Valid update
	exec(t, db, `UPDATE items SET qty = 5 WHERE id = 'i1'`)

	// Invalid update: qty < 0
	_, err := db.Query(context.Background(),
		`UPDATE items SET qty = -3 WHERE id = 'i1'`)
	if err == nil {
		t.Fatal("expected CHECK constraint failure for qty < 0")
	}
	t.Logf("CHECK update reject: %v", err)
}

func TestCheckConstraint_NamedConstraint(t *testing.T) {
	db := openTempDB(t, "check_named")
	defer db.Close()

	exec(t, db, `CREATE TABLE users (id TEXT PRIMARY KEY, age INTEGER, CONSTRAINT age_check CHECK (age >= 0))`)
	exec(t, db, `INSERT INTO users (id, age) VALUES ('u1', 25)`)

	_, err := db.Query(context.Background(),
		`INSERT INTO users (id, age) VALUES ('u2', -5)`)
	if err == nil {
		t.Fatal("expected CHECK constraint failure")
	}
	t.Logf("Named CHECK reject: %v", err)
}

func TestCheckConstraint_NotNull(t *testing.T) {
	db := openTempDB(t, "check_notnull")
	defer db.Close()

	exec(t, db, `CREATE TABLE profiles (id TEXT PRIMARY KEY, bio TEXT, CHECK (bio IS NOT NULL))`)
	exec(t, db, `INSERT INTO profiles (id, bio) VALUES ('p1', 'hello')`)

	// bio missing — should fail CHECK
	_, err := db.Query(context.Background(),
		`INSERT INTO profiles (id) VALUES ('p2')`)
	if err == nil {
		t.Fatal("expected CHECK (bio IS NOT NULL) failure")
	}
	t.Logf("CHECK IS NOT NULL reject: %v", err)
}

func TestDefaultLiteral_AppliedOnInsert(t *testing.T) {
	db := openTempDB(t, "default_insert")
	defer db.Close()

	exec(t, db, `CREATE TABLE config (id TEXT PRIMARY KEY, enabled BOOLEAN DEFAULT TRUE, status TEXT DEFAULT 'active', retries INTEGER DEFAULT 3)`)
	exec(t, db, `INSERT INTO config (id) VALUES ('cfg1')`)

	col := getColl(t, db, "config")
	rec, err := col.Get(context.Background(), "cfg1")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if rec.Metadata["enabled"] != true {
		t.Errorf("expected enabled=true, got %v", rec.Metadata["enabled"])
	}
	if rec.Metadata["status"] != "active" {
		t.Errorf("expected status='active', got %v", rec.Metadata["status"])
	}
	if rec.Metadata["retries"] != int64(3) {
		t.Errorf("expected retries=3, got %v (%T)", rec.Metadata["retries"], rec.Metadata["retries"])
	}
}

func TestDefaultLiteral_ExplicitOverridesDefault(t *testing.T) {
	db := openTempDB(t, "default_override")
	defer db.Close()

	exec(t, db, `CREATE TABLE settings (id TEXT PRIMARY KEY, mode TEXT DEFAULT 'auto')`)
	exec(t, db, `INSERT INTO settings (id, mode) VALUES ('s1', 'manual')`)

	col := getColl(t, db, "settings")
	rec, err := col.Get(context.Background(), "s1")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if rec.Metadata["mode"] != "manual" {
		t.Errorf("expected mode='manual', got %v", rec.Metadata["mode"])
	}
}

func TestDefaultLiteral_Null(t *testing.T) {
	db := openTempDB(t, "default_null")
	defer db.Close()

	exec(t, db, `CREATE TABLE sparse (id TEXT PRIMARY KEY, optional TEXT DEFAULT NULL)`)
	exec(t, db, `INSERT INTO sparse (id) VALUES ('sp1')`)

	col := getColl(t, db, "sparse")
	rec, err := col.Get(context.Background(), "sp1")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if v, exists := rec.Metadata["optional"]; exists && v != nil {
		t.Errorf("expected optional to be nil (DEFAULT NULL), got %v", v)
	}
}

func TestCheckDefault_CloseReopen(t *testing.T) {
	dir := t.TempDir()

	dbPath := filepath.Join(dir, "persist.libravdb")
	db1, err := Open(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	exec(t, db1, `CREATE TABLE persist (id TEXT PRIMARY KEY, score INTEGER DEFAULT 100, CHECK (score >= 0))`)
	db1.Close()

	// Reopen — DEFAULT and CHECK should survive.
	db2, err := Open(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("Reopen: %v", err)
	}
	defer db2.Close()

	// DEFAULT still works.
	exec(t, db2, `INSERT INTO persist (id) VALUES ('r1')`)

	col := getColl(t, db2, "persist")
	rec, err := col.Get(context.Background(), "r1")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if rec.Metadata["score"] != int64(100) {
		t.Errorf("expected score=100 after reopen, got %v", rec.Metadata["score"])
	}

	// CHECK still enforced.
	_, err = db2.Query(context.Background(),
		`INSERT INTO persist (id, score) VALUES ('r2', -10)`)
	if err == nil {
		t.Fatal("expected CHECK failure after reopen")
	}
	t.Logf("CHECK after reopen: %v", err)
}

func TestFK_SetNull_Delete(t *testing.T) {
	db := openTempDB(t, "fk_setnull")
	defer db.Close()

	exec(t, db, `CREATE TABLE parents (id TEXT PRIMARY KEY, name TEXT)`)
	exec(t, db, `CREATE TABLE children (id TEXT PRIMARY KEY, parent_id TEXT REFERENCES parents(id) ON DELETE SET NULL)`)
	exec(t, db, `INSERT INTO parents (id, name) VALUES ('p1', 'alpha')`)
	exec(t, db, `INSERT INTO children (id, parent_id) VALUES ('c1', 'p1')`)

	// Delete parent — child.parent_id should become NULL.
	exec(t, db, `DELETE FROM parents WHERE id = 'p1'`)

	col := getColl(t, db, "children")
	rec, err := col.Get(context.Background(), "c1")
	if err != nil {
		t.Fatalf("Get child: %v", err)
	}
	if v := rec.Metadata["parent_id"]; v != nil {
		t.Errorf("expected parent_id to be NULL, got %v", v)
	}
}

func TestFK_SetNull_NotNull_Rejected(t *testing.T) {
	db := openTempDB(t, "fk_setnull_nn")
	defer db.Close()

	exec(t, db, `CREATE TABLE parents2 (id TEXT PRIMARY KEY)`)

	// parent_id is NOT NULL — SET NULL should be rejected at DDL time.
	_, err := db.Query(context.Background(),
		`CREATE TABLE children2 (id TEXT PRIMARY KEY, parent_id TEXT NOT NULL REFERENCES parents2(id) ON DELETE SET NULL)`)
	if err == nil {
		t.Fatal("expected DDL error: SET NULL with NOT NULL column")
	}
	t.Logf("SET NULL+NOT NULL reject: %v", err)
}

func TestFK_SetDefault_Delete(t *testing.T) {
	db := openTempDB(t, "fk_setdef")
	defer db.Close()

	exec(t, db, `CREATE TABLE categories (id TEXT PRIMARY KEY, name TEXT)`)
	exec(t, db, `CREATE TABLE items (id TEXT PRIMARY KEY, cat_id TEXT DEFAULT 'default_cat' REFERENCES categories(id) ON DELETE SET DEFAULT)`)
	exec(t, db, `INSERT INTO categories (id, name) VALUES ('cat1', 'Electronics')`)
	exec(t, db, `INSERT INTO items (id, cat_id) VALUES ('item1', 'cat1')`)

	// Delete category — item.cat_id should become 'default_cat'.
	exec(t, db, `DELETE FROM categories WHERE id = 'cat1'`)

	col := getColl(t, db, "items")
	rec, err := col.Get(context.Background(), "item1")
	if err != nil {
		t.Fatalf("Get item: %v", err)
	}
	if rec.Metadata["cat_id"] != "default_cat" {
		t.Errorf("expected cat_id='default_cat', got %v", rec.Metadata["cat_id"])
	}
}

func TestFK_SetDefault_NoDefaultRejected(t *testing.T) {
	db := openTempDB(t, "fk_setdef_none")
	defer db.Close()

	exec(t, db, `CREATE TABLE parents3 (id TEXT PRIMARY KEY)`)

	// No DEFAULT on parent_id, but SET DEFAULT — should be rejected at DDL time.
	_, err := db.Query(context.Background(),
		`CREATE TABLE children3 (id TEXT PRIMARY KEY, parent_id TEXT REFERENCES parents3(id) ON DELETE SET DEFAULT)`)
	if err == nil {
		t.Fatal("expected DDL error: SET DEFAULT without DEFAULT value")
	}
	t.Logf("SET DEFAULT without DEFAULT reject: %v", err)
}

func TestFK_OnUpdateSetNull(t *testing.T) {
	db := openTempDB(t, "fk_upd_null")
	defer db.Close()

	exec(t, db, `CREATE TABLE depts (code TEXT PRIMARY KEY, name TEXT)`)
	exec(t, db, `CREATE TABLE employees (id TEXT PRIMARY KEY, dept_code TEXT REFERENCES depts(code) ON UPDATE SET NULL)`)
	exec(t, db, `INSERT INTO depts (code, name) VALUES ('ENG', 'Engineering')`)
	exec(t, db, `INSERT INTO employees (id, dept_code) VALUES ('e1', 'ENG')`)

	// Update parent PK — child.dept_code should become NULL.
	exec(t, db, `UPDATE depts SET code = 'ENGR' WHERE code = 'ENG'`)

	col := getColl(t, db, "employees")
	rec, err := col.Get(context.Background(), "e1")
	if err != nil {
		t.Fatalf("Get emp: %v", err)
	}
	if v, exists := rec.Metadata["dept_code"]; exists && v != nil {
		t.Errorf("expected dept_code to be NULL, got %v", v)
	}
}

func TestFK_OnUpdateSetDefault(t *testing.T) {
	db := openTempDB(t, "fk_upd_def")
	defer db.Close()

	exec(t, db, `CREATE TABLE statuses (code TEXT PRIMARY KEY)`)
	exec(t, db, `CREATE TABLE tickets (id TEXT PRIMARY KEY, status TEXT DEFAULT 'new' REFERENCES statuses(code) ON UPDATE SET DEFAULT)`)
	exec(t, db, `INSERT INTO statuses (code) VALUES ('open')`)
	exec(t, db, `INSERT INTO tickets (id, status) VALUES ('t1', 'open')`)

	// Update parent PK — ticket.status should become 'new' (DEFAULT).
	exec(t, db, `UPDATE statuses SET code = 'active' WHERE code = 'open'`)

	col := getColl(t, db, "tickets")
	rec, err := col.Get(context.Background(), "t1")
	if err != nil {
		t.Fatalf("Get ticket: %v", err)
	}
	if rec.Metadata["status"] != "new" {
		t.Errorf("expected status='new', got %v", rec.Metadata["status"])
	}
}

func TestCheckConstraint_TableLevel(t *testing.T) {
	db := openTempDB(t, "check_table")
	defer db.Close()

	exec(t, db, `CREATE TABLE orders (id TEXT PRIMARY KEY, total FLOAT, discount FLOAT, CHECK (discount <= total))`)
	exec(t, db, `INSERT INTO orders (id, total, discount) VALUES ('o1', 100, 10)`)

	_, err := db.Query(context.Background(),
		`INSERT INTO orders (id, total, discount) VALUES ('o2', 50, 75)`)
	if err == nil {
		t.Fatal("expected CHECK (discount <= total) failure")
	}
	t.Logf("Table-level CHECK reject: %v", err)
}

func TestDefaultLiteral_RejectsExpression(t *testing.T) {
	db := openTempDB(t, "default_expr")
	defer db.Close()

	_, err := db.Query(context.Background(),
		`CREATE TABLE bad_defaults (id TEXT PRIMARY KEY, ts TEXT DEFAULT NOW())`)
	if err == nil {
		t.Fatal("expected error for DEFAULT NOW() expression")
	}
	if !strings.Contains(err.Error(), "DEFAULT") {
		t.Errorf("expected DEFAULT-related error, got: %v", err)
	}
	t.Logf("DEFAULT expression reject: %v", err)
}
