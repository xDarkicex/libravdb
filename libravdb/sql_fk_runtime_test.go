package libravdb

import (
	"context"
	"os"
	"strings"
	"testing"
)

func TestSQLForeignKeys_RuntimeCrossDomain(t *testing.T) {
	db := openTempDB(t, "sql_fk_runtime")
	defer db.Close()

	exec(t, db, "CREATE TABLE products (sku TEXT, name TEXT, PRIMARY KEY (sku, name))")
	exec(t, db, "CREATE TABLE orders (id TEXT, product_sku TEXT, product_name TEXT, FOREIGN KEY (product_sku, product_name) REFERENCES products(sku, name))")
	exec(t, db, "INSERT INTO products (sku, name) VALUES ('sku-a', 'name-a')")
	exec(t, db, "INSERT INTO products (sku, name) VALUES ('sku-b', 'name-b')")

	_, err := db.Query(context.Background(), "INSERT INTO orders (id, product_sku, product_name) VALUES ('bad', 'sku-a', 'name-b')")
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "foreign key") {
		t.Fatalf("cross-pair composite FK should fail, got %v", err)
	}
}

func TestSQLForeignKeys_DeleteCascade(t *testing.T) {
	db := openTempDB(t, "sql_fk_delete_cascade")
	defer db.Close()
	exec(t, db, "CREATE TABLE customers (id TEXT PRIMARY KEY)")
	exec(t, db, "CREATE TABLE orders (id TEXT, customer_id TEXT REFERENCES customers(id) ON DELETE CASCADE)")
	exec(t, db, "INSERT INTO customers (id) VALUES ('c1')")
	exec(t, db, "INSERT INTO orders (id, customer_id) VALUES ('o1', 'c1')")
	if _, err := db.Query(context.Background(), "DELETE FROM customers WHERE id = 'c1'"); err != nil {
		t.Fatalf("cascade delete: %v", err)
	}
	rows, err := db.Query(context.Background(), "SELECT * FROM orders WHERE id = 'o1'")
	if err != nil {
		t.Fatalf("verify child deletion: %v", err)
	}
	if len(rows.Results) != 0 {
		t.Fatalf("cascade left %d child rows", len(rows.Results))
	}
}

func TestSQLForeignKeys_UpdateCascade(t *testing.T) {
	db := openTempDB(t, "sql_fk_update_cascade")
	defer db.Close()
	exec(t, db, "CREATE TABLE customers (id TEXT PRIMARY KEY, code TEXT UNIQUE)")
	exec(t, db, "CREATE TABLE orders (id TEXT, customer_code TEXT REFERENCES customers(code) ON UPDATE CASCADE)")
	exec(t, db, "INSERT INTO customers (id, code) VALUES ('c1', 'old')")
	exec(t, db, "INSERT INTO orders (id, customer_code) VALUES ('o1', 'old')")
	if _, err := db.Query(context.Background(), "UPDATE customers SET code = 'new' WHERE id = 'c1'"); err != nil {
		t.Fatalf("cascade update: %v", err)
	}
	rows, err := db.Query(context.Background(), "SELECT * FROM orders WHERE customer_code = 'new'")
	if err != nil {
		t.Fatalf("verify child update: %v", err)
	}
	if len(rows.Results) != 1 {
		t.Fatalf("expected cascaded child update, got %d rows", len(rows.Results))
	}
}

func TestSQLForeignKeys_CompositeSurvivesReopen(t *testing.T) {
	path := t.TempDir() + "/composite_fk_reopen"
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	exec(t, db, "CREATE TABLE products (sku TEXT, name TEXT, PRIMARY KEY (sku, name))")
	exec(t, db, "CREATE TABLE orders (id TEXT, product_sku TEXT, product_name TEXT, FOREIGN KEY (product_sku, product_name) REFERENCES products(sku, name))")
	exec(t, db, "INSERT INTO products (sku, name) VALUES ('sku-a', 'name-a')")
	db.Close()

	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	defer os.RemoveAll(path)
	_, err = db.Query(context.Background(), "INSERT INTO orders (id, product_sku, product_name) VALUES ('bad', 'sku-a', 'wrong')")
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "foreign key") {
		t.Fatalf("composite FK was not preserved after reopen: %v", err)
	}
}
