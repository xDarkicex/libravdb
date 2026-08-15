package libravdb

import (
	"context"
	"os"
	"strings"
	"testing"
)

func TestSQLPrimaryKeyRenameOnUpdateCascade(t *testing.T) {
	db := openTempDB(t, "sql_pk_rename_cascade")
	defer db.Close()
	exec(t, db, "CREATE TABLE customers (email TEXT PRIMARY KEY, name TEXT)")
	exec(t, db, "CREATE TABLE orders (id TEXT, customer_email TEXT REFERENCES customers(email) ON UPDATE CASCADE)")
	exec(t, db, "INSERT INTO customers (email, name) VALUES ('old@example.com', 'Alice')")
	exec(t, db, "INSERT INTO orders (id, customer_email) VALUES ('o1', 'old@example.com')")
	oldNode, err := db.GetNodeID(context.Background(), "customers", "__pk:5:email15:old@example.com|")
	if err != nil {
		t.Fatalf("resolve old graph node: %v", err)
	}

	if _, err := db.Query(context.Background(), "UPDATE customers SET email = 'new@example.com' WHERE email = 'old@example.com'"); err != nil {
		t.Fatalf("primary-key rename: %v", err)
	}
	rows, err := db.Query(context.Background(), "SELECT * FROM customers WHERE email = 'new@example.com'")
	if err != nil || len(rows.Results) != 1 {
		t.Fatalf("new primary-key row missing: rows=%d err=%v", len(rows.Results), err)
	}
	rows, err = db.Query(context.Background(), "SELECT * FROM customers WHERE email = 'old@example.com'")
	if err != nil || len(rows.Results) != 0 {
		t.Fatalf("old primary-key row remains: rows=%d err=%v", len(rows.Results), err)
	}
	rows, err = db.Query(context.Background(), "SELECT * FROM orders WHERE customer_email = 'new@example.com'")
	if err != nil || len(rows.Results) != 1 {
		t.Fatalf("child FK was not cascaded: rows=%d err=%v", len(rows.Results), err)
	}
	newNode, err := db.GetNodeID(context.Background(), "customers", "__pk:5:email15:new@example.com|")
	if err != nil || newNode != oldNode {
		t.Fatalf("primary-key rename changed graph identity: old=%d new=%d err=%v", oldNode, newNode, err)
	}
}

func TestSQLPrimaryKeyRenameRestrict(t *testing.T) {
	db := openTempDB(t, "sql_pk_rename_restrict")
	defer db.Close()
	exec(t, db, "CREATE TABLE customers (email TEXT PRIMARY KEY, name TEXT)")
	exec(t, db, "CREATE TABLE orders (id TEXT, customer_email TEXT REFERENCES customers(email) ON UPDATE RESTRICT)")
	exec(t, db, "INSERT INTO customers (email, name) VALUES ('old@example.com', 'Alice')")
	exec(t, db, "INSERT INTO orders (id, customer_email) VALUES ('o1', 'old@example.com')")
	_, err := db.Query(context.Background(), "UPDATE customers SET email = 'new@example.com' WHERE email = 'old@example.com'")
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "foreign key") {
		t.Fatalf("expected ON UPDATE RESTRICT failure, got %v", err)
	}
}

func TestSQLPrimaryKeyRenameComposite(t *testing.T) {
	db := openTempDB(t, "sql_pk_rename_composite")
	defer db.Close()
	exec(t, db, "CREATE TABLE products (sku TEXT, name TEXT, PRIMARY KEY (sku, name))")
	exec(t, db, "CREATE TABLE orders (id TEXT, sku TEXT, name TEXT, FOREIGN KEY (sku, name) REFERENCES products(sku, name) ON UPDATE CASCADE)")
	exec(t, db, "INSERT INTO products (sku, name) VALUES ('s1', 'n1')")
	exec(t, db, "INSERT INTO orders (id, sku, name) VALUES ('o1', 's1', 'n1')")
	if _, err := db.Query(context.Background(), "UPDATE products SET name = 'n2' WHERE sku = 's1'"); err != nil {
		t.Fatalf("composite primary-key rename: %v", err)
	}
	rows, err := db.Query(context.Background(), "SELECT * FROM orders WHERE sku = 's1'")
	if err != nil || len(rows.Results) != 1 {
		t.Fatalf("composite child missing: rows=%d err=%v", len(rows.Results), err)
	}
	rows, err = db.Query(context.Background(), "SELECT * FROM orders WHERE name = 'n2'")
	if err != nil || len(rows.Results) != 1 {
		t.Fatalf("composite child FK not cascaded: rows=%d err=%v", len(rows.Results), err)
	}
}

func TestSQLPrimaryKeyRenameEpochRollback(t *testing.T) {
	db := openTempDB(t, "sql_pk_rename_epoch")
	defer db.Close()
	exec(t, db, "CREATE TABLE customers (email TEXT PRIMARY KEY, name TEXT)")
	exec(t, db, "CREATE TABLE orders (id TEXT, customer_email TEXT REFERENCES customers(email) ON UPDATE CASCADE)")
	exec(t, db, "INSERT INTO customers (email, name) VALUES ('old@example.com', 'Alice')")
	exec(t, db, "INSERT INTO orders (id, customer_email) VALUES ('o1', 'old@example.com')")
	session, err := db.NewSQLSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	if err := session.Exec("BEGIN EPOCH TRANSACTION"); err != nil {
		t.Fatal(err)
	}
	if err := session.Exec("UPDATE customers SET email = 'branch@example.com' WHERE email = 'old@example.com'"); err != nil {
		t.Fatalf("epoch primary-key rename: %v", err)
	}
	rows, err := session.Query("SELECT * FROM orders WHERE customer_email = 'branch@example.com'")
	if err != nil || len(rows.Results) != 1 {
		t.Fatalf("epoch cascade missing: rows=%d err=%v", len(rows.Results), err)
	}
	if err := session.Exec("ROLLBACK"); err != nil {
		t.Fatal(err)
	}
	rows, err = db.Query(context.Background(), "SELECT * FROM customers WHERE email = 'old@example.com'")
	if err != nil || len(rows.Results) != 1 {
		t.Fatalf("rollback did not restore original key: rows=%d err=%v", len(rows.Results), err)
	}
}

func TestSQLPrimaryKeyRenameSurvivesReopen(t *testing.T) {
	path := t.TempDir() + "/sql_pk_rename_reopen"
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	exec(t, db, "CREATE TABLE customers (email TEXT PRIMARY KEY, name TEXT)")
	exec(t, db, "INSERT INTO customers (email, name) VALUES ('old@example.com', 'Alice')")
	oldNode, err := db.GetNodeID(context.Background(), "customers", "__pk:5:email15:old@example.com|")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(context.Background(), "UPDATE customers SET email = 'new@example.com' WHERE email = 'old@example.com'"); err != nil {
		t.Fatal(err)
	}
	db.Close()

	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	defer os.RemoveAll(path)
	newNode, err := db.GetNodeID(context.Background(), "customers", "__pk:5:email15:new@example.com|")
	if err != nil || newNode != oldNode {
		t.Fatalf("reopen lost renamed graph identity: old=%d new=%d err=%v", oldNode, newNode, err)
	}
}
