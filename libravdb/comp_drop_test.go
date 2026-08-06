package libravdb

import (
	"context"
	"strings"
	"testing"
)

func TestCompositeFKEnforcement(t *testing.T) {
	db := openTempDB(t, "comp_fk")
	defer db.Close()
	exec(t, db, "CREATE TABLE parent (a TEXT, b TEXT, PRIMARY KEY (a, b))")
	insertRecord(t, db, "parent", "x", "a", "x", "b", "y")
	exec(t, db, "CREATE TABLE child (id TEXT, ca TEXT, cb TEXT, FOREIGN KEY (ca, cb) REFERENCES parent(a, b))")
	child := getColl(t, db, "child")

	// Valid composite FK
	err := child.Insert(context.Background(), "c1", nil,
		map[string]interface{}{"ca": "x", "cb": "y"})
	if err != nil {
		t.Errorf("valid composite FK: %v", err)
	}

	// Invalid composite FK (second column doesn't match)
	err = child.Insert(context.Background(), "c2", nil,
		map[string]interface{}{"ca": "x", "cb": "z"})
	if err == nil {
		t.Fatal("expected FK violation")
	}
}

func TestDropTableFKProtection(t *testing.T) {
	db := openTempDB(t, "drop_fk")
	defer db.Close()
	exec(t, db, "CREATE TABLE customers (id TEXT PRIMARY KEY)")
	exec(t, db, "CREATE TABLE orders (id TEXT, customer_id UUID REFERENCES customers(id))")

	_, err := db.Query(context.Background(), "DROP TABLE customers")
	if err == nil {
		t.Fatal("expected FK protection error")
	}
	if !strings.Contains(err.Error(), "cannot drop") {
		t.Errorf("got: %v", err)
	}

	// Dropping the child should succeed.
	_, err = db.Query(context.Background(), "DROP TABLE orders")
	if err != nil {
		t.Errorf("DROP orders: %v", err)
	}
}
