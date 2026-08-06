package libravdb

import (
	"context"
	"strings"
	"testing"
)

func TestPKFK_EndToEnd(t *testing.T) {
	db := openTempDB(t, "pkfk_e2e")
	defer db.Close()

	// 1. Arbitrary PK + composite PK
	exec(t, db, "CREATE TABLE customers (email TEXT PRIMARY KEY, name TEXT)")
	exec(t, db, "CREATE TABLE products (sku TEXT, name TEXT, PRIMARY KEY (sku, name))")

	// 2. Single + composite FK
	exec(t, db, "CREATE TABLE orders (id TEXT, customer_email TEXT REFERENCES customers(email), product_sku TEXT, product_name TEXT, FOREIGN KEY (product_sku, product_name) REFERENCES products(sku, name))")

	// 3. INSERT parent records
	exec(t, db, "INSERT INTO customers (email, name) VALUES ('alice@ex.com', 'Alice')")
	exec(t, db, "INSERT INTO products (sku, name) VALUES ('SKU-1', 'Widget')")

	// 4. INSERT valid child
	_, err := db.Query(context.Background(),
		"INSERT INTO orders (id, customer_email, product_sku, product_name) VALUES ('o1', 'alice@ex.com', 'SKU-1', 'Widget')")
	if err != nil {
		t.Fatalf("valid INSERT: %v", err)
	}

	// 5. INSERT invalid child (FK violation)
	_, err = db.Query(context.Background(),
		"INSERT INTO orders (id, customer_email, product_sku, product_name) VALUES ('o2', 'ghost@ex.com', 'SKU-1', 'Widget')")
	if err == nil {
		t.Fatal("expected FK violation for ghost customer")
	}
	if !strings.Contains(err.Error(), "foreign key") {
		t.Errorf("got: %v", err)
	}

	// 6. INSERT invalid composite FK
	_, err = db.Query(context.Background(),
		"INSERT INTO orders (id, customer_email, product_sku, product_name) VALUES ('o3', 'alice@ex.com', 'SKU-1', 'Gadget')")
	if err == nil {
		t.Fatal("expected composite FK violation")
	}

	// 7. UNIQUE violation
	_, err = db.Query(context.Background(),
		"INSERT INTO customers (email, name) VALUES ('alice@ex.com', 'Alice2')")
	if err == nil {
		t.Fatal("expected UNIQUE violation")
	}

	// 8. DELETE RESTRICT
	_, err = db.Query(context.Background(), "DELETE FROM customers WHERE email = 'alice@ex.com'")
	if err == nil {
		t.Fatal("expected RESTRICT violation")
	}
	if !strings.Contains(err.Error(), "foreign key") {
		t.Errorf("got: %v", err)
	}

	// 9. DROP TABLE protection
	_, err = db.Query(context.Background(), "DROP TABLE customers")
	if err == nil {
		t.Fatal("expected DROP protection")
	}
	if !strings.Contains(err.Error(), "cannot drop") {
		t.Errorf("got: %v", err)
	}

	t.Log("✅ PK/FK end-to-end: all checks passed")
}
