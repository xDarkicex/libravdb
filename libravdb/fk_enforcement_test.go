package libravdb

import (
	"context"
	"strings"
	"testing"
)

// TestFKEnforcement_Insert validates FK constraints at insert time.
func TestFKEnforcement_Insert(t *testing.T) {
	t.Run("valid FK insert succeeds", func(t *testing.T) {
		db := openTempDB(t, "fk_valid")
		defer db.Close()
		exec(t, db, "CREATE TABLE customers (id TEXT PRIMARY KEY)")
		insertRecord(t, db, "customers", "cust-1")
		exec(t, db, "CREATE TABLE orders (id TEXT, customer_id UUID REFERENCES customers(id))")

		child := getColl(t, db, "orders")
		if err := child.Insert(context.Background(), "order-1", nil,
			map[string]interface{}{"customer_id": "cust-1"}); err != nil {
			t.Errorf("valid FK insert failed: %v", err)
		}
	})

	t.Run("invalid FK insert rejected", func(t *testing.T) {
		db := openTempDB(t, "fk_invalid")
		defer db.Close()
		exec(t, db, "CREATE TABLE customers (id TEXT PRIMARY KEY)")
		exec(t, db, "CREATE TABLE orders (id TEXT, customer_id UUID REFERENCES customers(id))")

		child := getColl(t, db, "orders")
		err := child.Insert(context.Background(), "order-1", nil,
			map[string]interface{}{"customer_id": "ghost"})
		if err == nil {
			t.Fatal("expected FK violation, got nil")
		}
		if !strings.Contains(err.Error(), "foreign key violation") {
			t.Errorf("expected 'foreign key violation', got: %v", err)
		}
	})

	t.Run("optional FK column succeeds", func(t *testing.T) {
		db := openTempDB(t, "fk_optional")
		defer db.Close()
		exec(t, db, "CREATE TABLE customers (id TEXT PRIMARY KEY)")
		exec(t, db, "CREATE TABLE orders (id TEXT, customer_id UUID REFERENCES customers(id))")

		child := getColl(t, db, "orders")
		if err := child.Insert(context.Background(), "order-1", nil, nil); err != nil {
			t.Errorf("optional FK insert failed: %v", err)
		}
	})

	t.Run("self-referencing FK", func(t *testing.T) {
		db := openTempDB(t, "fk_self")
		defer db.Close()
		exec(t, db, "CREATE TABLE employees (id TEXT, manager_id UUID REFERENCES employees(id))")
		coll := getColl(t, db, "employees")
		insertRecord(t, coll, "mgr-1")

		if err := coll.Insert(context.Background(), "emp-1", nil,
			map[string]interface{}{"manager_id": "mgr-1"}); err != nil {
			t.Errorf("self-ref FK failed: %v", err)
		}
		if err := coll.Insert(context.Background(), "emp-2", nil,
			map[string]interface{}{"manager_id": "ghost"}); err == nil {
			t.Fatal("expected self-ref FK violation, got nil")
		}
	})
}

// TestFKEnforcement_Update validates FK during updates.
func TestFKEnforcement_Update(t *testing.T) {
	t.Run("update to valid FK", func(t *testing.T) {
		db := openTempDB(t, "fk_upd_ok")
		defer db.Close()
		exec(t, db, "CREATE TABLE customers (id TEXT PRIMARY KEY)")
		insertRecord(t, db, "customers", "c1")
		insertRecord(t, db, "customers", "c2")
		exec(t, db, "CREATE TABLE orders (id TEXT, customer_id UUID REFERENCES customers(id))")
		child := getColl(t, db, "orders")
		insertRecord(t, child, "o1", "customer_id", "c1")

		if err := child.Update(context.Background(), "o1", nil,
			map[string]interface{}{"customer_id": "c2"}); err != nil {
			t.Errorf("valid FK update failed: %v", err)
		}
	})

	t.Run("update to invalid FK rejected", func(t *testing.T) {
		db := openTempDB(t, "fk_upd_bad")
		defer db.Close()
		exec(t, db, "CREATE TABLE customers (id TEXT PRIMARY KEY)")
		insertRecord(t, db, "customers", "c1")
		exec(t, db, "CREATE TABLE orders (id TEXT, customer_id UUID REFERENCES customers(id))")
		child := getColl(t, db, "orders")
		insertRecord(t, child, "o1", "customer_id", "c1")

		err := child.Update(context.Background(), "o1", nil,
			map[string]interface{}{"customer_id": "ghost"})
		if err == nil {
			t.Fatal("expected FK violation on update, got nil")
		}
		if !strings.Contains(err.Error(), "foreign key violation") {
			t.Errorf("got: %v", err)
		}
	})
}

// TestFKEnforcement_Delete validates ON DELETE RESTRICT and CASCADE.
func TestFKEnforcement_Delete(t *testing.T) {
	t.Run("RESTRICT rejects delete when children exist", func(t *testing.T) {
		db := openTempDB(t, "fk_del_restrict")
		defer db.Close()
		exec(t, db, "CREATE TABLE customers (id TEXT PRIMARY KEY)")
		exec(t, db, "CREATE TABLE orders (id TEXT, customer_id UUID REFERENCES customers(id))")
		insertRecord(t, db, "customers", "c1")
		child := getColl(t, db, "orders")
		insertRecord(t, child, "o1", "customer_id", "c1")

		parent := getColl(t, db, "customers")
		err := parent.Delete(context.Background(), "c1")
		if err == nil {
			t.Fatal("expected RESTRICT violation, got nil")
		}
		if !strings.Contains(err.Error(), "foreign key violation") {
			t.Errorf("got: %v", err)
		}
	})

	t.Run("delete with no children succeeds", func(t *testing.T) {
		db := openTempDB(t, "fk_del_ok")
		defer db.Close()
		exec(t, db, "CREATE TABLE customers (id TEXT PRIMARY KEY)")
		exec(t, db, "CREATE TABLE orders (id TEXT, customer_id UUID REFERENCES customers(id))")
		insertRecord(t, db, "customers", "c1")

		parent := getColl(t, db, "customers")
		if err := parent.Delete(context.Background(), "c1"); err != nil {
			t.Errorf("delete with no children failed: %v", err)
		}
	})

	t.Run("CASCADE deletes children", func(t *testing.T) {
		db := openTempDB(t, "fk_cascade")
		defer db.Close()
		exec(t, db, "CREATE TABLE customers (id TEXT PRIMARY KEY)")
		exec(t, db, "CREATE TABLE orders (id TEXT, customer_id UUID REFERENCES customers(id) ON DELETE CASCADE)")
		insertRecord(t, db, "customers", "c1")
		child := getColl(t, db, "orders")
		insertRecord(t, child, "o1", "customer_id", "c1")

		parent := getColl(t, db, "customers")
		if err := parent.Delete(context.Background(), "c1"); err != nil {
			t.Fatalf("cascade delete failed: %v", err)
		}

		// Child must be gone.
		idx := child.GetIndex()
		getter := idx.(interface {
			Get(context.Context, string) (uint32, uint32, uint64, error)
		})
		if _, _, _, err := getter.Get(context.Background(), "o1"); err == nil {
			t.Fatal("child should be cascade-deleted but still exists")
		}
	})
}

// TestFKEnforcement_NotNull validates NOT NULL on FK columns.
func TestFKEnforcement_NotNull(t *testing.T) {
	t.Run("NOT NULL FK with value passes", func(t *testing.T) {
		db := openTempDB(t, "fk_nn_ok")
		defer db.Close()
		exec(t, db, "CREATE TABLE customers (id TEXT PRIMARY KEY)")
		insertRecord(t, db, "customers", "c1")
		exec(t, db, "CREATE TABLE orders (id TEXT, customer_id UUID NOT NULL REFERENCES customers(id))")
		child := getColl(t, db, "orders")

		err := child.Insert(context.Background(), "o1", nil,
			map[string]interface{}{"customer_id": "c1"})
		if err != nil {
			t.Errorf("NOT NULL FK with value failed: %v", err)
		}
	})

	t.Run("NOT NULL FK with null rejected", func(t *testing.T) {
		db := openTempDB(t, "fk_nn_bad")
		defer db.Close()
		exec(t, db, "CREATE TABLE customers (id TEXT PRIMARY KEY)")
		exec(t, db, "CREATE TABLE orders (id TEXT, customer_id UUID NOT NULL REFERENCES customers(id))")
		child := getColl(t, db, "orders")

		err := child.Insert(context.Background(), "o1", nil, nil)
		if err == nil {
			t.Fatal("expected NOT NULL violation, got nil")
		}
		if !strings.Contains(err.Error(), "NOT NULL") {
			t.Errorf("got: %v", err)
		}
	})
}

// --- helpers ---

func exec(t *testing.T, db *Database, sql string) {
	t.Helper()
	if _, err := db.Query(context.Background(), sql); err != nil {
		t.Fatalf("SQL %q: %v", sql, err)
	}
}

func getColl(t *testing.T, db *Database, name string) *Collection {
	t.Helper()
	c, err := db.GetCollection(name)
	if err != nil {
		t.Fatalf("GetCollection(%q): %v", name, err)
	}
	return c
}

// insertRecord inserts a record. Accepts a Database+name or a Collection directly.
// Usage: insertRecord(t, db, "coll", "id") or insertRecord(t, coll, "id", "key", "val")
func insertRecord(t *testing.T, dbOrColl interface{}, args ...string) {
	t.Helper()
	var coll *Collection
	var id string
	var meta map[string]interface{}

	switch c := dbOrColl.(type) {
	case *Database:
		coll = getColl(t, c, args[0])
		id = args[1]
		if len(args) > 2 {
			meta = pairsToMeta(args[2:]...)
		}
	case *Collection:
		coll = c
		id = args[0]
		if len(args) > 1 {
			meta = pairsToMeta(args[1:]...)
		}
	default:
		t.Fatalf("unsupported type: %T", dbOrColl)
	}
	if err := coll.Insert(context.Background(), id, nil, meta); err != nil {
		t.Fatalf("Insert(%q): %v", id, err)
	}
}

func pairsToMeta(pairs ...string) map[string]interface{} {
	m := make(map[string]interface{}, len(pairs)/2)
	for i := 0; i+1 < len(pairs); i += 2 {
		m[pairs[i]] = pairs[i+1]
	}
	return m
}
