package libravdb

import (
	"context"
	"strings"
	"testing"
)

func TestForeignKeyCascadeCycleFailsAtomically(t *testing.T) {
	db := openTempDB(t, "fk_cascade_cycle")
	defer db.Close()
	ctx := context.Background()

	exec(t, db, "CREATE TABLE cycle_rows (id TEXT PRIMARY KEY, parent_id TEXT REFERENCES cycle_rows(id) ON DELETE CASCADE)")
	exec(t, db, "INSERT INTO cycle_rows (id) VALUES ('a1')")
	exec(t, db, "INSERT INTO cycle_rows (id, parent_id) VALUES ('b1', 'a1')")
	exec(t, db, "UPDATE cycle_rows SET parent_id = 'b1' WHERE id = 'a1'")

	if _, err := db.Query(ctx, "DELETE FROM cycle_rows WHERE id = 'a1'"); err == nil || !strings.Contains(strings.ToLower(err.Error()), "cascade cycle") {
		t.Fatalf("expected cascade-cycle error, got %v", err)
	}
	rows, err := db.Query(ctx, "SELECT id FROM cycle_rows")
	if err != nil {
		t.Fatal(err)
	}
	if len(rows.Results) != 2 {
		t.Fatalf("cycle delete partially mutated table: %#v", rows.Results)
	}
}

func TestForeignKeyCompositeUpdateCascadeIsTupleAtomic(t *testing.T) {
	db := openTempDB(t, "fk_composite_update_tuple")
	defer db.Close()
	ctx := context.Background()

	exec(t, db, "CREATE TABLE tuple_products (sku TEXT, name TEXT, PRIMARY KEY (sku, name))")
	exec(t, db, "CREATE TABLE tuple_orders (id TEXT PRIMARY KEY, product_sku TEXT, product_name TEXT, FOREIGN KEY (product_sku, product_name) REFERENCES tuple_products(sku, name) ON UPDATE CASCADE)")
	exec(t, db, "INSERT INTO tuple_products (sku, name) VALUES ('s1', 'n1')")
	exec(t, db, "INSERT INTO tuple_orders (id, product_sku, product_name) VALUES ('o1', 's1', 'n1')")

	if _, err := db.Query(ctx, "UPDATE tuple_products SET name = 'n2' WHERE sku = 's1'"); err != nil {
		t.Fatalf("composite tuple update: %v", err)
	}
	rows, err := db.Query(ctx, "SELECT id FROM tuple_orders WHERE product_sku = 's1' AND product_name = 'n2'")
	if err != nil {
		t.Fatal(err)
	}
	if len(rows.Results) != 1 {
		t.Fatalf("composite cascade did not update the complete tuple: %#v", rows.Results)
	}
	rows, err = db.Query(ctx, "SELECT id FROM tuple_orders WHERE product_sku = 's1' AND product_name = 'n1'")
	if err != nil {
		t.Fatal(err)
	}
	if len(rows.Results) != 0 {
		t.Fatalf("old composite tuple remained after cascade: %#v", rows.Results)
	}
}

func TestForeignKeyNoActionIsImmediate(t *testing.T) {
	db := openTempDB(t, "fk_no_action_immediate")
	defer db.Close()
	ctx := context.Background()

	exec(t, db, "CREATE TABLE no_action_parent (id TEXT PRIMARY KEY)")
	exec(t, db, "CREATE TABLE no_action_child (id TEXT PRIMARY KEY, parent_id TEXT REFERENCES no_action_parent(id) ON DELETE NO ACTION)")
	exec(t, db, "INSERT INTO no_action_parent (id) VALUES ('p1')")
	exec(t, db, "INSERT INTO no_action_child (id, parent_id) VALUES ('c1', 'p1')")

	if _, err := db.Query(ctx, "DELETE FROM no_action_parent WHERE id = 'p1'"); err == nil || !strings.Contains(strings.ToLower(err.Error()), "foreign key") {
		t.Fatalf("expected immediate NO ACTION rejection, got %v", err)
	}
	rows, err := db.Query(ctx, "SELECT id FROM no_action_parent WHERE id = 'p1'")
	if err != nil || len(rows.Results) != 1 {
		t.Fatalf("NO ACTION rejection mutated parent: rows=%#v err=%v", rows.Results, err)
	}
}
