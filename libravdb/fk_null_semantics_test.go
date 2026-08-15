package libravdb

import (
	"context"
	"testing"
)

func TestFKEmptyStringIsNotSQLNull(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "fk_null_semantics")
	defer db.Close()

	exec := func(sql string) {
		t.Helper()
		if _, err := db.Query(ctx, sql); err != nil {
			t.Fatalf("%s: %v", sql, err)
		}
	}
	exec("CREATE TABLE fk_parents (id TEXT PRIMARY KEY, code TEXT UNIQUE)")
	exec("CREATE TABLE fk_children (id TEXT PRIMARY KEY, parent_code TEXT REFERENCES fk_parents(code) ON DELETE CASCADE)")
	exec("INSERT INTO fk_parents (id, code) VALUES ('p1', '')")
	exec("INSERT INTO fk_children (id, parent_code) VALUES ('empty', '')")
	exec("INSERT INTO fk_children (id, parent_code) VALUES ('null', NULL)")

	// The empty string is a real FK value and therefore matches the parent;
	// the explicit SQL NULL is exempt from the FK and must remain untouched.
	exec("DELETE FROM fk_parents WHERE id = 'p1'")
	children, err := db.GetCollection("fk_children")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := children.Get(ctx, "empty"); err == nil {
		t.Fatal("empty-string child survived ON DELETE CASCADE")
	}
	if _, err := children.Get(ctx, "null"); err != nil {
		t.Fatalf("SQL NULL child was incorrectly cascaded: %v", err)
	}
}

func TestFKSetDefaultEmptyStringRemainsValue(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "fk_empty_default")
	defer db.Close()

	exec := func(sql string) {
		t.Helper()
		if _, err := db.Query(ctx, sql); err != nil {
			t.Fatalf("%s: %v", sql, err)
		}
	}
	exec("CREATE TABLE default_parents (id TEXT PRIMARY KEY, code TEXT UNIQUE)")
	exec("CREATE TABLE default_children (id TEXT PRIMARY KEY, parent_code TEXT DEFAULT '' REFERENCES default_parents(code) ON DELETE SET DEFAULT)")
	exec("INSERT INTO default_parents (id, code) VALUES ('empty-parent', '')")
	exec("INSERT INTO default_parents (id, code) VALUES ('old-parent', 'old')")
	exec("INSERT INTO default_children (id, parent_code) VALUES ('child', 'old')")
	exec("DELETE FROM default_parents WHERE id = 'old-parent'")

	children, err := db.GetCollection("default_children")
	if err != nil {
		t.Fatal(err)
	}
	child, err := children.Get(ctx, "child")
	if err != nil {
		t.Fatal(err)
	}
	value, ok := child.Metadata["parent_code"]
	if !ok || value == nil {
		t.Fatalf("SET DEFAULT empty string became SQL NULL: metadata=%#v", child.Metadata)
	}
	if value != "" {
		t.Fatalf("SET DEFAULT value = %#v, want empty string", value)
	}
}
