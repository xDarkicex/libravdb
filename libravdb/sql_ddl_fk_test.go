package libravdb

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/xDarkicex/libravdb/internal/catalog"
)

// TestSQL_DDLCreateTableFK verifies foreign key constraint DDL-time validation
// and catalog persistence. Runtime enforcement is not tested.
func TestSQL_DDLCreateTableFK(t *testing.T) {
	// Helper: create a parent table so FK references are valid at DDL time.
	createParent := func(t *testing.T, db *Database, name string) {
		t.Helper()
		_, err := db.Query(context.Background(),
			"CREATE TABLE "+name+" (id TEXT PRIMARY KEY)")
		if err != nil {
			t.Fatalf("CREATE TABLE %s: %v", name, err)
		}
	}

	t.Run("inline FK persisted", func(t *testing.T) {
		db := openTempDB(t, "fk_inline")
		defer db.Close()
		createParent(t, db, "customers")

		_, err := db.Query(context.Background(),
			"CREATE TABLE orders (id TEXT, customer_id UUID REFERENCES customers(id))")
		if err != nil {
			t.Fatalf("CREATE TABLE: %v", err)
		}

		fks := getCatalogFKs(t, db)
		if len(fks) != 1 {
			t.Fatalf("expected 1 FK, got %d", len(fks))
		}
		if fks[0].OnDelete != catalog.OnDeleteNoAction {
			t.Errorf("OnDelete: want NoAction(0), got %d", fks[0].OnDelete)
		}
	})

	t.Run("inline FK ON DELETE CASCADE", func(t *testing.T) {
		db := openTempDB(t, "fk_cascade")
		defer db.Close()
		createParent(t, db, "customers")

		_, err := db.Query(context.Background(),
			"CREATE TABLE orders (id TEXT, customer_id UUID REFERENCES customers(id) ON DELETE CASCADE)")
		if err != nil {
			t.Fatalf("CREATE TABLE: %v", err)
		}

		fks := getCatalogFKs(t, db)
		if len(fks) != 1 {
			t.Fatalf("expected 1 FK, got %d", len(fks))
		}
		if fks[0].OnDelete != catalog.OnDeleteCascade {
			t.Errorf("OnDelete: want Cascade(1), got %d", fks[0].OnDelete)
		}
	})

	t.Run("table-level FK persisted", func(t *testing.T) {
		db := openTempDB(t, "fk_table_level")
		defer db.Close()
		createParent(t, db, "customers")

		_, err := db.Query(context.Background(),
			"CREATE TABLE orders (id TEXT PRIMARY KEY, customer_id UUID, FOREIGN KEY (customer_id) REFERENCES customers(id))")
		if err != nil {
			t.Fatalf("CREATE TABLE: %v", err)
		}

		fks := getCatalogFKs(t, db)
		if len(fks) != 1 {
			t.Fatalf("expected 1 FK, got %d", len(fks))
		}
	})

	t.Run("named table-level FK persisted", func(t *testing.T) {
		db := openTempDB(t, "fk_named")
		defer db.Close()
		createParent(t, db, "customers")

		_, err := db.Query(context.Background(),
			"CREATE TABLE orders (id TEXT PRIMARY KEY, customer_id UUID, CONSTRAINT valid_customer FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE CASCADE)")
		if err != nil {
			t.Fatalf("CREATE TABLE: %v", err)
		}

		fks := getCatalogFKs(t, db)
		if len(fks) != 1 {
			t.Fatalf("expected 1 FK, got %d", len(fks))
		}
		if fks[0].OnDelete != catalog.OnDeleteCascade {
			t.Errorf("OnDelete: want Cascade(1), got %d", fks[0].OnDelete)
		}
	})

	t.Run("GRAPH_NODES target bypasses DDL validation", func(t *testing.T) {
		db := openTempDB(t, "fk_graph_nodes")
		defer db.Close()

		_, err := db.Query(context.Background(),
			"CREATE TABLE users (id BIGINT PRIMARY KEY, CONSTRAINT valid_neighbor FOREIGN KEY (id) REFERENCES GRAPH_NODES(id) ON DELETE CASCADE)")
		if err != nil {
			t.Fatalf("CREATE TABLE: %v", err)
		}

		fks := getCatalogFKs(t, db)
		if len(fks) != 1 {
			t.Fatalf("expected 1 FK, got %d", len(fks))
		}
	})

	t.Run("FK and VECTOR(1536) coexist", func(t *testing.T) {
		db := openTempDB(t, "fk_with_vector")
		defer db.Close()
		createParent(t, db, "users")

		_, err := db.Query(context.Background(),
			"CREATE TABLE docs (id TEXT, embedding VECTOR(1536), owner_id UUID REFERENCES users(id))")
		if err != nil {
			t.Fatalf("CREATE TABLE: %v", err)
		}

		coll, err := db.GetCollection("docs")
		if err != nil {
			t.Fatalf("GetCollection: %v", err)
		}
		if coll.Dimension() != 1536 {
			t.Errorf("dimension: want 1536, got %d", coll.Dimension())
		}

		fks := getCatalogFKs(t, db)
		if len(fks) != 1 {
			t.Fatalf("expected 1 FK, got %d", len(fks))
		}
	})

	t.Run("column PK flags preserved with FK", func(t *testing.T) {
		db := openTempDB(t, "fk_pk_flags")
		defer db.Close()
		createParent(t, db, "parent")

		_, err := db.Query(context.Background(),
			"CREATE TABLE t (id TEXT PRIMARY KEY NOT NULL, ref TEXT REFERENCES parent(id))")
		if err != nil {
			t.Fatalf("CREATE TABLE: %v", err)
		}

		_, err = db.GetCollection("t")
		if err != nil {
			t.Fatalf("GetCollection: %v", err)
		}
	})

	t.Run("FK survives close and reopen", func(t *testing.T) {
		path := t.TempDir() + "/fk_reopen"
		db, err := Open(WithStoragePath(path), WithMetrics(false))
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		createParent(t, db, "customers")

		_, err = db.Query(context.Background(),
			"CREATE TABLE orders (id TEXT PRIMARY KEY, customer_id UUID REFERENCES customers(id) ON DELETE CASCADE)")
		if err != nil {
			db.Close()
			t.Fatalf("CREATE TABLE: %v", err)
		}
		db.Close()

		db2, err := Open(WithStoragePath(path), WithMetrics(false))
		if err != nil {
			t.Fatalf("Reopen: %v", err)
		}
		defer db2.Close()
		defer os.RemoveAll(path)

		fks := getCatalogFKs(t, db2)
		if len(fks) != 1 {
			t.Fatalf("after reopen: expected 1 FK, got %d", len(fks))
		}
		if fks[0].OnDelete != catalog.OnDeleteCascade {
			t.Errorf("after reopen OnDelete: want Cascade(1), got %d", fks[0].OnDelete)
		}

		coll, err := db2.GetCollection("orders")
		if err != nil {
			t.Fatalf("GetCollection after reopen: %v", err)
		}
		if coll.Dimension() != 0 {
			t.Errorf("dimension after reopen: want 0 (metadata-only), got %d", coll.Dimension())
		}
	})

	t.Run("FK metadata in live catalog", func(t *testing.T) {
		db := openTempDB(t, "fk_live")
		defer db.Close()
		createParent(t, db, "customers")

		_, err := db.Query(context.Background(),
			"CREATE TABLE orders (id TEXT, customer_id UUID REFERENCES customers(id) ON DELETE CASCADE)")
		if err != nil {
			t.Fatalf("CREATE TABLE: %v", err)
		}

		fks := getCatalogFKs(t, db)
		if len(fks) != 1 {
			t.Fatalf("expected 1 FK in live catalog, got %d", len(fks))
		}
		if fks[0].OnDelete != catalog.OnDeleteCascade {
			t.Errorf("OnDelete: want Cascade(1), got %d", fks[0].OnDelete)
		}
	})
}

// TestSQL_DDLCreateTableFK_ValidationErrors verifies that invalid FK
// references are rejected at DDL time with clear error messages.
func TestSQL_DDLCreateTableFK_ValidationErrors(t *testing.T) {
	tests := []struct {
		name        string
		setup       string // SQL to run before the FK table (empty = none)
		ddl         string
		errContains string
	}{
		{
			name:        "nonexistent target table",
			ddl:         "CREATE TABLE orders (id TEXT, customer_id UUID REFERENCES no_such_table(id))",
			errContains: "does not exist",
		},
		{
			name:        "nonexistent target column",
			setup:       "CREATE TABLE customers (id TEXT PRIMARY KEY)",
			ddl:         "CREATE TABLE orders (id TEXT, customer_id UUID REFERENCES customers(no_such_column))",
			errContains: "does not exist",
		},
		{
			name:        "named FK to nonexistent table",
			setup:       "CREATE TABLE customers (id TEXT PRIMARY KEY)",
			ddl:         "CREATE TABLE orders (id TEXT, customer_id UUID CONSTRAINT bad_fk REFERENCES ghost_table(id))",
			errContains: "does not exist",
		},
		{
			name:        "table-level FK to nonexistent table",
			ddl:         "CREATE TABLE orders (id TEXT, customer_id UUID, FOREIGN KEY (customer_id) REFERENCES ghost(id))",
			errContains: "does not exist",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			db := openTempDB(t, "fk_val_"+strings.ReplaceAll(tt.name, " ", "_"))
			defer db.Close()

			if tt.setup != "" {
				if _, err := db.Query(context.Background(), tt.setup); err != nil {
					t.Fatalf("setup: %v", err)
				}
			}

			_, err := db.Query(context.Background(), tt.ddl)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tt.errContains)
			}
			if !strings.Contains(err.Error(), tt.errContains) {
				t.Errorf("expected error containing %q, got %q", tt.errContains, err.Error())
			}
		})
	}
}

// openTempDB creates a temporary in-memory database for testing.
func openTempDB(t *testing.T, name string) *Database {
	t.Helper()
	db, err := Open(WithStoragePath(":memory:"+name), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return db
}

// getCatalogFKs extracts all FK definitions from the catalog for testing.
func getCatalogFKs(t *testing.T, db *Database) []*catalog.ForeignKeyDef {
	t.Helper()
	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.catalog.AllForeignKeys()
}
