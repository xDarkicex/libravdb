package libravdb

import (
	"context"
	"os"
	"testing"

	"github.com/xDarkicex/libravdb/internal/catalog"
)

func TestSQLCompositePrimaryKeyDDLAndRuntime(t *testing.T) {
	db := openTempDB(t, "composite_pk")
	defer db.Close()
	ctx := context.Background()

	if _, err := db.Query(ctx, `CREATE TABLE memberships (
        tenant_id UUID,
        user_id UUID,
        label TEXT,
        PRIMARY KEY (tenant_id, user_id)
    )`); err != nil {
		t.Fatalf("CREATE TABLE: %v", err)
	}

	col, err := db.GetCollection("memberships")
	if err != nil {
		t.Fatalf("GetCollection: %v", err)
	}
	if got := col.Config().PrimaryKeyColumns; len(got) != 2 || got[0] != "tenant_id" || got[1] != "user_id" {
		t.Fatalf("PrimaryKeyColumns=%v, want [tenant_id user_id]", got)
	}

	db.mu.RLock()
	table, err := db.catalog.GetTable(catalog.HashIdentifier("memberships"))
	if err != nil {
		db.mu.RUnlock()
		t.Fatalf("GetTable: %v", err)
	}
	for _, name := range []string{"tenant_id", "user_id"} {
		column, columnErr := db.catalog.GetColumn(table, catalog.HashIdentifier(name))
		if columnErr != nil {
			db.mu.RUnlock()
			t.Fatalf("GetColumn(%s): %v", name, columnErr)
		}
		if column.Flags&catalog.ColFlagPrimaryKey == 0 || column.Flags&catalog.ColFlagNotNull == 0 {
			db.mu.RUnlock()
			t.Fatalf("column %s flags=%d, want PK and NOT NULL", name, column.Flags)
		}
	}
	db.mu.RUnlock()

	if _, err := db.Query(ctx, `INSERT INTO memberships (tenant_id, user_id, label)
        VALUES ('tenant-a', 'user-1', 'first')`); err != nil {
		t.Fatalf("first INSERT: %v", err)
	}
	if _, err := db.Query(ctx, `INSERT INTO memberships (tenant_id, user_id, label)
        VALUES ('tenant-a', 'user-2', 'second')`); err != nil {
		t.Fatalf("second INSERT: %v", err)
	}
	if _, err := db.Query(ctx, `INSERT INTO memberships (tenant_id, user_id, label)
        VALUES ('tenant-a', 'user-1', 'duplicate')`); err == nil {
		t.Fatal("duplicate composite primary key INSERT succeeded")
	}
	if _, err := db.Query(ctx, `UPDATE memberships SET user_id = 'user-9' WHERE tenant_id = 'tenant-a'`); err == nil {
		t.Fatal("UPDATE of composite primary key column succeeded")
	}
}

func TestSQLCompositePrimaryKeyMissingComponentRejected(t *testing.T) {
	db := openTempDB(t, "composite_pk_missing")
	defer db.Close()
	ctx := context.Background()
	if _, err := db.Query(ctx, `CREATE TABLE memberships (
        tenant_id TEXT,
        user_id TEXT,
        PRIMARY KEY (tenant_id, user_id)
    )`); err != nil {
		t.Fatalf("CREATE TABLE: %v", err)
	}
	if _, err := db.Query(ctx, `INSERT INTO memberships (tenant_id) VALUES ('tenant-a')`); err == nil {
		t.Fatal("INSERT missing composite PK component succeeded")
	}
}

func TestSQLCompositePrimaryKeyEnforcementAfterReopen(t *testing.T) {
	path := t.TempDir() + "/composite_pk_reopen"
	ctx := context.Background()
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := db.Query(ctx, `CREATE TABLE memberships (
        tenant_id TEXT,
        user_id TEXT,
        PRIMARY KEY (tenant_id, user_id)
    )`); err != nil {
		db.Close()
		t.Fatalf("CREATE TABLE: %v", err)
	}
	if _, err := db.Query(ctx, `INSERT INTO memberships (tenant_id, user_id)
        VALUES ('tenant-a', 'user-1')`); err != nil {
		db.Close()
		t.Fatalf("INSERT: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	db2, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("Reopen: %v", err)
	}
	defer db2.Close()
	defer os.RemoveAll(path)
	if _, err := db2.Query(ctx, `INSERT INTO memberships (tenant_id, user_id)
        VALUES ('tenant-a', 'user-2')`); err != nil {
		t.Fatalf("distinct composite key after reopen: %v", err)
	}
	if _, err := db2.Query(ctx, `INSERT INTO memberships (tenant_id, user_id)
        VALUES ('tenant-a', 'user-1')`); err == nil {
		t.Fatal("duplicate composite primary key succeeded after reopen")
	}
}
