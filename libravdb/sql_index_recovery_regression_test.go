package libravdb

import (
	"context"
	"path/filepath"
	"testing"
)

func TestSQLMetadataIndexReplayReplacementAfterCheckpoint(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "todos-source.libravdb")
	copyPath := filepath.Join(dir, "todos-replay.libravdb")

	db, err := Open(WithStoragePath(sourcePath), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `CREATE TABLE todos (id TEXT PRIMARY KEY, title TEXT, completed BOOLEAN DEFAULT false)`); err != nil {
		_ = db.Close()
		t.Fatal(err)
	}
	for _, id := range []string{"todo-1", "todo-2", "todo-3"} {
		if _, err := db.Query(ctx, "INSERT INTO todos (id, title) VALUES ('"+id+"', 'before')"); err != nil {
			_ = db.Close()
			t.Fatal(err)
		}
	}
	compactor, ok := db.storage.(interface{ Compact() error })
	if !ok {
		_ = db.Close()
		t.Fatal("storage engine does not support Compact")
	}
	if err := compactor.Compact(); err != nil {
		_ = db.Close()
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "UPDATE todos SET title = 'after' WHERE id = 'todo-3'"); err != nil {
		_ = db.Close()
		t.Fatal(err)
	}
	copyFile(t, sourcePath, copyPath)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	recovered, err := Open(WithStoragePath(copyPath), WithMetrics(false))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer recovered.Close()
	rows, err := recovered.Query(ctx, "SELECT id, title FROM todos WHERE id = 'todo-3'")
	if err != nil {
		t.Fatal(err)
	}
	if rows.Total != 1 || rows.Results[0].Metadata["title"] != "after" {
		t.Fatalf("recovered todo-3 = %#v", rows.Results)
	}
}
