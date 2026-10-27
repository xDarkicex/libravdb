package libravdb

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
)

func changeTestDB(t *testing.T) (*Database, *Collection) {
	t.Helper()
	db, err := Open(WithStoragePath(filepath.Join(t.TempDir(), "changes.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := db.Close(); err != nil {
			t.Error(err)
		}
	})
	graph, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := graph.Close(); err != nil {
			t.Error(err)
		}
	})
	col, err := db.CreateCollection(context.Background(), "docs", WithDimension(1), WithGraph(graph))
	if err != nil {
		t.Fatal(err)
	}
	return db, col
}
func changeCursor(t *testing.T, db *Database) string {
	t.Helper()
	cursor, err := db.ChangeCheckpoint(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return cursor
}
func changePage(t *testing.T, db *Database, cursor string) ChangePage {
	t.Helper()
	page, err := db.ReadChanges(context.Background(), cursor, 100)
	if err != nil {
		t.Fatal(err)
	}
	return page
}
func TestChangeFeedCombinedEpochAndRollback(t *testing.T) {
	db, col := changeTestDB(t)
	source, target := seedChangeGraph(t, db, col)
	cursor := changeCursor(t, db)
	changeEpoch(t, db, source, target, false)
	page := changePage(t, db, cursor)
	if len(page.Rows.Operations) != 2 || page.Rows.Operations[0] != "record_put" || page.Rows.Operations[1] != "edge_add" {
		t.Fatalf("epoch changes: %+v", page)
	}
	if page.Rows.CommitLSNs[0] != page.Rows.CommitLSNs[1] || page.Rows.Sources[1] != source || page.Rows.Targets[1] != target {
		t.Fatal("record and graph change lost atomic grouping")
	}
	cursor = page.Next
	changeEpoch(t, db, source, target, true)
	if page := changePage(t, db, cursor); len(page.Rows.Operations) != 0 {
		t.Fatal("rolled back epoch published changes")
	}
}
func changeEpoch(t *testing.T, db *Database, source, target uint64, rollback bool) {
	t.Helper()
	ctx := context.Background()
	tx, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := tx.Update(ctx, "docs", "a", []float32{2}, nil); err != nil {
		t.Fatal(err)
	}
	if err := tx.AddGraphEdge("docs", source, target, 1, 10); err != nil {
		t.Fatal(err)
	}
	if rollback {
		err = tx.Rollback(ctx)
	} else {
		err = tx.Commit(ctx)
	}
	if err != nil {
		t.Fatal(err)
	}
}
func TestChangeFeedErrors(t *testing.T) {
	db, _ := changeTestDB(t)
	_, err := db.ReadChanges(context.Background(), "bad", 1)
	if !errors.Is(err, ErrChangeCursor) {
		t.Fatalf("cursor error: %v", err)
	}
	for _, limit := range []int{0, -1, 10001} {
		if _, err := db.ReadChanges(context.Background(), "", limit); err == nil {
			t.Fatal("invalid limit accepted")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := db.ReadChanges(ctx, "", 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := db.ChangeCheckpoint(context.Background()); !errors.Is(err, ErrDatabaseClosed) {
		t.Fatalf("closed database: %v", err)
	}
}
func TestChangeFeedOutputIsOwnedAndReadOnly(t *testing.T) {
	db, col := changeTestDB(t)
	ctx := context.Background()
	cursor := changeCursor(t, db)
	if err := col.Insert(ctx, "exact", []float32{1}, nil); err != nil {
		t.Fatal(err)
	}
	before, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatal(err)
	}
	page := changePage(t, db, cursor)
	page.Rows.RecordIDs[0] = "corrupt"
	again := changePage(t, db, cursor)
	if again.Rows.RecordIDs[0] != "exact" {
		t.Fatal("caller corrupted feed storage")
	}
	after, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatal("read mutated durable store")
	}
}

func seedChangeGraph(t *testing.T, db *Database, col *Collection) (uint64, uint64) {
	t.Helper()
	ctx := context.Background()
	for _, id := range []string{"a", "b"} {
		if err := col.Insert(ctx, id, []float32{1}, nil); err != nil {
			t.Fatal(err)
		}
	}
	source, err := db.GetNodeID(ctx, "docs", "a")
	if err != nil {
		t.Fatal(err)
	}
	target, err := db.GetNodeID(ctx, "docs", "b")
	if err != nil {
		t.Fatal(err)
	}
	return source, target
}
