package libravdb

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestBetaSQLSurfaceFindingsRepro(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/beta-sql-surface.libravdb"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	if _, err := db.Query(ctx, `CREATE TABLE todos (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        completed BOOLEAN DEFAULT false,
        priority INTEGER DEFAULT 3,
        due_at TIMESTAMP,
        tags JSON
    )`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `INSERT INTO todos (id, title, completed, priority, due_at, tags)
        VALUES ('t1', 'Write docs', false, 1, '2026-08-12 10:00:00', '["docs","go"]')`); err != nil {
		t.Fatal(err)
	}

	all, err := db.Query(ctx, "SELECT * FROM todos")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := strings.Join(all.Columns, ","), "id,completed,due_at,priority,tags,title"; got != want {
		t.Fatalf("SELECT * columns=%q, want %q", got, want)
	}
	if len(all.Results) != 1 {
		t.Fatalf("SELECT * rows=%d, want 1", len(all.Results))
	}
	row := all.Results[0]
	if row.ID != "t1" || row.Metadata["title"] != "Write docs" || row.Metadata["completed"] != "false" || row.Metadata["priority"] != "1" || row.Metadata["due_at"] != "2026-08-12 10:00:00" {
		t.Fatalf("SELECT * row misaligned: %#v", row)
	}
	if tags, ok := row.Metadata["tags"].([]interface{}); !ok || len(tags) != 2 || tags[0] != "docs" || tags[1] != "go" {
		t.Fatalf("SELECT * tags=%#v", row.Metadata["tags"])
	}
	logSQLResults(t, "SELECT *", all)

	for _, sql := range []string{
		"SELECT id, completed FROM todos WHERE completed = true",
		"SELECT id FROM todos WHERE priority IN (1, 2)",
		"SELECT id FROM todos WHERE priority BETWEEN 1 AND 2",
	} {
		rows, queryErr := db.Query(ctx, sql)
		if queryErr != nil {
			t.Fatal(queryErr)
		}
		if sql == "SELECT id FROM todos WHERE priority IN (1, 2)" || sql == "SELECT id FROM todos WHERE priority BETWEEN 1 AND 2" {
			if rows.Total != 1 || rows.Results[0].ID != "t1" {
				t.Fatalf("%s rows=%#v, want t1", sql, rows.Results)
			}
		}
		t.Logf("%s -> rows=%d results=%#v err=%v", sql, rows.Total, rows.Results, queryErr)
	}

	_, err = db.Query(ctx, `INSERT INTO todos (id, title, completed, priority, due_at, tags) VALUES
        ('t2', 'Ship demo', true, 2, '2026-08-11 09:00:00', '["ship"]')`)
	if err != nil {
		t.Fatal(err)
	}
	trueRows, err := db.Query(ctx, "SELECT id, completed FROM todos WHERE completed = true")
	if err != nil {
		t.Fatal(err)
	}
	if trueRows.Total != 1 || trueRows.Results[0].ID != "t2" || trueRows.Results[0].Metadata["completed"] != "true" {
		t.Fatalf("WHERE completed = true rows=%#v", trueRows.Results)
	}
	logSQLResults(t, "WHERE completed = true", trueRows)

	if _, err := db.Query(ctx, "UPDATE todos SET completed = true WHERE id = 't1'"); err != nil {
		t.Fatal(err)
	}
	updated, err := db.GetCollection("todos")
	if err != nil {
		t.Fatal(err)
	}
	updatedRecord, err := updated.Get(ctx, "t1")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("after UPDATE native record=%#v", updatedRecord)
	afterUpdate, err := db.SnapshotAt(ctx, time.Now().UTC().Add(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	asOf, err := db.Query(ctx, "SELECT id, title, completed FROM todos AS OF TIMESTAMP '"+afterUpdate.Timestamp.Format(time.RFC3339Nano)+"' WHERE id = 't1'")
	afterUpdate.Close()
	if err != nil {
		t.Fatal(err)
	}
	if asOf.Total != 1 || asOf.Results[0].Metadata["completed"] != "true" {
		t.Fatalf("AS OF updated metadata=%#v", asOf.Results)
	}
	logSQLResults(t, "AS OF after UPDATE", asOf)
	if _, err := db.QueryWithParams(ctx, "UPDATE todos SET completed = NOT completed WHERE id = $1", QueryParams{"1": "t1"}); err != nil {
		t.Fatalf("parameterized UPDATE ... SET NOT: %v", err)
	}
	toggled, err := db.Query(ctx, "SELECT completed FROM todos WHERE id = 't1'")
	if err != nil {
		t.Fatal(err)
	}
	if toggled.Total != 1 || toggled.Results[0].Metadata["completed"] != "false" {
		t.Fatalf("UPDATE ... SET NOT completed result=%#v", toggled.Results)
	}
	all, err = db.Query(ctx, "SELECT * FROM todos ORDER BY id")
	if err != nil {
		t.Fatal(err)
	}
	logSQLResults(t, "after UPDATE SELECT *", all)

	_, err = db.Query(ctx, `INSERT INTO todos (id, title, completed, priority, due_at, tags) VALUES
        ('t3', 'Comma, title', false, 3, '2026-08-10 09:00:00', '["comma","json"]'),
        ('t4', 'Another', true, 4, '2026-08-10 10:00:00', '["one"]')`)
	t.Logf("multi-row insert err=%v", err)
	if err != nil {
		t.Fatal(err)
	}
}

func logSQLResults(t *testing.T, label string, results *SearchResults) {
	t.Helper()
	for i, row := range results.Results {
		t.Logf("%s columns=%v row[%d] id=%q metadata=%#v", label, results.Columns, i, row.ID, row.Metadata)
	}
}
