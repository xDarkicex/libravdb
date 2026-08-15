package pgwire

import (
	"context"
	"database/sql"
	"encoding/json"
	"net"
	"testing"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/xDarkicex/libravdb/libravdb"
)

func TestBetaSQLSurfaceThroughPgx(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/beta-sql-pgx"), libravdb.WithMetrics(false))
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

	srv := startTestServer(t, db)
	defer srv.Close()
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()

	if _, err := sqlDB.ExecContext(ctx, `INSERT INTO todos (id, title, completed, priority, due_at, tags)
        VALUES ('t1', 'Write docs', false, 1, '2026-08-12 10:00:00', '["docs","go"]')`); err != nil {
		t.Fatal(err)
	}
	var id, title string
	var completed bool
	var priority int
	var dueAt time.Time
	var tags json.RawMessage
	if err := sqlDB.QueryRowContext(ctx, "SELECT id, title, completed, priority, due_at, tags FROM todos").Scan(&id, &title, &completed, &priority, &dueAt, &tags); err != nil {
		t.Fatalf("SELECT * scan: %v", err)
	}
	if id != "t1" || title != "Write docs" || completed || priority != 1 || dueAt.UTC().Format("2006-01-02 15:04:05") != "2026-08-12 10:00:00" || string(tags) != `["docs","go"]` {
		t.Fatalf("SELECT * values: id=%q title=%q completed=%v priority=%d due_at=%s tags=%s", id, title, completed, priority, dueAt.UTC().Format("2006-01-02 15:04:05"), tags)
	}

	if _, err := sqlDB.ExecContext(ctx, `INSERT INTO todos (id, title, completed, priority) VALUES ('t2', 'Ship demo', true, 2)`); err != nil {
		t.Fatal(err)
	}
	var trueCount, inCount, betweenCount int
	if err := sqlDB.QueryRowContext(ctx, "SELECT COUNT(*) FROM todos WHERE completed = true").Scan(&trueCount); err != nil {
		t.Fatal(err)
	}
	if err := sqlDB.QueryRowContext(ctx, "SELECT COUNT(*) FROM todos WHERE priority IN (1, 2)").Scan(&inCount); err != nil {
		t.Fatal(err)
	}
	if err := sqlDB.QueryRowContext(ctx, "SELECT COUNT(*) FROM todos WHERE priority BETWEEN 1 AND 2").Scan(&betweenCount); err != nil {
		t.Fatal(err)
	}
	if trueCount != 1 || inCount != 2 || betweenCount != 2 {
		t.Fatalf("predicate counts: true=%d in=%d between=%d", trueCount, inCount, betweenCount)
	}

	if _, err := sqlDB.ExecContext(ctx, `INSERT INTO todos (id, title, completed, priority, due_at, tags) VALUES
        ('t3', 'Comma, title', false, 3, '2026-08-10 09:00:00', '["comma","json"]'),
        ('t4', 'Another', true, 4, '2026-08-10 10:00:00', '["one"]')`); err != nil {
		t.Fatal("multi-row JSON INSERT:", err)
	}
	if _, err := sqlDB.ExecContext(ctx, "UPDATE todos SET completed = false WHERE id = 't2'"); err != nil {
		t.Fatal(err)
	}
	// Use a timestamp derived from the write under test. A calendar date in
	// the test makes the snapshot expire as soon as the suite runs on a later
	// day, even though the database has correctly retained all history.
	asOf := time.Now().UTC().Add(time.Second).Format(time.RFC3339Nano)
	var asOfCompleted bool
	if err := sqlDB.QueryRowContext(ctx, "SELECT completed FROM todos AS OF TIMESTAMP '"+asOf+"' WHERE id = 't2'").Scan(&asOfCompleted); err != nil {
		t.Fatal("AS OF scan:", err)
	}
	if asOfCompleted {
		t.Fatal("AS OF returned stale completed=true after UPDATE completed=false")
	}
	if _, err := sqlDB.ExecContext(ctx, "UPDATE todos SET completed = NOT completed WHERE id = $1", "t2"); err != nil {
		t.Fatal("parameterized UPDATE ... SET NOT:", err)
	}
	if err := sqlDB.QueryRowContext(ctx, "SELECT completed FROM todos WHERE id = 't2'").Scan(&asOfCompleted); err != nil {
		t.Fatal("toggle scan:", err)
	}
	if !asOfCompleted {
		t.Fatal("parameterized UPDATE ... SET NOT did not toggle completed=true")
	}
}
