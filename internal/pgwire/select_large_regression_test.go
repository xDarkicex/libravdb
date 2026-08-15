package pgwire

import (
	"context"
	"database/sql"
	"fmt"
	"net"
	"testing"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/xDarkicex/libravdb/libravdb"
)

func TestPGWireSelectReturnsAllRowsOverLargeResult(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/large-select-pgwire"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	if _, err := db.Query(ctx, "CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT)"); err != nil {
		t.Fatal(err)
	}
	const want = 500
	for i := 0; i < want; i++ {
		if _, err := db.QueryWithParams(ctx, "INSERT INTO users (id, name) VALUES ($1, $2)", libravdb.QueryParams{
			"1": fmt.Sprintf("node-%03d", i),
			"2": fmt.Sprintf("name-%03d", want-i),
		}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
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

	var count int
	if err := sqlDB.QueryRowContext(ctx, "SELECT COUNT(*) FROM users").Scan(&count); err != nil {
		t.Fatal("count:", err)
	}
	if count != want {
		t.Fatalf("count=%d, want %d", count, want)
	}

	rows, err := sqlDB.QueryContext(ctx, "SELECT id FROM users ORDER BY name")
	if err != nil {
		t.Fatal("select:", err)
	}
	defer rows.Close()
	seen := make(map[string]struct{}, want)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			t.Fatal("scan:", err)
		}
		seen[id] = struct{}{}
	}
	if err := rows.Err(); err != nil {
		t.Fatal("rows:", err)
	}
	if len(seen) != want {
		t.Fatalf("SELECT returned %d unique rows, want %d", len(seen), want)
	}
}
