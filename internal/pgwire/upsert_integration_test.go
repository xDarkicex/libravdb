package pgwire

import (
	"context"
	"database/sql"
	"encoding/json"
	"math"
	"net"
	"testing"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/xDarkicex/libravdb/libravdb"
)

// TestPGWireSQLUpsert uses database/sql + pgx, not the internal protocol
// helpers, so ON CONFLICT is verified at the same public-wire boundary as an
// external application.
func TestPGWireSQLUpsert(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/pgwire_upsert"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE TABLE docs (id TEXT PRIMARY KEY, title TEXT, counter TEXT)",
		"INSERT INTO docs (id, title, counter) VALUES ('d1', 'old', '1')",
		"INSERT INTO docs (id, title, counter) VALUES ('d1', 'ignored', '99') ON CONFLICT (id) DO NOTHING",
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatalf("%s: %v", query, err)
		}
	}
	if _, err := sqlDB.ExecContext(ctx, "INSERT INTO docs (id, title, counter) VALUES ($1, $2, $3) ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title || '-wire', counter = counter + EXCLUDED.counter", "d1", "new", "2"); err != nil {
		t.Fatalf("parameterized ON CONFLICT DO UPDATE: %v", err)
	}
	var title, counter string
	if err := sqlDB.QueryRowContext(ctx, "SELECT title, counter FROM docs WHERE id = 'd1'").Scan(&title, &counter); err != nil {
		t.Fatal(err)
	}
	if title != "new-wire" || counter != "3" {
		t.Fatalf("title=%q counter=%q, want new-wire/3", title, counter)
	}
	if _, err := sqlDB.ExecContext(ctx, "CREATE TABLE counters (key TEXT PRIMARY KEY, value BIGINT NOT NULL DEFAULT 0)"); err != nil {
		t.Fatalf("create counters: %v", err)
	}
	const counterUpsert = "INSERT INTO counters (key, value) VALUES ('page_views_home', 123) ON CONFLICT (key) DO UPDATE SET value = counters.value + EXCLUDED.value"
	if _, err := sqlDB.ExecContext(ctx, counterUpsert); err != nil {
		t.Fatalf("counter first upsert: %v", err)
	}
	if _, err := sqlDB.ExecContext(ctx, counterUpsert); err != nil {
		t.Fatalf("counter second upsert: %v", err)
	}
	var counterValue int64
	if err := sqlDB.QueryRowContext(ctx, "SELECT value FROM counters WHERE key = 'page_views_home'").Scan(&counterValue); err != nil {
		t.Fatalf("read counter: %v", err)
	}
	if counterValue != 246 {
		t.Fatalf("counter value=%d, want 246", counterValue)
	}
}

func TestPGWireParameterizedJSONBUpsert(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/pgwire_json_upsert"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := sqlDB.ExecContext(ctx, "CREATE TABLE people (id TEXT PRIMARY KEY, metadata JSONB, vector VECTOR(3))"); err != nil {
		t.Fatalf("create: %v", err)
	}
	const upsert = `INSERT INTO people (id, metadata, vector)
VALUES ($1, $2::jsonb, $3)
ON CONFLICT (id) DO UPDATE SET metadata = EXCLUDED.metadata, vector = EXCLUDED.vector`
	first := json.RawMessage(`{"name":"Ada","roles":["admin"]}`)
	if _, err := sqlDB.ExecContext(ctx, upsert, "p1", first, "[1,0,0]"); err != nil {
		t.Fatalf("JSONB insert: %v", err)
	}
	second := json.RawMessage(`{"name":"Ada Lovelace","roles":["admin","owner"]}`)
	if _, err := sqlDB.ExecContext(ctx, upsert, "p1", second, "[0,1,0]"); err != nil {
		t.Fatalf("JSONB update: %v", err)
	}
	var metadata string
	if err := sqlDB.QueryRowContext(ctx, "SELECT metadata->>'name' FROM people WHERE id = 'p1'").Scan(&metadata); err != nil {
		t.Fatalf("read JSONB: %v", err)
	}
	if metadata != "Ada Lovelace" {
		t.Fatalf("metadata name=%q", metadata)
	}
}

func TestPGWireSQLLiteralEscapesAndScientificNumbers(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/pgwire_literals"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := sqlDB.ExecContext(ctx, "CREATE TABLE literal_rows (id TEXT PRIMARY KEY, note TEXT, amount FLOAT)"); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"INSERT INTO literal_rows (id, note, amount) VALUES ('row1', 'foo''bar', -5.0e-3)",
		"INSERT INTO literal_rows (id, note, amount) VALUES ('row2', 'foo\\'bar', 1.5E+2)",
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatal(err)
		}
	}
	rows, err := sqlDB.QueryContext(ctx, "SELECT note, amount FROM literal_rows ORDER BY id")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	want := []struct {
		note  string
		value float64
	}{
		{note: "foo'bar", value: -5e-3},
		{note: "foo'bar", value: 1.5e2},
	}
	for i := range want {
		if !rows.Next() {
			t.Fatalf("missing row %d", i)
		}
		var note string
		var value float64
		if err := rows.Scan(&note, &value); err != nil {
			t.Fatal(err)
		}
		if note != want[i].note || math.Abs(value-want[i].value) > 1e-12 {
			t.Fatalf("row %d=(%q,%v) want (%q,%v)", i, note, value, want[i].note, want[i].value)
		}
	}
	if rows.Next() {
		t.Fatal("unexpected extra row")
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
}

func TestPGWireSQLQuotedReservedIdentifiersAndUpsert(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/pgwire_quoted"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		`CREATE TABLE "order" ("key" TEXT PRIMARY KEY, "value" BIGINT NOT NULL, "select" TEXT)`,
		`INSERT INTO "order" ("key", "value", "select") VALUES ('a-1', 10, 'SELECT')`,
		`INSERT INTO "order" ("key", "value", "select") VALUES ('a-1', 10, 'FROM') ON CONFLICT ("key") DO UPDATE SET "value" = "order"."value" + EXCLUDED."value"`,
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatal(err)
		}
	}
	var value int64
	var selected string
	if err := sqlDB.QueryRowContext(ctx, `SELECT "value", "select" FROM "order" WHERE "key" = 'a-1'`).Scan(&value, &selected); err != nil {
		t.Fatal(err)
	}
	if value != 20 || selected != "SELECT" {
		t.Fatalf("quoted row=(%d,%q), want (20,SELECT)", value, selected)
	}
}

func TestPGWireSQLCommentsAcrossUpsert(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/pgwire_comments"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := sqlDB.ExecContext(ctx, "CREATE TABLE comment_rows (id TEXT PRIMARY KEY, value BIGINT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := sqlDB.ExecContext(ctx, "INSERT INTO comment_rows (id, value) VALUES ('c1', 1)"); err != nil {
		t.Fatal(err)
	}
	query := `-- leading comment
/* block comment with 'quotes' */
INSERT INTO comment_rows (id, value) -- between clauses
VALUES ('c1', 1) /* inline */
ON CONFLICT (id) DO UPDATE
SET value = comment_rows.value + EXCLUDED.value;`
	if _, err := sqlDB.ExecContext(ctx, query); err != nil {
		t.Fatal(err)
	}
	var value int64
	if err := sqlDB.QueryRowContext(ctx, "SELECT value FROM comment_rows WHERE id = 'c1'").Scan(&value); err != nil {
		t.Fatal(err)
	}
	if value != 2 {
		t.Fatalf("comment query value=%d", value)
	}
}

func TestPGWireSQLMultiRowNumericLiterals(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/pgwire_numeric"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := sqlDB.ExecContext(ctx, "CREATE TABLE numeric_rows (id TEXT PRIMARY KEY, value FLOAT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := sqlDB.ExecContext(ctx, "INSERT INTO numeric_rows (id, value) VALUES ('int_lit', 123), ('float_lit', 123.45), ('sci_lit', 1.23e10), ('neg_lit', -42)"); err != nil {
		t.Fatal(err)
	}
	rows, err := sqlDB.QueryContext(ctx, "SELECT id, value FROM numeric_rows ORDER BY id")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	want := map[string]float64{"float_lit": 123.45, "int_lit": 123, "neg_lit": -42, "sci_lit": 1.23e10}
	seen := 0
	for rows.Next() {
		var id string
		var value float64
		if err := rows.Scan(&id, &value); err != nil {
			t.Fatal(err)
		}
		wantValue, ok := want[id]
		if !ok || math.Abs(value-wantValue) > math.Max(1e-12, math.Abs(wantValue)*1e-12) {
			t.Fatalf("row=(%q,%v)", id, value)
		}
		seen++
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if seen != len(want) {
		t.Fatalf("rows=%d want=%d", seen, len(want))
	}
}

func TestPGWireSQLEscapeStringLiteral(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/pgwire_escape"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := sqlDB.ExecContext(ctx, "CREATE TABLE escape_rows (id TEXT PRIMARY KEY, note TEXT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := sqlDB.ExecContext(ctx, `INSERT INTO escape_rows (id, note) VALUES ('e1', E'path\nwith\ttabs')`); err != nil {
		t.Fatal(err)
	}
	rows, err := sqlDB.QueryContext(ctx, "SELECT id, note FROM escape_rows")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	if !rows.Next() {
		t.Fatal("escape row was not inserted")
	}
	var id, note string
	if err := rows.Scan(&id, &note); err != nil {
		t.Fatal(err)
	}
	if id != "e1" {
		t.Fatalf("escape row id=%q", id)
	}
	if note != "path\nwith\ttabs" {
		t.Fatalf("escape string=%q", note)
	}
}

func TestPGWireSQLConflictScalarExpressions(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/pgwire_conflict_scalar"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE TABLE scalar_rows (id TEXT PRIMARY KEY, value BIGINT, marker TEXT, updated_at TEXT, ratio FLOAT)",
		"INSERT INTO scalar_rows (id, value, marker) VALUES ('r1', 10, 'same')",
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatal(err)
		}
	}
	query := `INSERT INTO scalar_rows (id, value, marker, updated_at, ratio)
VALUES ('r1', 20, 'same', 'ignored', 0)
ON CONFLICT (id) DO UPDATE SET
  value = CASE WHEN scalar_rows.value < 100 THEN scalar_rows.value + EXCLUDED.value ELSE scalar_rows.value END,
  marker = NULLIF(scalar_rows.marker, EXCLUDED.marker),
  updated_at = NOW(),
  ratio = (scalar_rows.value + EXCLUDED.value)::float`
	if _, err := sqlDB.ExecContext(ctx, query); err != nil {
		t.Fatal(err)
	}
	var value int64
	var marker, updatedAt sql.NullString
	var ratio float64
	if err := sqlDB.QueryRowContext(ctx, "SELECT value, marker, updated_at, ratio FROM scalar_rows WHERE id = 'r1'").Scan(&value, &marker, &updatedAt, &ratio); err != nil {
		t.Fatal(err)
	}
	if marker.Valid {
		t.Fatalf("NULLIF marker=%q, want NULL", marker.String)
	}
	if value != 30 || !updatedAt.Valid || updatedAt.String == "" || ratio != 30 {
		t.Fatalf("row=(%d,%q,%v), want value=30 non-empty NOW ratio=30", value, updatedAt.String, ratio)
	}
	for _, query := range []string{
		"INSERT INTO scalar_rows (id, value) VALUES ('r1', 0) ON CONFLICT (id) DO UPDATE SET value = scalar_rows.value << 1",
		"INSERT INTO scalar_rows (id, value) VALUES ('r1', 0) ON CONFLICT (id) DO UPDATE SET value = scalar_rows.value >> 1",
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatal(err)
		}
	}
	if err := sqlDB.QueryRowContext(ctx, "SELECT value FROM scalar_rows WHERE id = 'r1'").Scan(&value); err != nil {
		t.Fatal(err)
	}
	if value != 30 {
		t.Fatalf("shift value=%d, want 30", value)
	}
}

func TestPGWireSQLConflictNamedConstraintAndWhere(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/pgwire_conflict_named"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE TABLE counters_named (id TEXT PRIMARY KEY, key TEXT, region TEXT, value BIGINT)",
		"CREATE UNIQUE INDEX idx_counters_key_region ON counters_named (key, region)",
		"INSERT INTO counters_named (id, key, region, value) VALUES ('r1', 'home', 'us', 10)",
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatal(err)
		}
	}
	query := `INSERT INTO counters_named (id, key, region, value)
VALUES ('r2', 'home', 'us', 5)
ON CONFLICT ON CONSTRAINT idx_counters_key_region DO UPDATE
SET value = counters_named.value + EXCLUDED.value
WHERE counters_named.value < 100`
	if _, err := sqlDB.ExecContext(ctx, query); err != nil {
		t.Fatal(err)
	}
	var value int64
	if err := sqlDB.QueryRowContext(ctx, "SELECT value FROM counters_named WHERE id = 'r1'").Scan(&value); err != nil {
		t.Fatal(err)
	}
	if value != 15 {
		t.Fatalf("named conflict value=%d, want 15", value)
	}
	if _, err := sqlDB.ExecContext(ctx, `INSERT INTO counters_named (id, key, region, value)
VALUES ('r3', 'home', 'us', 5)
ON CONFLICT ON CONSTRAINT idx_counters_key_region DO UPDATE
SET value = counters_named.value + EXCLUDED.value
WHERE counters_named.value < 10`); err != nil {
		t.Fatal(err)
	}
	if err := sqlDB.QueryRowContext(ctx, "SELECT value FROM counters_named WHERE id = 'r1'").Scan(&value); err != nil {
		t.Fatal(err)
	}
	if value != 15 {
		t.Fatalf("conflict WHERE false value=%d, want 15", value)
	}
}

func TestPGWireSQLInsertSelect(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/pgwire_insert_select"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE TABLE source_rows (id TEXT PRIMARY KEY, value BIGINT)",
		"CREATE TABLE copied_rows (id TEXT PRIMARY KEY, value BIGINT)",
		"INSERT INTO source_rows (id, value) VALUES ('a', 1), ('b', 2)",
		"INSERT INTO copied_rows (id, value) SELECT id, value FROM source_rows WHERE value > 1",
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatal(query, ": ", err)
		}
	}
	var value int64
	if err := sqlDB.QueryRowContext(ctx, "SELECT value FROM copied_rows WHERE id = 'b'").Scan(&value); err != nil {
		t.Fatal(err)
	}
	if value != 2 {
		t.Fatalf("copied value=%d, want 2", value)
	}
	if err := sqlDB.QueryRowContext(ctx, "SELECT COUNT(*) FROM copied_rows").Scan(&value); err != nil {
		t.Fatal(err)
	}
	if value != 1 {
		t.Fatalf("copied rows=%d, want 1", value)
	}
}

func TestPGWireSQLUnionAll(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/pgwire_union_all"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE TABLE union_rows (id TEXT PRIMARY KEY, value BIGINT)",
		"INSERT INTO union_rows (id, value) VALUES ('a', 1), ('b', 2)",
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatal(err)
		}
	}
	rows, err := sqlDB.QueryContext(ctx, "SELECT id, value FROM union_rows WHERE value > 0 UNION ALL SELECT id, value FROM union_rows WHERE value > 1")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	count := 0
	for rows.Next() {
		var id string
		var value int64
		if err := rows.Scan(&id, &value); err != nil {
			t.Fatal(err)
		}
		count++
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("UNION ALL rows=%d, want 3", count)
	}
}

func TestPGWireSQLSetOperations(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/pgwire_set_operations"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE TABLE set_rows (id TEXT PRIMARY KEY, value BIGINT)",
		"INSERT INTO set_rows (id, value) VALUES ('a', 1), ('b', 2), ('c', 3)",
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatal(query, ": ", err)
		}
	}
	queries := []struct {
		name  string
		query string
		want  int
	}{
		{"UNION", "SELECT id, value FROM set_rows WHERE value < 3 UNION SELECT id, value FROM set_rows WHERE value > 1", 3},
		{"UNION ALL", "SELECT id, value FROM set_rows WHERE value < 3 UNION ALL SELECT id, value FROM set_rows WHERE value > 1", 4},
		{"INTERSECT", "SELECT id, value FROM set_rows WHERE value < 3 INTERSECT SELECT id, value FROM set_rows WHERE value > 1", 1},
		{"INTERSECT ALL", "SELECT id, value FROM set_rows WHERE value < 3 INTERSECT ALL SELECT id, value FROM set_rows WHERE value > 1", 1},
		{"EXCEPT", "SELECT id, value FROM set_rows WHERE value < 3 EXCEPT SELECT id, value FROM set_rows WHERE value > 1", 1},
		{"EXCEPT ALL", "SELECT id, value FROM set_rows WHERE value < 3 EXCEPT ALL SELECT id, value FROM set_rows WHERE value > 1", 1},
	}
	for _, tc := range queries {
		rows, err := sqlDB.QueryContext(ctx, tc.query)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		count := 0
		for rows.Next() {
			var id string
			var value int64
			if err := rows.Scan(&id, &value); err != nil {
				rows.Close()
				t.Fatalf("%s scan: %v", tc.name, err)
			}
			count++
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			t.Fatalf("%s rows: %v", tc.name, err)
		}
		rows.Close()
		if count != tc.want {
			t.Fatalf("%s rows=%d, want %d", tc.name, count, tc.want)
		}
	}
}

func TestPGWireSQLPrepareExecute(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/pgwire_prepare_execute"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE TABLE prepared_rows (id TEXT PRIMARY KEY, value BIGINT)",
		"INSERT INTO prepared_rows (id, value) VALUES ('p1', 1)",
		"PREPARE bump AS INSERT INTO prepared_rows (id, value) VALUES ($1, $2) ON CONFLICT (id) DO UPDATE SET value = prepared_rows.value + EXCLUDED.value",
		"EXECUTE bump('p1', 4)",
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatal(query, ": ", err)
		}
	}
	var value int64
	if err := sqlDB.QueryRowContext(ctx, "SELECT value FROM prepared_rows WHERE id = 'p1'").Scan(&value); err != nil {
		t.Fatal(err)
	}
	if value != 5 {
		t.Fatalf("prepared value=%d, want 5", value)
	}
}
