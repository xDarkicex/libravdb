package pgwire

import (
	"context"
	"database/sql"
	"net"
	"testing"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/xDarkicex/libravdb/libravdb"
)

func TestPGWireScalarCaseAndCasts(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/scalar_projection"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE scalar_rows (id TEXT PRIMARY KEY, amount BIGINT, payload JSONB)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO scalar_rows (id, amount, payload) VALUES ('low', 5, '{\"kind\":\"a\"}'), ('high', 15, '{\"kind\":\"b\"}')"); err != nil {
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
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	rows, err := sqlDB.QueryContext(ctx, `SELECT id,
		CASE WHEN amount >= 10 THEN 'high' ELSE 'low' END AS tier,
		amount::float AS amount_float,
		'550e8400-e29b-41d4-a716-446655440000'::uuid AS row_uuid,
		'{"kind":"ok"}'::jsonb AS payload_copy
		FROM scalar_rows ORDER BY id`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	cols, err := rows.Columns()
	if err != nil {
		t.Fatal(err)
	}
	wantCols := []string{"id", "tier", "amount_float", "row_uuid", "payload_copy"}
	for i := range wantCols {
		if i >= len(cols) || cols[i] != wantCols[i] {
			t.Fatalf("columns=%v want=%v", cols, wantCols)
		}
	}
	type scalarRow struct {
		id, tier, uuid, payload string
		amount                  float64
	}
	var got []scalarRow
	for rows.Next() {
		var row scalarRow
		if err := rows.Scan(&row.id, &row.tier, &row.amount, &row.uuid, &row.payload); err != nil {
			t.Fatal(err)
		}
		got = append(got, row)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].tier != "high" || got[1].tier != "low" || got[0].amount != 15 || got[1].amount != 5 {
		t.Fatalf("scalar rows=%#v", got)
	}
	if got[0].uuid != "550e8400-e29b-41d4-a716-446655440000" || got[0].payload != `{"kind":"ok"}` {
		t.Fatalf("cast rows=%#v", got[0])
	}
	paramRows, err := sqlDB.QueryContext(ctx, `SELECT id FROM scalar_rows WHERE CASE WHEN amount >= $1 THEN TRUE ELSE FALSE END`, int64(10))
	if err != nil {
		t.Fatalf("typed CASE parameter query: %v", err)
	}
	defer paramRows.Close()
	var selected []string
	for paramRows.Next() {
		var id string
		if err := paramRows.Scan(&id); err != nil {
			t.Fatal(err)
		}
		selected = append(selected, id)
	}
	if err := paramRows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(selected) != 1 || selected[0] != "high" {
		t.Fatalf("typed parameter selected=%v", selected)
	}
}

func TestPGWireScalarCaseDescribeTypes(t *testing.T) {
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/scalar_describe"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(context.Background(), "CREATE TABLE scalar_describe (id TEXT PRIMARY KEY, amount BIGINT)"); err != nil {
		t.Fatal(err)
	}
	_, cols, err := describeStatement(db, `SELECT CASE WHEN amount > 0 THEN 'yes' ELSE 'no' END AS state, amount::float AS score, '550e8400-e29b-41d4-a716-446655440000'::uuid AS uid, NOW() AS observed_at, NULLIF(amount, 0) AS nullable_score FROM scalar_describe`, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := []ColumnMeta{{Name: "state", TypeOID: OIDText}, {Name: "score", TypeOID: OIDFloat8}, {Name: "uid", TypeOID: OIDUUID}, {Name: "observed_at", TypeOID: OIDTimestamptz}, {Name: "nullable_score", TypeOID: OIDInt8}}
	assertColumns(t, cols, want)
}

func TestPGWireScalarRejectsUnknownCast(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/scalar_unknown_cast"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE scalar_unknown_cast (id TEXT PRIMARY KEY, amount BIGINT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO scalar_unknown_cast (id, amount) VALUES ('row', 1)"); err != nil {
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
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	rows, err := sqlDB.QueryContext(ctx, "SELECT id, amount::not_a_real_type FROM scalar_unknown_cast")
	if err == nil {
		rows.Close()
		t.Fatal("unknown cast target unexpectedly succeeded over pgwire")
	}
}
