package pgwire

import (
	"context"
	"database/sql"
	"net"
	"testing"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/xDarkicex/libravdb/libravdb"
)

func TestPostgreSQLVectorOperators(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(":memory:pgwire_vector_operator"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "operator_docs", libravdb.WithDimension(3), libravdb.WithMetric(libravdb.CosineDistance))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for _, row := range []struct {
		id  string
		vec []float32
	}{
		{id: "a", vec: []float32{1, 0, 0}},
		{id: "b", vec: []float32{0.8, 0.6, 0}},
		{id: "c", vec: []float32{2, 1, 0}},
	} {
		if err := col.Insert(ctx, row.id, row.vec, nil); err != nil {
			t.Fatalf("Insert %s: %v", row.id, err)
		}
	}

	// Simple-query protocol exercises literal parsing, direct operator lowering,
	// RowDescription typing, result projection, and ORDER BY alias handling.
	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)
	sendSimpleQuery(conn, "SELECT id, embedding <-> '[1,0,0]' AS distance FROM operator_docs ORDER BY distance LIMIT 2")
	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("RowDescription: %v", err)
	}
	if msgType != msgRowDescription {
		t.Fatalf("first response: got %q, want RowDescription", msgType)
	}
	cols := decodeRowDescription(t, payload)
	if len(cols) != 2 || cols[0].Name != "id" || cols[1].Name != "distance" || cols[1].TypeOID != OIDFloat8 {
		t.Fatalf("vector operator RowDescription: %#v", cols)
	}
	rows := readDataRowsUntilComplete(t, conn)
	if len(rows) != 2 || rows[0][0] != "a" || rows[1][0] != "b" {
		t.Fatalf("<-> rows: %#v", rows)
	}
	consumeReadyForQuery(t, conn)

	// Bare operator ORDER BY is the pgvector ORM shape; it must work without
	// wrapping the operator in VECTOR_DISTANCE or selecting the score.
	sendSimpleQuery(conn, "SELECT id FROM operator_docs ORDER BY embedding <-> '[1,0,0]' LIMIT 2")
	msgType, payload, err = ReadMessage(conn)
	if err != nil || msgType != msgRowDescription {
		t.Fatalf("bare operator RowDescription: type=%q err=%v", msgType, err)
	}
	bareCols := decodeRowDescription(t, payload)
	if len(bareCols) != 1 || bareCols[0].Name != "id" {
		t.Fatalf("bare operator columns: %#v", bareCols)
	}
	bareRows := readDataRowsUntilComplete(t, conn)
	if len(bareRows) != 2 || bareRows[0][0] != "a" || bareRows[1][0] != "b" {
		t.Fatalf("bare <-> rows: %#v", bareRows)
	}
	consumeReadyForQuery(t, conn)

	// Extended protocol through the real pgx driver exercises typed float-array
	// parameter decoding without rewriting the SQL source.
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatalf("SplitHostPort: %v", err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatalf("Ping: %v", err)
	}
	queryRows, err := sqlDB.QueryContext(ctx, "SELECT id, embedding <#> $1 AS distance FROM operator_docs ORDER BY distance LIMIT 2", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("pgx <#> query: %v", err)
	}
	defer queryRows.Close()
	var ids []string
	var scores []float64
	for queryRows.Next() {
		var id string
		var score float64
		if err := queryRows.Scan(&id, &score); err != nil {
			t.Fatalf("scan <#>: %v", err)
		}
		ids = append(ids, id)
		scores = append(scores, score)
	}
	if err := queryRows.Err(); err != nil {
		t.Fatalf("pgx rows: %v", err)
	}
	if len(ids) != 2 || ids[0] != "c" || ids[1] != "a" || scores[0] >= scores[1] {
		t.Fatalf("<#> rows=%v scores=%v; want c,a with ascending negative inner product", ids, scores)
	}
}
