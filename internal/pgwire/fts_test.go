package pgwire

import (
	"context"
	"database/sql"
	"net"
	"testing"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/xDarkicex/libravdb/libravdb"
)

func TestPostgreSQLFTS_SimpleQuery(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(":memory:pgwire_fts"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "documents", libravdb.WithMetadataOnly(), libravdb.WithMetadataSchema(libravdb.MetadataSchema{
		"content": libravdb.StringField,
	}))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for _, row := range []struct{ id, content string }{
		{"d1", "security incident response"},
		{"d2", "gardening notes"},
	} {
		if err := col.Insert(ctx, row.id, nil, map[string]interface{}{"content": row.content}); err != nil {
			t.Fatalf("Insert %s: %v", row.id, err)
		}
	}

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	sendSimpleQuery(conn, "SELECT id, ts_rank(to_tsvector(content), plainto_tsquery('security incident')) AS rank FROM documents WHERE to_tsvector(content) @@ plainto_tsquery('security incident') ORDER BY rank DESC")
	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("RowDescription: %v", err)
	}
	if msgType != msgRowDescription {
		t.Fatalf("first response: got %q, want RowDescription", msgType)
	}
	cols := decodeRowDescription(t, payload)
	if len(cols) != 2 || cols[0].Name != "id" || cols[1].Name != "rank" || cols[1].TypeOID != OIDFloat8 {
		t.Fatalf("FTS RowDescription: %#v", cols)
	}
	rows := readDataRowsUntilComplete(t, conn)
	if len(rows) != 1 || rows[0][0] != "d1" {
		t.Fatalf("FTS DataRows: %#v", rows)
	}
	consumeReadyForQuery(t, conn)
}

func TestDescribeStatement_PostgreSQLFTS(t *testing.T) {
	db := openDescribeTestDB(t, "fts")
	defer db.Close()
	query := "SELECT ts_rank(to_tsvector(title), plainto_tsquery($query)) AS rank FROM docs WHERE to_tsvector(title) @@ plainto_tsquery($query)"
	params, cols, err := describeStatement(db, query, 1)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	if len(params) != 1 || params[0] != OIDText {
		t.Fatalf("FTS parameter OIDs: got %v, want [%d]", params, OIDText)
	}
	if len(cols) != 1 || cols[0].Name != "rank" || cols[0].TypeOID != OIDFloat8 {
		t.Fatalf("FTS columns: got %#v", cols)
	}
	params, _, err = describeStatement(db, "SELECT ts_rank(to_tsvector(title), plainto_tsquery($query), 2) AS rank FROM docs", 1)
	if err != nil {
		t.Fatalf("describe normalized FTS: %v", err)
	}
	if len(params) != 1 || params[0] != OIDText {
		t.Fatalf("normalized FTS parameter OIDs: got %v, want [%d]", params, OIDText)
	}
}

func TestPGWireSQLFTSParameterized(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/pgwire_fts_driver"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "documents", libravdb.WithMetadataOnly(), libravdb.WithMetadataSchema(libravdb.MetadataSchema{
		"content": libravdb.StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range []struct{ id, content string }{
		{"d1", "security incident response"},
		{"d2", "gardening notes"},
	} {
		if err := col.Insert(ctx, row.id, nil, map[string]interface{}{"content": row.content}); err != nil {
			t.Fatal(err)
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
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	rows, err := sqlDB.QueryContext(ctx, "SELECT id, ts_rank(to_tsvector(content), plainto_tsquery($1)) AS rank FROM documents WHERE to_tsvector(content) @@ plainto_tsquery($1) ORDER BY rank DESC", "security incident")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	if !rows.Next() {
		t.Fatal("parameterized FTS returned no rows")
	}
	var id string
	var rank float64
	if err := rows.Scan(&id, &rank); err != nil {
		t.Fatal(err)
	}
	if id != "d1" || rank <= 0 {
		t.Fatalf("parameterized FTS row id=%q rank=%v", id, rank)
	}
	if rows.Next() {
		t.Fatal("parameterized FTS returned an unrelated row")
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
}
