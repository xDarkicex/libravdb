package pgwire

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/libravdb"
)

// TestRRF_SimpleQuery exercises the actual PostgreSQL wire path, rather than
// only the database API or Describe. The query deliberately uses literals so
// this test validates parsing, lowering, reciprocal-rank fusion, RowDescription
// and DataRow encoding in one protocol exchange.
func TestRRF_SimpleQuery(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(
		libravdb.WithStoragePath(":memory:pgwire_rrf"),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "rrf_docs", libravdb.WithDimension(3), libravdb.WithMetadataSchema(libravdb.MetadataSchema{
		"content": libravdb.StringField,
	}))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for _, row := range []struct {
		id      string
		vector  []float32
		content string
	}{
		{id: "semantic", vector: []float32{1, 0, 0}, content: "security incident response"},
		{id: "lexical", vector: []float32{0.8, 0.2, 0}, content: "security security incident"},
		{id: "other", vector: []float32{0, 1, 0}, content: "unrelated gardening notes"},
	} {
		if err := col.Insert(ctx, row.id, row.vector, map[string]interface{}{"content": row.content}); err != nil {
			t.Fatalf("Insert %s: %v", row.id, err)
		}
	}

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	sendSimpleQuery(conn, "SELECT id, RRF(VECTOR_DISTANCE(embedding, '[1,0,0]'), FTS_RANK(content, 'security incident')) AS unified_relevance FROM rrf_docs ORDER BY unified_relevance DESC LIMIT 3")
	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("RowDescription: %v", err)
	}
	if msgType != msgRowDescription {
		t.Fatalf("first response: got %q, want RowDescription", msgType)
	}
	cols := decodeRowDescription(t, payload)
	if len(cols) != 2 || cols[0].Name != "id" || cols[1].Name != "unified_relevance" || cols[1].TypeOID != OIDFloat8 {
		t.Fatalf("RRF RowDescription: got %#v", cols)
	}
	rows := readDataRowsUntilComplete(t, conn)
	if len(rows) != 3 {
		t.Fatalf("RRF DataRows: got %d, want 3", len(rows))
	}
	if rows[2][0] != "other" {
		t.Fatalf("RRF order: last row %q, want other", rows[2][0])
	}
	consumeReadyForQuery(t, conn)
}
