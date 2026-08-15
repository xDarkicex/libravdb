package pgwire

import (
	"context"
	"encoding/binary"
	"net"
	"testing"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/libravdb"
)

// openDescribeTestDB opens an in-memory database with a "docs" collection whose
// catalog schema declares title (string), score (float), and rating (int)
// columns, so Describe can resolve real column types. A dimension is used so
// the relational executor can actually return rows in wire tests (a
// metadata-only collection has no index and returns an empty scan).
func openDescribeTestDB(t *testing.T, name string) *libravdb.Database {
	t.Helper()
	db, err := libravdb.Open(
		libravdb.WithStoragePath(":memory:describe_"+name),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	ctx := context.Background()
	if _, err := db.CreateCollection(ctx, "docs",
		libravdb.WithDimension(3),
		libravdb.WithMetadataSchema(libravdb.MetadataSchema{
			"title":  libravdb.StringField,
			"score":  libravdb.FloatField,
			"rating": libravdb.IntField,
		}),
	); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	return db
}

func TestInferParamOIDs_EdgeWeight(t *testing.T) {
	const query = "SELECT id FROM docs d WHERE MATCH (d)-[r:RELATES WHERE r.weight > $threshold]->(target)"
	src := []byte(query)
	doc := &parser.QueryDoc{}
	if err := parser.Parse(src, doc); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	oids := inferParamOIDs(doc, src, nil, nil, 1)
	if len(oids) != 1 || oids[0] != OIDFloat8 {
		t.Fatalf("edge weight parameter OIDs: want [%d], got %v", OIDFloat8, oids)
	}
	db := openDescribeTestDB(t, "edge_weight_oid")
	defer db.Close()
	described, _, err := describeStatement(db, query, 1)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	if len(described) != 1 || described[0] != OIDFloat8 {
		t.Fatalf("described edge weight parameter OIDs: want [%d], got %v", OIDFloat8, described)
	}
}

func TestInferParamOIDs_EdgePropertyBlock(t *testing.T) {
	const query = "SELECT id FROM docs d WHERE MATCH (d)-[r:RELATES {weight > $threshold, type: 'RELATES'}]->(target)"
	src := []byte(query)
	doc := &parser.QueryDoc{}
	if err := parser.Parse(src, doc); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	oids := inferParamOIDs(doc, src, nil, nil, 1)
	if len(oids) != 1 || oids[0] != OIDFloat8 {
		t.Fatalf("edge property-block parameter OIDs: want [%d], got %v", OIDFloat8, oids)
	}
	db := openDescribeTestDB(t, "edge_property_block_oid")
	defer db.Close()
	described, _, err := describeStatement(db, query, 1)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	if len(described) != 1 || described[0] != OIDFloat8 {
		t.Fatalf("described edge property-block parameter OIDs: want [%d], got %v", OIDFloat8, described)
	}
}

func TestInferParamOIDs_ArbitraryEdgeProperty(t *testing.T) {
	const query = "SELECT id FROM docs d WHERE MATCH (d)-[r:RELATES {cost >= $threshold}]->(target)"
	src := []byte(query)
	doc := &parser.QueryDoc{}
	if err := parser.Parse(src, doc); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	oids := inferParamOIDs(doc, src, nil, nil, 1)
	if len(oids) != 1 || oids[0] != OIDFloat8 {
		t.Fatalf("arbitrary edge property parameter OIDs: want [%d], got %v", OIDFloat8, oids)
	}
}

func TestDescribeAsyncpgTypeInfoBindsOIDArray(t *testing.T) {
	db := openDescribeTestDB(t, "asyncpg_typeinfo_oid_array")
	defer db.Close()
	query := `WITH RECURSIVE typeinfo_tree AS (
		SELECT typelem FROM pg_catalog.pg_type WHERE oid = any($1::oid[])
	) SELECT * FROM typeinfo_tree`
	paramOIDs, _, err := describeStatement(db, query, 1)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	if len(paramOIDs) != 1 || paramOIDs[0] != OIDOIDArray {
		t.Fatalf("asyncpg type-info parameter OIDs: want [%d], got %v", OIDOIDArray, paramOIDs)
	}
}

func TestDescribeAsyncpgJITRestoreBindsText(t *testing.T) {
	db := openDescribeTestDB(t, "asyncpg_jit_restore")
	defer db.Close()
	query := "SELECT current_setting('jit') AS cur, set_config('jit', $1, false) AS new"
	paramOIDs, columns, err := describeStatement(db, query, 1)
	if err != nil {
		t.Fatalf("describe asyncpg JIT restore: %v", err)
	}
	if len(paramOIDs) != 1 || paramOIDs[0] != OIDText {
		t.Fatalf("asyncpg JIT restore parameter OIDs: got %v, want [%d]", paramOIDs, OIDText)
	}
	assertColumns(t, columns, []ColumnMeta{{Name: "cur", TypeOID: OIDText}, {Name: "new", TypeOID: OIDText}})
}

func TestInferParamOIDsVectorInsertAlias(t *testing.T) {
	db := openDescribeTestDB(t, "vector_insert_alias_oid")
	defer db.Close()
	query := "INSERT INTO docs (id, embedding, rating) VALUES ($1, $2::vector, $3)"
	paramOIDs, _, err := describeStatement(db, query, 3)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	want := []uint32{OIDText, OIDVector, OIDInt4}
	if len(paramOIDs) != len(want) {
		t.Fatalf("vector insert parameter OIDs: got %v, want %v", paramOIDs, want)
	}
	for i := range want {
		if paramOIDs[i] != want[i] {
			t.Fatalf("vector insert parameter OIDs: got %v, want %v", paramOIDs, want)
		}
	}
}

func TestInferParamOIDsVectorOperatorExplicitCast(t *testing.T) {
	db := openDescribeTestDB(t, "vector_operator_cast_oid")
	defer db.Close()
	query := "SELECT id, embedding, embedding <-> $1::vector AS distance FROM docs ORDER BY distance, id LIMIT 128"
	paramOIDs, columns, err := describeStatement(db, query, 1)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	if len(paramOIDs) != 1 || paramOIDs[0] != OIDVector {
		t.Fatalf("vector operator parameter OIDs: got %v, want [%d]", paramOIDs, OIDVector)
	}
	assertColumns(t, columns, []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "embedding", TypeOID: OIDText},
		{Name: "distance", TypeOID: OIDFloat8},
	})
}

// =============================================================================
// Unit tests: describeStatement
// =============================================================================

func TestDescribeStatement_ColumnTypes(t *testing.T) {
	db := openDescribeTestDB(t, "columns")
	defer db.Close()

	query := "SELECT id, title, score, rating FROM docs"
	paramOIDs, cols, err := describeStatement(db, query, countParams(query))
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	if len(paramOIDs) != 0 {
		t.Errorf("paramOIDs: want 0, got %v", paramOIDs)
	}
	want := []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "title", TypeOID: OIDText},
		{Name: "score", TypeOID: OIDFloat8},
		{Name: "rating", TypeOID: OIDInt4},
	}
	assertColumns(t, cols, want)
}

func TestDescribeStatement_AliasProjection(t *testing.T) {
	db := openDescribeTestDB(t, "alias")
	defer db.Close()

	query := "SELECT title AS name, score FROM docs"
	_, cols, err := describeStatement(db, query, 0)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	want := []ColumnMeta{
		{Name: "name", TypeOID: OIDText},
		{Name: "score", TypeOID: OIDFloat8},
	}
	assertColumns(t, cols, want)
}

func TestDescribeStatement_VectorFuncProjection(t *testing.T) {
	db, err := libravdb.Open(
		libravdb.WithStoragePath(":memory:describe_vec"),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	ctx := context.Background()
	if _, err := db.CreateCollection(ctx, "vecs",
		libravdb.WithDimension(3),
		libravdb.WithMetadataSchema(libravdb.MetadataSchema{"title": libravdb.StringField}),
	); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	query := "SELECT title, SIMILARITY(embedding, '[1,0,0]') AS sim FROM vecs"
	_, cols, err := describeStatement(db, query, 0)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	want := []ColumnMeta{
		{Name: "title", TypeOID: OIDText},
		{Name: "sim", TypeOID: OIDFloat8},
	}
	assertColumns(t, cols, want)
}

func TestDescribeStatement_Aggregate(t *testing.T) {
	db := openDescribeTestDB(t, "agg")
	defer db.Close()

	tests := []struct {
		query string
		want  ColumnMeta
	}{
		{query: "SELECT COUNT(*) FROM docs", want: ColumnMeta{Name: "count", TypeOID: OIDInt8}},
		{query: "SELECT SUM(rating) FROM docs", want: ColumnMeta{Name: "sum", TypeOID: OIDFloat8}},
		{query: "SELECT AVG(score) FROM docs", want: ColumnMeta{Name: "avg", TypeOID: OIDFloat8}},
		// MIN/MAX inherit the aggregate column's type from the catalog.
		{query: "SELECT MIN(rating) FROM docs", want: ColumnMeta{Name: "min", TypeOID: OIDInt4}},
		{query: "SELECT MAX(score) FROM docs", want: ColumnMeta{Name: "max", TypeOID: OIDFloat8}},
		{query: "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) FROM docs", want: ColumnMeta{Name: "percentile_cont", TypeOID: OIDFloat8}},
		{query: "SELECT PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY rating) FROM docs", want: ColumnMeta{Name: "percentile_disc", TypeOID: OIDInt4}},
		{query: "SELECT MODE() WITHIN GROUP (ORDER BY title) FROM docs", want: ColumnMeta{Name: "mode", TypeOID: OIDText}},
	}
	for _, tt := range tests {
		t.Run(tt.query, func(t *testing.T) {
			_, cols, err := describeStatement(db, tt.query, 0)
			if err != nil {
				t.Fatalf("describeStatement: %v", err)
			}
			assertColumns(t, cols, []ColumnMeta{tt.want})
		})
	}
}

func TestDescribeStatement_SelectStar(t *testing.T) {
	db := openDescribeTestDB(t, "star")
	defer db.Close()

	// SELECT * expands to the relational collection schema. This is required
	// for database/sql clients such as GORM, which scan every model field.
	_, cols, err := describeStatement(db, "SELECT * FROM docs", 0)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	assertColumns(t, cols, []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "rating", TypeOID: OIDInt4},
		{Name: "score", TypeOID: OIDFloat8},
		{Name: "title", TypeOID: OIDText},
	})
}

func TestDescribeStatement_SelectStarWithColumn(t *testing.T) {
	db := openDescribeTestDB(t, "starmix")
	defer db.Close()

	// The optimizer drops the star and keeps the explicit column, so the
	// description reports only the explicit column.
	_, cols, err := describeStatement(db, "SELECT *, title FROM docs", 0)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	assertColumns(t, cols, []ColumnMeta{{Name: "title", TypeOID: OIDText}})
}

func TestDescribeStatement_WindowSelectStarUsesSchemaTypes(t *testing.T) {
	db := openDescribeTestDB(t, "window_star")
	defer db.Close()

	_, cols, err := describeStatement(db, "SELECT *, ROW_NUMBER() OVER (PARTITION BY title ORDER BY rating) AS rn FROM docs", 0)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	// The virtual window executor expands SELECT * in deterministic sorted
	// order, then appends the window projection. Types must come from the
	// collection schema rather than name-only fallbacks.
	assertColumns(t, cols, []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "rating", TypeOID: OIDInt4},
		{Name: "score", TypeOID: OIDFloat8},
		{Name: "title", TypeOID: OIDText},
		{Name: "rn", TypeOID: OIDInt8},
	})
}

func TestDescribeStatement_MixedAggregateWindow(t *testing.T) {
	db := openDescribeTestDB(t, "window_aggregate")
	defer db.Close()

	_, cols, err := describeStatement(db, "SELECT title, COUNT(*) AS cnt, ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rn FROM docs GROUP BY title", 0)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	assertColumns(t, cols, []ColumnMeta{
		{Name: "title", TypeOID: OIDText},
		{Name: "cnt", TypeOID: OIDInt8},
		{Name: "rn", TypeOID: OIDInt8},
	})
}

func TestDescribeStatement_NonRowProducing(t *testing.T) {
	db := openDescribeTestDB(t, "norows")
	defer db.Close()

	// Transaction controls and DML produce no rows → empty columns (NoData).
	for _, q := range []string{"BEGIN EPOCH TRANSACTION", "COMMIT", "ROLLBACK"} {
		_, cols, err := describeStatement(db, q, 0)
		if err != nil {
			t.Fatalf("describeStatement(%q): %v", q, err)
		}
		if len(cols) != 0 {
			t.Errorf("describeStatement(%q): want 0 columns, got %d", q, len(cols))
		}
	}
}

func TestDescribeStatement_DMLReturning(t *testing.T) {
	db := openDescribeTestDB(t, "dml_returning")
	defer db.Close()

	paramOIDs, columns, err := describeStatement(db, "INSERT INTO docs (id, title) VALUES ($1, $2) RETURNING id", 2)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	if len(paramOIDs) != 2 || len(columns) != 1 || columns[0].Name != "id" {
		t.Fatalf("DML RETURNING shape: params=%v columns=%v", paramOIDs, columns)
	}
}

func TestDescribeStatement_NonReturningDMLParameterOIDs(t *testing.T) {
	db := openDescribeTestDB(t, "dml_params")
	defer db.Close()

	paramOIDs, columns, err := describeStatement(db, "INSERT INTO docs (id, title) VALUES ($1, $2)", 2)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	want := []uint32{OIDText, OIDText}
	if len(paramOIDs) != len(want) {
		t.Fatalf("paramOIDs: want %v, got %v", want, paramOIDs)
	}
	for i := range want {
		if paramOIDs[i] != want[i] {
			t.Errorf("paramOIDs[%d]: want %d, got %d", i, want[i], paramOIDs[i])
		}
	}
	if len(columns) != 0 {
		t.Fatalf("columns: want NoData, got %v", columns)
	}
}

func TestDescribeStatement_ParameterOIDs(t *testing.T) {
	db := openDescribeTestDB(t, "params")
	defer db.Close()

	query := "SELECT * FROM docs WHERE title = $1 AND score > $2"
	paramOIDs, _, err := describeStatement(db, query, countParams(query))
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	want := []uint32{OIDText, OIDFloat8}
	if len(paramOIDs) != len(want) {
		t.Fatalf("paramOIDs: want %v, got %v", want, paramOIDs)
	}
	for i := range want {
		if paramOIDs[i] != want[i] {
			t.Errorf("paramOIDs[%d]: want %d, got %d", i, want[i], paramOIDs[i])
		}
	}
}

func TestDescribeStatement_UninferableParam(t *testing.T) {
	db := openDescribeTestDB(t, "uninf")
	defer db.Close()

	// $1 in a bare boolean position has no comparison context → OID 0
	// (clients send it as text).
	query := "SELECT title FROM docs WHERE $1"
	paramOIDs, _, err := describeStatement(db, query, countParams(query))
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	if len(paramOIDs) != 1 || paramOIDs[0] != 0 {
		t.Errorf("paramOIDs: want [0], got %v", paramOIDs)
	}
}

func TestDescribeStatement_NamedAtParam(t *testing.T) {
	db, err := libravdb.Open(
		libravdb.WithStoragePath(":memory:describe_at"),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	ctx := context.Background()
	if _, err := db.CreateCollection(ctx, "vecs", libravdb.WithDimension(3)); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	query := "SELECT VECTOR_DISTANCE(embedding, @query_vec) FROM vecs"
	paramOIDs, _, err := describeStatement(db, query, countParams(query))
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	// Vector operands use the native one-dimensional float4-array type.
	if len(paramOIDs) != 1 || paramOIDs[0] != OIDFloat4Array {
		t.Errorf("paramOIDs: want [%d], got %v", OIDFloat4Array, paramOIDs)
	}
}

func TestDescribeStatement_RRFProjection(t *testing.T) {
	db := openDescribeTestDB(t, "rrf")
	defer db.Close()
	query := "SELECT RRF(VECTOR_DISTANCE(embedding, $query_vec), FTS_RANK(title, $text_query)) AS unified_relevance FROM docs"
	paramOIDs, cols, err := describeStatement(db, query, countParams(query))
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	if len(paramOIDs) != 2 || paramOIDs[0] != OIDFloat4Array || paramOIDs[1] != OIDText {
		t.Fatalf("RRF parameter OIDs: got %v, want [%d %d]", paramOIDs, OIDFloat4Array, OIDText)
	}
	if len(cols) != 1 || cols[0].Name != "unified_relevance" || cols[0].TypeOID != OIDFloat8 {
		t.Fatalf("RRF columns: got %#v, want unified_relevance/float8", cols)
	}
}

func TestDescribeStatement_ComputeLeiden(t *testing.T) {
	db := openDescribeTestDB(t, "leiden")
	defer db.Close()

	query := "COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(t)"
	_, cols, err := describeStatement(db, query, 0)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	if len(cols) != 7 {
		t.Fatalf("COMPUTE LEIDEN: want 7 columns, got %d", len(cols))
	}
	if cols[0].Name != "node_id" || cols[0].TypeOID != OIDInt8 {
		t.Errorf("col[0]: want node_id/OIDInt8, got %q/%d", cols[0].Name, cols[0].TypeOID)
	}
	if cols[4].Name != "truncated" || cols[4].TypeOID != OIDBool {
		t.Errorf("col[4]: want truncated/OIDBool, got %q/%d", cols[4].Name, cols[4].TypeOID)
	}
}

func TestDescribeStatement_BindError(t *testing.T) {
	db := openDescribeTestDB(t, "binderr")
	defer db.Close()

	// A column the catalog does not know cannot execute, so Describe reports
	// the failure instead of a guessed RowDescription.
	_, _, err := describeStatement(db, "SELECT nonexistent FROM docs", 0)
	if err == nil {
		t.Fatal("describeStatement: want bind error, got nil")
	}
}

func TestDescribeStatement_PgCatalogTypes(t *testing.T) {
	db := openDescribeTestDB(t, "catalogtypes")
	defer db.Close()

	tests := []struct {
		query string
		want  []ColumnMeta
	}{
		{
			query: "SELECT oid, relname, relnamespace, relkind, reltuples FROM pg_class",
			want: []ColumnMeta{
				{Name: "oid", TypeOID: OIDOID},
				{Name: "relname", TypeOID: OIDName},
				{Name: "relnamespace", TypeOID: OIDOID},
				{Name: "relkind", TypeOID: OIDChar},
				{Name: "reltuples", TypeOID: OIDFloat4},
			},
		},
		{
			query: "SELECT attrelid, attname, atttypid, attnum, attnotnull FROM pg_attribute",
			want: []ColumnMeta{
				{Name: "attrelid", TypeOID: OIDOID},
				{Name: "attname", TypeOID: OIDName},
				{Name: "atttypid", TypeOID: OIDOID},
				{Name: "attnum", TypeOID: OIDInt2},
				{Name: "attnotnull", TypeOID: OIDBool},
			},
		},
		{
			query: "SELECT oid, typname, typlen FROM pg_type",
			want: []ColumnMeta{
				{Name: "oid", TypeOID: OIDOID},
				{Name: "typname", TypeOID: OIDName},
				{Name: "typlen", TypeOID: OIDInt2},
			},
		},
		{
			query: "SELECT oid, nspname, nspowner FROM pg_namespace",
			want: []ColumnMeta{
				{Name: "oid", TypeOID: OIDOID},
				{Name: "nspname", TypeOID: OIDName},
				{Name: "nspowner", TypeOID: OIDOID},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.query, func(t *testing.T) {
			_, cols, err := describeStatement(db, tt.query, 0)
			if err != nil {
				t.Fatalf("describeStatement: %v", err)
			}
			assertColumns(t, cols, tt.want)
		})
	}
}

func TestDescribeStatement_PgCatalogAggregateType(t *testing.T) {
	db := openDescribeTestDB(t, "catalogagg")
	defer db.Close()

	query := "SELECT MIN(relname) FROM pg_class"
	_, described, err := describeStatement(db, query, 0)
	if err != nil {
		t.Fatalf("describeStatement: %v", err)
	}
	assertColumns(t, described, []ColumnMeta{{Name: "min", TypeOID: OIDName}})

	results, err := db.Query(context.Background(), query)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	assertColumns(t, inferColumns(results), []ColumnMeta{{Name: "min", TypeOID: OIDName}})
}

func assertColumns(t *testing.T, got, want []ColumnMeta) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("columns: want %d (%+v), got %d (%+v)", len(want), want, len(got), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("column[%d]: want %+v, got %+v", i, want[i], got[i])
		}
	}
}

// =============================================================================
// Integration tests: Describe over the wire protocol
// =============================================================================

func TestDescribeStatement_OverWire(t *testing.T) {
	db := openDescribeTestDB(t, "wire")
	defer db.Close()

	ctx := context.Background()
	col, _ := db.GetCollection("docs")
	col.Insert(ctx, "d1", []float32{1, 0, 0}, map[string]interface{}{"title": "hello", "score": 1.5, "rating": 5})

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	// Parse a named prepared statement.
	sendParse(t, conn, "s1", "SELECT id, title, score FROM docs", nil)
	assertMessageType(t, conn, msgParseComplete, "ParseComplete")

	// Describe statement: ParameterDescription (0 params) then RowDescription.
	sendDescribe(t, conn, 'S', "s1")
	assertMessageType(t, conn, msgParameterDescription, "ParameterDescription")

	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("Describe(S) RowDescription: %v", err)
	}
	if msgType != msgRowDescription {
		t.Fatalf("Describe(S): want RowDescription (T), got %c", msgType)
	}
	cols := decodeRowDescription(t, payload)
	want := []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "title", TypeOID: OIDText},
		{Name: "score", TypeOID: OIDFloat8},
	}
	assertColumns(t, cols, want)

	// Bind and describe the portal — same RowDescription, no ParameterDescription.
	sendBind(t, conn, "p1", "s1", nil, nil, 0)
	assertMessageType(t, conn, msgBindComplete, "BindComplete")

	sendDescribe(t, conn, 'P', "p1")
	msgType, payload, err = ReadMessage(conn)
	if err != nil {
		t.Fatalf("Describe(P) RowDescription: %v", err)
	}
	if msgType != msgRowDescription {
		t.Fatalf("Describe(P): want RowDescription (T), got %c", msgType)
	}
	assertColumns(t, decodeRowDescription(t, payload), want)

	// Execute: the executor sends its own RowDescription + one DataRow.
	sendExecute(t, conn, "p1", 0)
	rows := readDataRowsUntilComplete(t, conn)
	if len(rows) != 1 {
		t.Fatalf("Execute: want 1 row, got %d", len(rows))
	}
	if rows[0][1] != "hello" {
		t.Errorf("row title: want hello, got %q", rows[0][1])
	}

	sendSync(t, conn)
	assertMessageType(t, conn, msgReadyForQuery, "ReadyForQuery")
}

func TestDescribeParameterOIDs_OverWire(t *testing.T) {
	db := openDescribeTestDB(t, "wireparams")
	defer db.Close()

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	sendParse(t, conn, "s1", "SELECT * FROM docs WHERE title = $1 AND score > $2", nil)
	assertMessageType(t, conn, msgParseComplete, "ParseComplete")

	sendDescribe(t, conn, 'S', "s1")

	// ParameterDescription with inferred OIDs: title→text, score→float8.
	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("Describe(S) ParameterDescription: %v", err)
	}
	if msgType != msgParameterDescription {
		t.Fatalf("want ParameterDescription (t), got %c", msgType)
	}
	oids := decodeParameterDescription(t, payload)
	want := []uint32{OIDText, OIDFloat8}
	if len(oids) != len(want) {
		t.Fatalf("param OIDs: want %v, got %v", want, oids)
	}
	for i := range want {
		if oids[i] != want[i] {
			t.Errorf("oid[%d]: want %d, got %d", i, want[i], oids[i])
		}
	}

	// RowDescription for SELECT * → default id/score.
	msgType, _, err = ReadMessage(conn)
	if err != nil {
		t.Fatalf("Describe(S) RowDescription: %v", err)
	}
	if msgType != msgRowDescription {
		t.Fatalf("want RowDescription (T), got %c", msgType)
	}

	sendSync(t, conn)
	assertMessageType(t, conn, msgReadyForQuery, "ReadyForQuery")
}

func TestDescribeNoData_OverWire(t *testing.T) {
	db := openDescribeTestDB(t, "wirenodata")
	defer db.Close()

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	// Non-row-producing statement → ParameterDescription(0) + NoData.
	sendParse(t, conn, "s1", "BEGIN EPOCH TRANSACTION", nil)
	assertMessageType(t, conn, msgParseComplete, "ParseComplete")

	sendDescribe(t, conn, 'S', "s1")
	assertMessageType(t, conn, msgParameterDescription, "ParameterDescription")
	assertMessageType(t, conn, msgNoData, "NoData for statement")

	// A bound portal for the same statement also describes as NoData.
	sendBind(t, conn, "p1", "s1", nil, nil, 0)
	assertMessageType(t, conn, msgBindComplete, "BindComplete")
	sendDescribe(t, conn, 'P', "p1")
	assertMessageType(t, conn, msgNoData, "NoData for portal")

	sendSync(t, conn)
	assertMessageType(t, conn, msgReadyForQuery, "ReadyForQuery")
}

func TestDescribeThenExecuteWithParams_OverWire(t *testing.T) {
	db := openDescribeTestDB(t, "wireexe")
	defer db.Close()

	ctx := context.Background()
	col, _ := db.GetCollection("docs")
	col.Insert(ctx, "d1", []float32{1, 0, 0}, map[string]interface{}{"title": "hello", "score": 1.5, "rating": 5})

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	// Parse a parameterized statement.
	sendParse(t, conn, "s1", "SELECT * FROM docs WHERE title = $1", nil)
	assertMessageType(t, conn, msgParseComplete, "ParseComplete")

	// Describe: the parameter is inferred as text from the title comparison.
	sendDescribe(t, conn, 'S', "s1")
	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("Describe(S) ParameterDescription: %v", err)
	}
	if msgType != msgParameterDescription {
		t.Fatalf("want ParameterDescription (t), got %c", msgType)
	}
	if oids := decodeParameterDescription(t, payload); len(oids) != 1 || oids[0] != OIDText {
		t.Fatalf("param OIDs: want [%d], got %v", OIDText, oids)
	}
	msgType, _, err = ReadMessage(conn)
	if err != nil {
		t.Fatalf("Describe(S) RowDescription: %v", err)
	}
	if msgType != msgRowDescription {
		t.Fatalf("want RowDescription (T), got %c", msgType)
	}

	// Bind with the parameter value, then execute through native QueryParams.
	sendBindParams(t, conn, "p1", "s1", [][]byte{[]byte("hello")})
	assertMessageType(t, conn, msgBindComplete, "BindComplete")

	sendExecute(t, conn, "p1", 0)
	rows := readDataRowsUntilComplete(t, conn)
	if len(rows) != 1 {
		t.Fatalf("Execute: want 1 row, got %d", len(rows))
	}

	sendSync(t, conn)
	assertMessageType(t, conn, msgReadyForQuery, "ReadyForQuery")
}

func TestDescribeUnknownStatement_OverWire(t *testing.T) {
	db := openDescribeTestDB(t, "wireunknown")
	defer db.Close()

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	// Describing a statement that was never parsed is an error.
	sendDescribe(t, conn, 'S', "ghost")
	assertMessageType(t, conn, msgErrorResponse, "ErrorResponse")

	sendSync(t, conn)
	assertMessageType(t, conn, msgReadyForQuery, "ReadyForQuery")
}

func TestDescribeAndExecutePgCatalogPrefix_OverWire(t *testing.T) {
	db := openDescribeTestDB(t, "wirepgcatalog")
	defer db.Close()

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	query := "SELECT oid, relname, relnamespace, relkind, reltuples FROM pg_catalog.pg_class WHERE relname = 'docs'"
	sendParse(t, conn, "s1", query, nil)
	assertMessageType(t, conn, msgParseComplete, "ParseComplete")

	// Describe must apply the same pg_catalog rewrite as execution.
	sendDescribe(t, conn, 'S', "s1")
	assertMessageType(t, conn, msgParameterDescription, "ParameterDescription")
	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("Describe(S): %v", err)
	}
	if msgType != msgRowDescription {
		t.Fatalf("Describe(S): want RowDescription (T), got %c", msgType)
	}
	assertColumns(t, decodeRowDescription(t, payload), []ColumnMeta{
		{Name: "oid", TypeOID: OIDOID},
		{Name: "relname", TypeOID: OIDName},
		{Name: "relnamespace", TypeOID: OIDOID},
		{Name: "relkind", TypeOID: OIDChar},
		{Name: "reltuples", TypeOID: OIDFloat4},
	})

	sendBind(t, conn, "p1", "s1", nil, nil, 0)
	assertMessageType(t, conn, msgBindComplete, "BindComplete")
	sendExecute(t, conn, "p1", 0)
	rows := readDataRowsUntilComplete(t, conn)
	if len(rows) != 1 || len(rows[0]) != 5 || rows[0][1] != "docs" {
		t.Fatalf("Execute pg_catalog query: want one docs row, got %v", rows)
	}

	sendSync(t, conn)
	assertMessageType(t, conn, msgReadyForQuery, "ReadyForQuery")
}

// =============================================================================
// Wire helpers
// =============================================================================

func decodeParameterDescription(t *testing.T, payload []byte) []uint32 {
	t.Helper()
	if len(payload) < 2 {
		t.Fatal("ParameterDescription too short")
	}
	n := int(binary.BigEndian.Uint16(payload[:2]))
	if len(payload) < 2+4*n {
		t.Fatalf("ParameterDescription truncated: want %d bytes, have %d", 2+4*n, len(payload))
	}
	oids := make([]uint32, n)
	for i := 0; i < n; i++ {
		oids[i] = binary.BigEndian.Uint32(payload[2+4*i:])
	}
	return oids
}

// sendBindParams binds a statement with actual parameter values (all text
// format), unlike the zero-parameter sendBind helper.
func sendBindParams(t *testing.T, conn net.Conn, portal, stmtName string, params [][]byte) {
	t.Helper()
	var buf []byte
	buf = append(buf, portal...)
	buf = append(buf, 0)
	buf = append(buf, stmtName...)
	buf = append(buf, 0)
	// Number of param format codes (0 = all text).
	buf = append(buf, 0, 0)
	// Number of param values.
	buf = append(buf, byte(len(params)>>8), byte(len(params)))
	for _, p := range params {
		off := len(buf)
		buf = append(buf, 0, 0, 0, 0)
		if p == nil {
			binary.BigEndian.PutUint32(buf[off:], 0xFFFFFFFF) // NULL
			continue
		}
		binary.BigEndian.PutUint32(buf[off:], uint32(len(p)))
		buf = append(buf, p...)
	}
	// Number of result format codes (0 = all text).
	buf = append(buf, 0, 0)
	if err := WriteMessage(conn, msgBind, buf); err != nil {
		t.Fatalf("sendBindParams: %v", err)
	}
}
