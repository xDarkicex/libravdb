package libravdb

import (
	"context"
	"math"
	"strconv"
	"testing"
)

func metadataFloat(v interface{}) (float64, bool) {
	s, ok := v.(string)
	if !ok {
		return 0, false
	}
	f, err := strconv.ParseFloat(s, 64)
	return f, err == nil
}

func TestSQLLiteralEscapesAndScientificNumbers(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_literal_numeric"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE literal_rows (id TEXT PRIMARY KEY, note TEXT, amount FLOAT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO literal_rows (id, note, amount) VALUES ('row1', 'foo''bar', -5.0e-3)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO literal_rows (id, note, amount) VALUES ('row2', 'foo\\'bar', 1.5E+2)"); err != nil {
		t.Fatal(err)
	}
	col, err := db.GetCollection("literal_rows")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		id   string
		note string
		want float64
	}{
		{id: "row1", note: "foo'bar", want: -5e-3},
		{id: "row2", note: "foo'bar", want: 1.5e2},
	} {
		record, err := col.Get(ctx, tc.id)
		if err != nil {
			t.Fatal(err)
		}
		if record.Metadata["note"] != tc.note {
			t.Fatalf("%s note=%#v want %q", tc.id, record.Metadata["note"], tc.note)
		}
		got, ok := metadataFloat(record.Metadata["amount"])
		if !ok || math.Abs(got-tc.want) > 1e-12 {
			t.Fatalf("%s amount=%#v parsed=%v want %v", tc.id, record.Metadata["amount"], got, tc.want)
		}
	}
}

func TestSQLQuotedReservedIdentifiersAndUpsert(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_quoted_identifiers"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, `CREATE TABLE "order" ("key" TEXT PRIMARY KEY, "value" BIGINT NOT NULL, "select" TEXT)`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `INSERT INTO "order" ("key", "value", "select") VALUES ('a-1', 10, 'SELECT')`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `INSERT INTO "order" ("key", "value", "select") VALUES ('a-1', 10, 'FROM') ON CONFLICT ("key") DO UPDATE SET "value" = "order"."value" + EXCLUDED."value"`); err != nil {
		t.Fatal(err)
	}
	col, err := db.GetCollection("order")
	if err != nil {
		t.Fatal(err)
	}
	record, err := col.Get(ctx, "__pk:3:key3:a-1|")
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata["value"] != "20" || record.Metadata["select"] != "SELECT" {
		t.Fatalf("quoted upsert metadata=%#v", record.Metadata)
	}
}

func TestSQLCommentsAcrossUpsert(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_comments"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE comment_rows (id TEXT PRIMARY KEY, value BIGINT)"); err != nil {
		t.Fatal(err)
	}
	query := `-- leading comment
/* block comment with 'quotes' */
INSERT INTO comment_rows (id, value) -- between clauses
VALUES ('c1', 1) /* inline */
ON CONFLICT (id) DO UPDATE
SET value = comment_rows.value + EXCLUDED.value;`
	if _, err := db.Query(ctx, query); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, query); err != nil {
		t.Fatal(err)
	}
	col, err := db.GetCollection("comment_rows")
	if err != nil {
		t.Fatal(err)
	}
	record, err := col.Get(ctx, "c1")
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata["value"] != "2" {
		t.Fatalf("comment query value=%#v", record.Metadata["value"])
	}
}

func TestSQLMultiRowNumericLiterals(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_numeric_matrix"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE numeric_rows (id TEXT PRIMARY KEY, value FLOAT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO numeric_rows (id, value) VALUES ('int_lit', 123), ('float_lit', 123.45), ('sci_lit', 1.23e10), ('neg_lit', -42)"); err != nil {
		t.Fatal(err)
	}
	col, err := db.GetCollection("numeric_rows")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		id   string
		want float64
	}{
		{id: "int_lit", want: 123},
		{id: "float_lit", want: 123.45},
		{id: "sci_lit", want: 1.23e10},
		{id: "neg_lit", want: -42},
	} {
		record, err := col.Get(ctx, tc.id)
		if err != nil {
			t.Fatal(err)
		}
		got, ok := metadataFloat(record.Metadata["value"])
		if !ok || math.Abs(got-tc.want) > math.Max(1e-12, math.Abs(tc.want)*1e-12) {
			t.Fatalf("%s value=%#v parsed=%v want=%v", tc.id, record.Metadata["value"], got, tc.want)
		}
	}
}

func TestSQLEscapeStringLiteral(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_escape_string"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE escape_rows (id TEXT PRIMARY KEY, note TEXT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `INSERT INTO escape_rows (id, note) VALUES ('e1', E'path\nwith\ttabs')`); err != nil {
		t.Fatal(err)
	}
	col, err := db.GetCollection("escape_rows")
	if err != nil {
		t.Fatal(err)
	}
	record, err := col.Get(ctx, "e1")
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata["note"] != "path\nwith\ttabs" {
		t.Fatalf("escape string=%#v", record.Metadata["note"])
	}
	results, err := db.Query(ctx, "SELECT id, note FROM escape_rows WHERE id = 'e1'")
	if err != nil {
		t.Fatal(err)
	}
	if results.Total != 1 {
		t.Fatalf("filtered escape query total=%d", results.Total)
	}
}

func TestSQLConflictScalarExpressions(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_conflict_scalar"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE scalar_rows (id TEXT PRIMARY KEY, value BIGINT, marker TEXT, updated_at TEXT, ratio FLOAT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO scalar_rows (id, value, marker) VALUES ('r1', 10, 'same')"); err != nil {
		t.Fatal(err)
	}
	query := `INSERT INTO scalar_rows (id, value, marker, updated_at, ratio)
VALUES ('r1', 20, 'same', 'ignored', 0)
ON CONFLICT (id) DO UPDATE SET
  value = CASE WHEN scalar_rows.value < 100 THEN scalar_rows.value + EXCLUDED.value ELSE scalar_rows.value END,
  marker = NULLIF(scalar_rows.marker, EXCLUDED.marker),
  updated_at = NOW(),
  ratio = (scalar_rows.value + EXCLUDED.value)::float`
	if _, err := db.Query(ctx, query); err != nil {
		t.Fatal(err)
	}
	col, err := db.GetCollection("scalar_rows")
	if err != nil {
		t.Fatal(err)
	}
	record, err := col.Get(ctx, "r1")
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata["value"] != "30" {
		t.Fatalf("CASE value=%#v, want 30", record.Metadata["value"])
	}
	if record.Metadata["marker"] != nil {
		t.Fatalf("NULLIF marker=%#v, want nil", record.Metadata["marker"])
	}
	if _, ok := record.Metadata["updated_at"].(string); !ok || record.Metadata["updated_at"] == "" {
		t.Fatalf("NOW updated_at=%#v", record.Metadata["updated_at"])
	}
	ratio, ok := metadataFloat(record.Metadata["ratio"])
	if !ok || ratio != 30 {
		t.Fatalf("cast ratio=%#v parsed=%v, want 30", record.Metadata["ratio"], ratio)
	}
	if _, err := db.Query(ctx, "INSERT INTO scalar_rows (id, value) VALUES ('r1', 0) ON CONFLICT (id) DO UPDATE SET value = scalar_rows.value << 1"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO scalar_rows (id, value) VALUES ('r1', 0) ON CONFLICT (id) DO UPDATE SET value = scalar_rows.value >> 1"); err != nil {
		t.Fatal(err)
	}
	record, err = col.Get(ctx, "r1")
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata["value"] != "30" {
		t.Fatalf("shift value=%#v, want 30", record.Metadata["value"])
	}
}

func TestSQLConflictNamedConstraintAndWhere(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_conflict_named"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE counters_named (id TEXT PRIMARY KEY, key TEXT, region TEXT, value BIGINT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "CREATE UNIQUE INDEX idx_counters_key_region ON counters_named (key, region)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO counters_named (id, key, region, value) VALUES ('r1', 'home', 'us', 10)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `INSERT INTO counters_named (id, key, region, value)
VALUES ('r2', 'home', 'us', 5)
ON CONFLICT ON CONSTRAINT idx_counters_key_region DO UPDATE
SET value = counters_named.value + EXCLUDED.value
WHERE counters_named.value < 100`); err != nil {
		t.Fatal(err)
	}
	col, err := db.GetCollection("counters_named")
	if err != nil {
		t.Fatal(err)
	}
	record, err := col.Get(ctx, "r1")
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata["value"] != "15" {
		t.Fatalf("named conflict value=%#v, want 15", record.Metadata["value"])
	}
	if _, err := db.Query(ctx, `INSERT INTO counters_named (id, key, region, value)
VALUES ('r3', 'home', 'us', 5)
ON CONFLICT ON CONSTRAINT idx_counters_key_region DO UPDATE
SET value = counters_named.value + EXCLUDED.value
WHERE counters_named.value < 10`); err != nil {
		t.Fatal(err)
	}
	record, err = col.Get(ctx, "r1")
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata["value"] != "15" {
		t.Fatalf("conflict WHERE false value=%#v, want 15", record.Metadata["value"])
	}
}

func TestSQLInsertSelect(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_insert_select"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, query := range []string{
		"CREATE TABLE source_rows (id TEXT PRIMARY KEY, value BIGINT)",
		"CREATE TABLE copied_rows (id TEXT PRIMARY KEY, value BIGINT)",
		"INSERT INTO source_rows (id, value) VALUES ('a', 1), ('b', 2)",
		"INSERT INTO copied_rows (id, value) SELECT id, value FROM source_rows WHERE value > 1",
	} {
		if _, err := db.Query(ctx, query); err != nil {
			t.Fatal(query, ": ", err)
		}
	}
	col, err := db.GetCollection("copied_rows")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := col.Get(ctx, "a"); err == nil {
		t.Fatal("filtered INSERT ... SELECT copied excluded row")
	}
	record, err := col.Get(ctx, "b")
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata["value"] != "2" {
		t.Fatalf("copied value=%#v, want 2", record.Metadata["value"])
	}
}

func TestSQLUnionAll(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_union_all"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE union_rows (id TEXT PRIMARY KEY, value BIGINT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO union_rows (id, value) VALUES ('a', 1), ('b', 2)"); err != nil {
		t.Fatal(err)
	}
	results, err := db.Query(ctx, "SELECT id, value FROM union_rows WHERE value > 0 UNION ALL SELECT id, value FROM union_rows WHERE value > 1")
	if err != nil {
		t.Fatal(err)
	}
	if results.Total != 3 || len(results.Results) != 3 {
		t.Fatalf("UNION ALL total=%d rows=%d, want 3", results.Total, len(results.Results))
	}
}

func TestSQLSetOperations(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_set_operations"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE set_rows (id TEXT PRIMARY KEY, value BIGINT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO set_rows (id, value) VALUES ('a', 1), ('b', 2), ('c', 3)"); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
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
	for _, tc := range cases {
		results, err := db.Query(ctx, tc.query)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if results.Total != tc.want {
			t.Fatalf("%s total=%d, want %d", tc.name, results.Total, tc.want)
		}
	}
}

func TestSQLPrepareExecute(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_prepare_execute"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	session, err := db.NewSQLSession(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	for _, query := range []string{
		"CREATE TABLE prepared_rows (id TEXT PRIMARY KEY, value BIGINT)",
		"INSERT INTO prepared_rows (id, value) VALUES ('p1', 1)",
		"PREPARE bump AS INSERT INTO prepared_rows (id, value) VALUES ($1, $2) ON CONFLICT (id) DO UPDATE SET value = prepared_rows.value + EXCLUDED.value",
	} {
		if err := session.Exec(query); err != nil {
			t.Fatal(query, ": ", err)
		}
	}
	if _, err := session.Query("EXECUTE bump('p1', 4)"); err != nil {
		t.Fatal(err)
	}
	col, err := db.GetCollection("prepared_rows")
	if err != nil {
		t.Fatal(err)
	}
	record, err := col.Get(ctx, "p1")
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata["value"] != "5" {
		t.Fatalf("prepared value=%#v, want 5", record.Metadata["value"])
	}
}
