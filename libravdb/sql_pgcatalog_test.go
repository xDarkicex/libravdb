package libravdb

import (
	"context"
	"testing"
)

func TestSQL_PgAttributeColumns(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:pgattr_test"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	ctx := context.Background()

	_, err = db.CreateCollection(ctx, "users", WithMetadataOnly(),
		WithMetadataSchema(map[string]FieldType{
			"name":  StringField,
			"age":   BigIntField,
			"score": FloatField,
		}),
	)
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	results, err := db.Query(ctx,
		"SELECT attname, atttypid, attnotnull FROM pg_attribute")
	if err != nil {
		t.Fatalf("Query pg_attribute: %v", err)
	}
	if len(results.Results) < 4 {
		t.Fatalf("expected >= 4 columns in pg_attribute, got %d", len(results.Results))
	}

	expectedCols := map[string]bool{"id": true, "name": true, "age": true, "score": true}
	for _, r := range results.Results {
		attname, _ := r.Metadata["attname"].(string)
		if attname == "" {
			continue
		}
		delete(expectedCols, attname)
	}
	if len(expectedCols) > 0 {
		t.Errorf("missing columns in pg_attribute: %v", expectedCols)
	}
	t.Logf("pg_attribute returned %d rows", len(results.Results))

	// Verify correct type OIDs.
	for _, r := range results.Results {
		attname, _ := r.Metadata["attname"].(string)
		atttypid, _ := r.Metadata["atttypid"].(int64)
		switch attname {
		case "id":
			if atttypid != 25 {
				t.Errorf("id atttypid = %d, want 25 (text)", atttypid)
			}
		case "age":
			if atttypid != 20 {
				t.Errorf("age atttypid = %d, want 20 (int8)", atttypid)
			}
		case "score":
			if atttypid != 701 {
				t.Errorf("score atttypid = %d, want 701 (float8)", atttypid)
			}
		}
	}
}

func TestSQL_PgTypeRows(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:pgtype_test"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	ctx := context.Background()

	results, err := db.Query(ctx, "SELECT oid, typname, typlen FROM pg_type")
	if err != nil {
		t.Fatalf("Query pg_type: %v", err)
	}
	if len(results.Results) == 0 {
		t.Fatal("expected non-empty pg_type")
	}

	expected := map[string]bool{
		"int4": false, "int8": false, "float8": false,
		"text": false, "bool": false,
	}
	for _, r := range results.Results {
		typname, _ := r.Metadata["typname"].(string)
		if _, ok := expected[typname]; ok {
			expected[typname] = true
		}
	}
	for typ, found := range expected {
		if !found {
			t.Errorf("pg_type missing expected type: %s", typ)
		}
	}
	t.Logf("pg_type returned %d rows", len(results.Results))
}

func TestSQL_PgNamespaceRows(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:pgns_test"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	ctx := context.Background()

	results, err := db.Query(ctx,
		"SELECT nspname FROM pg_namespace WHERE nspname = 'public'")
	if err != nil {
		t.Fatalf("Query pg_namespace: %v", err)
	}
	if len(results.Results) != 1 {
		t.Fatalf("expected 1 row, got %d", len(results.Results))
	}
	nspname, _ := results.Results[0].Metadata["nspname"].(string)
	if nspname != "public" {
		t.Errorf("nspname = %q, want 'public'", nspname)
	}

	results, err = db.Query(ctx, "SELECT oid, nspname FROM pg_namespace")
	if err != nil {
		t.Fatalf("Query all pg_namespace: %v", err)
	}
	if len(results.Results) != 3 {
		t.Fatalf("expected 3 namespaces, got %d", len(results.Results))
	}
	seen := make(map[string]bool)
	for _, r := range results.Results {
		if n, ok := r.Metadata["nspname"].(string); ok {
			seen[n] = true
		}
	}
	for _, ns := range []string{"pg_catalog", "public", "information_schema"} {
		if !seen[ns] {
			t.Errorf("missing namespace: %s", ns)
		}
	}
	t.Logf("pg_namespace names: %v", seen)
}

func TestSQL_PgIndexesRows(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:pgindexes_test"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	ctx := context.Background()

	if _, err := db.Query(ctx, `CREATE TABLE index_catalog_rows (id TEXT PRIMARY KEY, location TEXT, age INTEGER)`); err != nil {
		t.Fatalf("CREATE TABLE: %v", err)
	}
	if _, err := db.Query(ctx, `CREATE INDEX index_catalog_location_idx ON index_catalog_rows (location)`); err != nil {
		t.Fatalf("CREATE INDEX: %v", err)
	}

	results, err := db.Query(ctx, `
		SELECT schemaname, tablename, indexname, tablespace, indexdef
		FROM pg_catalog.pg_indexes
		WHERE tablename = 'index_catalog_rows'
		ORDER BY indexname`)
	if err != nil {
		t.Fatalf("Query pg_indexes: %v", err)
	}
	if len(results.Results) != 2 {
		t.Fatalf("pg_indexes rows=%d, want primary key and secondary index: %#v", len(results.Results), results.Results)
	}

	seen := make(map[string]map[string]interface{}, len(results.Results))
	for _, row := range results.Results {
		name, _ := row.Metadata["indexname"].(string)
		seen[name] = row.Metadata
	}
	for _, name := range []string{"index_catalog_rows_pkey", "index_catalog_location_idx"} {
		metadata, ok := seen[name]
		if !ok {
			t.Fatalf("pg_indexes missing %q: %#v", name, seen)
		}
		if metadata["schemaname"] != "public" || metadata["tablename"] != "index_catalog_rows" {
			t.Fatalf("pg_indexes identity for %q = %#v", name, metadata)
		}
		if metadata["tablespace"] != nil {
			t.Fatalf("pg_indexes tablespace for %q = %#v, want NULL", name, metadata["tablespace"])
		}
		definition, _ := metadata["indexdef"].(string)
		if definition == "" {
			t.Fatalf("pg_indexes indexdef for %q is empty", name)
		}
	}

	results, err = db.QueryWithParams(ctx,
		`SELECT indexname FROM pg_indexes WHERE indexname = $name`,
		QueryParams{"name": "index_catalog_location_idx"})
	if err != nil || len(results.Results) != 1 {
		t.Fatalf("parameterized pg_indexes lookup rows=%#v err=%v", results, err)
	}
}

// TestSQL_PgCatalogViaSchemaPrefix validates that native SQL accepts the
// PostgreSQL-qualified catalog relation form as well as the bare form.
func TestSQL_PgCatalogViaSchemaPrefix(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:pgcatschema_test"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	ctx := context.Background()

	_, err = db.CreateCollection(ctx, "products", WithMetadataOnly())
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	results, err := db.Query(ctx,
		"SELECT relname FROM pg_catalog.pg_class WHERE relname = 'products'")
	if err != nil {
		t.Fatalf("Query pg_class: %v", err)
	}
	if len(results.Results) != 1 {
		t.Fatalf("expected 1 row, got %d", len(results.Results))
	}
	relname, _ := results.Results[0].Metadata["relname"].(string)
	if relname != "products" {
		t.Errorf("relname = %q, want 'products'", relname)
	}
	t.Logf("pg_class WHERE relname='products': %v", relname)
}
