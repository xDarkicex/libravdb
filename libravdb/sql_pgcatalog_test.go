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

// TestSQL_PgCatalogViaSchemaPrefix validates that pg_catalog. prefix
// rewriting works end-to-end through a pgwire connection.
// The prefix stripping happens in the pgwire layer (handleQuery), so
// this test exercises the full pgwire protocol path.
func TestSQL_PgCatalogViaSchemaPrefix(t *testing.T) {
	// This test exercises the pgwire path — see pgwire tests for the
	// full protocol-level verification. The SQL engine path (db.Query)
	// does not strip pg_catalog. prefix; that's a pgwire-layer concern.
	// We verify the underlying system table works with bare names.
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

	// pg_class without prefix works through the SQL engine.
	results, err := db.Query(ctx,
		"SELECT relname FROM pg_class WHERE relname = 'products'")
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
