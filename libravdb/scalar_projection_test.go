package libravdb

import (
	"context"
	"math"
	"testing"
)

func TestSQLScalarCaseAndCasts(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_scalar_projection"), WithMetrics(false))
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
	results, err := db.Query(ctx, `SELECT id,
		CASE WHEN amount >= 10 THEN 'high' ELSE 'low' END AS tier,
		amount::float AS amount_float,
		'550e8400-e29b-41d4-a716-446655440000'::uuid AS row_uuid,
		'{"kind":"ok"}'::jsonb AS payload_copy
		FROM scalar_rows ORDER BY id`)
	if err != nil {
		t.Fatal(err)
	}
	if len(results.Results) != 2 || len(results.Columns) != 5 {
		t.Fatalf("results shape columns=%v rows=%d", results.Columns, len(results.Results))
	}
	if results.Columns[1] != "tier" || results.Columns[2] != "amount_float" || results.Columns[3] != "row_uuid" || results.Columns[4] != "payload_copy" {
		t.Fatalf("columns=%v", results.Columns)
	}
	if results.Results[0].Metadata["tier"] != "high" || results.Results[1].Metadata["tier"] != "low" {
		t.Fatalf("case values: %#v %#v", results.Results[0].Metadata, results.Results[1].Metadata)
	}
	for _, row := range results.Results {
		amount, ok := row.Metadata["amount_float"].(float64)
		if !ok || math.Abs(amount-15) > 1e-9 && math.Abs(amount-5) > 1e-9 {
			t.Fatalf("cast amount=%#v", row.Metadata["amount_float"])
		}
		if row.Metadata["row_uuid"] != "550e8400-e29b-41d4-a716-446655440000" {
			t.Fatalf("uuid cast=%#v", row.Metadata["row_uuid"])
		}
		payload, ok := row.Metadata["payload_copy"].(map[string]interface{})
		if !ok || payload["kind"] != "ok" {
			t.Fatalf("jsonb cast=%#v", row.Metadata["payload_copy"])
		}
	}
}

func TestSQLScalarCaseNullAndFunctions(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_scalar_null"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE scalar_null_rows (id TEXT PRIMARY KEY, value BIGINT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO scalar_null_rows (id, value) VALUES ('nullish', NULL), ('present', 3)"); err != nil {
		t.Fatal(err)
	}
	results, err := db.Query(ctx, `SELECT id,
		CASE WHEN value IS NULL THEN 'missing' ELSE 'present' END AS state,
		NULLIF(value, 3) AS without_three,
		NOW() AS observed_at
		FROM scalar_null_rows ORDER BY id`)
	if err != nil {
		t.Fatal(err)
	}
	if len(results.Results) != 2 {
		t.Fatalf("rows=%d", len(results.Results))
	}
	if results.Results[0].Metadata["state"] != "missing" || results.Results[1].Metadata["state"] != "present" {
		t.Fatalf("states=%#v %#v", results.Results[0].Metadata, results.Results[1].Metadata)
	}
	if results.Results[0].Metadata["without_three"] != nil || results.Results[1].Metadata["without_three"] != nil {
		t.Fatalf("NULLIF values=%#v %#v", results.Results[0].Metadata["without_three"], results.Results[1].Metadata["without_three"])
	}
	if _, ok := results.Results[0].Metadata["observed_at"].(interface{}); !ok {
		t.Fatalf("NOW result missing: %#v", results.Results[0].Metadata)
	}
}

func TestSQLScalarRejectsUnknownCast(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_scalar_unknown_cast"), WithMetrics(false))
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
	if _, err := db.Query(ctx, "SELECT id, amount::not_a_real_type FROM scalar_unknown_cast"); err == nil {
		t.Fatal("unknown cast target unexpectedly succeeded")
	}
}
