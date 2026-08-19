package libravdb

import (
	"context"
	"reflect"
	"testing"

	apexjson "github.com/xDarkicex/apexJSON/v2"
)

func TestSQLParameterizedJSONBUpsert(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/json_params"
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := db.Query(ctx, "CREATE TABLE people (id TEXT PRIMARY KEY, metadata JSONB, vector VECTOR(3))"); err != nil {
		db.Close()
		t.Fatalf("create: %v", err)
	}
	const upsert = `INSERT INTO people (id, metadata, vector)
VALUES ($1, $2::jsonb, $3)
ON CONFLICT (id) DO UPDATE SET
  metadata = EXCLUDED.metadata,
  vector = EXCLUDED.vector
RETURNING id, metadata`

	first := map[string]interface{}{
		"name": "Ada",
		"profile": map[string]interface{}{
			"active": true,
			"roles":  []interface{}{"admin", "reviewer"},
		},
	}
	if _, err := db.QueryWithParams(ctx, upsert, QueryParams{
		"1": "p1",
		"2": first,
		"3": []float32{1, 0, 0},
	}); err != nil {
		db.Close()
		t.Fatalf("parameterized JSONB insert: %v", err)
	}

	second := map[string]interface{}{
		"name": "Ada Lovelace",
		"profile": map[string]interface{}{
			"active": false,
			"roles":  []interface{}{"admin", "owner"},
		},
		"scores": []interface{}{1, 2.5, 3},
	}
	returned, err := db.QueryWithParams(ctx, upsert, QueryParams{
		"1": "p1",
		"2": second,
		"3": []float32{0, 1, 0},
	})
	if err != nil {
		db.Close()
		t.Fatalf("parameterized JSONB update: %v", err)
	}
	if returned.Total != 1 || returned.Columns == nil || len(returned.Columns) != 2 {
		t.Fatalf("RETURNING shape: %#v", returned)
	}

	col, err := db.GetCollection("people")
	if err != nil {
		db.Close()
		t.Fatal(err)
	}
	record, err := col.Get(ctx, "p1")
	if err != nil {
		db.Close()
		t.Fatal(err)
	}
	gotJSON, _ := apexjson.Marshal(record.Metadata["metadata"])
	wantJSON, _ := apexjson.Marshal(second)
	if string(gotJSON) != string(wantJSON) {
		t.Fatalf("stored JSONB=%#v, want %#v", record.Metadata["metadata"], second)
	}
	if !reflect.DeepEqual(record.Vector, []float32{0, 1, 0}) {
		t.Fatalf("stored vector=%v", record.Vector)
	}

	if _, err := db.QueryWithParams(ctx,
		"INSERT INTO people (id, metadata, vector) VALUES ($1, $2::jsonb, $3) ON CONFLICT (id) DO NOTHING",
		QueryParams{"1": "p1", "2": map[string]interface{}{"ignored": true}, "3": []float32{0, 0, 1}}); err != nil {
		db.Close()
		t.Fatalf("parameterized JSONB DO NOTHING: %v", err)
	}
	if _, err := db.QueryWithParams(ctx,
		"INSERT INTO people (id, metadata, vector) VALUES ($1, $2::jsonb, $3)",
		QueryParams{"1": "p2", "2": "not-json", "3": []float32{0, 0, 1}}); err == nil {
		db.Close()
		t.Fatal("invalid parameterized JSONB was accepted")
	}
	db.Close()

	reopened, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	after, err := reopened.Query(ctx, "SELECT metadata FROM people WHERE id = 'p1'")
	if err != nil {
		t.Fatalf("reopen JSONB read: %v", err)
	}
	afterJSON, _ := apexjson.Marshal(after.Results[0].Metadata["metadata"])
	if after.Total != 1 || string(afterJSON) != string(wantJSON) {
		t.Fatalf("reopened JSONB=%#v, want %#v", after, second)
	}
}
