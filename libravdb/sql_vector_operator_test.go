package libravdb

import (
	"context"
	"fmt"
	"testing"
)

func TestSQLVectorOperatorOrderByAndProjection(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_vector_operator"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	// Deliberately configure cosine indexing while exercising all three SQL
	// operators. The operator, not the collection default, must determine the
	// distance semantics.
	col, err := db.CreateCollection(ctx, "operator_docs", WithDimension(3), WithMetric(CosineDistance), WithMetadataSchema(MetadataSchema{
		"title": StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range []struct {
		id  string
		vec []float32
	}{
		{"a", []float32{1, 0, 0}},
		{"b", []float32{0.8, 0.6, 0}},
		{"c", []float32{2, 0, 0}},
	} {
		if err := col.Insert(ctx, row.id, row.vec, map[string]interface{}{"title": row.id}); err != nil {
			t.Fatalf("insert %s: %v", row.id, err)
		}
	}

	assertIDs := func(label string, result *SearchResults, want ...string) {
		t.Helper()
		if result == nil || len(result.Results) != len(want) {
			t.Fatalf("%s rows=%v want=%v", label, result, want)
		}
		for i, id := range want {
			if result.Results[i].ID != id {
				t.Fatalf("%s row %d=%q want %q; rows=%v", label, i, result.Results[i].ID, id, result.Results)
			}
		}
	}

	l2, err := db.Query(ctx, "SELECT id FROM operator_docs ORDER BY embedding <-> '[1,0,0]' LIMIT 3")
	if err != nil {
		t.Fatalf("L2 operator: %v", err)
	}
	assertIDs("L2", l2, "a", "b", "c")

	ip, err := db.Query(ctx, "SELECT id FROM operator_docs ORDER BY embedding <#> '[1,0,0]' LIMIT 3")
	if err != nil {
		t.Fatalf("inner-product operator: %v", err)
	}
	assertIDs("inner product", ip, "c", "a", "b")
	if ip.Results[0].Score >= ip.Results[1].Score {
		t.Fatalf("<#> scores must retain negative-inner-product ordering: %v, %v", ip.Results[0].Score, ip.Results[1].Score)
	}

	cosine, err := db.QueryWithParams(ctx, "SELECT id FROM operator_docs ORDER BY embedding <=> $query LIMIT 3", QueryParams{"query": []float32{1, 0, 0}})
	if err != nil {
		t.Fatalf("cosine operator parameter: %v", err)
	}
	assertIDs("cosine", cosine, "a", "c", "b")

	projected, err := db.Query(ctx, "SELECT id, embedding, embedding <-> '[1,0,0]' AS distance FROM operator_docs ORDER BY distance LIMIT 2")
	if err != nil {
		t.Fatalf("operator projection: %v", err)
	}
	assertIDs("projection", projected, "a", "b")
	if value := projected.Results[0].Metadata["distance"]; fmt.Sprint(value) == "" {
		t.Fatalf("operator projection distance missing: %#v", projected.Results[0].Metadata)
	}
	if value := projected.Results[0].Metadata["embedding"]; value == nil || fmt.Sprint(value) == "" {
		t.Fatalf("operator projection embedding missing: %#v", projected.Results[0].Metadata)
	}
}
