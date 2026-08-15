package libravdb

import (
	"context"
	"math"
	"testing"
)

func TestSQLVectorAvg(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:vector_avg"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	col, err := db.CreateCollection(ctx, "vector_avg_docs", WithDimension(3), WithMetadataSchema(MetadataSchema{
		"category": StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range []struct {
		id, category string
		vector       []float32
	}{
		{"a1", "a", []float32{1, 2, 3}},
		{"a2", "a", []float32{3, 4, 5}},
		{"b1", "b", []float32{10, 20, 30}},
	} {
		if err := col.Insert(ctx, row.id, row.vector, map[string]interface{}{"category": row.category}); err != nil {
			t.Fatal(err)
		}
	}

	result, err := db.Query(ctx, "SELECT VECTOR_AVG(embedding) AS centroid FROM vector_avg_docs")
	if err != nil {
		t.Fatal(err)
	}
	if result.Total != 1 || result.Columns[0] != "centroid" {
		t.Fatalf("vector average shape: total=%d columns=%v", result.Total, result.Columns)
	}
	centroid, ok := result.Results[0].Metadata["centroid"].([]float32)
	if !ok {
		t.Fatalf("vector average type: %T", result.Results[0].Metadata["centroid"])
	}
	for i, want := range []float32{14.0 / 3, 26.0 / 3, 38.0 / 3} {
		if math.Abs(float64(centroid[i]-want)) > 1e-5 {
			t.Fatalf("centroid[%d]=%v, want %v", i, centroid[i], want)
		}
	}
	empty, err := db.Query(ctx, "SELECT VECTOR_AVG(embedding) AS centroid FROM vector_avg_docs WHERE category = 'missing'")
	if err != nil {
		t.Fatal(err)
	}
	if empty.Total != 1 || empty.Results[0].Metadata["centroid"] != nil {
		t.Fatalf("empty vector average=%#v, want SQL NULL", empty.Results[0].Metadata["centroid"])
	}

	grouped, err := db.Query(ctx, "SELECT category, VECTOR_AVG(embedding) AS centroid FROM vector_avg_docs GROUP BY category ORDER BY category")
	if err != nil {
		t.Fatal(err)
	}
	if grouped.Total != 2 || len(grouped.Results) != 2 {
		t.Fatalf("grouped vector average rows=%d", grouped.Total)
	}
	if got := grouped.Results[0].Metadata["centroid"].([]float32); got[0] != 2 || got[1] != 3 || got[2] != 4 {
		t.Fatalf("group a centroid=%v", got)
	}
	if got := grouped.Results[1].Metadata["centroid"].([]float32); got[0] != 10 || got[1] != 20 || got[2] != 30 {
		t.Fatalf("group b centroid=%v", got)
	}
}
