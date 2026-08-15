package libravdb

import (
	"context"
	"testing"
	"time"
)

func TestTemporalVectorOperatorParameterizedSnapshot(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir()+"/temporal-vector-operator.libravdb"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	col, err := db.CreateCollection(ctx, "temporal_operator_docs", WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := col.Insert(ctx, "historical", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("historical insert: %v", err)
	}
	snapshot, err := db.SnapshotAt(ctx, time.Now().UTC().Add(time.Second))
	if err != nil {
		t.Fatalf("SnapshotAt: %v", err)
	}
	cutoff := snapshot.Timestamp.Format(time.RFC3339Nano)
	snapshot.Close()

	// The live row is changed after the cutoff. A temporal operator query must
	// still score the historical vector, not the current vector.
	if err := col.Update(ctx, "historical", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("post-snapshot update: %v", err)
	}
	if err := col.Insert(ctx, "future", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("post-snapshot insert: %v", err)
	}

	result, err := db.QueryWithParams(ctx,
		"SELECT id, embedding <-> $query_vec AS distance "+
			"FROM temporal_operator_docs AS OF TIMESTAMP $as_of d "+
			"ORDER BY distance ASC LIMIT 1",
		QueryParams{
			"as_of":     cutoff,
			"query_vec": []float32{1, 0, 0},
		})
	if err != nil {
		t.Fatalf("parameterized temporal operator query: %v", err)
	}
	if result == nil || result.Total != 1 || len(result.Results) != 1 {
		t.Fatalf("temporal operator rows=%v, want one historical row", result)
	}
	row := result.Results[0]
	if row.ID != "historical" {
		t.Fatalf("temporal operator id=%q, want historical", row.ID)
	}
	if distance, ok := row.Metadata["distance"].(float32); !ok || distance != 0 {
		t.Fatalf("historical operator distance=%#v, want float32(0)", row.Metadata["distance"])
	}
}
