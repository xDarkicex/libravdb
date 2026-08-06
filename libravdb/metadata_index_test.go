package libravdb

import (
	"context"
	"testing"
)

func TestIndexedMetadataPostingsTrackMutationsAndTransactions(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:metadata_postings"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "indexed", WithDimension(2),
		WithMetadataSchema(MetadataSchema{"category": StringField}),
		WithIndexedFields("category"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := collection.Insert(ctx, "a", []float32{1, 0}, map[string]interface{}{"category": "old"}); err != nil {
		t.Fatal(err)
	}
	if err := collection.Insert(ctx, "b", []float32{0, 1}, map[string]interface{}{"category": "old"}); err != nil {
		t.Fatal(err)
	}

	assertIDs := func(value string, want ...string) {
		t.Helper()
		records, err := collection.ListByMetadata(ctx, "category", value)
		if err != nil {
			t.Fatal(err)
		}
		got := make(map[string]bool, len(records))
		for _, record := range records {
			got[record.ID] = true
		}
		if len(got) != len(want) {
			t.Fatalf("category %q IDs = %v, want %v", value, got, want)
		}
		for _, id := range want {
			if !got[id] {
				t.Fatalf("category %q IDs = %v, missing %q", value, got, id)
			}
		}
	}

	assertIDs("old", "a", "b")
	if collection.metadataIndexBuiltAt != collection.metadataMutationEpoch.Load() {
		t.Fatal("posting index was not published at the current mutation epoch")
	}

	if err := collection.Update(ctx, "a", nil, map[string]interface{}{"category": "updated"}); err != nil {
		t.Fatal(err)
	}
	if collection.metadataIndexBuiltAt == collection.metadataMutationEpoch.Load() {
		t.Fatal("update did not invalidate metadata postings")
	}
	assertIDs("old", "b")
	assertIDs("updated", "a")

	if err := db.WithTx(ctx, func(tx Tx) error {
		return tx.Update(ctx, collection.name, "b", nil, map[string]interface{}{"category": "transaction"})
	}); err != nil {
		t.Fatal(err)
	}
	if collection.metadataIndexBuiltAt == collection.metadataMutationEpoch.Load() {
		t.Fatal("transaction commit did not invalidate metadata postings")
	}
	assertIDs("old")
	assertIDs("transaction", "b")

	if err := collection.Delete(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	assertIDs("updated")

	numeric, err := db.CreateCollection(ctx, "numeric", WithDimension(2), WithIndexedFields("value"))
	if err != nil {
		t.Fatal(err)
	}
	if err := numeric.Insert(ctx, "forty-two", []float32{1, 1}, map[string]interface{}{"value": int64(42)}); err != nil {
		t.Fatal(err)
	}
	records, err := numeric.ListByMetadata(ctx, "value", float64(42))
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 1 || records[0].ID != "forty-two" {
		t.Fatalf("cross-type numeric posting lookup = %+v, want forty-two", records)
	}
}
