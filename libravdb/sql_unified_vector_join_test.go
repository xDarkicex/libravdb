package libravdb

import (
	"context"
	"testing"
)

func TestSQLJoinAndAggregateOverVectorCollection(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_unified_vector_join"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.CreateCollection(ctx, "authors", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{"name": StringField})); err != nil {
		t.Fatal(err)
	}
	if _, err := db.CreateCollection(ctx, "documents", WithDimension(3), WithMetadataSchema(MetadataSchema{
		"author_id": StringField,
		"title":     StringField,
	})); err != nil {
		t.Fatal(err)
	}
	authors, _ := db.GetCollection("authors")
	if err := authors.Insert(ctx, "a1", nil, map[string]interface{}{"name": "Ada"}); err != nil {
		t.Fatal(err)
	}
	docs, _ := db.GetCollection("documents")
	for _, row := range []struct {
		id, author, title string
	}{
		{"d1", "a1", "one"},
		{"d2", "a1", "two"},
	} {
		if err := docs.Insert(ctx, row.id, []float32{1, 0, 0}, map[string]interface{}{"author_id": row.author, "title": row.title}); err != nil {
			t.Fatal(err)
		}
	}
	joined, err := db.Query(ctx, "SELECT d.id, a.name FROM documents d JOIN authors a ON d.author_id = a.id ORDER BY d.id")
	if err != nil {
		t.Fatal(err)
	}
	if len(joined.Results) != 2 || len(joined.Columns) != 2 || joined.Columns[1] != "name" {
		t.Fatalf("joined shape columns=%v rows=%d", joined.Columns, len(joined.Results))
	}
	if joined.Results[0].ID != "d1" || joined.Results[0].Metadata["name"] != "Ada" {
		t.Fatalf("joined row=%+v", joined.Results[0])
	}
	agg, err := db.Query(ctx, "SELECT COUNT(*) FROM documents")
	if err != nil {
		t.Fatal(err)
	}
	if len(agg.Results) != 1 || agg.Results[0].Metadata["count"] != int64(2) {
		t.Fatalf("aggregate over vector collection=%+v", agg.Results)
	}
}
