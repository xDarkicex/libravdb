package libravdb

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func TestCollectionDeclarationsSurviveReopenAndEnsureDoesNotDiscardOptions(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/declarations.libravdb"
	schema := MetadataSchema{
		"from_id": StringField,
		"to_id":   StringField,
	}
	indexed := []string{"from_id", "to_id"}

	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("open initial database: %v", err)
	}
	collection, err := db.CreateCollection(ctx, "__causal_edges_v2",
		WithDimension(1),
		WithFlat(),
		WithMetadataSchema(schema),
		WithIndexedFields(indexed...),
	)
	if err != nil {
		db.Close()
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.Insert(ctx, "edge-1", []float32{1}, map[string]interface{}{
		"from_id": "alice",
		"to_id":   "bob",
	}); err != nil {
		db.Close()
		t.Fatalf("insert edge: %v", err)
	}

	if err := db.Close(); err != nil {
		t.Fatalf("close initial database: %v", err)
	}

	reopened, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	reloaded, err := reopened.GetCollection("__causal_edges_v2")
	if err != nil {
		t.Fatalf("load collection after reopen: %v", err)
	}
	config := reloaded.Config()
	if config.Dimension != 1 {
		t.Fatalf("reloaded dimension = %d, want 1", config.Dimension)
	}
	if !reflect.DeepEqual(config.MetadataSchema, schema) {
		t.Fatalf("reloaded metadata schema = %#v, want %#v", config.MetadataSchema, schema)
	}
	if !reflect.DeepEqual(config.IndexedFields, indexed) {
		t.Fatalf("reloaded indexed fields = %#v, want %#v", config.IndexedFields, indexed)
	}

	matches, err := reloaded.ListByMetadata(ctx, "from_id", "alice")
	if err != nil {
		t.Fatalf("indexed lookup after reopen: %v", err)
	}
	if len(matches) != 1 || matches[0].ID != "edge-1" {
		t.Fatalf("indexed lookup after reopen = %+v, want edge-1", matches)
	}
	if reloaded.metadataIndex == nil || reloaded.metadataIndex["from_id"] == nil {
		t.Fatal("indexed lookup after reopen did not rebuild the configured posting list")
	}

	ensured, err := reopened.EnsureCollection(ctx, "__causal_edges_v2", 1,
		WithFlat(),
		WithMetadataSchema(schema),
		WithIndexedFields(indexed...),
	)
	if err != nil {
		t.Fatalf("EnsureCollection with matching declarations: %v", err)
	}
	if ensured != reloaded {
		t.Fatal("EnsureCollection returned a different instance for an existing collection")
	}

	_, err = reopened.EnsureCollection(ctx, "__causal_edges_v2", 1,
		WithFlat(),
		WithMetadataSchema(schema),
		WithIndexedFields("to_id"),
	)
	if !errors.Is(err, ErrCollectionConfigurationMismatch) {
		t.Fatalf("EnsureCollection mismatch error = %v, want ErrCollectionConfigurationMismatch", err)
	}
}
