package libravdb

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"testing"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
)

func TestHNSWStorageOwnedReopenAndMetadataRegression(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	collection, err := db.CreateCollection(ctx, "storage_owned", WithDimension(3), WithMetric(L2Distance), WithHNSW(16, 100, 50))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	entries := []VectorEntry{
		{ID: "a", Vector: []float32{1, 0, 0}, Metadata: map[string]interface{}{"sessionId": "s1", "kind": "spec"}},
		{ID: "b", Vector: []float32{0, 1, 0}, Metadata: map[string]interface{}{"sessionId": "s1", "kind": "chat"}},
		{ID: "c", Vector: []float32{0, 0, 1}, Metadata: map[string]interface{}{"sessionId": "s2", "kind": "spec"}},
	}
	if err := collection.InsertBatch(ctx, entries); err != nil {
		t.Fatalf("insert batch: %v", err)
	}

	if raw := collection.DebugRawVectorStoreProfile(); raw != nil {
		t.Fatalf("expected no raw vector store profile for provider-backed HNSW, got %v", raw)
	}

	assertCollectionState := func(c *Collection) {
		t.Helper()

		count, err := c.Count(ctx)
		if err != nil {
			t.Fatalf("count: %v", err)
		}
		if count != len(entries) {
			t.Fatalf("expected count %d, got %d", len(entries), count)
		}

		results, err := c.Search(ctx, []float32{1, 0, 0}, 2)
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(results.Results) == 0 || results.Results[0].ID != "a" {
			t.Fatalf("expected nearest result a, got %+v", results.Results)
		}
		if results.Results[0].Metadata["sessionId"] != "s1" {
			t.Fatalf("expected metadata hydrated from storage, got %+v", results.Results[0].Metadata)
		}

		all, err := c.ListAll(ctx)
		if err != nil {
			t.Fatalf("list all: %v", err)
		}
		if len(all) != len(entries) {
			t.Fatalf("expected %d records, got %d", len(entries), len(all))
		}

		specMatches, err := c.ListByMetadata(ctx, "kind", "spec")
		if err != nil {
			t.Fatalf("list by metadata: %v", err)
		}
		if len(specMatches) != 2 {
			t.Fatalf("expected 2 metadata matches, got %d", len(specMatches))
		}

		iterated := 0
		if err := c.Iterate(ctx, func(record Record) error {
			iterated++
			if len(record.Vector) != 3 {
				t.Fatalf("expected hydrated vector in iteration, got %v", record.Vector)
			}
			return nil
		}); err != nil {
			t.Fatalf("iterate: %v", err)
		}
		if iterated != len(entries) {
			t.Fatalf("expected %d iterated records, got %d", len(entries), iterated)
		}
	}

	assertCollectionState(collection)

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	reloaded, err := reopened.GetCollection("storage_owned")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}

	if raw := reloaded.DebugRawVectorStoreProfile(); raw != nil {
		t.Fatalf("expected no raw vector store profile after reopen, got %v", raw)
	}

	assertCollectionState(reloaded)
}

func TestHNSWStableOrdinalAcrossUpdateDeleteInsert(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "ordinals", WithDimension(3), WithMetric(L2Distance), WithHNSW(16, 100, 50))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	if err := collection.Insert(ctx, "keep", []float32{1, 0, 0}, map[string]interface{}{"version": 1}); err != nil {
		t.Fatalf("insert keep: %v", err)
	}
	if err := collection.Insert(ctx, "gone", []float32{0, 1, 0}, map[string]interface{}{"version": 1}); err != nil {
		t.Fatalf("insert gone: %v", err)
	}

	before := ordinalSnapshot(t, db, "ordinals")

	if err := collection.Update(ctx, "keep", []float32{1, 1, 0}, map[string]interface{}{"version": 2}); err != nil {
		t.Fatalf("update keep: %v", err)
	}
	if err := collection.Delete(ctx, "gone"); err != nil {
		t.Fatalf("delete gone: %v", err)
	}
	if err := collection.Insert(ctx, "new", []float32{0, 0, 1}, map[string]interface{}{"version": 1}); err != nil {
		t.Fatalf("insert new: %v", err)
	}

	after := ordinalSnapshot(t, db, "ordinals")
	if before["keep"] != after["keep"] {
		t.Fatalf("expected keep ordinal to stay stable across update, before=%d after=%d", before["keep"], after["keep"])
	}
	if _, ok := after["gone"]; ok {
		t.Fatalf("expected deleted ID to disappear from ordinal snapshot: %+v", after)
	}
	if after["new"] == after["keep"] {
		t.Fatalf("expected new ID to receive distinct ordinal, got %+v", after)
	}

	results, err := collection.Search(ctx, []float32{1, 1, 0}, 3)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results.Results) == 0 || results.Results[0].ID != "keep" {
		t.Fatalf("expected keep to remain searchable after update, got %+v", results.Results)
	}
	for _, result := range results.Results {
		if result.ID == "gone" {
			t.Fatalf("deleted ID returned in results: %+v", results.Results)
		}
	}
}

func TestHNSWProviderBackedQuantizationRegression(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	collection, err := db.CreateCollection(
		ctx,
		"quantized_provider",
		WithDimension(16),
		WithMetric(L2Distance),
		WithHNSW(16, 100, 50),
		WithScalarQuantization(8, 1.0),
	)
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	vectors := make([][]float32, 64)
	for i := range vectors {
		vec := make([]float32, 16)
		for d := range vec {
			vec[d] = float32(i*3+d) / 10
		}
		vectors[i] = vec
		if err := collection.Insert(ctx, fmt.Sprintf("q_%d", i), vec, map[string]interface{}{"bucket": i / 8}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	searchAndCheck := func(c *Collection) {
		t.Helper()
		results, err := c.Search(ctx, vectors[0], 5)
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(results.Results) == 0 || results.Results[0].ID != "q_0" {
			t.Fatalf("expected quantized nearest match q_0, got %+v", results.Results)
		}
		if len(results.Results[0].Vector) != 16 {
			t.Fatalf("expected hydrated raw vector from storage, got len=%d", len(results.Results[0].Vector))
		}
		if results.Results[0].Metadata["bucket"] == nil {
			t.Fatalf("expected metadata from storage, got %+v", results.Results[0].Metadata)
		}
	}

	searchAndCheck(collection)

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	reloaded, err := reopened.GetCollection("quantized_provider")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}
	searchAndCheck(reloaded)
}

func TestHNSW2500MemoryAcceptance(t *testing.T) {
	if os.Getenv("LIBRAVDB_RUN_MEMORY_ACCEPTANCE") != "1" {
		t.Skip("set LIBRAVDB_RUN_MEMORY_ACCEPTANCE=1 to run hnsw_2500_memory acceptance")
	}

	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "hnsw_2500_memory", WithDimension(64), WithMetric(L2Distance), WithHNSW(16, 100, 50))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	entries := make([]VectorEntry, 2500)
	for i := range entries {
		vec := make([]float32, 64)
		for d := range vec {
			vec[d] = float32(i+d) / 100
		}
		entries[i] = VectorEntry{
			ID:       fmt.Sprintf("m_%d", i),
			Vector:   vec,
			Metadata: map[string]interface{}{"group": i / 100},
		}
	}

	if err := collection.InsertBatch(ctx, entries); err != nil {
		t.Fatalf("insert batch: %v", err)
	}

	if raw := collection.DebugRawVectorStoreProfile(); raw != nil {
		t.Fatalf("expected no raw vector duplication profile, got %v", raw)
	}

	usage, err := collection.GetMemoryUsage()
	if err != nil {
		t.Fatalf("get memory usage: %v", err)
	}
	if usage.Total <= 0 {
		t.Fatalf("expected positive total memory usage, got %d", usage.Total)
	}

	stats := collection.Stats()
	if stats.MemoryStats == nil {
		t.Fatalf("expected memory stats")
	}
	if stats.MemoryStats.Storage <= 0 {
		t.Fatalf("expected storage memory to be tracked, got %+v", stats.MemoryStats)
	}
	if stats.RawVectorStoreStats != nil {
		t.Fatalf("expected no raw vector store stats for provider-backed collection, got %+v", stats.RawVectorStoreStats)
	}

	results, err := collection.Search(ctx, entries[0].Vector, 10)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results.Results) == 0 || results.Results[0].ID != entries[0].ID {
		t.Fatalf("expected nearest result %s, got %+v", entries[0].ID, results.Results)
	}

	if profilePath := os.Getenv("LIBRAVDB_HEAP_PROFILE"); profilePath != "" {
		runtime.GC()
		file, err := os.Create(filepath.Clean(profilePath))
		if err != nil {
			t.Fatalf("create heap profile: %v", err)
		}
		defer file.Close()
		if err := pprof.WriteHeapProfile(file); err != nil {
			t.Fatalf("write heap profile: %v", err)
		}
	}
}

func ordinalSnapshot(t *testing.T, db *Database, name string) map[string]uint32 {
	t.Helper()

	fileEngine, ok := db.storage.(interface {
		GetCollectionWithConfig(name string) (storage.Collection, *storage.CollectionConfig, error)
	})
	if !ok {
		t.Fatalf("storage engine does not expose collection config")
	}
	storageCollection, _, err := fileEngine.GetCollectionWithConfig(name)
	if err != nil {
		t.Fatalf("get storage collection: %v", err)
	}

	snapshot := make(map[string]uint32)
	if err := storageCollection.Iterate(context.Background(), func(entry *index.VectorEntry) error {
		snapshot[entry.ID] = entry.Ordinal
		return nil
	}); err != nil {
		t.Fatalf("iterate storage collection: %v", err)
	}
	return snapshot
}
