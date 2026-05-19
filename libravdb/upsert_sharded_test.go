package libravdb

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"
)

// -- sharded mirror of existing non-sharded upsert tests --

func TestShardedUpsertInsertOnNewID(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, fmt.Sprintf("sharded_upsert_new_%d", time.Now().UnixNano()), WithDimension(3), WithSharding(true))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	err = collection.Upsert(ctx, "id-001", []float32{1, 2, 3}, map[string]interface{}{"k": "v"})
	if err != nil {
		t.Fatalf("Upsert on new ID: %v", err)
	}

	record, err := collection.Get(ctx, "id-001")
	if err != nil {
		t.Fatalf("Get after upsert: %v", err)
	}
	if record.ID != "id-001" {
		t.Fatalf("expected ID 'id-001', got %q", record.ID)
	}
	if record.Metadata["k"] != "v" {
		t.Fatalf("expected metadata k='v', got %v", record.Metadata["k"])
	}
}

func TestShardedUpsertReplaceOnExistingID(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, fmt.Sprintf("sharded_upsert_replace_%d", time.Now().UnixNano()), WithDimension(3), WithSharding(true))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	if err := collection.Insert(ctx, "id-001", []float32{1, 2, 3}, map[string]interface{}{"k": "old"}); err != nil {
		t.Fatalf("seed insert: %v", err)
	}

	err = collection.Upsert(ctx, "id-001", []float32{4, 5, 6}, map[string]interface{}{"k": "new"})
	if err != nil {
		t.Fatalf("Upsert on existing ID: %v", err)
	}

	record, err := collection.Get(ctx, "id-001")
	if err != nil {
		t.Fatalf("Get after upsert: %v", err)
	}

	if record.Vector[0] != 4 || record.Vector[1] != 5 || record.Vector[2] != 6 {
		t.Fatalf("expected updated vector {4,5,6}, got %v", record.Vector)
	}
	if record.Metadata["k"] != "new" {
		t.Fatalf("expected metadata k='new', got %v", record.Metadata["k"])
	}
}

func TestShardedUpsertMetadataReplaceNotMerge(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, fmt.Sprintf("sharded_upsert_meta_%d", time.Now().UnixNano()), WithDimension(3), WithSharding(true))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	if err := collection.Insert(ctx, "id-001", []float32{1, 1, 1}, map[string]interface{}{"a": 1, "b": 2}); err != nil {
		t.Fatalf("seed insert: %v", err)
	}

	err = collection.Upsert(ctx, "id-001", []float32{2, 2, 2}, map[string]interface{}{"b": 99, "c": 3})
	if err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	record, err := collection.Get(ctx, "id-001")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}

	if _, ok := record.Metadata["a"]; ok {
		t.Fatalf("key 'a' must NOT be present (replace, not merge), got %v", record.Metadata)
	}
	if record.Metadata["b"] != 99 {
		t.Fatalf("expected b=99, got %v", record.Metadata["b"])
	}
	if record.Metadata["c"] != 3 {
		t.Fatalf("expected c=3, got %v", record.Metadata["c"])
	}
	if len(record.Metadata) != 2 {
		t.Fatalf("expected exactly 2 metadata keys, got %d: %v", len(record.Metadata), record.Metadata)
	}
}

func TestShardedUpsertOrdinalPreservedOnReplace(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, fmt.Sprintf("sharded_ordinal_%d", time.Now().UnixNano()), WithDimension(3), WithSharding(true))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	if err := collection.Insert(ctx, "id-001", []float32{1, 0, 0}, map[string]interface{}{"tag": "original"}); err != nil {
		t.Fatalf("seed insert: %v", err)
	}

	first, err := collection.Get(ctx, "id-001")
	if err != nil {
		t.Fatalf("Get after insert: %v", err)
	}
	originalOrdinal := first.Ordinal

	err = collection.Upsert(ctx, "id-001", []float32{4, 5, 6}, map[string]interface{}{"k": "v"})
	if err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	second, err := collection.Get(ctx, "id-001")
	if err != nil {
		t.Fatalf("Get after upsert: %v", err)
	}
	if second.Ordinal != originalOrdinal {
		t.Fatalf("expected ordinal %d to be preserved, got %d", originalOrdinal, second.Ordinal)
	}
}

func TestShardedUpsertConcurrentSameID(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, fmt.Sprintf("sharded_upsert_conc_%d", time.Now().UnixNano()), WithDimension(3), WithSharding(true))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	var wg sync.WaitGroup
	errs := make(chan error, 10)

	for i := 0; i < 10; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			vec := []float32{float32(i), float32(i), float32(i)}
			meta := map[string]interface{}{"writer": i}
			if err := collection.Upsert(ctx, "id-001", vec, meta); err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)

	for e := range errs {
		t.Fatalf("concurrent Upsert error: %v", e)
	}

	record, err := collection.Get(ctx, "id-001")
	if err != nil {
		t.Fatalf("Get after concurrent upserts: %v", err)
	}
	if record.ID != "id-001" {
		t.Fatalf("expected ID 'id-001', got %q", record.ID)
	}

	count, err := collection.Count(ctx)
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if count != 1 {
		t.Fatalf("expected exactly 1 record, got %d", count)
	}
}

func TestShardedUpsertConcurrentDifferentShards(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)), WithMaxWriteQueueDepth(80))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, fmt.Sprintf("sharded_upsert_multi_%d", time.Now().UnixNano()), WithDimension(3), WithSharding(true))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	const totalIDs = 40
	ids := make([]string, totalIDs)
	for i := 0; i < totalIDs; i++ {
		ids[i] = fmt.Sprintf("multi-%03d", i)
	}

	var wg sync.WaitGroup
	errs := make(chan error, totalIDs)

	for i := 0; i < totalIDs; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			f := float32(i)
			vec := []float32{f, f, f}
			meta := map[string]interface{}{"idx": i}
			if err := collection.Upsert(ctx, ids[i], vec, meta); err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)

	for e := range errs {
		t.Fatalf("concurrent Upsert error: %v", e)
	}

	for _, id := range ids {
		record, err := collection.Get(ctx, id)
		if err != nil {
			t.Fatalf("Get %s: %v", id, err)
		}
		if record.ID != id {
			t.Fatalf("expected ID %s, got %s", id, record.ID)
		}
	}

	count, err := collection.Count(ctx)
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if count != totalIDs {
		t.Fatalf("expected %d records, got %d", totalIDs, count)
	}
}

func TestShardedUpsertMultipleReplacesSameID(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, fmt.Sprintf("sharded_upsert_seq_%d", time.Now().UnixNano()), WithDimension(3), WithSharding(true))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	for i := 1; i <= 5; i++ {
		vec := []float32{float32(i), float32(i), float32(i)}
		err := collection.Upsert(ctx, "id-001", vec, map[string]interface{}{"iteration": i})
		if err != nil {
			t.Fatalf("Upsert iteration %d: %v", i, err)
		}
	}

	record, err := collection.Get(ctx, "id-001")
	if err != nil {
		t.Fatalf("Get after 5 upserts: %v", err)
	}

	expected := []float32{5, 5, 5}
	if record.Vector[0] != expected[0] || record.Vector[1] != expected[1] || record.Vector[2] != expected[2] {
		t.Fatalf("expected last-write-wins vector {5,5,5}, got %v", record.Vector)
	}
	if record.Metadata["iteration"] != 5 {
		t.Fatalf("expected iteration=5, got %v", record.Metadata["iteration"])
	}

	count, err := collection.Count(ctx)
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if count != 1 {
		t.Fatalf("expected exactly 1 record, got %d", count)
	}
}
