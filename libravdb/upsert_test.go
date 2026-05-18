package libravdb

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestUpsertInsertOnNewID(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, fmt.Sprintf("upsert_new_%d", time.Now().UnixNano()), WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	err = collection.Upsert(ctx, "new-id", []float32{1, 2, 3}, map[string]interface{}{"key": "val"})
	if err != nil {
		t.Fatalf("Upsert on new ID: %v", err)
	}

	record, err := collection.Get(ctx, "new-id")
	if err != nil {
		t.Fatalf("Get after upsert: %v", err)
	}
	if record.ID != "new-id" {
		t.Fatalf("expected ID 'new-id', got %q", record.ID)
	}
	if record.Metadata["key"] != "val" {
		t.Fatalf("expected metadata key='val', got %v", record.Metadata["key"])
	}

	results, err := collection.Search(ctx, []float32{1, 2, 3}, 1)
	if err != nil {
		t.Fatalf("Search after upsert: %v", err)
	}
	if len(results.Results) != 1 || results.Results[0].ID != "new-id" {
		t.Fatalf("expected search to find 'new-id', got %+v", results.Results)
	}
}

func TestUpsertReplaceOnExistingID(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, fmt.Sprintf("upsert_replace_%d", time.Now().UnixNano()), WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	if err := collection.Insert(ctx, "r1", []float32{1, 0, 0}, map[string]interface{}{"a": 1, "b": 2}); err != nil {
		t.Fatalf("seed insert: %v", err)
	}

	err = collection.Upsert(ctx, "r1", []float32{0, 1, 0}, map[string]interface{}{"b": 99, "c": 3})
	if err != nil {
		t.Fatalf("Upsert on existing ID: %v", err)
	}

	record, err := collection.Get(ctx, "r1")
	if err != nil {
		t.Fatalf("Get after upsert: %v", err)
	}

	if record.Vector[0] != 0 || record.Vector[1] != 1 || record.Vector[2] != 0 {
		t.Fatalf("expected updated vector, got %v", record.Vector)
	}
	if _, ok := record.Metadata["a"]; ok {
		t.Fatalf("key 'a' must not survive replace, got %v", record.Metadata)
	}
	if record.Metadata["b"] != 99 {
		t.Fatalf("expected b=99, got %v", record.Metadata["b"])
	}
	if record.Metadata["c"] != 3 {
		t.Fatalf("expected c=3, got %v", record.Metadata["c"])
	}

	results, err := collection.Search(ctx, []float32{0, 1, 0}, 1)
	if err != nil {
		t.Fatalf("Search after upsert: %v", err)
	}
	if len(results.Results) != 1 || results.Results[0].ID != "r1" {
		t.Fatalf("expected search to find 'r1', got %+v", results.Results)
	}
}

func TestUpsertMetadataReplaceNotMerge(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, fmt.Sprintf("upsert_meta_%d", time.Now().UnixNano()), WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	if err := collection.Insert(ctx, "m1", []float32{1, 1, 1}, map[string]interface{}{"a": 1, "b": 2}); err != nil {
		t.Fatalf("seed insert: %v", err)
	}

	err = collection.Upsert(ctx, "m1", []float32{2, 2, 2}, map[string]interface{}{"b": 99, "c": 3})
	if err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	record, err := collection.Get(ctx, "m1")
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

func TestUpsertConcurrentSameID(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, fmt.Sprintf("upsert_conc_%d", time.Now().UnixNano()), WithDimension(3))
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
			if err := collection.Upsert(ctx, "concurrent", vec, meta); err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Fatalf("concurrent Upsert error: %v", err)
	}

	record, err := collection.Get(ctx, "concurrent")
	if err != nil {
		t.Fatalf("Get after concurrent upserts: %v", err)
	}
	if record.ID != "concurrent" {
		t.Fatalf("expected ID 'concurrent', got %q", record.ID)
	}

	count, err := collection.Count(ctx)
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if count != 1 {
		t.Fatalf("expected exactly 1 record, got %d", count)
	}
}

func TestTransactionUpsertRollsBackCleanly(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, fmt.Sprintf("tx_upsert_rollback_%d", time.Now().UnixNano()), WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	// Seed an existing record.
	if err := collection.Insert(ctx, "existing", []float32{1, 0, 0}, map[string]interface{}{"original": true}); err != nil {
		t.Fatalf("seed insert: %v", err)
	}

	// Upsert inside a transaction, then force rollback.
	tx, err := db.BeginTx(ctx)
	if err != nil {
		t.Fatalf("begin tx: %v", err)
	}
	if err := tx.Upsert(ctx, collection.name, "existing", []float32{0, 1, 0}, map[string]interface{}{"should_not": "persist"}); err != nil {
		t.Fatalf("stage upsert in tx: %v", err)
	}
	if err := tx.Upsert(ctx, collection.name, "new-one", []float32{0, 0, 1}, map[string]interface{}{"should_not": "persist_either"}); err != nil {
		t.Fatalf("stage upsert new in tx: %v", err)
	}
	if err := tx.Rollback(ctx); err != nil {
		t.Fatalf("rollback tx: %v", err)
	}

	// Existing record must be unchanged.
	record, err := collection.Get(ctx, "existing")
	if err != nil {
		t.Fatalf("Get existing after rollback: %v", err)
	}
	if record.Metadata["original"] != true {
		t.Fatalf("expected original metadata to survive rollback, got %v", record.Metadata)
	}
	if record.Vector[1] != 0 {
		t.Fatalf("expected original vector to survive rollback, got %v", record.Vector)
	}

	// New record must not exist.
	if _, err := collection.Get(ctx, "new-one"); err == nil {
		t.Fatal("expected 'new-one' to not exist after rollback")
	}
}

func TestBatchUpsertTrueOnExistingRecord(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, fmt.Sprintf("batch_upsert_%d", time.Now().UnixNano()), WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	if err := collection.Insert(ctx, "b1", []float32{1, 0, 0}, map[string]interface{}{"old": "data"}); err != nil {
		t.Fatalf("seed insert: %v", err)
	}

	batch := collection.NewBatchUpdate([]*VectorUpdate{
		{ID: "b1", Vector: []float32{0, 1, 0}, Metadata: map[string]interface{}{"new": "replaced"}, Upsert: true},
	}, &BatchOptions{FailFast: true})

	result, err := batch.Execute(ctx)
	if err != nil {
		t.Fatalf("batch Execute: %v", err)
	}
	if result.Failed != 0 {
		t.Fatalf("expected 0 failures, got %d: %v", result.Failed, result.Errors)
	}

	record, err := collection.Get(ctx, "b1")
	if err != nil {
		t.Fatalf("Get after batch upsert: %v", err)
	}
	if _, ok := record.Metadata["old"]; ok {
		t.Fatalf("old metadata key must be gone after upsert, got %v", record.Metadata)
	}
	if record.Metadata["new"] != "replaced" {
		t.Fatalf("expected new='replaced', got %v", record.Metadata["new"])
	}
}
