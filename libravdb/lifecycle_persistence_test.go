package libravdb

import (
	"context"
	"sort"
	"strings"
	"testing"
)

func TestListCollectionsPersistsAcrossReopen(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	names := []string{"session:s1", "user:u1:64d", "global"}
	for _, name := range names {
		if _, err := db.CreateCollection(ctx, name, WithDimension(3)); err != nil {
			t.Fatalf("create collection %s: %v", name, err)
		}
	}

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	got, err := reopened.ListCollectionsWithContext(ctx)
	if err != nil {
		t.Fatalf("list collections with context: %v", err)
	}
	sort.Strings(names)
	if len(got) != len(names) {
		t.Fatalf("expected %d collections, got %d: %v", len(names), len(got), got)
	}

	for i := range names {
		if got[i] != names[i] {
			t.Fatalf("expected collections %v, got %v", names, got)
		}
	}
}

func TestCollectionIterationAndMetadataPersistAcrossReopen(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	collection, err := db.CreateCollection(ctx, "user:u1", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	entries := []VectorEntry{
		{ID: "r1", Vector: []float32{1, 0, 0}, Metadata: map[string]interface{}{"sessionId": "s1", "source": "spec"}},
		{ID: "r2", Vector: []float32{0, 1, 0}, Metadata: map[string]interface{}{"sessionId": "s1", "source": "chat"}},
		{ID: "r3", Vector: []float32{0, 0, 1}, Metadata: map[string]interface{}{"sessionId": "s2", "source": "spec"}},
	}
	if err := collection.InsertBatch(ctx, entries); err != nil {
		t.Fatalf("insert batch: %v", err)
	}

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	reloaded, err := reopened.GetCollection("user:u1")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}

	records, err := reloaded.ListAll(ctx)
	if err != nil {
		t.Fatalf("list all: %v", err)
	}
	if len(records) != len(entries) {
		t.Fatalf("expected %d records, got %d", len(entries), len(records))
	}

	byID := make(map[string]Record, len(records))
	for _, record := range records {
		byID[record.ID] = record
	}

	for _, entry := range entries {
		record, ok := byID[entry.ID]
		if !ok {
			t.Fatalf("missing record %s after reopen", entry.ID)
		}
		if len(record.Vector) != len(entry.Vector) {
			t.Fatalf("vector length mismatch for %s", entry.ID)
		}
		if record.Metadata["sessionId"] != entry.Metadata["sessionId"] {
			t.Fatalf("sessionId mismatch for %s: got %v want %v", entry.ID, record.Metadata["sessionId"], entry.Metadata["sessionId"])
		}
		if record.Metadata["source"] != entry.Metadata["source"] {
			t.Fatalf("source mismatch for %s: got %v want %v", entry.ID, record.Metadata["source"], entry.Metadata["source"])
		}
	}

	results, err := reloaded.Search(ctx, []float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("search after reopen: %v", err)
	}
	if len(results.Results) == 0 {
		t.Fatalf("expected search results after reopen")
	}
}

func TestDeleteCollectionPersistsAcrossReopen(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	for _, name := range []string{"user:u1", "user:u1:256d"} {
		if _, err := db.CreateCollection(ctx, name, WithDimension(3)); err != nil {
			t.Fatalf("create collection %s: %v", name, err)
		}
	}

	if err := db.DeleteCollection(ctx, "user:u1"); err != nil {
		t.Fatalf("delete collection: %v", err)
	}

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	got := reopened.ListCollections()
	if len(got) != 1 || got[0] != "user:u1:256d" {
		t.Fatalf("expected only surviving collection, got %v", got)
	}
}

func TestDeleteCollectionsPersistsAcrossReopen(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	for _, name := range []string{"user:u1", "user:u1:64d", "global"} {
		if _, err := db.CreateCollection(ctx, name, WithDimension(3)); err != nil {
			t.Fatalf("create collection %s: %v", name, err)
		}
	}

	if err := db.DeleteCollections(ctx, []string{"user:u1", "global"}); err != nil {
		t.Fatalf("delete collections: %v", err)
	}

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	got, err := reopened.ListCollectionsWithContext(ctx)
	if err != nil {
		t.Fatalf("list collections with context: %v", err)
	}
	if len(got) != 1 || got[0] != "user:u1:64d" {
		t.Fatalf("expected only surviving collection, got %v", got)
	}
}

func TestDeleteBatchRemovesRecordsFromIterationAndSearch(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "turns:t1", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	entries := []VectorEntry{
		{ID: "keep", Vector: []float32{1, 0, 0}, Metadata: map[string]interface{}{"userId": "u1"}},
		{ID: "drop1", Vector: []float32{0, 1, 0}, Metadata: map[string]interface{}{"userId": "u1"}},
		{ID: "drop2", Vector: []float32{0, 0, 1}, Metadata: map[string]interface{}{"userId": "u2"}},
	}
	if err := collection.InsertBatch(ctx, entries); err != nil {
		t.Fatalf("insert batch: %v", err)
	}

	if err := collection.DeleteBatch(ctx, []string{"drop1", "drop2"}); err != nil {
		t.Fatalf("delete batch: %v", err)
	}

	records, err := collection.ListAll(ctx)
	if err != nil {
		t.Fatalf("list all: %v", err)
	}
	if len(records) != 1 || records[0].ID != "keep" {
		t.Fatalf("expected only keep record, got %v", records)
	}

	results, err := collection.Search(ctx, []float32{0, 1, 0}, 3)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	for _, result := range results.Results {
		if result.ID == "drop1" || result.ID == "drop2" {
			t.Fatalf("deleted record %s still returned by search", result.ID)
		}
	}
}

func TestListByMetadataWorksBeforeAndAfterReopen(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	collection, err := db.CreateCollection(ctx, "session:s1", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	entries := []VectorEntry{
		{ID: "a", Vector: []float32{1, 0, 0}, Metadata: map[string]interface{}{"sessionId": "s1", "source": "spec"}},
		{ID: "b", Vector: []float32{0, 1, 0}, Metadata: map[string]interface{}{"sessionId": "s1", "source": "notes"}},
		{ID: "c", Vector: []float32{0, 0, 1}, Metadata: map[string]interface{}{"sessionId": "s2", "source": "spec"}},
	}
	if err := collection.InsertBatch(ctx, entries); err != nil {
		t.Fatalf("insert batch: %v", err)
	}

	checkMatches := func(c *Collection) {
		t.Helper()

		sessionMatches, err := c.ListByMetadata(ctx, "sessionId", "s1")
		if err != nil {
			t.Fatalf("list by sessionId: %v", err)
		}
		if len(sessionMatches) != 2 {
			t.Fatalf("expected 2 session matches, got %d", len(sessionMatches))
		}

		sourceMatches, err := c.ListByMetadata(ctx, "source", "spec")
		if err != nil {
			t.Fatalf("list by source: %v", err)
		}
		if len(sourceMatches) != 2 {
			t.Fatalf("expected 2 source matches, got %d", len(sourceMatches))
		}
	}

	checkMatches(collection)

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	reloaded, err := reopened.GetCollection("session:s1")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}

	checkMatches(reloaded)
}

func TestCollectionCountReflectsPersistedState(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	collection, err := db.CreateCollection(ctx, "global", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	if err := collection.InsertBatch(ctx, []VectorEntry{
		{ID: "1", Vector: []float32{1, 0, 0}},
		{ID: "2", Vector: []float32{0, 1, 0}},
		{ID: "3", Vector: []float32{0, 0, 1}},
	}); err != nil {
		t.Fatalf("insert batch: %v", err)
	}

	count, err := collection.Count(ctx)
	if err != nil {
		t.Fatalf("count before reopen: %v", err)
	}
	if count != 3 {
		t.Fatalf("expected count 3 before reopen, got %d", count)
	}

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	reloaded, err := reopened.GetCollection("global")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}

	count, err = reloaded.Count(ctx)
	if err != nil {
		t.Fatalf("count after reopen: %v", err)
	}
	if count != 3 {
		t.Fatalf("expected count 3 after reopen, got %d", count)
	}
}

func TestQueryBuilderListSupportsMetadataOnlyListing(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	collection, err := db.CreateCollection(ctx, "session:q1", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	if err := collection.InsertBatch(ctx, []VectorEntry{
		{ID: "1", Vector: []float32{1, 0, 0}, Metadata: map[string]interface{}{"sessionId": "s1", "rank": 1}},
		{ID: "2", Vector: []float32{0, 1, 0}, Metadata: map[string]interface{}{"sessionId": "s1", "rank": 2}},
		{ID: "3", Vector: []float32{0, 0, 1}, Metadata: map[string]interface{}{"sessionId": "s2", "rank": 3}},
	}); err != nil {
		t.Fatalf("insert batch: %v", err)
	}

	checkMatches := func(c *Collection) {
		t.Helper()

		records, err := c.Query(ctx).Eq("sessionId", "s1").Limit(10).List()
		if err != nil {
			t.Fatalf("query builder metadata list: %v", err)
		}
		if len(records) != 2 {
			t.Fatalf("expected 2 records, got %d", len(records))
		}

		records, err = c.Query(ctx).Between("rank", 2, 3).Limit(10).List()
		if err != nil {
			t.Fatalf("query builder numeric list: %v", err)
		}
		if len(records) != 2 {
			t.Fatalf("expected 2 numeric records, got %d", len(records))
		}
	}

	checkMatches(collection)

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	reloaded, err := reopened.GetCollection("session:q1")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}

	checkMatches(reloaded)
}

func TestQueryBuilderListHonorsThresholdWithVectorQuery(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	collection, err := db.CreateCollection(ctx, "session:threshold", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	if err := collection.InsertBatch(ctx, []VectorEntry{
		{ID: "exact", Vector: []float32{1, 0, 0}, Metadata: map[string]interface{}{"sessionId": "s1"}},
		{ID: "close", Vector: []float32{0.9, 0.1, 0}, Metadata: map[string]interface{}{"sessionId": "s1"}},
		{ID: "far", Vector: []float32{0, 1, 0}, Metadata: map[string]interface{}{"sessionId": "s1"}},
	}); err != nil {
		t.Fatalf("insert batch: %v", err)
	}

	checkMatches := func(c *Collection) {
		t.Helper()

		results, err := c.Query(ctx).
			WithVector([]float32{1, 0, 0}).
			WithThreshold(0.8).
			Limit(10).
			Execute()
		if err != nil {
			t.Fatalf("query builder threshold execute: %v", err)
		}

		records, err := c.Query(ctx).
			WithVector([]float32{1, 0, 0}).
			WithThreshold(0.8).
			Limit(10).
			List()
		if err != nil {
			t.Fatalf("query builder threshold list: %v", err)
		}
		if len(records) != len(results.Results) {
			t.Fatalf("threshold list/execute mismatch: list=%d execute=%d", len(records), len(results.Results))
		}

		recordIDs := make(map[string]struct{}, len(records))
		for _, record := range records {
			recordIDs[record.ID] = struct{}{}
		}
		for _, result := range results.Results {
			if _, ok := recordIDs[result.ID]; !ok {
				t.Fatalf("threshold list missing execute result %q: list=%v execute=%v", result.ID, records, results.Results)
			}
		}
	}

	checkMatches(collection)

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	reloaded, err := reopened.GetCollection("session:threshold")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}

	checkMatches(reloaded)
}

func TestShardedCollectionReopen(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	collection, err := db.CreateCollection(ctx, "sharded_test", WithDimension(3), WithSharding(true))
	if err != nil {
		t.Fatalf("create sharded collection: %v", err)
	}

	entries := []VectorEntry{
		{ID: "a", Vector: []float32{1, 0, 0}},
		{ID: "b", Vector: []float32{0, 1, 0}},
		{ID: "c", Vector: []float32{0, 0, 1}},
	}
	if err := collection.InsertBatch(ctx, entries); err != nil {
		t.Fatalf("insert batch: %v", err)
	}

	results, err := collection.Search(ctx, []float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("search before close: %v", err)
	}
	if len(results.Results) == 0 {
		t.Fatal("expected search results before close")
	}

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	reloaded, err := reopened.GetCollection("sharded_test")
	if err != nil {
		t.Fatalf("get collection after reopen: %v", err)
	}

	results, err = reloaded.Search(ctx, []float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("search after reopen: %v", err)
	}
	if len(results.Results) == 0 {
		t.Fatal("expected search results after reopen")
	}

	count, err := reloaded.Count(ctx)
	if err != nil {
		t.Fatalf("count after reopen: %v", err)
	}
	if count != 3 {
		t.Fatalf("expected count 3, got %d", count)
	}
}

func TestShardedCollectionListHidesShardNames(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	_, err = db.CreateCollection(ctx, "sharded_parent", WithDimension(3), WithSharding(true))
	if err != nil {
		t.Fatalf("create sharded collection: %v", err)
	}

	_, err = db.CreateCollection(ctx, "normal_collection", WithDimension(3))
	if err != nil {
		t.Fatalf("create normal collection: %v", err)
	}

	got := db.ListCollections()
	hasShardName := false
	for _, name := range got {
		if strings.Contains(name, "__shard__") {
			hasShardName = true
			break
		}
	}
	if hasShardName {
		t.Fatalf("ListCollections should not expose shard names, got %v", got)
	}

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	got = reopened.ListCollections()
	hasShardName = false
	for _, name := range got {
		if strings.Contains(name, "__shard__") {
			hasShardName = true
			break
		}
	}
	if hasShardName {
		t.Fatalf("ListCollections after reopen should not expose shard names, got %v", got)
	}
}

func TestShardedCollectionDeleteRemovesAllShards(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	_, err = db.CreateCollection(ctx, "sharded_to_delete", WithDimension(3), WithSharding(true))
	if err != nil {
		t.Fatalf("create sharded collection: %v", err)
	}

	if err := db.DeleteCollection(ctx, "sharded_to_delete"); err != nil {
		t.Fatalf("delete collection: %v", err)
	}

	got := db.ListCollections()
	for _, name := range got {
		if name == "sharded_to_delete" {
			t.Fatal("deleted collection should not appear in list")
		}
	}

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	_, err = reopened.GetCollection("sharded_to_delete")
	if err == nil {
		t.Fatal("expected error when getting deleted collection after reopen")
	}
}

func TestShardedCollectionCloseCleansUpResources(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	collection, err := db.CreateCollection(ctx, "sharded_close_test", WithDimension(3), WithSharding(true))
	if err != nil {
		t.Fatalf("create sharded collection: %v", err)
	}

	entries := []VectorEntry{
		{ID: "x", Vector: []float32{1, 0, 0}},
		{ID: "y", Vector: []float32{0, 1, 0}},
	}
	if err := collection.InsertBatch(ctx, entries); err != nil {
		t.Fatalf("insert batch: %v", err)
	}

	// Close the collection (in-memory resources only, not persisted data)
	if err := collection.Close(); err != nil {
		t.Fatalf("close collection: %v", err)
	}

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	// Reopen - collection should still exist since we didn't delete it
	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	// Verify collection can still be accessed after just Close()
	reloaded, err := reopened.GetCollection("sharded_close_test")
	if err != nil {
		t.Fatalf("expected collection to exist after close, got: %v", err)
	}

	count, err := reloaded.Count(ctx)
	if err != nil {
		t.Fatalf("count after reopen: %v", err)
	}
	if count != 2 {
		t.Fatalf("expected count 2, got %d", count)
	}

	// Now delete it and verify it's gone
	if err := reopened.DeleteCollection(ctx, "sharded_close_test"); err != nil {
		t.Fatalf("delete collection: %v", err)
	}

	if err := reopened.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened2, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database after delete: %v", err)
	}
	defer reopened2.Close()

	_, err = reopened2.GetCollection("sharded_close_test")
	if err == nil {
		t.Fatal("expected error getting deleted collection after reopen")
	}
}
