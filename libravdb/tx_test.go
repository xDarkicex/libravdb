package libravdb

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
)

func TestCASSuccessIncrementsVersion(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "cas_success", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.Insert(ctx, "x", []float32{1, 0, 0}, map[string]interface{}{"v": 1}); err != nil {
		t.Fatalf("seed insert: %v", err)
	}

	before, err := collection.Get(ctx, "x")
	if err != nil {
		t.Fatalf("get before: %v", err)
	}
	if before.Version != 1 {
		t.Fatalf("expected initial version 1, got %d", before.Version)
	}

	if err := collection.UpdateIfVersion(ctx, "x", []float32{0, 1, 0}, map[string]interface{}{"v": 2}, 1); err != nil {
		t.Fatalf("cas update: %v", err)
	}

	after, err := collection.Get(ctx, "x")
	if err != nil {
		t.Fatalf("get after: %v", err)
	}
	if after.Version != 2 {
		t.Fatalf("expected version 2 after cas update, got %d", after.Version)
	}
}

func TestCASStaleConflictLeavesRowIntact(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "cas_conflict", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.Insert(ctx, "x", []float32{1, 0, 0}, map[string]interface{}{"v": "initial"}); err != nil {
		t.Fatalf("seed insert: %v", err)
	}
	if err := collection.UpdateIfVersion(ctx, "x", []float32{0, 1, 0}, map[string]interface{}{"v": "first"}, 1); err != nil {
		t.Fatalf("first cas update: %v", err)
	}

	err = collection.UpdateIfVersion(ctx, "x", []float32{0, 0, 1}, map[string]interface{}{"v": "stale"}, 1)
	if err == nil {
		t.Fatal("expected stale cas conflict")
	}
	if !strings.Contains(err.Error(), "version conflict") {
		t.Fatalf("expected version conflict error, got %v", err)
	}

	record, err := collection.Get(ctx, "x")
	if err != nil {
		t.Fatalf("get record: %v", err)
	}
	if record.Version != 2 {
		t.Fatalf("expected row version 2 to remain intact, got %d", record.Version)
	}
	if got := record.Metadata["v"]; got != "first" {
		t.Fatalf("expected first update to remain, got %v", got)
	}
}

func TestCASMetadataAndVectorOnlyUpdates(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "cas_partial", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.Insert(ctx, "x", []float32{1, 0, 0}, map[string]interface{}{"tag": "a"}); err != nil {
		t.Fatalf("seed insert: %v", err)
	}

	if err := collection.UpdateIfVersion(ctx, "x", nil, map[string]interface{}{"tag": "b"}, 1); err != nil {
		t.Fatalf("metadata-only cas update: %v", err)
	}
	mid, err := collection.Get(ctx, "x")
	if err != nil {
		t.Fatalf("get after metadata cas: %v", err)
	}
	if mid.Version != 2 || mid.Vector[0] != 1 || mid.Metadata["tag"] != "b" {
		t.Fatalf("unexpected metadata-only cas result: %+v", mid)
	}

	if err := collection.UpdateIfVersion(ctx, "x", []float32{0, 1, 0}, nil, 2); err != nil {
		t.Fatalf("vector-only cas update: %v", err)
	}
	after, err := collection.Get(ctx, "x")
	if err != nil {
		t.Fatalf("get after vector cas: %v", err)
	}
	if after.Version != 3 || after.Vector[1] != 1 || after.Metadata["tag"] != "b" {
		t.Fatalf("unexpected vector-only cas result: %+v", after)
	}
}

func TestCASDeleteSuccessAndStaleConflict(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "cas_delete", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.Insert(ctx, "keep", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("seed keep: %v", err)
	}
	if err := collection.Insert(ctx, "drop", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("seed drop: %v", err)
	}
	if err := collection.Update(ctx, "keep", []float32{0, 0, 1}, nil); err != nil {
		t.Fatalf("update keep: %v", err)
	}

	if err := collection.DeleteIfVersion(ctx, "drop", 1); err != nil {
		t.Fatalf("delete with correct version: %v", err)
	}
	if _, err := collection.Get(ctx, "drop"); err == nil {
		t.Fatal("expected deleted row to be gone")
	}

	err = collection.DeleteIfVersion(ctx, "keep", 1)
	if err == nil {
		t.Fatal("expected stale delete conflict")
	}
	record, err := collection.Get(ctx, "keep")
	if err != nil {
		t.Fatalf("get surviving row: %v", err)
	}
	if record.Version != 2 {
		t.Fatalf("expected stale delete to leave row intact at version 2, got %d", record.Version)
	}
}

func TestWithTxCommitsAcrossCollectionsAtomically(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	views, err := db.CreateCollection(ctx, "session_view:test", WithDimension(3))
	if err != nil {
		t.Fatalf("create views collection: %v", err)
	}
	edges, err := db.CreateCollection(ctx, "session_edge:test", WithDimension(3))
	if err != nil {
		t.Fatalf("create edges collection: %v", err)
	}

	if err := db.WithTx(ctx, func(tx Tx) error {
		if err := tx.Insert(ctx, "session_view:test", "summary-1", []float32{1, 0, 0}, map[string]interface{}{"kind": "summary"}); err != nil {
			return err
		}
		if err := tx.Insert(ctx, "session_edge:test", "edge-1", []float32{0, 1, 0}, map[string]interface{}{"kind": "coverage"}); err != nil {
			return err
		}
		return nil
	}); err != nil {
		t.Fatalf("commit transaction: %v", err)
	}

	viewCount, err := views.Count(ctx)
	if err != nil {
		t.Fatalf("count views: %v", err)
	}
	edgeCount, err := edges.Count(ctx)
	if err != nil {
		t.Fatalf("count edges: %v", err)
	}
	if viewCount != 1 || edgeCount != 1 {
		t.Fatalf("expected one record in each collection, got views=%d edges=%d", viewCount, edgeCount)
	}
}

func TestTransactionRollbackLeavesNoVisibleWrites(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	for _, name := range []string{"a", "b"} {
		if _, err := db.CreateCollection(ctx, name, WithDimension(3)); err != nil {
			t.Fatalf("create collection %s: %v", name, err)
		}
	}

	tx, err := db.BeginTx(ctx)
	if err != nil {
		t.Fatalf("begin tx: %v", err)
	}
	if err := tx.Insert(ctx, "a", "r1", []float32{1, 0, 0}, map[string]interface{}{"phase": "staged"}); err != nil {
		t.Fatalf("stage insert a: %v", err)
	}
	if err := tx.Insert(ctx, "b", "r2", []float32{0, 1, 0}, map[string]interface{}{"phase": "staged"}); err != nil {
		t.Fatalf("stage insert b: %v", err)
	}
	if err := tx.Rollback(ctx); err != nil {
		t.Fatalf("rollback tx: %v", err)
	}

	for _, name := range []string{"a", "b"} {
		collection, err := db.GetCollection(name)
		if err != nil {
			t.Fatalf("get collection %s: %v", name, err)
		}
		count, err := collection.Count(ctx)
		if err != nil {
			t.Fatalf("count %s: %v", name, err)
		}
		if count != 0 {
			t.Fatalf("expected %s to remain empty after rollback, got count=%d", name, count)
		}
	}
}

func TestTransactionDurabilityAcrossReopen(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	for _, name := range []string{"view", "edge", "state"} {
		if _, err := db.CreateCollection(ctx, name, WithDimension(3)); err != nil {
			t.Fatalf("create collection %s: %v", name, err)
		}
	}

	if err := db.WithTx(ctx, func(tx Tx) error {
		if err := tx.Insert(ctx, "view", "summary", []float32{1, 0, 0}, map[string]interface{}{"type": "summary"}); err != nil {
			return err
		}
		if err := tx.Insert(ctx, "edge", "coverage", []float32{0, 1, 0}, map[string]interface{}{"type": "edge"}); err != nil {
			return err
		}
		if err := tx.Insert(ctx, "state", "checkpoint", []float32{0, 0, 1}, map[string]interface{}{"rev": "1"}); err != nil {
			return err
		}
		return nil
	}); err != nil {
		t.Fatalf("commit tx: %v", err)
	}

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	for _, tc := range []struct {
		collection string
		id         string
	}{
		{collection: "view", id: "summary"},
		{collection: "edge", id: "coverage"},
		{collection: "state", id: "checkpoint"},
	} {
		collection, err := reopened.GetCollection(tc.collection)
		if err != nil {
			t.Fatalf("get collection %s: %v", tc.collection, err)
		}
		record, err := collection.storage.Get(ctx, tc.id)
		if err != nil {
			t.Fatalf("get %s/%s after reopen: %v", tc.collection, tc.id, err)
		}
		if record.ID != tc.id {
			t.Fatalf("expected %s after reopen, got %s", tc.id, record.ID)
		}
	}
}

func TestTransactionSameKeySemantics(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "items", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	if err := db.WithTx(ctx, func(tx Tx) error {
		if err := tx.Insert(ctx, "items", "temp", []float32{1, 0, 0}, map[string]interface{}{"version": "1"}); err != nil {
			return err
		}
		return tx.Delete(ctx, "items", "temp")
	}); err != nil {
		t.Fatalf("insert then delete tx: %v", err)
	}

	count, err := collection.Count(ctx)
	if err != nil {
		t.Fatalf("count after insert+delete: %v", err)
	}
	if count != 0 {
		t.Fatalf("expected insert then delete to leave no row, got %d", count)
	}

	if err := collection.Insert(ctx, "replace", []float32{1, 1, 0}, map[string]interface{}{"version": "old"}); err != nil {
		t.Fatalf("seed replace record: %v", err)
	}

	if err := db.WithTx(ctx, func(tx Tx) error {
		if err := tx.Delete(ctx, "items", "replace"); err != nil {
			return err
		}
		if err := tx.Insert(ctx, "items", "replace", []float32{0, 1, 1}, map[string]interface{}{"version": "new"}); err != nil {
			return err
		}
		return tx.Update(ctx, "items", "replace", nil, map[string]interface{}{"source": "tx"})
	}); err != nil {
		t.Fatalf("delete then insert tx: %v", err)
	}

	record, err := collection.storage.Get(ctx, "replace")
	if err != nil {
		t.Fatalf("get replaced record: %v", err)
	}
	if got := record.Metadata["version"]; got != "new" {
		t.Fatalf("expected replaced version metadata, got %v", got)
	}
	if got := record.Metadata["source"]; got != "tx" {
		t.Fatalf("expected merged update metadata, got %v", got)
	}
}

func TestTransactionStagedInsertOverwriteIsDeterministic(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "overwrite", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	if err := db.WithTx(ctx, func(tx Tx) error {
		if err := tx.Insert(ctx, "overwrite", "item", []float32{1, 0, 0}, map[string]interface{}{"version": "v1"}); err != nil {
			return err
		}
		return tx.Insert(ctx, "overwrite", "item", []float32{0, 1, 0}, map[string]interface{}{"version": "v2"})
	}); err != nil {
		t.Fatalf("commit tx: %v", err)
	}

	record, err := collection.storage.Get(ctx, "item")
	if err != nil {
		t.Fatalf("get item: %v", err)
	}
	if got := record.Metadata["version"]; got != "v2" {
		t.Fatalf("expected later staged insert to win, got %v", got)
	}
	if record.Vector[1] != 1 {
		t.Fatalf("expected later staged vector to win, got %v", record.Vector)
	}
}

func TestTransactionReadersNeverObserveOddCounts(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "pairs", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	const txCount = 20
	errCh := make(chan error, 2)
	done := make(chan struct{})

	go func() {
		defer close(done)
		for i := 0; i < txCount; i++ {
			err := db.WithTx(ctx, func(tx Tx) error {
				if err := tx.Insert(ctx, "pairs", fmt.Sprintf("left-%d", i), []float32{1, 0, 0}, map[string]interface{}{"pair": i}); err != nil {
					return err
				}
				if err := tx.Insert(ctx, "pairs", fmt.Sprintf("right-%d", i), []float32{0, 1, 0}, map[string]interface{}{"pair": i}); err != nil {
					return err
				}
				return nil
			})
			if err != nil {
				errCh <- err
				return
			}
		}
	}()

	go func() {
		for {
			select {
			case <-done:
				return
			default:
			}

			count, err := collection.Count(ctx)
			if err != nil {
				errCh <- err
				return
			}
			if count%2 != 0 {
				errCh <- fmt.Errorf("observed torn count %d", count)
				return
			}

			records, err := collection.ListAll(ctx)
			if err != nil {
				errCh <- err
				return
			}
			if len(records)%2 != 0 {
				errCh <- fmt.Errorf("observed torn iteration length %d", len(records))
				return
			}

			results, err := collection.Search(ctx, []float32{1, 0, 0}, 1000)
			if err != nil {
				if !strings.Contains(err.Error(), "index is empty") {
					errCh <- err
					return
				}
				continue
			}
			if len(results.Results)%2 != 0 {
				errCh <- fmt.Errorf("observed torn search length %d", len(results.Results))
				return
			}
		}
	}()

	var once sync.Once
	select {
	case err := <-errCh:
		once.Do(func() { t.Fatalf("concurrency check failed: %v", err) })
	case <-done:
	}

	finalCount, err := collection.Count(ctx)
	if err != nil {
		t.Fatalf("final count: %v", err)
	}
	if finalCount != txCount*2 {
		t.Fatalf("expected final count %d, got %d", txCount*2, finalCount)
	}
}

func TestTransactionManyCommitsRemainCorrectAfterReopen(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}

	for _, name := range []string{"summary", "edges"} {
		if _, err := db.CreateCollection(ctx, name, WithDimension(3)); err != nil {
			t.Fatalf("create collection %s: %v", name, err)
		}
	}

	const txCount = 25
	for i := 0; i < txCount; i++ {
		i := i
		if err := db.WithTx(ctx, func(tx Tx) error {
			if err := tx.Insert(ctx, "summary", fmt.Sprintf("summary-%d", i), []float32{1, 0, 0}, map[string]interface{}{"tx": i}); err != nil {
				return err
			}
			return tx.Insert(ctx, "edges", fmt.Sprintf("edge-%d", i), []float32{0, 1, 0}, map[string]interface{}{"tx": i})
		}); err != nil {
			t.Fatalf("commit tx %d: %v", i, err)
		}
	}

	if err := db.Close(); err != nil {
		t.Fatalf("close db: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen db: %v", err)
	}
	defer reopened.Close()

	for _, name := range []string{"summary", "edges"} {
		collection, err := reopened.GetCollection(name)
		if err != nil {
			t.Fatalf("get collection %s: %v", name, err)
		}
		count, err := collection.Count(ctx)
		if err != nil {
			t.Fatalf("count %s: %v", name, err)
		}
		if count != txCount {
			t.Fatalf("expected %d rows in %s after reopen, got %d", txCount, name, count)
		}
	}
}

func TestCASConflictAbortsWholeTransaction(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	left, err := db.CreateCollection(ctx, "left", WithDimension(3))
	if err != nil {
		t.Fatalf("create left: %v", err)
	}
	right, err := db.CreateCollection(ctx, "right", WithDimension(3))
	if err != nil {
		t.Fatalf("create right: %v", err)
	}
	if err := left.Insert(ctx, "row", []float32{1, 0, 0}, map[string]interface{}{"state": "base"}); err != nil {
		t.Fatalf("seed left: %v", err)
	}
	if err := left.Update(ctx, "row", []float32{0, 1, 0}, map[string]interface{}{"state": "newer"}); err != nil {
		t.Fatalf("bump left version: %v", err)
	}

	err = db.WithTx(ctx, func(tx Tx) error {
		if err := tx.UpdateIfVersion(ctx, "left", "row", []float32{0, 0, 1}, map[string]interface{}{"state": "stale"}, 1); err != nil {
			return err
		}
		return tx.Insert(ctx, "right", "other", []float32{1, 1, 0}, map[string]interface{}{"state": "should_not_commit"})
	})
	if err == nil {
		t.Fatal("expected transaction-wide abort on CAS conflict")
	}

	leftRow, err := left.Get(ctx, "row")
	if err != nil {
		t.Fatalf("get left row: %v", err)
	}
	if leftRow.Version != 2 || leftRow.Metadata["state"] != "newer" {
		t.Fatalf("left row unexpectedly changed: %+v", leftRow)
	}

	if _, err := right.Get(ctx, "other"); err == nil {
		t.Fatal("expected right-side insert to be rolled back")
	}
}

func TestCASConcurrentWritersOnlyOneSucceeds(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "cas_race", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.Insert(ctx, "x", []float32{1, 0, 0}, map[string]interface{}{"winner": "none"}); err != nil {
		t.Fatalf("seed insert: %v", err)
	}

	errCh := make(chan error, 2)
	var wg sync.WaitGroup
	for _, label := range []string{"a", "b"} {
		label := label
		wg.Add(1)
		go func() {
			defer wg.Done()
			errCh <- collection.UpdateIfVersion(ctx, "x", nil, map[string]interface{}{"winner": label}, 1)
		}()
	}
	wg.Wait()
	close(errCh)

	successes := 0
	conflicts := 0
	for err := range errCh {
		if err == nil {
			successes++
			continue
		}
		if strings.Contains(err.Error(), "version conflict") {
			conflicts++
			continue
		}
		t.Fatalf("unexpected concurrent CAS error: %v", err)
	}
	if successes != 1 || conflicts != 1 {
		t.Fatalf("expected exactly one success and one conflict, got successes=%d conflicts=%d", successes, conflicts)
	}
}

func TestCASVersionPersistsAcrossReopen(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("new db: %v", err)
	}
	collection, err := db.CreateCollection(ctx, "cas_reopen", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.Insert(ctx, "x", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("seed insert: %v", err)
	}
	if err := collection.UpdateIfVersion(ctx, "x", []float32{0, 1, 0}, nil, 1); err != nil {
		t.Fatalf("cas update before reopen: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("close db: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen db: %v", err)
	}
	defer reopened.Close()

	reloaded, err := reopened.GetCollection("cas_reopen")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}
	record, err := reloaded.Get(ctx, "x")
	if err != nil {
		t.Fatalf("get after reopen: %v", err)
	}
	if record.Version != 2 {
		t.Fatalf("expected version 2 after reopen, got %d", record.Version)
	}
	if err := reloaded.UpdateIfVersion(ctx, "x", []float32{0, 0, 1}, nil, 1); err == nil {
		t.Fatal("expected stale expected-version to fail after reopen")
	}
}

func TestCASSameKeyConflictingExpectedVersionsInOneTxFails(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new db: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "cas_same_key_conflict", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.Insert(ctx, "x", []float32{1, 0, 0}, map[string]interface{}{"state": "base"}); err != nil {
		t.Fatalf("seed insert: %v", err)
	}

	err = db.WithTx(ctx, func(tx Tx) error {
		if err := tx.UpdateIfVersion(ctx, "cas_same_key_conflict", "x", nil, map[string]interface{}{"step": "one"}, 1); err != nil {
			return err
		}
		return tx.UpdateIfVersion(ctx, "cas_same_key_conflict", "x", nil, map[string]interface{}{"step": "two"}, 2)
	})
	if err == nil {
		t.Fatal("expected conflicting expected versions in one tx to fail")
	}
	if !strings.Contains(err.Error(), "conflicting expected versions") {
		t.Fatalf("expected conflicting expected versions error, got %v", err)
	}

	record, err := collection.Get(ctx, "x")
	if err != nil {
		t.Fatalf("get row after failed tx: %v", err)
	}
	if record.Version != 1 || record.Metadata["state"] != "base" {
		t.Fatalf("row changed despite conflicting staged expectations: %+v", record)
	}
}

func TestCASSameKeyRepeatedSameExpectedVersionInOneTxSucceeds(t *testing.T) {
	ctx := context.Background()
	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("new db: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "cas_same_key_repeat", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.Insert(ctx, "x", []float32{1, 0, 0}, map[string]interface{}{"state": "base"}); err != nil {
		t.Fatalf("seed insert: %v", err)
	}

	err = db.WithTx(ctx, func(tx Tx) error {
		if err := tx.UpdateIfVersion(ctx, "cas_same_key_repeat", "x", nil, map[string]interface{}{"step": "one"}, 1); err != nil {
			return err
		}
		return tx.UpdateIfVersion(ctx, "cas_same_key_repeat", "x", []float32{0, 1, 0}, map[string]interface{}{"step": "two"}, 1)
	})
	if err != nil {
		t.Fatalf("expected repeated same-version CAS in one tx to succeed, got %v", err)
	}

	record, err := collection.Get(ctx, "x")
	if err != nil {
		t.Fatalf("get row after commit: %v", err)
	}
	if record.Version != 2 {
		t.Fatalf("expected final version 2, got %d", record.Version)
	}
	if got := record.Metadata["step"]; got != "two" {
		t.Fatalf("expected later staged state to win, got %v", got)
	}
	if record.Vector[1] != 1 {
		t.Fatalf("expected later staged vector to win, got %v", record.Vector)
	}
}
