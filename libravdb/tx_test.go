package libravdb

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
)

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
