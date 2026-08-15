package libravdb

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"testing"
)

func TestMixedTransactionalAndDirectWritesKeepOrdinalsUnique(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/mixed-ordinal.libravdb"
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	col, err := db.CreateCollection(ctx, "docs", WithDimension(1), WithFlat(),
		WithMetadataSchema(MetadataSchema{"kind": StringField}), WithIndexedFields("kind"))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := db.WithTxReceipt(ctx, func(tx ReceiptTx) error {
		return tx.Insert(ctx, "docs", "tx-before", []float32{0}, map[string]interface{}{"kind": "target"})
	}); err != nil {
		t.Fatalf("transactional insert: %v", err)
	}
	if err := col.Upsert(ctx, "direct-one", []float32{0}, map[string]interface{}{"kind": "other"}); err != nil {
		t.Fatalf("direct upsert: %v", err)
	}
	if err := col.Insert(ctx, "direct-two", []float32{0}, map[string]interface{}{"kind": "other"}); err != nil {
		t.Fatalf("direct insert: %v", err)
	}
	if _, err := db.WithTxReceipt(ctx, func(tx ReceiptTx) error {
		return tx.Insert(ctx, "docs", "tx-after", []float32{0}, map[string]interface{}{"kind": "target"})
	}); err != nil {
		t.Fatalf("second transactional insert: %v", err)
	}

	// A rolled-back transaction must not make a later direct write reuse a
	// committed ordinal. The implementation may leave a gap, but never a
	// collision.
	tx, err := db.BeginTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := tx.Insert(ctx, "docs", "rolled-back", []float32{0}, map[string]interface{}{"kind": "discarded"}); err != nil {
		t.Fatal(err)
	}
	if err := tx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if err := col.Upsert(ctx, "direct-after-rollback", []float32{0}, map[string]interface{}{"kind": "other"}); err != nil {
		t.Fatalf("direct write after rollback: %v", err)
	}

	assertOrdinalSet(t, ctx, col, []string{"tx-before", "direct-one", "direct-two", "tx-after", "direct-after-rollback"})
	targets, err := col.ListByMetadata(ctx, "kind", "target")
	if err != nil {
		t.Fatal(err)
	}
	if got := recordIDs(targets); fmt.Sprint(got) != "[tx-after tx-before]" {
		t.Fatalf("target metadata IDs=%v, want [tx-after tx-before]", got)
	}

	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer db.Close()
	reopened, err := db.GetCollection("docs")
	if err != nil {
		t.Fatal(err)
	}
	assertOrdinalSet(t, ctx, reopened, []string{"tx-before", "direct-one", "direct-two", "tx-after", "direct-after-rollback"})
}

func TestConcurrentTransactionalAndDirectOrdinalReservations(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:ordinal-concurrency"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "docs", WithDimension(1), WithFlat())
	if err != nil {
		t.Fatal(err)
	}

	const writers = 24
	errs := make(chan error, writers)
	var wg sync.WaitGroup
	for i := 0; i < writers; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			id := fmt.Sprintf("row-%02d", i)
			if i%2 == 0 {
				_, err := db.WithTxReceipt(ctx, func(tx ReceiptTx) error {
					return tx.Insert(ctx, "docs", id, []float32{0}, nil)
				})
				errs <- err
				return
			}
			errs <- col.Upsert(ctx, id, []float32{0}, nil)
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	assertOrdinalSet(t, ctx, col, func() []string {
		ids := make([]string, writers)
		for i := range ids {
			ids[i] = fmt.Sprintf("row-%02d", i)
		}
		return ids
	}())
}

type ordinalCollection interface {
	Count(context.Context) (int, error)
	ListAll(context.Context) ([]Record, error)
	Get(context.Context, string) (Record, error)
}

func assertOrdinalSet(t *testing.T, ctx context.Context, col ordinalCollection, expected []string) {
	t.Helper()
	all, err := col.ListAll(ctx)
	if err != nil {
		t.Fatal(err)
	}
	count, err := col.Count(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if count != len(all) || len(all) != len(expected) {
		t.Fatalf("count=%d list=%d expected=%d records=%#v", count, len(all), len(expected), all)
	}
	seenIDs := make(map[string]struct{}, len(all))
	seenOrdinals := make(map[uint32]string, len(all))
	for _, record := range all {
		seenIDs[record.ID] = struct{}{}
		if previous, exists := seenOrdinals[record.Ordinal]; exists {
			t.Fatalf("ordinal %d assigned to both %s and %s", record.Ordinal, previous, record.ID)
		}
		seenOrdinals[record.Ordinal] = record.ID
		if _, err := col.Get(ctx, record.ID); err != nil {
			t.Fatalf("Get(%s): %v", record.ID, err)
		}
	}
	for _, id := range expected {
		if _, ok := seenIDs[id]; !ok {
			t.Fatalf("ListAll omitted %s: %#v", id, all)
		}
	}
}

func recordIDs(records []Record) []string {
	ids := make([]string, 0, len(records))
	for _, record := range records {
		ids = append(ids, record.ID)
	}
	sort.Strings(ids)
	return ids
}
