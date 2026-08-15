package libravdb

import (
	"context"
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"testing"
)

func TestSQLSelectReturnsAllRows(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:large-select-regression"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	if _, err := db.Query(ctx, "CREATE TABLE large_users (id TEXT PRIMARY KEY, name TEXT)"); err != nil {
		t.Fatal(err)
	}
	const want = 500
	for i := 0; i < want; i++ {
		if _, err := db.QueryWithParams(ctx, "INSERT INTO large_users (id, name) VALUES ($1, $2)", QueryParams{
			"1": fmt.Sprintf("node-%03d", i),
			"2": fmt.Sprintf("name-%03d", want-i),
		}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	assertSQLResultIDs(t, db, ctx, "large_users", want)
}

func TestSQLSelectAfterMixedTransactionalAndDirectWritesReturnsAllRows(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/mixed-large-select.libravdb"
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	col, err := db.CreateCollection(ctx, "mixed_users", WithDimension(1), WithFlat(),
		WithMetadataSchema(MetadataSchema{"name": StringField}), WithIndexedFields("name"))
	if err != nil {
		t.Fatal(err)
	}

	const want = 316
	if err := db.WithTx(ctx, func(tx Tx) error {
		for i := 0; i < want/2; i++ {
			if err := tx.Insert(ctx, "mixed_users", fmt.Sprintf("tx-%03d", i), []float32{0}, map[string]interface{}{
				"name": fmt.Sprintf("name-%03d", i),
			}); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatal("transactional inserts:", err)
	}
	for i := want / 2; i < want; i++ {
		if err := col.Upsert(ctx, fmt.Sprintf("direct-%03d", i), []float32{0}, map[string]interface{}{
			"name": fmt.Sprintf("name-%03d", i),
		}); err != nil {
			t.Fatalf("direct insert %d: %v", i, err)
		}
	}

	assertSQLResultIDs(t, db, ctx, "mixed_users", want)

	if err := db.Close(); err != nil {
		t.Fatal("close:", err)
	}
	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal("reopen:", err)
	}
	defer db.Close()
	assertSQLResultIDs(t, db, ctx, "mixed_users", want)
}

func TestSQLMultiRowInsertThenSelectReturnsAllRows(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:multi-row-large-select"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT)"); err != nil {
		t.Fatal(err)
	}

	const want = 316
	values := make([]string, 0, want)
	for i := 0; i < want; i++ {
		values = append(values, fmt.Sprintf("('node-%03d','name-%03d')", i, want-i))
	}
	query := "INSERT INTO users (id, name) VALUES " + strings.Join(values, ",")
	if _, err := db.Query(ctx, query); err != nil {
		t.Fatal("multi-row insert:", err)
	}
	assertSQLResultIDs(t, db, ctx, "users", want)
}

func TestSQLSelectVariableLengthIDsAfterReopenReturnsAllRows(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/variable-ids.libravdb"
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT)"); err != nil {
		t.Fatal(err)
	}

	const want = 500
	ids := make([]string, 0, want)
	for i := 0; i < want; i++ {
		ids = append(ids, fmt.Sprintf("city-%d-%s", i, strings.Repeat("x", i%17)))
	}
	// Use a deterministic non-key-sorted insertion order to exercise B-tree
	// splits and merges across varied key lengths.
	rand.New(rand.NewSource(42)).Shuffle(len(ids), func(i, j int) { ids[i], ids[j] = ids[j], ids[i] })
	for i, id := range ids {
		if _, err := db.QueryWithParams(ctx, "INSERT INTO users (id, name) VALUES ($1, $2)", QueryParams{
			"1": id,
			"2": fmt.Sprintf("place-%04d", i),
		}); err != nil {
			t.Fatalf("insert %d (%s): %v", i, id, err)
		}
	}
	assertSQLResultIDs(t, db, ctx, "users", want)

	if err := db.Close(); err != nil {
		t.Fatal("close:", err)
	}
	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal("reopen:", err)
	}
	defer db.Close()
	assertSQLResultIDs(t, db, ctx, "users", want)

	result, err := db.Query(ctx, "SELECT id FROM users ORDER BY id")
	if err != nil {
		t.Fatal("ordered select:", err)
	}
	got := make([]string, 0, len(result.Results))
	for _, row := range result.Results {
		got = append(got, row.ID)
	}
	wantIDs := append([]string(nil), ids...)
	sort.Strings(wantIDs)
	if fmt.Sprint(got) != fmt.Sprint(wantIDs) {
		t.Fatalf("ordered IDs differ: got %d rows, want %d", len(got), len(wantIDs))
	}
}

func assertSQLResultIDs(t *testing.T, db *Database, ctx context.Context, table string, want int) {
	t.Helper()
	countResult, err := db.Query(ctx, "SELECT COUNT(*) FROM "+table)
	if err != nil {
		t.Fatal("count:", err)
	}
	if len(countResult.Results) != 1 || countResult.Results[0].Metadata["count"] != int64(want) {
		t.Fatalf("count result=%#v, want %d", countResult.Results, want)
	}

	result, err := db.Query(ctx, "SELECT id FROM "+table+" ORDER BY name")
	if err != nil {
		t.Fatal("select:", err)
	}
	if len(result.Results) != want {
		t.Fatalf("SELECT returned %d rows, want %d", len(result.Results), want)
	}
	seen := make(map[string]struct{}, want)
	for _, row := range result.Results {
		if row.ID == "" {
			t.Fatalf("SELECT returned empty row ID: %#v", row)
		}
		if _, exists := seen[row.ID]; exists {
			t.Fatalf("SELECT returned duplicate row ID %q", row.ID)
		}
		seen[row.ID] = struct{}{}
	}
	if len(seen) != want {
		t.Fatalf("SELECT returned %d unique IDs, want %d", len(seen), want)
	}
}
