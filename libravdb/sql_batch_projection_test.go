package libravdb

import (
	"context"
	"fmt"
	"testing"
)

func TestSQLBatchPrimaryKeyProjectionUsesBTree(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_batch_projection"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	if _, err := db.Query(ctx, "CREATE TABLE profiles (id TEXT PRIMARY KEY, name TEXT)"); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 256; i++ {
		if _, err := db.QueryWithParams(ctx,
			"INSERT INTO profiles (id, name) VALUES ($1, $2)",
			QueryParams{"1": fmt.Sprintf("profile-%03d", i), "2": fmt.Sprintf("Profile %03d", i)},
		); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	db.ResetSQLStats()
	rows, err := db.QueryWithParams(ctx,
		"SELECT id, name FROM profiles WHERE id IN ($1, $2, $3) ORDER BY id",
		QueryParams{"1": "profile-200", "2": "profile-017", "3": "missing"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if rows.Total != 2 || len(rows.Results) != 2 {
		t.Fatalf("batch projection rows=%#v total=%d, want two rows", rows.Results, rows.Total)
	}
	if rows.Results[0].ID != "profile-017" || rows.Results[1].ID != "profile-200" {
		t.Fatalf("batch projection IDs=%q,%q, want profile-017,profile-200", rows.Results[0].ID, rows.Results[1].ID)
	}

	stats := db.SQLStats()
	if stats.IndexHits != 3 {
		t.Fatalf("batch projection index_hits=%d, want one B-tree probe per requested ID", stats.IndexHits)
	}
	if stats.RowsExamined != 3 {
		t.Fatalf("batch projection rows_examined=%d, want three point probes", stats.RowsExamined)
	}
}
