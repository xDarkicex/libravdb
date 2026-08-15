package libravdb

import (
	"context"
	"strings"
	"testing"
)

func TestSQLQualifiedOrderBySelectStar(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:gorm-qualified-order"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, `CREATE TABLE gorm_catalog_probe (id TEXT PRIMARY KEY, name TEXT, score BIGINT)`); err != nil {
		t.Fatal(err)
	}
	if col, err := db.GetCollection("gorm_catalog_probe"); err != nil {
		t.Fatal(err)
	} else if col.Dimension() != 0 {
		t.Fatalf("metadata-only CREATE TABLE dimension=%d, want 0", col.Dimension())
	}
	if _, err := db.Query(ctx, `INSERT INTO gorm_catalog_probe (id, name, score) VALUES ('gorm-1', 'catalog probe', 7)`); err != nil {
		t.Fatal(err)
	}
	results, err := db.Query(ctx, `SELECT * FROM "gorm_catalog_probe" WHERE id = 'gorm-1' ORDER BY "gorm_catalog_probe"."id" LIMIT 1`)
	if err != nil {
		t.Fatal(err)
	}
	if len(results.Results) != 1 {
		t.Fatalf("rows=%d, want 1", len(results.Results))
	}
	if got, want := strings.Join(results.Columns, ","), "id,name,score"; got != want {
		t.Fatalf("columns=%v, want %s", results.Columns, want)
	}
	t.Logf("columns=%v metadata=%#v", results.Columns, results.Results[0].Metadata)
}
