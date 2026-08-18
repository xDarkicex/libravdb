package libravdb

import (
	"context"
	"testing"
)

func TestSQLSecondaryIndexSurvivesReopenAndRoutesEquality(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/secondary-index.libravdb"
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `CREATE TABLE person_core (id TEXT PRIMARY KEY, location TEXT, age INTEGER, gender TEXT)`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `INSERT INTO person_core (id, location, age, gender) VALUES ('a', 'SF', 30, 'x'), ('b', 'NY', 30, 'y')`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `CREATE INDEX person_core_location_idx ON person_core (location)`); err != nil {
		t.Fatal(err)
	}
	rows, err := db.Query(ctx, `SELECT id FROM person_core WHERE location = 'SF'`)
	if err != nil || rows.Total != 1 || rows.Results[0].ID != "a" {
		t.Fatalf("indexed query rows=%#v err=%v", rows, err)
	}
	db.Close()

	reopened, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	col, err := reopened.GetCollection("person_core")
	if err != nil {
		t.Fatal(err)
	}
	if !col.hasIndexedMetadataField("location") {
		t.Fatalf("location index declaration missing after reopen: %+v", col.Config())
	}
	rows, err = reopened.Query(ctx, `SELECT id FROM person_core WHERE location = 'SF'`)
	if err != nil || rows.Total != 1 || rows.Results[0].ID != "a" {
		t.Fatalf("reopened indexed query rows=%#v err=%v", rows, err)
	}
	if _, err := reopened.Query(ctx, `DROP INDEX person_core_location_idx`); err != nil {
		t.Fatalf("DROP INDEX: %v", err)
	}
	if col.hasIndexedMetadataField("location") {
		t.Fatalf("location index declaration remained after DROP INDEX: %+v", col.Config())
	}
	if cfg := col.Config(); len(cfg.SQLIndexes) != 0 || len(cfg.SQLIndexedFields) != 0 {
		t.Fatalf("SQL index declarations remained after DROP INDEX: %+v", cfg)
	}
}
