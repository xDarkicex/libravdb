package libravdb

import (
	"context"
	"testing"
)

func TestSQLArrayAggAndStringAgg(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:collection-aggregates"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.CreateCollection(ctx, "aggregate_rows", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{
		"category": StringField,
		"name":     StringField,
	})); err != nil {
		t.Fatal(err)
	}
	col, _ := db.GetCollection("aggregate_rows")
	for _, row := range []struct {
		id, category string
		name         interface{}
	}{
		{"a1", "a", "Ada"},
		{"a2", "a", "Grace"},
		{"a3", "a", nil},
		{"b1", "b", "Linus"},
	} {
		if err := col.Insert(ctx, row.id, nil, map[string]interface{}{"category": row.category, "name": row.name}); err != nil {
			t.Fatal(err)
		}
	}

	all, err := db.Query(ctx, "SELECT ARRAY_AGG(name) AS names, STRING_AGG(name, '|') AS joined FROM aggregate_rows")
	if err != nil {
		t.Fatalf("collection aggregates: %v", err)
	}
	if all.Total != 1 {
		t.Fatalf("collection aggregate rows=%d", all.Total)
	}
	values, ok := all.Results[0].Metadata["names"].([]interface{})
	if !ok || len(values) != 4 || values[2] != nil {
		t.Fatalf("array_agg=%#v", all.Results[0].Metadata["names"])
	}
	if got := all.Results[0].Metadata["joined"]; got != "Ada|Grace|Linus" {
		t.Fatalf("string_agg=%#v", got)
	}

	grouped, err := db.Query(ctx, "SELECT category, ARRAY_AGG(name) AS names, STRING_AGG(name, ',') AS joined FROM aggregate_rows GROUP BY category ORDER BY category")
	if err != nil {
		t.Fatalf("grouped collection aggregates: %v", err)
	}
	if grouped.Total != 2 || grouped.Results[0].Metadata["joined"] != "Ada,Grace" || grouped.Results[1].Metadata["joined"] != "Linus" {
		t.Fatalf("grouped collection aggregates=%#v", grouped)
	}
}
