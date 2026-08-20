package libravdb

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

func TestCypherINListParameter(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:cypher_in_param"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	col, err := db.CreateCollection(ctx, "people", WithMetadataOnly(), WithGraph(g), WithMetadataSchema(MetadataSchema{
		"group_id": StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for id, group := range map[string]string{"alice": "g1", "bob": "g2", "carol": "g3"} {
		if err := col.Insert(ctx, id, nil, map[string]interface{}{"group_id": group}); err != nil {
			t.Fatal(err)
		}
	}

	rows, err := db.QueryWithParams(ctx, `
		MATCH (e)
		WHERE e.group_id IN $group_ids
		RETURN e.id AS id
		ORDER BY id`, QueryParams{"group_ids": []string{"g1", "g2"}})
	if err != nil {
		t.Fatalf("IN list parameter: %v", err)
	}
	got := map[string]bool{}
	for _, row := range rows.Results {
		got[fmt.Sprint(row.Metadata["id"])] = true
	}
	if rows.Total != 2 || !got["alice"] || !got["bob"] {
		t.Fatalf("IN list rows=%#v, want alice and bob", got)
	}

	rows, err = db.QueryWithParams(ctx, `
		SELECT id
		FROM people
		WHERE group_id IN $group_ids
		ORDER BY id`, QueryParams{"group_ids": []string{"g1", "g2"}})
	if err != nil {
		t.Fatalf("relational IN list parameter: %v", err)
	}
	if rows.Total != 2 || rows.Results[0].ID != "alice" || rows.Results[1].ID != "bob" {
		t.Fatalf("relational IN rows=%#v, want alice and bob", rows.Results)
	}

	rows, err = db.QueryWithParams(ctx, `
		MATCH (e)
		WHERE e.group_id NOT IN $group_ids
		RETURN e.id AS id`, QueryParams{"group_ids": []string{"g1", "g2"}})
	if err != nil {
		t.Fatalf("NOT IN list parameter: %v", err)
	}
	if rows.Total != 1 || fmt.Sprint(rows.Results[0].Metadata["id"]) != "carol" {
		t.Fatalf("NOT IN rows=%#v, want carol", rows.Results)
	}

	rows, err = db.QueryWithParams(ctx, `
		MATCH (e)
		WHERE e.group_id IN $group_ids
		RETURN e.id AS id`, QueryParams{"group_ids": []string{}})
	if err != nil {
		t.Fatalf("empty IN list: %v", err)
	}
	if rows.Total != 0 {
		t.Fatalf("empty IN rows=%#v, want none", rows.Results)
	}

	_, err = db.QueryWithParams(ctx, `
		MATCH (e)
		WHERE e.group_id IN $group_id
		RETURN e.id`, QueryParams{"group_id": "g1"})
	if err == nil || !strings.Contains(err.Error(), "IN expects list parameter") {
		t.Fatalf("scalar IN error=%v, want explicit list-parameter error", err)
	}
}
