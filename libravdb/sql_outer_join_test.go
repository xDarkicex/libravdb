package libravdb

import (
	"context"
	"testing"
)

func TestSQLOuterJoinSemantics(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:outer-joins"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.CreateCollection(ctx, "join_left", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{"label": StringField, "right_id": StringField})); err != nil {
		t.Fatal(err)
	}
	if _, err := db.CreateCollection(ctx, "join_right", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{"right_label": StringField})); err != nil {
		t.Fatal(err)
	}
	left, _ := db.GetCollection("join_left")
	right, _ := db.GetCollection("join_right")
	for _, row := range []struct{ id, label, rightID string }{{"l1", "left-one", "r1"}, {"l2", "left-two", "missing"}} {
		if err := left.Insert(ctx, row.id, nil, map[string]interface{}{"label": row.label, "right_id": row.rightID}); err != nil {
			t.Fatal(err)
		}
	}
	for _, row := range []struct{ id, label string }{{"r1", "right-one"}, {"r2", "right-two"}} {
		if err := right.Insert(ctx, row.id, nil, map[string]interface{}{"right_label": row.label}); err != nil {
			t.Fatal(err)
		}
	}

	leftJoin, err := db.Query(ctx, "SELECT l.label, r.right_label FROM join_left l LEFT JOIN join_right r ON l.right_id = r.id ORDER BY l.label")
	if err != nil {
		t.Fatal("LEFT JOIN", err)
	}
	if leftJoin.Total != 2 {
		t.Fatalf("LEFT JOIN rows=%#v", leftJoin)
	}
	if leftJoin.Results[1].Metadata["right_label"] != nil {
		t.Fatalf("LEFT JOIN expected NULL right field: %#v", leftJoin.Results[1])
	}

	rightJoin, err := db.Query(ctx, "SELECT r.right_label, l.right_id FROM join_left l RIGHT JOIN join_right r ON l.right_id = r.id ORDER BY r.right_label")
	if err != nil {
		t.Fatal("RIGHT JOIN", err)
	}
	if rightJoin.Total != 2 {
		t.Fatalf("RIGHT JOIN rows=%#v", rightJoin)
	}
	if _, ok := rightJoin.Results[1].Metadata["right_id"]; !ok || rightJoin.Results[1].Metadata["right_id"] != nil {
		t.Fatalf("RIGHT JOIN expected NULL left field: %#v", rightJoin.Results[1])
	}

	fullJoin, err := db.Query(ctx, "SELECT l.right_id, r.right_label FROM join_left l FULL OUTER JOIN join_right r ON l.right_id = r.id")
	if err != nil {
		t.Fatal("FULL JOIN", err)
	}
	if fullJoin.Total != 3 {
		t.Fatalf("FULL JOIN rows=%#v", fullJoin)
	}

	crossJoin, err := db.Query(ctx, "SELECT l.label, r.right_label FROM join_left l CROSS JOIN join_right r")
	if err != nil {
		t.Fatal("CROSS JOIN", err)
	}
	if crossJoin.Total != 4 {
		t.Fatalf("CROSS JOIN rows=%#v", crossJoin)
	}

	if _, err := db.CreateCollection(ctx, "join_third", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{"left_id": StringField, "third_label": StringField})); err != nil {
		t.Fatal(err)
	}
	third, _ := db.GetCollection("join_third")
	if err := third.Insert(ctx, "t1", nil, map[string]interface{}{"left_id": "r1", "third_label": "third-one"}); err != nil {
		t.Fatal(err)
	}
	multi, err := db.Query(ctx, "SELECT l.label, r.right_label, t.third_label FROM join_left l LEFT JOIN join_right r ON l.right_id = r.id LEFT JOIN join_third t ON r.id = t.left_id ORDER BY l.label")
	if err != nil {
		t.Fatal("multiple LEFT JOIN", err)
	}
	if multi.Total != 2 {
		t.Fatalf("multiple JOIN rows=%#v", multi)
	}
}
