package libravdb

import (
	"context"
	"testing"
)

func TestSQLWindowFunctions(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:window-functions"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.CreateCollection(ctx, "window_rows", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{
		"category": StringField,
		"score":    IntField,
	})); err != nil {
		t.Fatal(err)
	}
	col, _ := db.GetCollection("window_rows")
	for _, row := range []struct {
		id, category string
		score        int64
	}{
		{"a1", "a", 10}, {"a2", "a", 20}, {"a3", "a", 20},
		{"b1", "b", 7}, {"b2", "b", 5},
	} {
		if err := col.Insert(ctx, row.id, nil, map[string]interface{}{"category": row.category, "score": row.score}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.CreateCollection(ctx, "window_nulls", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{
		"score": IntField,
	})); err != nil {
		t.Fatal(err)
	}
	nullCol, _ := db.GetCollection("window_nulls")
	for id, metadata := range map[string]map[string]interface{}{
		"null-score": {"score": nil},
		"score-10":   {"score": int64(10)},
		"score-20":   {"score": int64(20)},
	} {
		if err := nullCol.Insert(ctx, id, nil, metadata); err != nil {
			t.Fatal(err)
		}
	}

	ranking, err := db.Query(ctx, "SELECT id, category, score, ROW_NUMBER() OVER (PARTITION BY category ORDER BY score DESC) AS rn, RANK() OVER (PARTITION BY category ORDER BY score DESC) AS rnk, DENSE_RANK() OVER (PARTITION BY category ORDER BY score DESC) AS drnk FROM window_rows ORDER BY id")
	if err != nil {
		t.Fatalf("ranking windows: %v", err)
	}
	if ranking.Total != 5 {
		t.Fatalf("ranking rows=%d", ranking.Total)
	}
	want := []struct {
		id      string
		rn, rnk int64
		drnk    int64
	}{
		{"a1", 3, 3, 2}, {"a2", 1, 1, 1}, {"a3", 2, 1, 1},
		{"b1", 1, 1, 1}, {"b2", 2, 2, 2},
	}
	for i, expected := range want {
		result := ranking.Results[i]
		if result.ID != expected.id || result.Metadata["rn"] != expected.rn || result.Metadata["rnk"] != expected.rnk || result.Metadata["drnk"] != expected.drnk {
			t.Fatalf("ranking[%d]=%#v want id=%s rn=%d rank=%d dense=%d", i, result, expected.id, expected.rn, expected.rnk, expected.drnk)
		}
	}
	noOrder, err := db.Query(ctx, "SELECT id, RANK() OVER (PARTITION BY category) AS rnk FROM window_rows ORDER BY id")
	if err != nil {
		t.Fatalf("rank without order: %v", err)
	}
	for _, result := range noOrder.Results {
		if result.Metadata["rnk"] != int64(1) {
			t.Fatalf("rank without order for %s = %#v, want 1", result.ID, result.Metadata["rnk"])
		}
	}
	secondary, err := db.Query(ctx, "SELECT id, ROW_NUMBER() OVER (PARTITION BY category ORDER BY score DESC, id DESC) AS rn FROM window_rows ORDER BY id")
	if err != nil {
		t.Fatalf("window secondary order: %v", err)
	}
	for _, result := range secondary.Results {
		if result.ID == "a2" && result.Metadata["rn"] != int64(2) {
			t.Fatalf("secondary order a2 rn=%#v, want 2", result.Metadata["rn"])
		}
		if result.ID == "a3" && result.Metadata["rn"] != int64(1) {
			t.Fatalf("secondary order a3 rn=%#v, want 1", result.Metadata["rn"])
		}
	}
	framed, err := db.Query(ctx, "SELECT id, ROW_NUMBER() OVER (PARTITION BY category ORDER BY score DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rn FROM window_rows ORDER BY id")
	if err != nil {
		t.Fatalf("window frame: %v", err)
	}
	if framed.Total != 5 {
		t.Fatalf("window frame rows=%d", framed.Total)
	}
	mixed, err := db.Query(ctx, "SELECT category, COUNT(*) AS cnt, ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC, category ASC) AS rn FROM window_rows GROUP BY category ORDER BY category")
	if err != nil {
		t.Fatalf("aggregate/window projection: %v", err)
	}
	if mixed.Total != 2 || mixed.Results[0].Metadata["category"] != "a" || mixed.Results[0].Metadata["cnt"] != int64(3) || mixed.Results[0].Metadata["rn"] != int64(1) || mixed.Results[1].Metadata["rn"] != int64(2) {
		t.Fatalf("aggregate/window rows=%#v", mixed.Results)
	}
	scalarMixed, err := db.Query(ctx, "SELECT COUNT(*) AS cnt, ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rn FROM window_rows")
	if err != nil {
		t.Fatalf("scalar aggregate/window projection: %v", err)
	}
	if scalarMixed.Total != 1 || scalarMixed.Results[0].Metadata["cnt"] != int64(5) || scalarMixed.Results[0].Metadata["rn"] != int64(1) {
		t.Fatalf("scalar aggregate/window row=%#v", scalarMixed.Results)
	}
	windowAggregate, err := db.Query(ctx, "SELECT id, SUM(score) OVER (PARTITION BY category ORDER BY score ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_score, COUNT(*) OVER (PARTITION BY category) AS partition_count FROM window_rows ORDER BY id")
	if err != nil || windowAggregate.Total != 5 {
		t.Fatalf("aggregate window rows=%#v err=%v", windowAggregate, err)
	}
	for _, result := range windowAggregate.Results {
		wantRunning := map[string]float64{"a1": 10, "a2": 30, "a3": 50, "b1": 12, "b2": 5}[result.ID]
		if result.Metadata["running_score"] != wantRunning || result.Metadata["partition_count"] != map[string]int64{"a1": 3, "a2": 3, "a3": 3, "b1": 2, "b2": 2}[result.ID] {
			t.Fatalf("aggregate window row=%#v want running=%v", result, wantRunning)
		}
	}
	namedWindow, err := db.Query(ctx, "SELECT id, ROW_NUMBER() OVER ranked AS rn FROM window_rows WINDOW ranked AS (PARTITION BY category ORDER BY score DESC) ORDER BY id")
	if err != nil || namedWindow.Total != 5 {
		t.Fatalf("named window rows=%#v err=%v", namedWindow, err)
	}
	namedAggregateWindow, err := db.Query(ctx, "SELECT id, SUM(score) OVER ranked AS running_score FROM window_rows WINDOW ranked AS (PARTITION BY category ORDER BY score) ORDER BY id")
	if err != nil || namedAggregateWindow.Total != 5 {
		t.Fatalf("named aggregate window rows=%#v err=%v", namedAggregateWindow, err)
	}

	shift, err := db.Query(ctx, "SELECT id, score, LAG(score) OVER (PARTITION BY category ORDER BY score) AS previous_score, LEAD(score, 1, -1) OVER (PARTITION BY category ORDER BY score) AS next_score FROM window_rows ORDER BY id")
	if err != nil {
		t.Fatalf("lag/lead windows: %v", err)
	}
	if shift.Total != 5 {
		t.Fatalf("lag/lead rows=%d", shift.Total)
	}
	for _, result := range shift.Results {
		score := result.Metadata["score"].(int64)
		previous, next := result.Metadata["previous_score"], result.Metadata["next_score"]
		switch result.ID {
		case "a1":
			if previous != nil || next != int64(20) {
				t.Fatalf("a1 lag/lead=(%#v,%#v)", previous, next)
			}
		case "a2":
			if previous != int64(10) || next != int64(20) {
				t.Fatalf("a2 lag/lead=(%#v,%#v)", previous, next)
			}
		case "a3":
			if previous != int64(20) || next != int64(-1) {
				t.Fatalf("a3 lag/lead=(%#v,%#v)", previous, next)
			}
		case "b1":
			if previous != int64(5) || next != int64(-1) || score != 7 {
				t.Fatalf("b1 lag/lead=(%#v,%#v)", previous, next)
			}
		case "b2":
			if previous != nil || next != int64(7) {
				t.Fatalf("b2 lag/lead=(%#v,%#v)", previous, next)
			}
		}
	}
	star, err := db.Query(ctx, "SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY score DESC) AS rn FROM window_rows ORDER BY id")
	if err != nil || star.Total != 5 || star.Results[0].Metadata["category"] != "a" || star.Results[0].Metadata["rn"] == nil {
		t.Fatalf("window SELECT * result=%#v err=%v", star, err)
	}
	rangeWindow, err := db.Query(ctx, "SELECT id, SUM(score) OVER (PARTITION BY category ORDER BY score RANGE BETWEEN 5 PRECEDING AND CURRENT ROW) AS range_sum FROM window_rows ORDER BY id")
	if err != nil || rangeWindow.Total != 5 {
		t.Fatalf("numeric RANGE window rows=%#v err=%v", rangeWindow, err)
	}
	for _, result := range rangeWindow.Results {
		wantRange := map[string]float64{"a1": 10, "a2": 40, "a3": 40, "b1": 12, "b2": 5}[result.ID]
		if result.Metadata["range_sum"] != wantRange {
			t.Fatalf("numeric RANGE %s=%#v want %v", result.ID, result.Metadata["range_sum"], wantRange)
		}
	}
	descRange, err := db.Query(ctx, "SELECT id, SUM(score) OVER (PARTITION BY category ORDER BY score DESC RANGE BETWEEN 5 PRECEDING AND CURRENT ROW) AS range_sum FROM window_rows ORDER BY id")
	if err != nil || descRange.Total != 5 {
		t.Fatalf("descending numeric RANGE rows=%#v err=%v", descRange, err)
	}
	for _, result := range descRange.Results {
		wantRange := map[string]float64{"a1": 10, "a2": 40, "a3": 40, "b1": 7, "b2": 12}[result.ID]
		if result.Metadata["range_sum"] != wantRange {
			t.Fatalf("descending numeric RANGE %s=%#v want %v", result.ID, result.Metadata["range_sum"], wantRange)
		}
	}
	followingRange, err := db.Query(ctx, "SELECT id, SUM(score) OVER (PARTITION BY category ORDER BY score RANGE BETWEEN CURRENT ROW AND 5 FOLLOWING) AS range_sum FROM window_rows ORDER BY id")
	if err != nil || followingRange.Total != 5 {
		t.Fatalf("following numeric RANGE rows=%#v err=%v", followingRange, err)
	}
	for _, result := range followingRange.Results {
		wantRange := map[string]float64{"a1": 10, "a2": 40, "a3": 40, "b1": 7, "b2": 12}[result.ID]
		if result.Metadata["range_sum"] != wantRange {
			t.Fatalf("following numeric RANGE %s=%#v want %v", result.ID, result.Metadata["range_sum"], wantRange)
		}
	}
	distribution, err := db.Query(ctx, "SELECT id, PERCENT_RANK() OVER (PARTITION BY category ORDER BY score) AS percent_rank, CUME_DIST() OVER (PARTITION BY category ORDER BY score) AS cume_dist, NTILE(2) OVER (PARTITION BY category ORDER BY score) AS tile FROM window_rows ORDER BY id")
	if err != nil || distribution.Total != 5 {
		t.Fatalf("distribution windows rows=%#v err=%v", distribution, err)
	}
	for _, result := range distribution.Results {
		wantPercent := map[string]float64{"a1": 0, "a2": 0.5, "a3": 0.5, "b1": 1, "b2": 0}[result.ID]
		wantCume := map[string]float64{"a1": 1.0 / 3, "a2": 1, "a3": 1, "b1": 1, "b2": 0.5}[result.ID]
		wantTile := map[string]int64{"a1": 1, "a2": 1, "a3": 2, "b1": 2, "b2": 1}[result.ID]
		if result.Metadata["percent_rank"] != wantPercent || result.Metadata["cume_dist"] != wantCume || result.Metadata["tile"] != wantTile {
			t.Fatalf("distribution %s=%#v want=(%v,%v,%d)", result.ID, result.Metadata, wantPercent, wantCume, wantTile)
		}
	}
	nullOrdering, err := db.Query(ctx, "SELECT id, ROW_NUMBER() OVER (ORDER BY score ASC) AS asc_default, ROW_NUMBER() OVER (ORDER BY score DESC) AS desc_default, ROW_NUMBER() OVER (ORDER BY score ASC NULLS FIRST) AS asc_first, ROW_NUMBER() OVER (ORDER BY score DESC NULLS LAST) AS desc_last FROM window_nulls ORDER BY id")
	if err != nil || nullOrdering.Total != 3 {
		t.Fatalf("NULL ordering rows=%#v err=%v", nullOrdering, err)
	}
	for _, result := range nullOrdering.Results {
		want := map[string][4]int64{
			"null-score": {3, 1, 1, 3},
			"score-10":   {1, 3, 2, 2},
			"score-20":   {2, 2, 3, 1},
		}[result.ID]
		got := [4]int64{result.Metadata["asc_default"].(int64), result.Metadata["desc_default"].(int64), result.Metadata["asc_first"].(int64), result.Metadata["desc_last"].(int64)}
		if got != want {
			t.Fatalf("NULL ordering %s=%v want=%v", result.ID, got, want)
		}
	}
	if _, err := db.Query(ctx, "SELECT SUM(score) OVER (PARTITION BY category ORDER BY score, id RANGE BETWEEN 5 PRECEDING AND CURRENT ROW) FROM window_rows"); err == nil {
		t.Fatal("multi-key RANGE offset should be rejected")
	}
	if _, err := db.Query(ctx, "SELECT SUM(score) OVER (ORDER BY category RANGE BETWEEN 5 PRECEDING AND CURRENT ROW) FROM window_rows"); err == nil {
		t.Fatal("non-numeric RANGE offset should be rejected")
	}
	orderedSet, err := db.Query(ctx, "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) AS p50, PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY score) AS d50, MODE() WITHIN GROUP (ORDER BY category) AS common_category FROM window_rows")
	if err != nil || orderedSet.Total != 1 {
		t.Fatalf("ordered-set scalar result=%#v err=%v", orderedSet, err)
	}
	if orderedSet.Results[0].Metadata["p50"] != float64(10) || orderedSet.Results[0].Metadata["d50"] != int64(10) || orderedSet.Results[0].Metadata["common_category"] != "a" {
		t.Fatalf("ordered-set scalar metadata=%#v", orderedSet.Results[0].Metadata)
	}
	groupedOrderedSet, err := db.Query(ctx, "SELECT category, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) AS p50 FROM window_rows GROUP BY category ORDER BY category")
	if err != nil || groupedOrderedSet.Total != 2 {
		t.Fatalf("ordered-set grouped result=%#v err=%v", groupedOrderedSet, err)
	}
	if groupedOrderedSet.Results[0].Metadata["p50"] != float64(20) || groupedOrderedSet.Results[1].Metadata["p50"] != float64(6) {
		t.Fatalf("ordered-set grouped metadata=%#v", groupedOrderedSet.Results)
	}
}
