package libravdb

import (
	"context"
	"math"
	"testing"
	"time"
)

func TestSQLGroupedAggregateExpressionAndBoundedTemporalCTE(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/calibration.libravdb"

	open := func() *Database {
		db, err := Open(WithStoragePath(path), WithMetrics(false))
		if err != nil {
			t.Fatal(err)
		}
		return db
	}

	db := open()
	col, err := db.CreateCollection(ctx, "__next_context_transitions_v1",
		WithMetadataOnly(),
		WithMetadataSchema(MetadataSchema{
			"namespace":     StringField,
			"provider_id":   StringField,
			"event_class":   StringField,
			"history_class": StringField,
			"alpha":         FloatField,
			"beta":          FloatField,
			"usefulness":    FloatField,
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range []struct {
		id, namespace, provider, event, history string
		alpha, beta, usefulness                 float64
	}{
		{"r1", "ns", "p1", "event", "history", 1, 1, 0.1},
		{"r2", "ns", "p1", "event", "history", 3, 1, 0.2},
		{"r3", "other", "p1", "event", "history", 9, 1, 0.3},
	} {
		if err := col.Insert(ctx, row.id, nil, map[string]interface{}{
			"namespace":     row.namespace,
			"provider_id":   row.provider,
			"event_class":   row.event,
			"history_class": row.history,
			"alpha":         row.alpha,
			"beta":          row.beta,
			"usefulness":    row.usefulness,
		}); err != nil {
			t.Fatalf("insert %s: %v", row.id, err)
		}
	}

	snapshot, err := db.SnapshotAt(ctx, time.Now().UTC().Add(time.Second))
	if err != nil {
		t.Fatal("snapshot: ", err)
	}
	end := snapshot.Timestamp.Format(time.RFC3339Nano)
	snapshot.Close()

	assertGrouped := func(label string, current *Database) {
		t.Helper()
		result, queryErr := current.Query(ctx, `SELECT
            SUM(alpha) / SUM(alpha + beta) AS beta_mean
            FROM "__next_context_transitions_v1"
            GROUP BY namespace, provider_id, event_class, history_class`)
		if queryErr != nil {
			t.Fatalf("%s grouped aggregate: %v", label, queryErr)
		}
		if result.Total != 2 || len(result.Results) != 2 {
			t.Fatalf("%s grouped rows=%d results=%#v, want two groups", label, result.Total, result.Results)
		}
		seen := make(map[string]float64)
		for _, row := range result.Results {
			mean, ok := row.Metadata["beta_mean"].(float64)
			if !ok {
				t.Fatalf("%s beta_mean type=%T value=%#v row=%#v columns=%v", label, row.Metadata["beta_mean"], row.Metadata["beta_mean"], row, result.Columns)
			}
			seen[row.ID] = mean
		}
		if math.Abs(seen["r1"]-2.0/3.0) > 1e-9 {
			t.Fatalf("%s ns beta_mean=%v, want %v", label, seen["r1"], 2.0/3.0)
		}
		if math.Abs(seen["r3"]-0.9) > 1e-9 {
			t.Fatalf("%s other beta_mean=%v, want 0.9", label, seen["r3"])
		}
	}

	assertParameterizedAggregates := func(label string, current *Database) {
		t.Helper()
		parameterizedSQL := `SELECT namespace,
               MIN($p_threshold) AS admission_threshold,
               MIN($break_even_accuracy) AS break_even_accuracy,
               SUM($p_threshold) AS threshold_total
        FROM "__next_context_transitions_v1"
		GROUP BY namespace`
		result, queryErr := current.QueryWithParams(ctx, parameterizedSQL, QueryParams{
			"p_threshold":         0.75,
			"break_even_accuracy": 0.82,
		})
		if queryErr != nil {
			t.Fatalf("%s parameterized aggregates: %v", label, queryErr)
		}
		if result.Total != 2 || len(result.Results) != 2 {
			t.Fatalf("%s parameterized aggregate rows=%d results=%#v, want two groups", label, result.Total, result.Results)
		}
		for _, row := range result.Results {
			threshold, thresholdOK := row.Metadata["admission_threshold"].(float64)
			breakEven, breakEvenOK := row.Metadata["break_even_accuracy"].(float64)
			total, totalOK := row.Metadata["threshold_total"].(float64)
			if !thresholdOK || !breakEvenOK || !totalOK {
				t.Fatalf("%s parameterized aggregate columns=%v row=%#v, want float64 values", label, result.Columns, row)
			}
			if threshold != 0.75 || breakEven != 0.82 {
				t.Errorf("%s namespace=%q constants=(%v,%v), want (0.75,0.82)", label, row.ID, threshold, breakEven)
			}
			wantTotal := 0.75
			if row.ID == "r1" {
				wantTotal = 1.5
			}
			if total != wantTotal {
				t.Errorf("%s namespace=%q threshold_total=%v, want %v", label, row.ID, total, wantTotal)
			}
		}

		empty, queryErr := current.QueryWithParams(ctx, `SELECT MIN($p_threshold) AS admission_threshold,
               SUM($p_threshold) AS threshold_total
        FROM "__next_context_transitions_v1"
        WHERE namespace = $missing_namespace`, QueryParams{
			"p_threshold":       0.75,
			"missing_namespace": "does-not-exist",
		})
		if queryErr != nil {
			t.Fatalf("%s empty parameterized aggregates: %v", label, queryErr)
		}
		if empty.Total != 1 || empty.Results[0].Metadata["admission_threshold"] != nil || empty.Results[0].Metadata["threshold_total"] != nil {
			t.Fatalf("%s empty parameterized aggregates=%#v, want one row with NULL values", label, empty.Results)
		}
	}

	assertBoundedTemporalCTE := func(label string, current *Database) {
		t.Helper()
		result, queryErr := current.QueryWithParams(ctx, `WITH bounded AS (
            SELECT namespace, provider_id, event_class, history_class,
                   alpha, beta, usefulness
            FROM "__next_context_transitions_v1"
                 AS OF TIMESTAMP $end
            WHERE namespace = $libravdbd_namespace
            ORDER BY id ASC
            LIMIT $catalog_input_limit
        )
        SELECT namespace, provider_id, event_class, history_class,
               SUM(alpha) AS alpha_total
        FROM bounded
        GROUP BY namespace, provider_id, event_class, history_class
        LIMIT $catalog_limit`, QueryParams{
			"end":                 end,
			"libravdbd_namespace": "ns",
			"catalog_input_limit": 1,
			"catalog_limit":       10,
		})
		if queryErr != nil {
			t.Fatalf("%s bounded temporal CTE: %v", label, queryErr)
		}
		if result.Total != 1 || len(result.Results) != 1 {
			t.Fatalf("%s bounded rows=%d results=%#v, want one group", label, result.Total, result.Results)
		}
		if got := result.Results[0].Metadata["alpha_total"]; got != float64(1) {
			t.Fatalf("%s alpha_total=%#v, want 1", label, got)
		}
	}

	assertParameterizedBoundedTemporalCTE := func(label string, current *Database) {
		t.Helper()
		result, queryErr := current.QueryWithParams(ctx, `WITH bounded AS (
            SELECT namespace, alpha, beta
            FROM "__next_context_transitions_v1"
                 AS OF TIMESTAMP $end
            WHERE namespace = $libravdbd_namespace
            LIMIT $catalog_input_limit
        )
        SELECT namespace,
               MIN($p_threshold) AS admission_threshold,
               MIN($break_even_accuracy) AS break_even_accuracy,
               SUM($p_threshold) AS threshold_total
        FROM bounded
        GROUP BY namespace`, QueryParams{
			"end":                 end,
			"libravdbd_namespace": "ns",
			"catalog_input_limit": 1,
			"p_threshold":         0.75,
			"break_even_accuracy": 0.82,
		})
		if queryErr != nil {
			t.Fatalf("%s bounded temporal parameterized aggregates: %v", label, queryErr)
		}
		if result.Total != 1 || len(result.Results) != 1 {
			t.Fatalf("%s bounded temporal parameterized rows=%d results=%#v, want one group", label, result.Total, result.Results)
		}
		row := result.Results[0]
		if row.Metadata["admission_threshold"] != float64(0.75) || row.Metadata["break_even_accuracy"] != float64(0.82) || row.Metadata["threshold_total"] != float64(0.75) {
			t.Fatalf("%s bounded temporal parameterized row=%#v, want threshold=0.75 break_even=0.82 total=0.75", label, row)
		}
	}

	assertGrouped("before reopen", db)
	assertParameterizedAggregates("before reopen", db)
	assertBoundedTemporalCTE("before reopen", db)
	assertParameterizedBoundedTemporalCTE("before reopen", db)
	if err := db.Close(); err != nil {
		t.Fatal("close: ", err)
	}

	reopened := open()
	defer reopened.Close()
	assertGrouped("after reopen", reopened)
	assertParameterizedAggregates("after reopen", reopened)
	assertBoundedTemporalCTE("after reopen", reopened)
	assertParameterizedBoundedTemporalCTE("after reopen", reopened)
}
