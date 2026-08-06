package libravdb

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

func TestCostModelScaffoldRetainsHybridObservation(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:costmodel_observation"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	col, err := db.CreateCollection(ctx, "costmodel", WithDimension(2),
		WithMetadataSchema(MetadataSchema{"kind": IntField}), WithIndexedFields("kind"))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 50; i++ {
		if err := col.Insert(ctx, idStr(i), []float32{float32(i), 0}, map[string]interface{}{"kind": int64(i % 2)}); err != nil {
			t.Fatal(err)
		}
	}

	plan := &optimizer.PhysicalPlan{
		CollectionName:     "costmodel",
		Kind:               optimizer.QueryKindKNN,
		HasVectorSearch:    true,
		HasRelationalQuery: true,
		Predicates:         []optimizer.RelationalPredicate{{Column: "kind", Operator: 12, Value: []byte("1")}},
		QueryVector:        []float32{0, 0},
		Limit:              3,
		RecallContract:     optimizer.RecallBounded,
	}
	if _, err := newExecutor(db).Execute(ctx, plan); err != nil {
		t.Fatal(err)
	}

	observations := db.CostModelObservations()
	if len(observations) != 1 {
		t.Fatalf("cost-model observations = %d, want 1", len(observations))
	}
	metrics := observations[0].Metrics
	if metrics.EstimateSource != "provisional_heuristic" || metrics.EstimateConfidence <= 0 {
		t.Fatalf("estimate metadata = source %q confidence %f", metrics.EstimateSource, metrics.EstimateConfidence)
	}
	if len(metrics.EstimateAssumptions) == 0 || metrics.ActScalarCandidates == 0 {
		t.Fatalf("observation missing assumptions/actuals: %+v", metrics)
	}
}

func TestAnalyzeCollectionPersistsAndInvalidatesStatistics(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "costmodel.db")
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	col, err := db.CreateCollection(ctx, "costmodel_persist", WithDimension(2),
		WithMetadataSchema(MetadataSchema{"kind": IntField}))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		kind := int64(2)
		if i < 8 {
			kind = 1
		}
		if err := col.Insert(ctx, idStr(i), []float32{float32(i), 0}, map[string]interface{}{"kind": kind}); err != nil {
			t.Fatal(err)
		}
	}
	stats, err := db.AnalyzeCollection(ctx, "costmodel_persist")
	if err != nil {
		t.Fatal(err)
	}
	if stats.Version != costModelStatisticsVersion || stats.RowCount != 10 || stats.DataLSN == 0 {
		t.Fatalf("unexpected analyzed statistics: %+v", stats)
	}

	plan := &optimizer.PhysicalPlan{
		CollectionName:     "costmodel_persist",
		Kind:               optimizer.QueryKindKNN,
		HasVectorSearch:    true,
		HasRelationalQuery: true,
		Predicates:         []optimizer.RelationalPredicate{{Column: "kind", Operator: 12, Value: []byte("1")}},
		QueryVector:        []float32{0, 0},
		Limit:              3,
		RecallContract:     optimizer.RecallBounded,
	}
	_, _, metrics := newExecutor(db).dispatchHybrid(ctx, plan)
	if metrics.EstimateSource != "analyzed_collection_statistics" || metrics.EstScalarCandidates != 8 {
		t.Fatalf("analyzed estimate = source %q candidates %d, want analyzed/8", metrics.EstimateSource, metrics.EstScalarCandidates)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	_, _, metrics = newExecutor(db).dispatchHybrid(ctx, plan)
	if metrics.EstimateSource != "analyzed_collection_statistics" || metrics.EstScalarCandidates != 8 {
		t.Fatalf("recovered estimate = source %q candidates %d, want analyzed/8", metrics.EstimateSource, metrics.EstScalarCandidates)
	}
	col, err = db.GetCollection("costmodel_persist")
	if err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "new", []float32{11, 0}, map[string]interface{}{"kind": int64(1)}); err != nil {
		t.Fatal(err)
	}
	_, _, metrics = newExecutor(db).dispatchHybrid(ctx, plan)
	if metrics.EstimateSource != "provisional_heuristic" {
		t.Fatalf("estimate after mutation = %q, want stale statistics rejected", metrics.EstimateSource)
	}
}

func TestAnalyzeCollectionUsesBoundedFieldSynopsis(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:costmodel_bounded"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "bounded", WithDimension(2))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 300; i++ {
		if err := col.Insert(ctx, idStr(i), []float32{float32(i), 0}, map[string]interface{}{"n": i}); err != nil {
			t.Fatal(err)
		}
	}
	stats, err := db.AnalyzeCollection(ctx, "bounded")
	if err != nil {
		t.Fatal(err)
	}
	field := stats.Fields["n"]
	if field.Distinct < 200 || len(field.TopValues) > costModelMaxTopValues {
		t.Fatalf("bounded synopsis = distinct %d top values %d", field.Distinct, len(field.TopValues))
	}
	if len(field.Histogram) == 0 || len(field.Histogram) > costModelHistogramBuckets {
		t.Fatalf("numeric histogram buckets = %d", len(field.Histogram))
	}
}

func TestCostModelFeedbackPersistsGraphRankAndCalibration(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "feedback.db")
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	col, err := db.CreateCollection(ctx, "feedback", WithDimension(2))
	if err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "one", []float32{1, 0}, map[string]interface{}{"kind": "a"}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.AnalyzeCollection(ctx, "feedback"); err != nil {
		t.Fatal(err)
	}
	plan := &optimizer.PhysicalPlan{CollectionName: "feedback", HasGraphTraversal: true, GraphEdges: []optimizer.GraphEdgePlan{{EdgeKind: 1, Direction: 1, QuantMin: 1, QuantMax: 1}}}
	for i := 0; i < 8; i++ {
		db.recordCostModelFeedback(plan, &QueryMetrics{PlanChosen: DispatchExactCandidateScan, ActConjunctionCandidates: 100, GraphSeeds: 2, GraphVertices: 9, ActGraphCandidates: 7, ExecutionNanos: 1_000})
		db.recordCostModelFeedback(plan, &QueryMetrics{PlanChosen: DispatchFilteredANN, ActConjunctionCandidates: 100, ANNEf: 100, FilterValidHits: 10, ExecutionNanos: 10_000})
	}
	stats, ok := col.costModel.snapshot()
	if !ok {
		t.Fatal("expected live statistics")
	}
	graph := stats.Graph.PatternSamples[costModelGraphPatternKey(plan)]
	if graph.Observations != 16 || graph.Candidates != 56 {
		t.Fatalf("graph sample = %+v", graph)
	}
	if len(stats.RankBucketYields) != 1 || stats.RankBucketYields[0].Candidates != 800 {
		t.Fatalf("rank buckets = %+v", stats.RankBucketYields)
	}
	profile := stats.Calibrations[costModelCalibrationKey(stats)]
	if profile.ExactSamples != 8 || profile.FilteredSamples != 8 {
		t.Fatalf("calibration = %+v", profile)
	}
	if cap := calibratedExactCandidateCap(stats); cap <= 0 {
		t.Fatal("expected calibrated exact cap")
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	col, err = db.GetCollection("feedback")
	if err != nil {
		t.Fatal(err)
	}
	if recovered, ok := col.costModel.snapshot(); !ok || recovered.Calibrations[costModelCalibrationKey(recovered)].FilteredSamples != 8 {
		t.Fatalf("feedback did not recover: %+v", recovered)
	}
}

func TestCostModelScaffoldRingIsBoundedAndCopySafe(t *testing.T) {
	stats := newCostModelStats(2)
	for i := 0; i < 3; i++ {
		stats.record("c", &QueryMetrics{Transitions: []string{"t"}})
	}
	observations := stats.snapshot()
	if len(observations) != 2 {
		t.Fatalf("ring length = %d, want 2", len(observations))
	}
	observations[0].Metrics.Transitions[0] = "mutated"
	if stats.snapshot()[0].Metrics.Transitions[0] == "mutated" {
		t.Fatal("snapshot exposed internal transition storage")
	}
}
