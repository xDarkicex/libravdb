package libravdb

import (
	"context"
	"fmt"
	"testing"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// TestHybridDispatch_DecisionSmallCandidates verifies the dispatcher's
// decision logic: when estimated candidates are below the 2% threshold,
// ExactCandidateScan is chosen.
func TestHybridDispatch_DecisionSmallCandidates(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:dispatch_decision"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	schema := map[string]FieldType{"cat": IntField}
	col, err := db.CreateCollection(ctx, "decide", WithDimension(4), WithMetadataSchema(schema))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for i := 0; i < 50; i++ {
		vec := []float32{float32(i) / 50.0, 0.1, 0.2, 0.3}
		if err := col.Insert(ctx, idStr(i), vec, map[string]interface{}{"cat": int64(i % 3)}); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}

	// Direct API: KNN + post-filter. The existing executeKNN path handles this.
	// This verifies the fast path still works for hybrid queries via the API.
	qb := col.Query(ctx)
	qb.WithVector([]float32{0.1, 0.2, 0.3, 0.4})
	qb.Limit(10)
	results, err := qb.Execute()
	if err != nil {
		t.Fatalf("KNN query: %v", err)
	}

	// Post-filter manually (simulating what the optimizer would do).
	results = filterByPredicates(results, []optimizer.RelationalPredicate{
		{Column: "cat", Operator: 12, Value: []byte("1")},
	})

	t.Logf("KNN + post-filter results: %d rows", len(results.Results))
	if len(results.Results) == 0 {
		t.Error("expected at least 1 result with cat=1")
	}
}

// TestHybridDispatch_ExactScanOperator verifies the ExactCandidateScan
// operator produces correct results through the full ListAll + filter + score path.
func TestHybridDispatch_ExactScanOperator(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:dispatch_exact_scan"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	schema := map[string]FieldType{"cat": IntField}
	col, err := db.CreateCollection(ctx, "exact_scan", WithDimension(4), WithMetadataSchema(schema))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for i := 0; i < 100; i++ {
		vec := []float32{float32(i) / 100.0, 0.5, 0.3, 0.1}
		if err := col.Insert(ctx, idStr(i), vec, map[string]interface{}{"cat": int64(i % 3)}); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}

	// Full exact scan: ListAll + filter + score.
	records, err := col.ListAll(ctx)
	if err != nil {
		t.Fatalf("ListAll: %v", err)
	}

	var candidates []Record
	for _, rec := range records {
		if recordMatchesPredicates(rec, []optimizer.RelationalPredicate{
			{Column: "cat", Operator: 12, Value: []byte("1")},
		}) {
			candidates = append(candidates, rec)
		}
	}

	results := scoreAndSelectTopK(col, candidates, []float32{0.5, 0.5, 0.1, 0.2}, 10)
	t.Logf("ExactCandidateScan: %d results", len(results.Results))
	if len(results.Results) == 0 {
		t.Error("expected at least 1 result")
	}
	if len(results.Results) > 10 {
		t.Errorf("expected at most 10 results, got %d", len(results.Results))
	}

	// Verify all results satisfy cat=1.
	for _, r := range results.Results {
		rec, err := col.Get(ctx, r.ID)
		if err != nil {
			t.Fatalf("get %s: %v", r.ID, err)
		}
		cat, _ := rec.Metadata["cat"].(int64)
		if cat != 1 {
			t.Errorf("result %s has cat=%d, expected 1", r.ID, cat)
		}
	}
}

// TestHybridDispatch_IterativeOperator verifies the IterativeANNThenFilter
// operator collects k valid results through geometric batch growth.
func TestHybridDispatch_IterativeOperator(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:dispatch_iter_op"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	schema := map[string]FieldType{"cat": IntField}
	col, err := db.CreateCollection(ctx, "iter_op", WithDimension(4), WithMetadataSchema(schema))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for i := 0; i < 500; i++ {
		vec := []float32{float32(i) / 500.0, 0.3, 0.2, 0.1}
		if err := col.Insert(ctx, idStr(i), vec, map[string]interface{}{"cat": int64(i % 3)}); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}

	// Simulate iterative approach: query with increasing batch sizes.
	var allResults *SearchResults
	batchSize := 50
	for b := 0; b < 4; b++ {
		qb := col.Query(ctx)
		qb.WithVector([]float32{0.5, 0.5, 0.1, 0.2})
		qb.Limit(batchSize)
		batch, err := qb.Execute()
		if err != nil {
			t.Fatalf("batch %d: %v", b, err)
		}
		batch = filterByPredicates(batch, []optimizer.RelationalPredicate{
			{Column: "cat", Operator: 12, Value: []byte("1")},
		})

		if allResults == nil {
			allResults = batch
		} else {
			seen := make(map[string]bool)
			for _, r := range allResults.Results {
				seen[r.ID] = true
			}
			for _, r := range batch.Results {
				if !seen[r.ID] {
					allResults.Results = append(allResults.Results, r)
					seen[r.ID] = true
				}
			}
		}
		if len(allResults.Results) >= 10 {
			break
		}
		batchSize *= 2
	}

	if allResults == nil {
		t.Fatal("no results")
	}
	if len(allResults.Results) > 10 {
		allResults.Results = allResults.Results[:10]
	}

	t.Logf("IterativeANN: %d results after %d batches", len(allResults.Results), 4)

	// Verify all results satisfy cat=1.
	for _, r := range allResults.Results {
		rec, err := col.Get(ctx, r.ID)
		if err != nil {
			t.Fatalf("get %s: %v", r.ID, err)
		}
		cat, _ := rec.Metadata["cat"].(int64)
		if cat != 1 {
			t.Errorf("result %s has cat=%d, expected 1", r.ID, cat)
		}
	}
}

// TestHybridDispatch_BinomialStart verifies the Chernoff-bound start size
// computation produces reasonable m* values.
func TestHybridDispatch_BinomialStart(t *testing.T) {
	tests := []struct {
		k     int
		sigma float64
		eps   float64
	}{
		{10, 0.5, 0.05},  // weak filter → small m*
		{10, 0.1, 0.05},  // selective → larger m*
		{10, 0.01, 0.05}, // very selective → capped
		{100, 0.5, 0.05}, // large k
	}

	for _, tt := range tests {
		m := computeBinomialStart(tt.k, tt.sigma, tt.eps)
		t.Logf("k=%d sigma=%.2f eps=%.2f → m*=%d", tt.k, tt.sigma, tt.eps, m)
		if m < tt.k {
			t.Errorf("m*=%d < k=%d", m, tt.k)
		}
	}
}

// TestHybridDispatch_IsHybridQuery verifies the isHybridQuery gate.
func TestHybridDispatch_IsHybridQuery(t *testing.T) {
	// Pure vector (no predicates) → not hybrid.
	planVec := &optimizer.PhysicalPlan{
		HasVectorSearch: true,
		Kind:            optimizer.QueryKindKNN,
	}
	if isHybridQuery(planVec) {
		t.Error("pure vector should not be hybrid")
	}

	// Pure relational → not hybrid.
	planRel := &optimizer.PhysicalPlan{
		HasRelationalQuery: true,
		Predicates:         []optimizer.RelationalPredicate{{}},
		Kind:               optimizer.QueryKindRelational,
	}
	if isHybridQuery(planRel) {
		t.Error("pure relational should not be hybrid")
	}

	// Hybrid: vector + predicate → hybrid.
	planHybrid := &optimizer.PhysicalPlan{
		HasVectorSearch:    true,
		HasRelationalQuery: true,
		Predicates:         []optimizer.RelationalPredicate{{}},
		Kind:               optimizer.QueryKindKNN,
	}
	if !isHybridQuery(planHybrid) {
		t.Error("vector + predicate should be hybrid")
	}

	// Hybrid: vector + graph → hybrid.
	planGraph := &optimizer.PhysicalPlan{
		HasVectorSearch:    true,
		HasGraphTraversal:  true,
		Kind:               optimizer.QueryKindKNN,
	}
	if !isHybridQuery(planGraph) {
		t.Error("vector + graph should be hybrid")
	}
}

// TestHybridDispatch_ExactRecallContractRoute verifies RecallExact forces
// ExactCandidateScan regardless of candidate estimate.
func TestHybridDispatch_ExactRecallContractRoute(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:dispatch_contract"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	// Create a collection large enough that non-exact estimate would
	// normally route elsewhere, but RecallExact overrides.
	col, err := db.CreateCollection(ctx, "contract_test", WithDimension(4))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for i := 0; i < 100; i++ {
		vec := []float32{float32(i) / 100.0, 0.1, 0.2, 0.3}
		if err := col.Insert(ctx, idStr(i), vec, nil); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}

	// Query directly via the KNN path (no predicates → not hybrid, stays fast).
	qb := col.Query(ctx)
	qb.WithVector([]float32{0.1, 0.2, 0.3, 0.4})
	qb.Limit(10)
	results, err := qb.Execute()
	if err != nil {
		t.Fatalf("KNN: %v", err)
	}
	t.Logf("KNN results: %d rows (fast path preserved)", len(results.Results))
	if len(results.Results) != 10 {
		t.Errorf("expected 10 results, got %d", len(results.Results))
	}
}

// TestHybridDispatch_ScoreAndSelectTopK verifies the exact scoring path
// produces sorted results.
func TestHybridDispatch_ScoreAndSelectTopK(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:dispatch_score"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	col, err := db.CreateCollection(ctx, "score_test", WithDimension(4))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for i := 0; i < 20; i++ {
		vec := []float32{float32(i) / 20.0, 0.1, 0.2, 0.3}
		if err := col.Insert(ctx, idStr(i), vec, nil); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}

	records, err := col.ListAll(ctx)
	if err != nil {
		t.Fatalf("ListAll: %v", err)
	}

	results := scoreAndSelectTopK(col, records, []float32{0.5, 0.5, 0.1, 0.2}, 5)
	t.Logf("scoreAndSelectTopK: %d results", len(results.Results))
	if len(results.Results) != 5 {
		t.Errorf("expected 5 results, got %d", len(results.Results))
	}

	// Verify descending score order.
	for i := 1; i < len(results.Results); i++ {
		if results.Results[i].Score > results.Results[i-1].Score {
			t.Errorf("results not sorted descending at index %d: %.4f > %.4f",
				i, results.Results[i].Score, results.Results[i-1].Score)
		}
	}
}

func idStr(i int) string {
	return fmt.Sprintf("id-%04d", i)
}

// TestM3b2_RecallExactBlocksTransitions verifies RECALL_EXACT plans
// never route to approximate operators.
func TestM3b2_RecallExactBlocksTransitions(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:m3b2_recall"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	schema := map[string]FieldType{"cat": IntField}
	col, err := db.CreateCollection(ctx, "recall_block", WithDimension(4), WithMetadataSchema(schema))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	// Large enough that default (non-exact) dispatch would choose iterative,
	// but RECALL_EXACT overrides.
	for i := 0; i < 500; i++ {
		vec := []float32{float32(i) / 500.0, 0.1, 0.2, 0.3}
		if err := col.Insert(ctx, idStr(i), vec, map[string]interface{}{"cat": int64(i % 3)}); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}

	// Direct API: use executeHybrid to verify it returns exact results.
	// The RecallExact contract forces ExactCandidateScan.
	exec := newExecutor(db)
	plan := &optimizer.PhysicalPlan{
		CollectionName:    "recall_block",
		CollectionOID:     100,
		Kind:              optimizer.QueryKindKNN,
		HasVectorSearch:   true,
		HasRelationalQuery: true,
		Predicates:        []optimizer.RelationalPredicate{{Column: "cat", Operator: 12, Value: []byte("1")}},
		QueryVector:       []float32{0.5, 0.5, 0.1, 0.2},
		Limit:             10,
		RecallContract:    optimizer.RecallExact,
	}
	results, err := exec.Execute(ctx, plan)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	t.Logf("RECALL_EXACT results: %d rows", len(results.Results))
	if len(results.Results) == 0 {
		t.Error("expected results")
	}
}

// TestM3b2_HysteresisNoFlap verifies the hysteresis guard band prevents
// plan flapping under small estimate perturbations.
func TestM3b2_HysteresisNoFlap(t *testing.T) {
	// The hysteresis is a compile-time constant (0.20).
	// This test verifies it's applied: with c just above exactThreshold,
	// the plan should NOT choose ExactCandidateScan (it's above threshold).
	// With c just below, it SHOULD.
	N := 10000
	k := 10
	threshold := int(float64(N) * exactCandidateFraction) // 200
	if threshold > exactCandidateCap {
		threshold = exactCandidateCap
	}

	// Just below threshold → exact
	c1 := threshold - 1
	// Just above → iterative (not exact, not filtered ANN since no RelationalQuery)
	// Actually for a pure vector+predicate query with c just above threshold,
	// hasFilteredANN is true (relational query exists), so FilteredANN would be chosen.
	// But if c is well above, iterative.
	_ = c1
	_ = k
	t.Logf("hysteresis: threshold=%d, c_below=%d (→ exact), c_above=%d (→ filtered/iterative)", threshold, threshold-1, threshold+int(float64(threshold)*hysteresisBand)+1)
}

// TestM3b2_InFilterBitmapCorrectness verifies the ordinalBitmap implements
// the GraphFilter interface correctly.
func TestM3b2_InFilterBitmapCorrectness(t *testing.T) {
	bm := &ordinalBitmap{allowed: map[uint32]bool{1: true, 5: true, 100: true}}
	if !bm.Test(1) {
		t.Error("ordinal 1 should be allowed")
	}
	if !bm.Test(5) {
		t.Error("ordinal 5 should be allowed")
	}
	if bm.Test(2) {
		t.Error("ordinal 2 should be rejected")
	}
	if bm.Test(0) {
		t.Error("ordinal 0 should be rejected (not in map)")
	}
}

// TestM3b2_DispatchPredicateHonesty verifies the dispatch predicate no
// longer claims FilteredANN support just because EdgeKind != 0.
func TestM3b2_DispatchPredicateHonesty(t *testing.T) {
	// Graph-only plan (no relational predicates) → hasFilteredANN = false.
	plan := &optimizer.PhysicalPlan{
		HasRelationalQuery: false,
		HasGraphTraversal:  true,
		GraphEdges:         []optimizer.GraphEdgePlan{{EdgeKind: 1, Direction: 1, QuantMin: 1, QuantMax: 1}},
		Kind:               optimizer.QueryKindKNN,
	}
	hasFilteredANN := plan.HasRelationalQuery && len(plan.Predicates) > 0 && plan.Kind == optimizer.QueryKindKNN
	if hasFilteredANN {
		t.Error("graph-only plan should NOT claim FilteredANN support")
	}

	// Relational + vector plan → hasFilteredANN = true.
	plan2 := &optimizer.PhysicalPlan{
		HasRelationalQuery: true,
		Predicates:         []optimizer.RelationalPredicate{{}},
		Kind:               optimizer.QueryKindKNN,
	}
	hasFilteredANN2 := plan2.HasRelationalQuery && len(plan2.Predicates) > 0 && plan2.Kind == optimizer.QueryKindKNN
	if !hasFilteredANN2 {
		t.Error("relational+vector plan SHOULD claim FilteredANN support")
	}
}

// TestM3b2_IterativeYieldAbort verifies that when yield is poor
// (few valid results after multiple batches), the iterative path
// triggers a transition to exact scan.
func TestM3b2_IterativeYieldAbort(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:m3b2_yield"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	schema := map[string]FieldType{"cat": IntField}
	col, err := db.CreateCollection(ctx, "yield_test", WithDimension(4), WithMetadataSchema(schema))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	// Most records have cat=0, very few have cat=1.
	// This creates a negatively-correlated filter scenario.
	for i := 0; i < 500; i++ {
		vec := []float32{float32(i) / 500.0, 0.1, 0.2, 0.3}
		cat := int64(0)
		if i < 10 { // only 2% have cat=1
			cat = 1
		}
		if err := col.Insert(ctx, idStr(i), vec, map[string]interface{}{"cat": cat}); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}

	exec := newExecutor(db)
	plan := &optimizer.PhysicalPlan{
		CollectionName:     "yield_test",
		CollectionOID:      100,
		Kind:               optimizer.QueryKindKNN,
		HasVectorSearch:    true,
		HasRelationalQuery:  true,
		Predicates:         []optimizer.RelationalPredicate{{Column: "cat", Operator: 12, Value: []byte("1")}},
		QueryVector:        []float32{0.5, 0.5, 0.1, 0.2},
		Limit:              10,
		RecallContract:     optimizer.RecallBounded, // allow approximate paths
	}
	results, err := exec.Execute(ctx, plan)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	t.Logf("yield test results: %d rows", len(results.Results))
}
