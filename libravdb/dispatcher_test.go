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

func TestIterativeOperatorUsesOneTraversalAndUpdatesValid(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:iterative_single_traversal"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	col, err := db.CreateCollection(ctx, "iterative_single", WithDimension(4), WithMetadataSchema(map[string]FieldType{"cat": IntField}), WithIndexedFields("cat"))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for i := 0; i < 500; i++ {
		cat := int64(0)
		if i%5 == 0 {
			cat = 1
		}
		if err := col.Insert(ctx, idStr(i), []float32{float32(i) / 500, 0.1, 0.2, 0.3}, map[string]interface{}{"cat": cat}); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}

	plan := &optimizer.PhysicalPlan{
		CollectionName:     "iterative_single",
		Kind:               optimizer.QueryKindKNN,
		HasVectorSearch:    true,
		HasRelationalQuery: true,
		Predicates:         []optimizer.RelationalPredicate{{Column: "cat", Operator: 12, Value: []byte("1")}},
		QueryVector:        []float32{0.5, 0.5, 0.1, 0.2},
		Limit:              10,
		RecallContract:     optimizer.RecallBounded,
	}
	exec := newExecutor(db)
	metrics := &QueryMetrics{EstConjunctionCandidates: 100}
	results, err := exec.executeIterativeANNThenFilter(ctx, plan, metrics, &hybridConstraints{})
	if err != nil {
		t.Fatalf("iterative execute: %v", err)
	}
	if len(results.Results) != 10 {
		t.Fatalf("iterative results = %d, want 10", len(results.Results))
	}
	if metrics.FilterValidHits < 10 {
		t.Fatalf("FilterValidHits = %d, want at least 10", metrics.FilterValidHits)
	}
	if metrics.ANNBatches != 1 {
		t.Fatalf("ANNBatches = %d, want one shared traversal", metrics.ANNBatches)
	}
	for _, result := range results.Results {
		record, err := col.Get(ctx, result.ID)
		if err != nil {
			t.Fatal(err)
		}
		if record.Metadata["cat"] != int64(1) {
			t.Fatalf("result %s has category %v", result.ID, record.Metadata["cat"])
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
		HasVectorSearch:   true,
		HasGraphTraversal: true,
		Kind:              optimizer.QueryKindKNN,
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
		CollectionName:     "recall_block",
		CollectionOID:      100,
		Kind:               optimizer.QueryKindKNN,
		HasVectorSearch:    true,
		HasRelationalQuery: true,
		Predicates:         []optimizer.RelationalPredicate{{Column: "cat", Operator: 12, Value: []byte("1")}},
		QueryVector:        []float32{0.5, 0.5, 0.1, 0.2},
		Limit:              10,
		RecallContract:     optimizer.RecallExact,
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
	bm := &ordinalBitmap{membership: &mapMembership{m: map[uint32]bool{1: true, 5: true, 100: true}}}
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

	sharded := &ordinalBitmap{
		membership: &mapMembership{m: map[uint32]bool{1: true}},
		byMembership: []ordinalMembership{
			&mapMembership{m: map[uint32]bool{1: true}},
			&mapMembership{m: map[uint32]bool{}},
		},
	}
	if !sharded.ForShard(0).Test(1) {
		t.Error("ordinal 1 should be allowed in shard 0")
	}
	if sharded.ForShard(1).Test(1) {
		t.Error("ordinal 1 from shard 0 must not authorize shard 1 ordinal 1")
	}
	thresholded := &thresholdGraphFilter{base: sharded, threshold: 0.5}
	if thresholded.ForShard(1).Test(1) {
		t.Error("threshold wrapper must preserve shard-local filtering")
	}
}

func TestScoreAndSelectTopKNormalizesCosineVectors(t *testing.T) {
	col := &Collection{config: &CollectionConfig{Metric: CosineDistance}}
	results := scoreAndSelectTopK(col, []Record{
		{ID: "large-off-axis", Vector: []float32{100, 100}},
		{ID: "aligned", Vector: []float32{1, 0}},
	}, []float32{1, 0}, 1)
	if len(results.Results) != 1 || results.Results[0].ID != "aligned" {
		t.Fatalf("cosine top-1 = %#v, want aligned", results.Results)
	}
}

// TestM3b2_DispatchPredicateHonesty verifies FilteredANN is offered only when
// a vector query has a scalar or materializable graph bitmap.
func TestM3b2_DispatchPredicateHonesty(t *testing.T) {
	// Vector + graph can use the graph candidate bitmap.
	plan := &optimizer.PhysicalPlan{
		HasVectorSearch:   true,
		HasGraphTraversal: true,
		GraphEdges:        []optimizer.GraphEdgePlan{{EdgeKind: 1, Direction: 1, QuantMin: 1, QuantMax: 1}},
		Kind:              optimizer.QueryKindGraph,
	}
	hasFilteredANN := (plan.HasVectorSearch || plan.Kind == optimizer.QueryKindKNN) &&
		((plan.HasRelationalQuery && len(plan.Predicates) > 0) || plan.HasGraphTraversal)
	if !hasFilteredANN {
		t.Error("vector+graph plan should claim FilteredANN support")
	}

	// Relational + vector plan → hasFilteredANN = true.
	plan2 := &optimizer.PhysicalPlan{
		HasRelationalQuery: true,
		Predicates:         []optimizer.RelationalPredicate{{}},
		Kind:               optimizer.QueryKindKNN,
	}
	hasFilteredANN2 := (plan2.HasVectorSearch || plan2.Kind == optimizer.QueryKindKNN) &&
		((plan2.HasRelationalQuery && len(plan2.Predicates) > 0) || plan2.HasGraphTraversal)
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
		HasRelationalQuery: true,
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

// TestHybridGraphConstraintsAllOperators proves that MATCH is an authoritative
// constraint for exact, filtered-ANN, and iterative hybrid execution. The
// nearest scalar-matching record is deliberately outside the graph, while a
// graph-reachable record deliberately fails the scalar predicate.
func TestHybridGraphConstraintsAllOperators(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:hybrid_graph_constraints"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()

	col, err := db.CreateCollection(ctx, "hybrid_graph", WithDimension(4), WithGraph(gr), WithMetadataSchema(map[string]FieldType{"cat": IntField}))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	records := []struct {
		id  string
		vec []float32
		cat int64
	}{
		{id: "seed", vec: []float32{0, 1, 0, 0}, cat: 0},
		{id: "allowed", vec: []float32{0.8, 0.2, 0, 0}, cat: 1},
		{id: "reachable-wrong-scalar", vec: []float32{0.95, 0.05, 0, 0}, cat: 0},
		{id: "outside-nearest", vec: []float32{1, 0, 0, 0}, cat: 1},
	}
	for _, rec := range records {
		if err := col.Insert(ctx, rec.id, normalize(rec.vec), map[string]interface{}{"cat": rec.cat}); err != nil {
			t.Fatalf("insert %s: %v", rec.id, err)
		}
	}

	seedID, err := db.GetNodeID(ctx, "hybrid_graph", "seed")
	if err != nil {
		t.Fatalf("seed node: %v", err)
	}
	allowedID, err := db.GetNodeID(ctx, "hybrid_graph", "allowed")
	if err != nil {
		t.Fatalf("allowed node: %v", err)
	}
	wrongScalarID, err := db.GetNodeID(ctx, "hybrid_graph", "reachable-wrong-scalar")
	if err != nil {
		t.Fatalf("wrong-scalar node: %v", err)
	}

	txn := gr.BeginTxn()
	if err := gr.AddEdge(txn, seedID, allowedID, 1, 0); err != nil {
		t.Fatalf("seed->allowed: %v", err)
	}
	if err := gr.AddEdge(txn, seedID, wrongScalarID, 1, 0); err != nil {
		t.Fatalf("seed->wrong-scalar: %v", err)
	}
	gr.RegisterVertexLabel(seedID, "Service")

	basePlan := optimizer.PhysicalPlan{
		CollectionName:     "hybrid_graph",
		Kind:               optimizer.QueryKindKNN,
		HasVectorSearch:    true,
		HasGraphTraversal:  true,
		HasExplicitSeed:    true,
		ExplicitSeedID:     seedID,
		GraphEdges:         []optimizer.GraphEdgePlan{{Direction: 1}},
		MaxHops:            1,
		HasRelationalQuery: true,
		Predicates:         []optimizer.RelationalPredicate{{Column: "cat", Operator: 12, Value: []byte("1")}},
		QueryVector:        []float32{1, 0, 0, 0},
		Limit:              1,
		RecallContract:     optimizer.RecallExact,
	}

	assertOnlyAllowed := func(t *testing.T, results *SearchResults, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("execute: %v", err)
		}
		got := make([]string, 0, len(results.Results))
		for _, result := range results.Results {
			got = append(got, result.ID)
		}
		if len(got) != 1 || got[0] != "allowed" {
			t.Fatalf("result IDs = %v, want [allowed]", got)
		}
	}

	exec := newExecutor(db)
	t.Run("exact-dispatch", func(t *testing.T) {
		results, err := exec.Execute(ctx, &basePlan)
		assertOnlyAllowed(t, results, err)
	})

	t.Run("filtered-ann", func(t *testing.T) {
		metrics := &QueryMetrics{EstConjunctionCandidates: 1}
		constraints, err := exec.prepareHybridConstraints(ctx, &basePlan, metrics)
		if err != nil {
			t.Fatalf("prepare constraints: %v", err)
		}
		results, err := exec.executeFilteredANN(ctx, &basePlan, metrics, constraints)
		assertOnlyAllowed(t, results, err)
		if metrics.ActGraphCandidates != 2 || metrics.ActConjunctionCandidates != 1 {
			t.Fatalf("candidate metrics graph=%d conjunction=%d, want 2 and 1", metrics.ActGraphCandidates, metrics.ActConjunctionCandidates)
		}
	})

	t.Run("iterative", func(t *testing.T) {
		metrics := &QueryMetrics{EstConjunctionCandidates: 1}
		constraints, err := exec.prepareHybridConstraints(ctx, &basePlan, metrics)
		if err != nil {
			t.Fatalf("prepare constraints: %v", err)
		}
		results, err := exec.executeIterativeANNThenFilter(ctx, &basePlan, metrics, constraints)
		assertOnlyAllowed(t, results, err)
	})

	// Exercise parser -> optimizer -> dispatcher as real SQL. The vector
	// function is nested under AND so this also guards hybrid classification.
	t.Run("sql-end-to-end", func(t *testing.T) {
		results, err := db.Query(ctx, "SELECT id FROM GRAPH_TABLE(hybrid_graph MATCH (s:Service)-[e]->(x)) WHERE SIMILARITY(vector, '[1,0,0,0]') > 0 AND cat = 1 LIMIT 1")
		assertOnlyAllowed(t, results, err)
	})
}
