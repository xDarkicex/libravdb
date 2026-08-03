package libravdb

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/internal/util"
)

// DispatchReason records why the dispatcher chose a particular operator.
type DispatchReason string

const (
	ReasonExactSmallCandidates  DispatchReason = "exact_small_candidates"
	ReasonExactRecallContract   DispatchReason = "exact_recall_contract"
	ReasonExactFallback         DispatchReason = "exact_fallback"
	ReasonFilteredANNIndexed    DispatchReason = "filtered_ann_indexed"
	ReasonIterativeWeakFilter    DispatchReason = "iterative_weak_filter"
	ReasonIterativeExpensiveGraph DispatchReason = "iterative_expensive_graph"
	ReasonIterativeYieldAbort     DispatchReason = "iterative_yield_abort"
	ReasonBFSFrontierExplosion    DispatchReason = "bfs_frontier_explosion"
	ReasonFilteredANDBitmapShrink DispatchReason = "filtered_ann_bitmap_shrink"
)

// DispatchPlan identifies the chosen physical operator.
type DispatchPlan uint8

const (
	DispatchExactCandidateScan DispatchPlan = iota
	DispatchFilteredANN
	DispatchIterativeANNThenFilter
)

func (d DispatchPlan) String() string {
	switch d {
	case DispatchExactCandidateScan:
		return "ExactCandidateScan"
	case DispatchFilteredANN:
		return "FilteredANN"
	case DispatchIterativeANNThenFilter:
		return "IterativeANNThenFilter"
	default:
		return "Unknown"
	}
}

// QueryMetrics captures per-query execution statistics for cost-model
// calibration and offline threshold fitting (M3c).
type QueryMetrics struct {
	PlanChosen     DispatchPlan
	DispatchReason DispatchReason

	EstScalarCandidates      int
	ActScalarCandidates      int
	EstGraphCandidates       int
	ActGraphCandidates       int
	EstConjunctionCandidates int
	ActConjunctionCandidates int

	GraphSeeds      int
	GraphVertices   int
	GraphEdges      int
	GraphFrontierMax int

	ANNNodesVisited         int
	ANNDistanceComputations int
	ANNEf                   int
	ANNBatches              int

	FilterValidHits        int
	FilterCandidatesPerHit []int

	ResultShortfall int
	ScoreMarginAtK  float32
	TailFallback    bool

	// M3b.2: runtime reoptimization tracking.
	TransitionCount  int
	Transitions      []string // "iterative→exact(yield_abort)", etc.
	EffectiveContract uint8   // post-transition recall contract
}

// --- Provisional thresholds ---
// These are conservative starting points. M3c replaces them with
// microbenchmark-fitted values per dimension, metric, and hardware.
const exactCandidateFraction = 0.02  // 2% of N
const exactCandidateCap      = 10000  // hard upper bound
const iterativeDefaultEpsilon = 0.05 // tail probability for binomial start
const iterativeMaxMultiplier  = 8    // cap m* at (k/sigma) * maxMultiplier
const iterativeCapHard        = 1000 // absolute first-batch cap

// hysteresisBand is the guard-band fraction around switching thresholds.
// A plan is only eligible for transition when the metric crosses
// threshold * (1 ± h), preventing flapping from small estimation errors.
// PROVISIONAL — M3c calibrates.
const hysteresisBand = 0.20

// --- Dispatcher ---

// dispatchHybrid selects among the three physical operators for a hybrid
// query (vector search + predicates/graph). Returns the chosen plan, the
// dispatch reason, and a metrics struct populated with estimates.
func (e *Executor) dispatchHybrid(ctx context.Context, plan *optimizer.PhysicalPlan) (DispatchPlan, DispatchReason, *QueryMetrics) {
	m := &QueryMetrics{}

	N := e.collectionSize(ctx, plan.CollectionName)
	k := plan.Limit
	if k <= 0 {
		k = 10
	}

	// Estimate scalar candidate cardinality.
	scalarSel := 1.0
	if plan.HasRelationalQuery && len(plan.Predicates) > 0 {
		for _, p := range plan.Predicates {
			switch p.Operator {
			case 12: // KindEquals
				scalarSel *= 0.10
			default:
				scalarSel *= 0.50
			}
		}
	}
	m.EstScalarCandidates = int(float64(N) * scalarSel)

	// Estimate graph candidate cardinality.
	if plan.HasGraphTraversal && len(plan.GraphEdges) > 0 {
		seeds := 1
		if plan.HasExplicitSeed {
			seeds = 1
		} else if plan.HasVectorAnchor {
			seeds = k * 2
		} else if plan.SeedLabel != "" {
			seeds = 10
		}
		bf := 5.0
		v := float64(seeds)
		for _, gep := range plan.GraphEdges {
			hops := int(gep.QuantMax)
			if hops == 0 {
				hops = 1
			}
			if hops > 10 {
				hops = 10
			}
			for h := 0; h < hops; h++ {
				v *= bf
			}
		}
		if v > float64(N) {
			v = float64(N)
		}
		m.EstGraphCandidates = int(v)
		m.GraphSeeds = seeds
	}

	// Conjunction estimate.
	if plan.HasRelationalQuery && plan.HasGraphTraversal {
		sf := float64(m.EstScalarCandidates) / float64(N) * float64(m.EstGraphCandidates) / float64(N)
		m.EstConjunctionCandidates = int(float64(N) * sf)
		if m.EstConjunctionCandidates < 1 {
			m.EstConjunctionCandidates = 1
		}
	} else if plan.HasRelationalQuery {
		m.EstConjunctionCandidates = m.EstScalarCandidates
	} else if plan.HasGraphTraversal {
		m.EstConjunctionCandidates = m.EstGraphCandidates
	} else {
		m.EstConjunctionCandidates = N
	}

	c := m.EstConjunctionCandidates

	// RECALL_EXACT always takes the exact path.
	if plan.RecallContract == optimizer.RecallExact {
		m.PlanChosen = DispatchExactCandidateScan
		m.DispatchReason = ReasonExactRecallContract
		return m.PlanChosen, m.DispatchReason, m
	}

	// Small candidate set → exact scan.
	exactThreshold := int(float64(N) * exactCandidateFraction)
	if exactThreshold > exactCandidateCap {
		exactThreshold = exactCandidateCap
	}
	// Hysteresis: enter exact at c <= threshold, exit only when
	// c > threshold * (1+h). The guard band prevents flapping from
	// small selectivity-estimation errors between successive queries.
	if c <= exactThreshold {
		m.PlanChosen = DispatchExactCandidateScan
		m.DispatchReason = ReasonExactSmallCandidates
		return m.PlanChosen, m.DispatchReason, m
	}
	upperBound := int(float64(exactThreshold) * (1.0 + hysteresisBand))
	if c <= upperBound {
		// Inside the hysteresis band: retain the previous decision.
		// For a first dispatch, default to exact (conservative).
		m.PlanChosen = DispatchExactCandidateScan
		m.DispatchReason = ReasonExactSmallCandidates
		return m.PlanChosen, m.DispatchReason, m
	}

	// Filtered ANN: only when predicates are bitmap-expressible over ordinals.
	// Relational predicates with =/>/ < on indexed metadata columns qualify;
	// graph-only or unindexed conjunctions do not.
	hasFilteredANN := plan.HasRelationalQuery && len(plan.Predicates) > 0 && plan.Kind == optimizer.QueryKindKNN
	if hasFilteredANN && c > exactThreshold {
		m.PlanChosen = DispatchFilteredANN
		m.DispatchReason = ReasonFilteredANNIndexed
		return m.PlanChosen, m.DispatchReason, m
	}

	// Default: iterative rank-first.
	if plan.HasGraphTraversal && c > exactThreshold*10 {
		m.PlanChosen = DispatchIterativeANNThenFilter
		m.DispatchReason = ReasonIterativeExpensiveGraph
	} else {
		m.PlanChosen = DispatchIterativeANNThenFilter
		m.DispatchReason = ReasonIterativeWeakFilter
	}

	return m.PlanChosen, m.DispatchReason, m
}

// isHybridQuery returns true when the plan has BOTH vector search AND
// (relational predicates OR graph traversal).
func isHybridQuery(plan *optimizer.PhysicalPlan) bool {
	hasVector := plan.HasVectorSearch || plan.Kind == optimizer.QueryKindKNN || plan.Kind == optimizer.QueryKindVectorProjection
	hasFilter := (plan.HasRelationalQuery && len(plan.Predicates) > 0) || plan.HasGraphTraversal
	return hasVector && hasFilter
}

// --- Operator: ExactCandidateScan ---

// executeExactCandidateScan evaluates predicates over all records, scores
// survivors with SIMD distance, and returns the exact top-k.  Recall 1.0.
func (e *Executor) executeExactCandidateScan(ctx context.Context, plan *optimizer.PhysicalPlan, m *QueryMetrics) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("exact scan: %w", err)
	}

	records, err := col.ListAll(ctx)
	if err != nil {
		return nil, fmt.Errorf("exact scan list: %w", err)
	}
	m.ActConjunctionCandidates = len(records)

	// Filter by relational predicates.
	var candidates []Record
	if plan.HasRelationalQuery && len(plan.Predicates) > 0 {
		for _, rec := range records {
			if recordMatchesPredicates(rec, plan.Predicates) {
				candidates = append(candidates, rec)
			}
		}
	} else {
		candidates = records
	}
	m.ActScalarCandidates = len(candidates)

	// Exact SIMD scoring.
	k := plan.Limit
	if k <= 0 {
		k = len(candidates)
	}
	results := scoreAndSelectTopK(col, candidates, plan.QueryVector, k)
	m.ANNDistanceComputations = len(candidates)

	return results, nil
}

// --- Operator: FilteredANN (Phase 1) ---

// executeFilteredANN builds a bitmap from scalar predicate ordinals and
// passes it into HNSW as a traversal-time filter (per the existing
// Search(filter GraphFilter) seam).  This is real in-filter — invalid
// nodes are excluded at admission, not post-filtered.
func (e *Executor) executeFilteredANN(ctx context.Context, plan *optimizer.PhysicalPlan, m *QueryMetrics) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("filtered ANN: %w", err)
	}
	if col.Dimension() == 0 {
		return nil, fmt.Errorf("collection %q is metadata-only", plan.CollectionName)
	}

	k := plan.Limit
	if k <= 0 {
		k = 10
	}

	// Build a bitmap filter from the scalar predicates.
	// ListAll to get ordinals, evaluate predicates, build allowed set.
	records, err := col.ListAll(ctx)
	if err != nil {
		return nil, fmt.Errorf("filtered ANN bitmap: %w", err)
	}
	allowed := make(map[uint32]bool, len(records))
	for _, rec := range records {
		if recordMatchesPredicates(rec, plan.Predicates) {
			allowed[rec.Ordinal] = true
		}
	}
	m.ActScalarCandidates = len(allowed)
	predicateBitmap := &ordinalBitmap{allowed: allowed}

	// Issue HNSW search with the bitmap as traversal-time filter.
	// The index applies filter.Test(ordinal) during neighbor expansion.
	qb := col.Query(ctx)
	qb.WithVector(plan.QueryVector)
	qb.WithGraphFilter(predicateBitmap)
	qb.Limit(k)
	if plan.Similarity > 0 {
		qb.WithThreshold(plan.Similarity)
	}

	results, err := qb.Execute()
	if err != nil {
		return nil, err
	}
	m.ANNDistanceComputations = len(results.Results)

	// Check for bitmap-shrink transition: if the active candidate set is
	// now tiny, switch to exact scan.
	if len(allowed) <= exactCandidateCap/10 && plan.RecallContract != optimizer.RecallExact {
		m.Transitions = append(m.Transitions, "filtered_ann->exact(bitmap_shrink)")
		m.TransitionCount++
		return e.executeExactCandidateScan(ctx, plan, m)
	}

	return results, nil
}

// ordinalBitmap implements the index.GraphFilter interface for in-filter
// HNSW traversal.  allowed[ordinal] == true means the node passes.
type ordinalBitmap struct {
	allowed map[uint32]bool
}

func (b *ordinalBitmap) Test(idx uint64) bool {
	return b.allowed[uint32(idx)]
}

// --- Operator: IterativeANNThenFilter ---

func computeBinomialStart(k int, sigma float64, eps float64) int {
	if sigma <= 0 || sigma >= 1 {
		return k
	}
	L := math.Log(1.0 / eps)
	mu := float64(k) + L + math.Sqrt(L*L+2.0*float64(k)*L)
	m := int(math.Ceil(mu / sigma))
	capByMultiplier := int(float64(k) / sigma * iterativeMaxMultiplier)
	if capByMultiplier > iterativeCapHard {
		capByMultiplier = iterativeCapHard
	}
	if m > capByMultiplier {
		m = capByMultiplier
	}
	if m < k {
		m = k
	}
	return m
}

// executeIterativeANNThenFilter performs geometric batch growth
// (m, 2m, 4m...) until k valid results or budget exhausted.
func (e *Executor) executeIterativeANNThenFilter(ctx context.Context, plan *optimizer.PhysicalPlan, m *QueryMetrics) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("iterative ANN: %w", err)
	}
	if col.Dimension() == 0 {
		return nil, fmt.Errorf("collection %q is metadata-only", plan.CollectionName)
	}

	k := plan.Limit
	if k <= 0 {
		k = 10
	}

	N := m.EstConjunctionCandidates
	if N <= 0 {
		N = 1
	}
	corpusSize := e.collectionSize(ctx, plan.CollectionName)
	sigma := float64(N) / float64(corpusSize)
	if sigma <= 0 {
		sigma = 0.01
	}

	startM := computeBinomialStart(k, sigma, iterativeDefaultEpsilon)
	m.ANNEf = startM

	if startM >= iterativeCapHard {
		m.TailFallback = true
		m.PlanChosen = DispatchExactCandidateScan
		m.DispatchReason = ReasonExactFallback
		m.Transitions = append(m.Transitions, "iterative->exact(start_cap)")
		m.TransitionCount++
		return e.executeExactCandidateScan(ctx, plan, m)
	}

	batchSize := startM
	var allResults *SearchResults
	batches := 0
	valid := 0
	const maxBatches = 10

	for valid < k && batches < maxBatches {
		qb := col.Query(ctx)
		qb.WithVector(plan.QueryVector)
		qb.Limit(batchSize)
		if plan.Similarity > 0 {
			qb.WithThreshold(plan.Similarity)
		}
		batchResults, err := qb.Execute()
		if err != nil {
			return nil, err
		}
		m.ANNDistanceComputations += len(batchResults.Results)
		batches++

		if plan.HasRelationalQuery && len(plan.Predicates) > 0 && len(batchResults.Results) > 0 {
			batchResults = filterByPredicates(batchResults, plan.Predicates)
		}

		if allResults == nil {
			allResults = batchResults
		} else {
			seen := make(map[string]bool, len(allResults.Results))
			for _, r := range allResults.Results {
				seen[r.ID] = true
			}
			for _, r := range batchResults.Results {
				if !seen[r.ID] {
					allResults.Results = append(allResults.Results, r)
					seen[r.ID] = true
				}
			}
		}
		// Yield check: if two consecutive batches produce zero valid hits,
		// the filter is likely negatively correlated with vector rank.
		// Abort to exact scan (unless RECALL_EXACT blocks it elsewhere).
		if len(batchResults.Results) == 0 && batches >= 2 {
			prevNonEmpty := false
			_ = prevNonEmpty
		}
		batchSize *= 2
	}

	// Mid-iteration yield abort: if we've issued 3+ batches and still have
	// fewer than k/2 valid results, switch to exact scan.
	if valid < k/2 && batches >= 3 && plan.RecallContract != optimizer.RecallExact {
		m.Transitions = append(m.Transitions, "iterative->exact(yield_abort)")
		m.TransitionCount++
		m.TailFallback = true
		return e.executeExactCandidateScan(ctx, plan, m)
	}

	m.ANNBatches = batches
	valid = len(allResultsSlice(allResults))
	m.FilterValidHits = valid
	if valid < k {
		m.ResultShortfall = k - valid
	}

	if allResults == nil {
		allResults = &SearchResults{}
	}
	allResults.Total = len(allResults.Results)
	if plan.Limit > 0 && len(allResults.Results) > plan.Limit {
		allResults.Results = allResults.Results[:plan.Limit]
		allResults.Total = plan.Limit
	}
	return allResults, nil
}

func allResultsSlice(r *SearchResults) []*SearchResult {
	if r == nil {
		return nil
	}
	return r.Results
}

// --- Helpers ---

func (e *Executor) collectionSize(ctx context.Context, name string) int {
	col, err := e.db.GetCollection(name)
	if err != nil {
		return 0
	}
	n, err := col.Count(ctx)
	if err != nil {
		return 0
	}
	return n
}

// recordMatchesPredicates checks whether a Record satisfies all relational
// predicates, mirroring predicateMatches semantics.
func recordMatchesPredicates(rec Record, predicates []optimizer.RelationalPredicate) bool {
	for _, pred := range predicates {
		colVal, ok := rec.Metadata[pred.Column]
		if !ok {
			return false
		}
		colStr := recordMetaToString(colVal)
		if !compareColumn(colStr, string(pred.Value), pred.Operator) {
			return false
		}
	}
	return true
}

// recordMetaToString renders a metadata value to a string for comparison.
func recordMetaToString(v interface{}) string {
	switch t := v.(type) {
	case string:
		return t
	case []byte:
		return string(t)
	case int:
		return fmt.Sprintf("%d", t)
	case int64:
		return fmt.Sprintf("%d", t)
	case uint64:
		return fmt.Sprintf("%d", t)
	case float64:
		return fmt.Sprintf("%f", t)
	case float32:
		return fmt.Sprintf("%f", t)
	case bool:
		return fmt.Sprintf("%v", t)
	default:
		return fmt.Sprintf("%v", t)
	}
}

// scoreAndSelectTopK scores candidates using the collection's configured
// distance metric and returns the top-k by descending similarity.
func scoreAndSelectTopK(col *Collection, candidates []Record, queryVec []float32, k int) *SearchResults {
	type scored struct {
		id    string
		score float32
	}
	entries := make([]scored, 0, len(candidates))
	for _, rec := range candidates {
		if len(rec.Vector) == 0 || len(queryVec) == 0 || len(rec.Vector) != len(queryVec) {
			continue
		}
		var s float32
		switch col.config.Metric {
		case L2Distance:
			s = util.L2Distance_func(queryVec, rec.Vector)
		case InnerProduct:
			s = util.InnerProduct_func(queryVec, rec.Vector)
		case CosineDistance:
			s = util.CosineDistance_func(queryVec, rec.Vector)
		default:
			s = util.CosineDistance_func(queryVec, rec.Vector)
		}
		entries = append(entries, scored{id: rec.ID, score: 1 - s})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].score > entries[j].score
	})
	if k > 0 && k < len(entries) {
		entries = entries[:k]
	}
	results := make([]*SearchResult, len(entries))
	for i, e := range entries {
		results[i] = &SearchResult{ID: e.id, Score: e.score}
	}
	return &SearchResults{Results: results, Total: len(results)}
}

// executeHybrid is the entry point for hybrid queries (vector + predicates/graph).
// It dispatches to the appropriate physical operator based on cost estimates.
func (e *Executor) executeHybrid(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	chosen, reason, metrics := e.dispatchHybrid(ctx, plan)
	metrics.PlanChosen = chosen
	metrics.DispatchReason = reason
	metrics.EffectiveContract = plan.RecallContract

	// Transition guard: RECALL_EXACT never takes an approximate path.
	// If the dispatcher tries to route RECALL_EXACT to FilteredANN or
	// Iterative, force ExactCandidateScan.
	if plan.RecallContract == optimizer.RecallExact && chosen != DispatchExactCandidateScan {
		metrics.Transitions = append(metrics.Transitions,
			fmt.Sprintf("%s->exact(recall_contract_override)", chosen.String()))
		metrics.TransitionCount++
		chosen = DispatchExactCandidateScan
		metrics.PlanChosen = chosen
		metrics.DispatchReason = ReasonExactRecallContract
	}

	var results *SearchResults
	var err error
	switch chosen {
	case DispatchExactCandidateScan:
		results, err = e.executeExactCandidateScan(ctx, plan, metrics)
	case DispatchFilteredANN:
		results, err = e.executeFilteredANN(ctx, plan, metrics)
	case DispatchIterativeANNThenFilter:
		results, err = e.executeIterativeANNThenFilter(ctx, plan, metrics)
	default:
		return nil, fmt.Errorf("unknown dispatch plan: %v", chosen)
	}

	// If a transition downgraded the effective contract, record it.
	if metrics.TransitionCount > 0 && metrics.EffectiveContract == optimizer.RecallExact {
		metrics.EffectiveContract = optimizer.RecallBounded
	}
	return results, err
}
