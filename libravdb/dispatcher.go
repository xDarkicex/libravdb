package libravdb

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/graph"
	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/internal/util"
)

// DispatchReason records why the dispatcher chose a particular operator.
type DispatchReason string

const (
	ReasonExactSmallCandidates    DispatchReason = "exact_small_candidates"
	ReasonExactRecallContract     DispatchReason = "exact_recall_contract"
	ReasonExactFallback           DispatchReason = "exact_fallback"
	ReasonFilteredANNIndexed      DispatchReason = "filtered_ann_indexed"
	ReasonIterativeWeakFilter     DispatchReason = "iterative_weak_filter"
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
// calibration and offline threshold fitting.
type QueryMetrics struct {
	PlanChosen     DispatchPlan
	DispatchReason DispatchReason

	// Explainability fields identify which estimator produced the cardinality
	// inputs below.
	EstimateSource      string
	EstimateConfidence  float64
	EstimateAssumptions []string

	EstScalarCandidates      int
	ActScalarCandidates      int
	EstGraphCandidates       int
	ActGraphCandidates       int
	EstConjunctionCandidates int
	ActConjunctionCandidates int

	GraphSeeds       int
	GraphVertices    int
	GraphEdges       int
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
	ExecutionNanos  uint64

	// M3b.2: runtime reoptimization tracking.
	TransitionCount   int
	Transitions       []string // "iterative→exact(yield_abort)", etc.
	EffectiveContract uint8    // post-transition recall contract
}

// --- Provisional thresholds ---
// These are conservative defaults used until the persisted per-hardware,
// per-dimension calibration profile has enough samples to take precedence.
const exactCandidateFraction = 0.02  // 2% of N
const exactCandidateCap = 10000      // hard upper bound
const iterativeDefaultEpsilon = 0.05 // tail probability for binomial start
const iterativeMaxMultiplier = 8     // cap m* at (k/sigma) * maxMultiplier
const iterativeCapHard = 1000        // absolute first-batch cap

// hysteresisBand is the guard-band fraction around switching thresholds.
// A plan is only eligible for transition when the metric crosses
// threshold * (1 ± h), preventing flapping from small estimation errors.
// PROVISIONAL — offline calibration may refine these values.
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

	estimate := e.hybridCardinalityEstimate(plan, N, k)
	m.EstimateSource = estimate.source
	m.EstimateConfidence = estimate.confidence
	m.EstimateAssumptions = append([]string(nil), estimate.assumptions...)
	m.EstScalarCandidates = estimate.scalarCandidates
	m.EstGraphCandidates = estimate.graphCandidates
	m.EstConjunctionCandidates = estimate.conjunctionCandidates
	m.GraphSeeds = estimate.graphSeeds

	c := m.EstConjunctionCandidates

	// RECALL_EXACT always takes the exact path.
	if plan.RecallContract == optimizer.RecallExact {
		m.PlanChosen = DispatchExactCandidateScan
		m.DispatchReason = ReasonExactRecallContract
		return m.PlanChosen, m.DispatchReason, m
	}

	// Small candidate set → exact scan.
	exactThreshold := int(float64(N) * exactCandidateFraction)
	cap := exactCandidateCap
	if collection, err := e.db.GetCollection(plan.CollectionName); err == nil {
		if stats, ok := collection.costModel.snapshot(); ok {
			if calibratedCap := calibratedExactCandidateCap(stats); calibratedCap > 0 {
				cap = calibratedCap
				m.EstimateSource = "calibrated_collection_statistics"
				m.EstimateConfidence = max(m.EstimateConfidence, 0.80)
				m.EstimateAssumptions = append(m.EstimateAssumptions, "hardware_dimension_calibration")
			}
		}
	}
	if exactThreshold > cap {
		exactThreshold = cap
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

	// Filtered ANN is available when the vector query has either scalar
	// predicates or a materializable graph candidate set. Both become one
	// ordinal bitmap before the index search begins.
	hasVectorKNN := plan.HasVectorSearch || plan.Kind == optimizer.QueryKindKNN
	hasBitmapPredicate := (plan.HasRelationalQuery && planHasPredicates(plan)) || plan.HasGraphTraversal
	hasFilteredANN := hasVectorKNN && hasBitmapPredicate
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
	hasFilter := (plan.HasRelationalQuery && planHasPredicates(plan)) || plan.HasGraphTraversal
	return hasVector && hasFilter
}

// hybridConstraints contains the authoritative candidate sets materialized
// before a hybrid physical operator runs. A nil graphRecordIDs map means the
// query has no graph predicate; a non-nil empty map means MATCH produced no
// candidates and therefore no record is eligible.
type hybridConstraints struct {
	graphRecordIDs map[string]struct{}
}

func (c *hybridConstraints) graphAllows(recordID string) bool {
	if c == nil || c.graphRecordIDs == nil {
		return true
	}
	_, ok := c.graphRecordIDs[recordID]
	return ok
}

// hybridCandidateRecords chooses the narrowest authoritative source available
// before vector scoring. Equality predicates on WithIndexedFields use the
// collection posting index; graph-only constraints resolve MATCH record IDs
// directly. ListAll is the correctness fallback for unindexed predicates.
func (e *Executor) hybridCandidateRecords(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan, constraints *hybridConstraints) ([]Record, error) {
	if plan.HasRelationalQuery && len(plan.PredicateAlternatives) == 0 {
		for _, predicate := range plan.Predicates {
			if predicate.Operator != 12 || predicate.ValueIsNull || predicate.NullTest != optimizer.NullTestNone { // lexer.KindEquals
				continue
			}
			records, indexed, err := col.lookupIndexedMetadata(ctx, predicate.Column, predicate.PredicateValue().Bytes())
			if err != nil {
				return nil, err
			}
			if indexed {
				return records, nil
			}
		}
	}

	if constraints != nil && constraints.graphRecordIDs != nil {
		records := make([]Record, 0, len(constraints.graphRecordIDs))
		for id := range constraints.graphRecordIDs {
			record, err := col.Get(ctx, id)
			if err != nil {
				if errors.Is(err, ErrRecordNotFound) || isNotFoundError(err) {
					continue
				}
				return nil, err
			}
			records = append(records, record)
		}
		return records, nil
	}

	return col.ListAll(ctx)
}

// prepareHybridConstraints executes graph predicates once and shares the
// resulting record-ID set with every hybrid operator, including fallbacks.
func (e *Executor) prepareHybridConstraints(ctx context.Context, plan *optimizer.PhysicalPlan, m *QueryMetrics) (*hybridConstraints, error) {
	c := &hybridConstraints{}
	if !plan.HasGraphTraversal {
		return c, nil
	}

	graphIDs, err := e.materializeGraphCandidateIDs(ctx, plan, m)
	if err != nil {
		return nil, err
	}
	c.graphRecordIDs = graphIDs
	return c, nil
}

// materializeGraphCandidateIDs performs the complete MATCH traversal without
// applying the query's final LIMIT. Limiting before vector scoring would make
// RECALL_EXACT approximate by discarding graph-qualified candidates early.
func (e *Executor) materializeGraphCandidateIDs(ctx context.Context, plan *optimizer.PhysicalPlan, m *QueryMetrics) (map[string]struct{}, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("hybrid graph candidates: %w", err)
	}
	g := col.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("collection %q has no graph", plan.CollectionName)
	}

	seeds, err := e.hybridGraphSeeds(ctx, col, plan)
	if err != nil {
		return nil, err
	}
	if len(seeds) == 0 {
		return nil, fmt.Errorf(
			"hybrid graph query requires either an explicit seed, a vector anchor, or a labeled start vertex")
	}
	m.GraphSeeds = len(seeds)

	bitset, err := g.GetBitset()
	if err != nil {
		return nil, fmt.Errorf("hybrid graph bitset: %w", err)
	}
	defer g.PutBitset(bitset)

	frontier, err := g.GetFrontierBuf()
	if err != nil {
		return nil, fmt.Errorf("hybrid graph frontier: %w", err)
	}
	defer g.PutFrontierBuf(frontier)

	edges := make([]EdgePlan, len(plan.GraphEdges))
	for i, gep := range plan.GraphEdges {
		minHops := int(gep.QuantMin)
		maxHops := int(gep.QuantMax)
		if maxHops == 0 {
			if gep.QuantMin == 0 {
				// The parser encodes an unquantified edge as (0, 0), but
				// its SQL semantics are exactly one hop.
				minHops = 1
				maxHops = 1
			} else {
				maxHops = 1 << 20
			}
		}
		edges[i] = EdgePlan{Dir: gep.Direction, Min: minHops, Max: maxHops, Weight: gep.Weight, Predicate: gep.Predicate}
		if gep.EdgeKind != 0 {
			edges[i].KindSet.Set(gep.EdgeKind)
		}
	}

	matchedNodes := make(map[uint64]struct{})
	lastBand := len(edges) - 1
	for _, seed := range seeds {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		err := g.BFSPattern(seed, edges, plan.MaxHops, func(nodeID uint64, band int, step int) bool {
			m.GraphVertices++
			trackSQLGraphExpansion(ctx, 1)
			// Only endpoints satisfying the complete path are candidates.
			// Seeds, partial paths, and intermediate bands are traversal state,
			// not successful MATCH rows.
			if band == lastBand && step >= edges[band].Min {
				matchedNodes[nodeID] = struct{}{}
			}
			return true
		}, bitset, frontier)
		if err != nil {
			return nil, fmt.Errorf("hybrid graph traversal: %w", err)
		}
	}

	recordIDs := make(map[string]struct{}, len(matchedNodes))
	for nodeID := range matchedNodes {
		collectionName, recordID, err := e.db.ResolveNodeID(ctx, nodeID)
		if err != nil {
			continue
		}
		// A hybrid vector operator can only score records from its vector
		// collection. Keeping the collection identity here also prevents equal
		// record IDs in another collection from leaking through MATCH.
		if collectionName == plan.CollectionName {
			recordIDs[recordID] = struct{}{}
		}
	}
	m.ActGraphCandidates = len(recordIDs)
	return recordIDs, nil
}

func (e *Executor) hybridGraphSeeds(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan) ([]uint64, error) {
	if plan.HasExplicitSeed {
		if _, _, err := e.db.ResolveNodeID(ctx, plan.ExplicitSeedID); err != nil {
			return nil, fmt.Errorf("explicit graph seed %d: %w", plan.ExplicitSeedID, err)
		}
		return []uint64{plan.ExplicitSeedID}, nil
	}

	// A labeled first vertex is a declarative MATCH constraint, so it takes
	// precedence over the optional vector anchor used for unlabeled patterns.
	if plan.SeedLabel != "" {
		return col.GetGraph().GetLabelNodes(plan.SeedLabel), nil
	}

	if plan.HasVectorAnchor {
		anchor := plan.GraphAnchorVector
		if len(anchor) == 0 {
			anchor = plan.QueryVector
		}
		if len(anchor) == 0 {
			return nil, fmt.Errorf("hybrid graph vector anchor is empty")
		}

		if plan.RecallContract == optimizer.RecallExact {
			records, err := col.ListAll(ctx)
			if err != nil {
				return nil, fmt.Errorf("hybrid exact vector-anchor scan: %w", err)
			}
			results := scoreAndSelectTopK(col, records, anchor, len(records))
			seeds := make([]uint64, 0, len(results.Results))
			for _, result := range results.Results {
				if plan.Similarity > 0 && result.Score < plan.Similarity {
					continue
				}
				nodeID, err := e.db.GetNodeID(ctx, plan.CollectionName, result.ID)
				if err == nil {
					seeds = append(seeds, nodeID)
				}
			}
			return seeds, nil
		}

		seedLimit := plan.Limit
		if seedLimit <= 0 {
			seedLimit = 10
		}
		results, err := col.SearchWithGraphFilter(ctx, anchor, seedLimit, nil)
		if err != nil {
			return nil, fmt.Errorf("hybrid vector-anchored seed search: %w", err)
		}
		seeds := make([]uint64, 0, len(results.Results))
		for _, result := range results.Results {
			nodeID, err := e.db.GetNodeID(ctx, plan.CollectionName, result.ID)
			if err == nil {
				seeds = append(seeds, nodeID)
			}
		}
		return seeds, nil
	}

	return nil, nil
}

func filterByHybridConstraints(results *SearchResults, plan *optimizer.PhysicalPlan, constraints *hybridConstraints) *SearchResults {
	if results == nil {
		return nil
	}
	if plan.HasRelationalQuery && planHasPredicates(plan) && len(results.Results) > 0 {
		if len(plan.PredicateAlternatives) > 0 {
			filtered := results.Results[:0]
			for _, result := range results.Results {
				if searchResultMatchesPlan(plan, result) {
					filtered = append(filtered, result)
				}
			}
			results.Results = filtered
			results.Total = len(filtered)
		} else {
			results = filterByPredicates(results, plan.Predicates)
		}
	}
	if constraints == nil || constraints.graphRecordIDs == nil || len(results.Results) == 0 {
		return results
	}

	filtered := make([]*SearchResult, 0, len(results.Results))
	for _, result := range results.Results {
		if constraints.graphAllows(result.ID) {
			filtered = append(filtered, result)
		}
	}
	results.Results = filtered
	results.Total = len(filtered)
	return results
}

// --- Operator: ExactCandidateScan ---

// executeExactCandidateScan evaluates predicates over all records, scores
// survivors with SIMD distance, and returns the exact top-k.  Recall 1.0.
func (e *Executor) executeExactCandidateScan(ctx context.Context, plan *optimizer.PhysicalPlan, m *QueryMetrics, constraints *hybridConstraints) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("exact scan: %w", err)
	}

	records, err := e.hybridCandidateRecords(ctx, col, plan, constraints)
	if err != nil {
		return nil, fmt.Errorf("exact candidate enumeration: %w", err)
	}
	m.ActScalarCandidates = 0
	m.ActConjunctionCandidates = 0

	// Intersect relational and graph predicates before authoritative scoring.
	candidates := make([]Record, 0, len(records))
	for _, rec := range records {
		scalarMatch := !plan.HasRelationalQuery || !planHasPredicates(plan) || planMatchesRecord(plan, rec)
		if scalarMatch {
			m.ActScalarCandidates++
		}
		if scalarMatch && constraints.graphAllows(rec.ID) {
			candidates = append(candidates, rec)
		}
	}
	m.ActConjunctionCandidates = len(candidates)

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

func (e *Executor) buildHybridOrdinalBitmap(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan, constraints *hybridConstraints, m *QueryMetrics) (*ordinalBitmap, int, error) {
	records, err := e.hybridCandidateRecords(ctx, col, plan, constraints)
	if err != nil {
		return nil, 0, err
	}
	m.ActScalarCandidates = 0
	m.ActConjunctionCandidates = 0
	allowedMap := make(map[uint32]bool, len(records))
	membership := &mapMembership{m: allowedMap}
	var byMembership []ordinalMembership
	if col.shards != nil {
		byMembership = make([]ordinalMembership, len(col.shards))
		for i := range byMembership {
			byMembership[i] = &mapMembership{m: make(map[uint32]bool)}
		}
	}
	matchCount := 0
	for _, rec := range records {
		scalarMatch := !plan.HasRelationalQuery || !planHasPredicates(plan) || planMatchesRecord(plan, rec)
		if scalarMatch {
			m.ActScalarCandidates++
		}
		if scalarMatch && constraints.graphAllows(rec.ID) {
			membership.set(rec.Ordinal)
			matchCount++
			if byMembership != nil {
				byMembership[shardForID(rec.ID)].set(rec.Ordinal)
			}
		}
	}
	corpusSize := e.collectionSize(ctx, plan.CollectionName)
	selectivity := 1.0
	if corpusSize > 0 {
		selectivity = float64(matchCount) / float64(corpusSize)
	}
	m.ActConjunctionCandidates = matchCount
	return &ordinalBitmap{membership: membership, byMembership: byMembership, selectivity: selectivity}, matchCount, nil
}

// executeFilteredANN builds a bitmap from predicate-qualified ordinals and
// passes it into HNSW. HNSW keeps its ordinary unfiltered routing beam for
// connectivity and a separate valid-result heap fed by the same tuned
// assembly/SIMD distance batches.
func (e *Executor) executeFilteredANN(ctx context.Context, plan *optimizer.PhysicalPlan, m *QueryMetrics, constraints *hybridConstraints) (*SearchResults, error) {
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

	predicateBitmap, matchCount, err := e.buildHybridOrdinalBitmap(ctx, col, plan, constraints, m)
	if err != nil {
		return nil, fmt.Errorf("filtered ANN bitmap: %w", err)
	}
	m.ActConjunctionCandidates = matchCount

	// If the bitmap has already shrunk to an exact-scan-sized set, avoid an
	// ANN round trip entirely.
	if matchCount <= exactCandidateCap/10 && plan.RecallContract != optimizer.RecallExact {
		m.Transitions = append(m.Transitions, "filtered_ann->exact(bitmap_shrink)")
		m.TransitionCount++
		m.PlanChosen = DispatchExactCandidateScan
		return e.executeExactCandidateScan(ctx, plan, m, constraints)
	}

	sigma := predicateBitmap.selectivity
	searchEf := e.calibratedBinomialStart(plan.CollectionName, k, sigma)
	m.ANNEf = searchEf

	// Search the navigable graph normally, but only admit bitmap members to
	// the result set. The selectivity-aware ef supplies enough candidates for
	// sparse filters without disconnecting traversal.
	qb := col.Query(ctx)
	trackSQLIndexHit(ctx, 1)
	qb.WithVector(plan.QueryVector)
	qb.WithGraphFilter(predicateBitmap)
	qb.WithEfSearch(searchEf)
	qb.Limit(k)
	if plan.Similarity > 0 {
		qb.WithThreshold(plan.Similarity)
	}

	results, err := qb.Execute()
	if err != nil {
		return nil, err
	}
	m.ANNDistanceComputations = len(results.Results)
	m.FilterValidHits = len(results.Results)
	if len(results.Results) > 0 {
		m.FilterCandidatesPerHit = append(m.FilterCandidatesPerHit, max(1, searchEf/len(results.Results)))
	}

	return results, nil
}

// ordinalBitmap implements the index.GraphFilter interface for filtered ANN
// result admission. The membership is either map-backed (simple hybrid queries)
// or bitset-backed (zero-alloc multimodal path).
type ordinalBitmap struct {
	membership      ordinalMembership
	byMembership    []ordinalMembership
	selectivity     float64
	ownedMembership *bitsetMembership
	ownedShards     []*bitsetMembership
	pooled          bool
}

var ordinalBitmapPool = sync.Pool{New: func() any { return &ordinalBitmap{} }}

func (b *ordinalBitmap) release() {
	if b == nil || !b.pooled {
		return
	}
	releaseBitsetMembership(b.ownedMembership)
	for _, membership := range b.ownedShards {
		releaseBitsetMembership(membership)
	}
	*b = ordinalBitmap{}
	ordinalBitmapPool.Put(b)
}

func (b *ordinalBitmap) Test(idx uint64) bool {
	return b.membership.test(uint32(idx))
}

// Selectivity lets HNSW selectively enable bounded ACORN-lite rescue only
// when the ordinal filter is sparse enough to justify extra graph expansion.
func (b *ordinalBitmap) Selectivity() float64 { return b.selectivity }

// ForShard prevents local ordinals from one shard authorizing the same local
// ordinal in another shard.
func (b *ordinalBitmap) ForShard(shard int) GraphFilter {
	if shard < 0 || shard >= len(b.byMembership) {
		return &ordinalBitmap{membership: &mapMembership{m: map[uint32]bool{}}, selectivity: b.selectivity}
	}
	return &ordinalBitmap{membership: b.byMembership[shard], selectivity: b.selectivity}
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

// calibratedBinomialStart uses the lower of the current bitmap selectivity
// and observed ANN-rank yield. The lower value is conservative when predicate
// eligibility is negatively correlated with vector rank; absent enough saved
// evidence it exactly preserves the original binomial policy.
func (e *Executor) calibratedBinomialStart(collectionName string, k int, sigma float64) int {
	if e == nil || e.db == nil {
		return computeBinomialStart(k, sigma, iterativeDefaultEpsilon)
	}
	collection, err := e.db.GetCollection(collectionName)
	if err != nil {
		return computeBinomialStart(k, sigma, iterativeDefaultEpsilon)
	}
	statistics, ok := collection.costModel.snapshot()
	if !ok {
		return computeBinomialStart(k, sigma, iterativeDefaultEpsilon)
	}
	yield, samples := rankBucketYield(statistics.RankBucketYields)
	if samples < 64 || yield <= 0 {
		return computeBinomialStart(k, sigma, iterativeDefaultEpsilon)
	}
	return computeBinomialStart(k, min(sigma, yield), iterativeDefaultEpsilon)
}

func rankBucketYield(buckets []CostModelRankBucket) (float64, uint64) {
	var candidates, valid uint64
	for _, bucket := range buckets {
		candidates += bucket.Candidates
		valid += bucket.Valid
	}
	if candidates == 0 {
		return 0, 0
	}
	return float64(valid) / float64(candidates), candidates
}

// executeIterativeANNThenFilter evaluates geometric result prefixes
// (m, 2m, 4m...) from one maximum-budget filtered HNSW traversal. This keeps
// the operator's yield accounting while avoiding a fresh ANN traversal for
// every prefix.
func (e *Executor) executeIterativeANNThenFilter(ctx context.Context, plan *optimizer.PhysicalPlan, m *QueryMetrics, constraints *hybridConstraints) (*SearchResults, error) {
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

	predicateBitmap, matchCount, err := e.buildHybridOrdinalBitmap(ctx, col, plan, constraints, m)
	if err != nil {
		return nil, fmt.Errorf("iterative ANN bitmap: %w", err)
	}
	if matchCount == 0 {
		m.FilterValidHits = 0
		return &SearchResults{}, nil
	}

	N := matchCount
	if N <= 0 {
		N = m.EstConjunctionCandidates
	}
	if N <= 0 {
		N = 1
	}
	corpusSize := e.collectionSize(ctx, plan.CollectionName)
	sigma := float64(N) / float64(corpusSize)
	if sigma <= 0 {
		sigma = 0.01
	}

	startM := e.calibratedBinomialStart(plan.CollectionName, k, sigma)
	m.ANNEf = startM

	if startM >= iterativeCapHard {
		m.TailFallback = true
		m.PlanChosen = DispatchExactCandidateScan
		m.DispatchReason = ReasonExactFallback
		m.Transitions = append(m.Transitions, "iterative->exact(start_cap)")
		m.TransitionCount++
		return e.executeExactCandidateScan(ctx, plan, m, constraints)
	}

	// Run one HNSW traversal at the largest geometric budget, then consume
	// geometric prefixes from that result stream. Reissuing Query.Execute for
	// every prefix would restart HNSW and repeat distance work.
	maxBatch := startM
	const maxBatches = 10
	for batch := 1; batch < maxBatches && maxBatch < corpusSize; batch++ {
		next := maxBatch * 2
		if next <= maxBatch {
			break
		}
		if corpusSize > 0 && next > corpusSize {
			next = corpusSize
		}
		maxBatch = next
	}
	m.ANNEf = maxBatch

	qb := col.Query(ctx)
	qb.WithVector(plan.QueryVector)
	qb.WithGraphFilter(predicateBitmap)
	qb.WithEfSearch(maxBatch)
	qb.Limit(maxBatch)
	if plan.Similarity > 0 {
		qb.WithThreshold(plan.Similarity)
	}
	allResults, err := qb.Execute()
	if err != nil {
		return nil, err
	}
	if allResults == nil {
		allResults = &SearchResults{}
	}
	m.ANNDistanceComputations += len(allResults.Results)
	allResults = filterByHybridConstraints(allResults, plan, constraints)

	// Evaluate the geometric prefixes without rerunning the index. The result
	// stream is already predicate-valid because the bitmap was passed into
	// HNSW; this preserves the operator's batch semantics for metrics and
	// yield decisions while sharing one traversal state.
	valid := 0
	batches := 0
	for batchSize := startM; batches < maxBatches; batchSize *= 2 {
		if batchSize <= 0 {
			break
		}
		batches++
		valid = min(batchSize, len(allResults.Results))
		if valid >= k || batchSize >= maxBatch || valid == len(allResults.Results) {
			break
		}
	}

	// If the single traversal exhausts its budget with a poor yield, switch to
	// exact scan. The three-prefix guard preserves the previous reoptimization
	// behavior for genuinely weak filters.
	if valid < k/2 && (batches >= 3 || len(allResults.Results) < maxBatch) && plan.RecallContract != optimizer.RecallExact {
		m.Transitions = append(m.Transitions, "iterative->exact(yield_abort)")
		m.TransitionCount++
		m.TailFallback = true
		m.PlanChosen = DispatchExactCandidateScan
		return e.executeExactCandidateScan(ctx, plan, m, constraints)
	}

	m.ANNBatches = batches
	validHits := len(allResults.Results)
	m.FilterValidHits = validHits
	if validHits < k {
		m.ResultShortfall = k - validHits
	}

	allResults.Total = len(allResults.Results)
	if plan.Limit > 0 && len(allResults.Results) > plan.Limit {
		allResults.Results = allResults.Results[:plan.Limit]
		allResults.Total = plan.Limit
	}
	return allResults, nil
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
		if strings.EqualFold(pred.Column, "id") {
			// The physical record key is not stored in Metadata. Epoch
			// full-scans use this helper for SQL WHERE id predicates because
			// the live B-tree fast path is intentionally bypassed.
			if pred.NullTest == optimizer.NullTestIsNull {
				return false
			}
			if pred.NullTest == optimizer.NullTestNotNull {
				continue
			}
			if !scalarPredicateMatches(rec.ID, pred) {
				return false
			}
			continue
		}

		colVal, ok := rec.Metadata[pred.Column]
		isNull := !ok || colVal == nil
		if pred.NullTest == optimizer.NullTestIsNull {
			if !isNull {
				return false
			}
			continue
		}
		if pred.NullTest == optimizer.NullTestNotNull {
			if isNull {
				return false
			}
			continue
		}
		if isNull || !scalarPredicateMatches(colVal, pred) {
			return false
		}
	}
	return true
}

func recordMatchesFTSPredicates(rec Record, predicates []optimizer.FTSPredicate) bool {
	for _, predicate := range predicates {
		text := predicate.Text
		if predicate.Column != "" {
			value, ok := recordMetadataValue(rec.Metadata, predicate.Column)
			if !ok || value == nil {
				return false
			}
			text = recordMetaToString(value)
		}
		if ftsRankTextConfig(text, predicate.Query, predicate.QueryMode, predicate.Config) <= 0 {
			return false
		}
	}
	return true
}

// recordMetaToString renders a metadata value to a string for comparison.
func recordMetaToString(v interface{}) string {
	switch t := v.(type) {
	case util.JSONNull:
		return "null"
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
	indexQuery := vectorForIndex(col.config.Metric, queryVec)
	for _, rec := range candidates {
		if len(rec.Vector) == 0 || len(queryVec) == 0 || len(rec.Vector) != len(queryVec) {
			continue
		}
		indexVector := vectorForIndex(col.config.Metric, rec.Vector)
		var rawDistance float32
		switch col.config.Metric {
		case L2Distance:
			rawDistance = util.L2Distance_func(indexQuery, indexVector)
		case InnerProduct:
			rawDistance = util.InnerProduct_func(indexQuery, indexVector)
		case CosineDistance:
			rawDistance = util.CosineDistance_func(indexQuery, indexVector)
		default:
			rawDistance = util.CosineDistance_func(indexQuery, indexVector)
		}
		entries = append(entries, scored{id: rec.ID, score: publicScore(col.config.Metric, rawDistance)})
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
	startedAt := time.Now()

	// Epoch guard: force exact candidate scan when inside an epoch transaction.
	// Live HNSW contains post-S0 vectors, cannot contain staged vectors, and
	// cannot enforce snapshot graph topology. Future work may add an immutable
	// snapshot ANN cache to accelerate epoch hybrid queries.
	if epoch := epochFromContext(ctx); epoch != nil {
		return e.executeHybridEpoch(ctx, plan, epoch, startedAt)
	}

	chosen, reason, metrics := e.dispatchHybrid(ctx, plan)
	// Retain a bounded observation after execution so future calibration can
	// compare predicted and actual cardinalities without affecting results.
	defer func() {
		metrics.ExecutionNanos = uint64(time.Since(startedAt))
		e.db.costModelStats.record(plan.CollectionName, metrics)
		e.db.recordCostModelFeedback(plan, metrics)
	}()
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

	constraints, err := e.prepareHybridConstraints(ctx, plan, metrics)
	if err != nil {
		return nil, err
	}

	var results *SearchResults
	switch chosen {
	case DispatchExactCandidateScan:
		results, err = e.executeExactCandidateScan(ctx, plan, metrics, constraints)
	case DispatchFilteredANN:
		results, err = e.executeFilteredANN(ctx, plan, metrics, constraints)
	case DispatchIterativeANNThenFilter:
		results, err = e.executeIterativeANNThenFilter(ctx, plan, metrics, constraints)
	default:
		return nil, fmt.Errorf("unknown dispatch plan: %v", chosen)
	}

	// If a transition downgraded the effective contract, record it.
	if metrics.TransitionCount > 0 && metrics.EffectiveContract == optimizer.RecallExact {
		metrics.EffectiveContract = optimizer.RecallBounded
	}
	return results, err
}

// =============================================================================
// Epoch-aware hybrid query execution
// =============================================================================

// executeHybridEpoch runs a hybrid query entirely within the epoch's isolated
// view using exact SIMD scoring. HNSW is intentionally bypassed.
func (e *Executor) executeHybridEpoch(ctx context.Context, plan *optimizer.PhysicalPlan, epoch *EpochTx, startedAt time.Time) (*SearchResults, error) {
	metrics := &QueryMetrics{}
	defer func() {
		metrics.ExecutionNanos = uint64(time.Since(startedAt))
		metrics.PlanChosen = DispatchExactCandidateScan
		metrics.DispatchReason = "epoch_exact_candidate_scan"
		metrics.EffectiveContract = optimizer.RecallExact
		e.db.costModelStats.record(plan.CollectionName, metrics)
		e.db.recordCostModelFeedback(plan, metrics)
	}()

	constraints, err := e.prepareHybridConstraintsEpoch(ctx, plan, epoch)
	if err != nil {
		return nil, err
	}
	return e.executeExactCandidateScanEpoch(ctx, plan, epoch, constraints)
}

func (e *Executor) prepareHybridConstraintsEpoch(ctx context.Context, plan *optimizer.PhysicalPlan, epoch *EpochTx) (*hybridConstraints, error) {
	c := &hybridConstraints{}
	if !plan.HasGraphTraversal {
		return c, nil
	}
	graphIDs, err := e.materializeGraphCandidateIDsEpoch(ctx, plan, epoch)
	if err != nil {
		return nil, err
	}
	c.graphRecordIDs = graphIDs
	return c, nil
}

func (e *Executor) materializeGraphCandidateIDsEpoch(ctx context.Context, plan *optimizer.PhysicalPlan, epoch *EpochTx) (map[string]struct{}, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("hybrid graph candidates: %w", err)
	}
	seeds, err := e.hybridGraphSeedsEpoch(ctx, col, plan, epoch)
	if err != nil {
		return nil, err
	}
	if len(seeds) == 0 {
		return nil, fmt.Errorf("hybrid graph query requires a seed or anchor vertex")
	}
	gtx, err := epoch.GraphTxn(plan.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("epoch graph txn: %w", err)
	}
	edges := make([]graph.EdgePlan, len(plan.GraphEdges))
	for i, gep := range plan.GraphEdges {
		minHops := int(gep.QuantMin)
		maxHops := int(gep.QuantMax)
		if maxHops == 0 {
			if gep.QuantMin == 0 {
				minHops = 1
				maxHops = 1
			} else {
				maxHops = 1 << 20
			}
		}
		edges[i] = graph.EdgePlan{Dir: gep.Direction, Min: minHops, Max: maxHops, Weight: gep.Weight, Predicate: gep.Predicate}
		if gep.EdgeKind != 0 {
			edges[i].KindSet.Set(gep.EdgeKind)
		}
	}
	matchedNodes := make(map[uint64]struct{})
	lastBand := len(edges) - 1
	for _, seed := range seeds {
		type bfsState struct {
			nid        uint64
			band, step int
		}
		queue := []bfsState{{nid: seed, band: 0, step: 0}}
		visited := make(map[uint64]bool)
		visited[seed] = true
		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]
			if cur.step >= edges[cur.band].Min && cur.band == lastBand {
				matchedNodes[cur.nid] = struct{}{}
			}
			if cur.step >= edges[cur.band].Max || cur.band >= len(edges) {
				continue
			}
			advanceBand := cur.step >= edges[cur.band].Min && cur.band < lastBand
			useInbound := cur.band < len(edges) && edges[cur.band].Dir == -1
			var neighbors []graph.Edge
			if useInbound {
				neighbors, _ = gtx.InboundNeighborsOverlay(cur.nid)
			} else {
				neighbors, _ = gtx.NeighborsOverlay(cur.nid)
			}
			for _, nb := range neighbors {
				if cur.band >= len(edges) || !edges[cur.band].Matches(nb) {
					continue
				}
				if visited[nb.Target] {
					continue
				}
				visited[nb.Target] = true
				nextBand := cur.band
				nextStep := cur.step + 1
				if advanceBand && cur.step >= edges[cur.band].Max-1 {
					nextBand = cur.band + 1
					nextStep = 0
				}
				queue = append(queue, bfsState{nid: nb.Target, band: nextBand, step: nextStep})
			}
		}
	}
	recordIDs := make(map[string]struct{}, len(matchedNodes))
	for nodeID := range matchedNodes {
		_, recordID, err := e.resolveNodeIDInContext(ctx, nodeID)
		if err != nil {
			continue
		}
		recordIDs[recordID] = struct{}{}
	}
	return recordIDs, nil
}

func (e *Executor) hybridGraphSeedsEpoch(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan, epoch *EpochTx) ([]uint64, error) {
	var seeds []uint64
	if plan.HasExplicitSeed {
		_, _, err := e.resolveNodeIDInContext(ctx, plan.ExplicitSeedID)
		if err != nil {
			return nil, fmt.Errorf("explicit graph seed: %w", err)
		}
		seeds = append(seeds, plan.ExplicitSeedID)
		return seeds, nil
	}
	records, err := recordsVisibleInContext(ctx, col)
	if err != nil {
		return nil, err
	}
	for _, rec := range records {
		if plan.SeedLabel != "" {
			g := col.GetGraph()
			if g == nil {
				continue
			}
			labelNodes := g.GetLabelNodes(plan.SeedLabel)
			found := false
			for _, ln := range labelNodes {
				_, rid, rerr := e.resolveNodeIDInContext(ctx, ln)
				if rerr == nil && rid == rec.ID {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		nodeID, err := e.lookupNodeIDInContext(ctx, col.name, rec.ID)
		if err == nil {
			seeds = append(seeds, nodeID)
		}
	}
	return seeds, nil
}

func (e *Executor) executeExactCandidateScanEpoch(ctx context.Context, plan *optimizer.PhysicalPlan, epoch *EpochTx, constraints *hybridConstraints) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	records, err := recordsVisibleInContext(ctx, col)
	if err != nil {
		return nil, err
	}
	var candidates []Record
	for _, rec := range records {
		if constraints != nil && constraints.graphRecordIDs != nil {
			if _, ok := constraints.graphRecordIDs[rec.ID]; !ok {
				continue
			}
		}
		candidates = append(candidates, rec)
	}
	if plan.HasVectorSearch && len(plan.QueryVector) > 0 {
		type recScore struct {
			rec   Record
			score float32
		}
		var scoredList []recScore
		for _, rec := range candidates {
			if len(rec.Vector) == 0 {
				continue
			}
			var dist float32
			switch col.Config().Metric {
			case L2Distance:
				dist = util.L2Distance_func(plan.QueryVector, rec.Vector)
			case InnerProduct:
				dist = util.InnerProduct_func(plan.QueryVector, rec.Vector)
			case CosineDistance:
				dist = util.CosineDistance_func(plan.QueryVector, rec.Vector)
			default:
				dist = util.L2Distance_func(plan.QueryVector, rec.Vector)
			}
			scoredList = append(scoredList, recScore{rec: rec, score: dist})
		}
		sort.Slice(scoredList, func(i, j int) bool {
			if plan.IsDesc {
				return scoredList[i].score > scoredList[j].score
			}
			return scoredList[i].score < scoredList[j].score
		})
		if plan.Limit > 0 && len(scoredList) > plan.Limit {
			scoredList = scoredList[:plan.Limit]
		}
		results := &SearchResults{}
		for _, s := range scoredList {
			results.Results = append(results.Results, &SearchResult{
				ID: s.rec.ID, Score: s.score, Vector: s.rec.Vector, Metadata: s.rec.Metadata,
			})
		}
		results.Total = len(results.Results)
		return results, nil
	}
	results := &SearchResults{}
	for _, rec := range candidates {
		results.Results = append(results.Results, &SearchResult{ID: rec.ID, Score: 1.0})
		if plan.Limit > 0 && len(results.Results) >= plan.Limit {
			break
		}
	}
	results.Total = len(results.Results)
	return results, nil
}
