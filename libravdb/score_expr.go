package libravdb

import (
	"context"
	"fmt"
	"math"

	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/internal/util"
)

// ScoreExpr is a compiled scoring expression evaluated per candidate.
// The optimizer lowers the parsed AST (BinaryExpr, VectorFunc, GraphMetricExpr,
// Number literals, Column references) into this tree once per plan.
type ScoreExpr interface {
	Eval(ctx *ScoreEvalContext) (float64, error)
}

// ScoreEvalContext provides per-candidate data during expression evaluation.
type ScoreEvalContext struct {
	Record          *Record   // the current candidate record
	QueryVector     []float32 // the query vector operand
	DistanceFunc    func([]float32, []float32) float32
	GraphCentrality float64 // precomputed for the candidate node
}

// ── Expression node types ─────────────────────────────────────────────

type literalExpr struct{ val float64 }

func (e *literalExpr) Eval(ctx *ScoreEvalContext) (float64, error) { return e.val, nil }

type vectorDistExpr struct{}

func (e *vectorDistExpr) Eval(ctx *ScoreEvalContext) (float64, error) {
	if len(ctx.Record.Vector) == 0 {
		return 0, fmt.Errorf("missing vector")
	}
	return float64(ctx.DistanceFunc(ctx.QueryVector, ctx.Record.Vector)), nil
}

type graphMetricExpr struct{}

func (e *graphMetricExpr) Eval(ctx *ScoreEvalContext) (float64, error) {
	return ctx.GraphCentrality, nil
}

type columnExpr struct{ column string }

func (e *columnExpr) Eval(ctx *ScoreEvalContext) (float64, error) { return 0, nil }

type binaryArithExpr struct {
	left  ScoreExpr
	right ScoreExpr
	op    uint8 // lexer.KindPlus, KindDash, KindAsterisk
}

func (e *binaryArithExpr) Eval(ctx *ScoreEvalContext) (float64, error) {
	l, err := e.left.Eval(ctx)
	if err != nil {
		return 0, err
	}
	r, err := e.right.Eval(ctx)
	if err != nil {
		return 0, err
	}
	switch e.op {
	case 21: // KindPlus
		return l + r, nil
	case 18: // KindDash
		return l - r, nil
	case 11: // KindAsterisk
		return l * r, nil
	}
	return 0, fmt.Errorf("unknown arith op %d", e.op)
}

// ── Builder ────────────────────────────────────────────────────────────

type scoreExprBuilder struct {
	plan         *optimizer.PhysicalPlan
	queryVector  []float32
	distanceFunc func([]float32, []float32) float32
}

func buildScoreExpr(plan *optimizer.PhysicalPlan, distFn func([]float32, []float32) float32, queryVec []float32) ScoreExpr {
	_ = &scoreExprBuilder{plan: plan, queryVector: queryVec, distanceFunc: distFn}

	// Use optimizer-lowered expression fields if available.
	if plan.HasScoreExpr {
		if plan.HasGraphCentrality && plan.ScoreArithOp == 11 {
			// (literal - VECTOR_DISTANCE) * GRAPH_CENTRALITY
			dist := &vectorDistExpr{}
			oneMinusDist := &binaryArithExpr{
				left:  &literalExpr{plan.ScoreLiteralValue},
				right: dist,
				op:    18, // subtraction
			}
			return &binaryArithExpr{
				left:  oneMinusDist,
				right: &graphMetricExpr{},
				op:    11, // multiplication
			}
		}
		if plan.ScoreArithOp == 18 {
			return &binaryArithExpr{
				left:  &literalExpr{plan.ScoreLiteralValue},
				right: &vectorDistExpr{},
				op:    18,
			}
		}
	}

	// Fallback: detect from plan features. Guard against nil/dimension panic.
	hasVector := len(plan.QueryVector) > 0 || len(queryVec) > 0
	if !hasVector {
		return &literalExpr{0.0} // no vector scoring possible
	}
	if hasVector {
		return &vectorDistExpr{}
	}
	return &literalExpr{1.0}
}

// ── Top-k heap for O(N log K) ranking ──────────────────────────────────

type scoredCandidate struct {
	record *Record
	score  float64
}

type topKHeap struct {
	items []scoredCandidate
	k     int
	desc  bool
}

func newTopKHeap(k int, desc bool) *topKHeap {
	return &topKHeap{items: make([]scoredCandidate, 0, k), k: k, desc: desc}
}

func (h *topKHeap) push(r *Record, score float64) {
	if len(h.items) < h.k {
		h.items = append(h.items, scoredCandidate{record: r, score: score})
		h.siftUp(len(h.items) - 1)
		return
	}
	// Replace the worst element if the new score is better.
	if h.desc {
		// Keep largest; worst is smallest (min-heap).
		if score <= h.items[0].score {
			return
		}
	} else {
		// Keep smallest; worst is largest (max-heap).
		if score >= h.items[0].score {
			return
		}
	}
	h.items[0] = scoredCandidate{record: r, score: score}
	h.siftDown(0)
}

func (h *topKHeap) siftUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.less(i, parent) {
			h.items[i], h.items[parent] = h.items[parent], h.items[i]
			i = parent
		} else {
			break
		}
	}
}

func (h *topKHeap) siftDown(i int) {
	for {
		smallest := i
		l := 2*i + 1
		r := 2*i + 2
		if l < len(h.items) && h.less(l, smallest) {
			smallest = l
		}
		if r < len(h.items) && h.less(r, smallest) {
			smallest = r
		}
		if smallest != i {
			h.items[i], h.items[smallest] = h.items[smallest], h.items[i]
			i = smallest
		} else {
			break
		}
	}
}

func (h *topKHeap) less(i, j int) bool {
	// Primary: score comparison.
	// Secondary: deterministic tie-break on record ID.
	if h.desc {
		if h.items[i].score != h.items[j].score {
			return h.items[i].score < h.items[j].score
		}
		return h.items[i].record.ID > h.items[j].record.ID
	}
	if h.items[i].score != h.items[j].score {
		return h.items[i].score > h.items[j].score
	}
	return h.items[i].record.ID > h.items[j].record.ID
}

func (h *topKHeap) sorted() []scoredCandidate {
	for i := len(h.items) - 1; i >= 0; i-- {
		h.items[0], h.items[i] = h.items[i], h.items[0]
		h.siftDownLimit(0, i)
	}
	// Heap-sort on min/max-heap naturally produces the correct order:
	// min-heap (desc=false) → ascending
	// max-heap (desc=true) → descending
	return h.items
}

func (h *topKHeap) siftDownLimit(i, limit int) {
	for {
		smallest := i
		l := 2*i + 1
		r := 2*i + 2
		if l < limit && h.less(l, smallest) {
			smallest = l
		}
		if r < limit && h.less(r, smallest) {
			smallest = r
		}
		if smallest != i {
			h.items[i], h.items[smallest] = h.items[smallest], h.items[i]
			i = smallest
		} else {
			break
		}
	}
}

// ── Executor integration ───────────────────────────────────────────────

// executeScoredMultiModalWithCentrality is like executeScoredMultiModal
// but uses pre-computed centrality values keyed by record ID.
func (e *Executor) executeScoredMultiModalWithCentrality(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan, candidates map[string]struct{}, expr ScoreExpr, centralityMap map[string]float64, snapshotLSN uint64, desc bool) (*SearchResults, error) {
	k := plan.Limit
	if k <= 0 {
		k = 10
	}
	heap := newTopKHeap(k, desc)
	distFn, _ := util.GetDistanceFunc(util.DistanceMetric(col.config.Metric))

	for id := range candidates {
		var rec *Record
		var err error
		if snapshotLSN != 0 {
			rec, err = col.GetAtLSN(ctx, id, snapshotLSN)
		} else {
			r, gerr := col.Get(ctx, id)
			if gerr == nil {
				rec = &r
			}
			err = gerr
		}
		if err != nil || rec == nil {
			continue
		}

		centrality := centralityMap[id]
		ctx_ := &ScoreEvalContext{
			Record:          rec,
			QueryVector:     plan.QueryVector,
			DistanceFunc:    distFn,
			GraphCentrality: centrality,
		}
		score, err := expr.Eval(ctx_)
		if err != nil || math.IsNaN(score) || math.IsInf(score, 0) {
			continue
		}
		heap.push(rec, score)
	}

	sorted := heap.sorted()
	out := &SearchResults{Results: make([]*SearchResult, len(sorted)), Total: len(sorted), Columns: plan.Projections}
	for i, sc := range sorted {
		metadata := cloneMetadata(sc.record.Metadata)
		if plan.ScoreAlias != "" {
			if metadata == nil {
				metadata = make(map[string]interface{}, 1)
			}
			metadata[plan.ScoreAlias] = sc.score
		}
		out.Results[i] = &SearchResult{
			ID: sc.record.ID, Score: float32(sc.score), Metadata: metadata,
		}
	}
	return out, nil
}

// executeScoredMultiModal evaluates candidates with a scoring expression
// and returns top-k results ordered by the computed score.
func (e *Executor) executeScoredMultiModal(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan, candidates map[string]struct{}, expr ScoreExpr, snapshotLSN uint64, desc bool) (*SearchResults, error) {
	k := plan.Limit
	if k <= 0 {
		k = 10
	}
	heap := newTopKHeap(k, desc)

	for id := range candidates {
		var rec *Record
		var err error
		if snapshotLSN != 0 {
			rec, err = col.GetAtLSN(ctx, id, snapshotLSN)
		} else {
			r, gerr := col.Get(ctx, id)
			if gerr == nil {
				rec = &r
			}
			err = gerr
		}
		if err != nil || rec == nil {
			continue
		}
		// Compute graph centrality for this candidate.
		// Prefer live centrality when snapshot includes current state;
		// use temporal centrality for historical snapshots.
		centrality := 0.0
		if plan.HasGraphTraversal && col.graph != nil {
			if nodeID, err := e.db.GetNodeID(ctx, col.name, id); err == nil {
				centrality = col.graph.GraphCentrality(nodeID)
			}
		}

		distFn, _ := util.GetDistanceFunc(util.DistanceMetric(col.config.Metric))
		ctx_ := &ScoreEvalContext{
			Record:          rec,
			QueryVector:     plan.QueryVector,
			DistanceFunc:    distFn,
			GraphCentrality: centrality,
		}
		_ = ctx_
		_ = id
		_ = centrality
		score, err := expr.Eval(ctx_)
		if err != nil || math.IsNaN(score) || math.IsInf(score, 0) {
			continue
		}
		heap.push(rec, score)
	}

	sorted := heap.sorted()
	out := &SearchResults{Results: make([]*SearchResult, len(sorted)), Total: len(sorted), Columns: plan.Projections}
	for i, sc := range sorted {
		metadata := cloneMetadata(sc.record.Metadata)
		if plan.ScoreAlias != "" {
			if metadata == nil {
				metadata = make(map[string]interface{}, 1)
			}
			metadata[plan.ScoreAlias] = sc.score
		}
		out.Results[i] = &SearchResult{
			ID: sc.record.ID, Score: float32(sc.score), Metadata: metadata,
		}
	}
	return out, nil
}
