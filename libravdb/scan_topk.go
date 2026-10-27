package libravdb

import (
	"context"
	"errors"
	"strings"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

var errScanTopKFallback = errors.New("ordered scan needs the general comparison path")

// Retain only K owned snapshots. Metadata maps are reused when a better row
// replaces the heap root. Borrowed mutable maps/slices never escape the callback;
// immutable scalar interfaces can be retained without allocating another box.
type scanTopK struct {
	ids                                []string
	ordinals                           []uint32
	versions                           []uint64
	metadata                           []map[string]interface{}
	vectors                            [][]float32
	scores                             []float64
	heap                               []int
	columns                            []string
	rank                               *preparedFTSPredicate
	limit                              int
	descending, allColumns, needVector bool
}

func (e *Executor) newScanTopK(plan *optimizer.PhysicalPlan) *scanTopK {
	k := plan.Limit + plan.Offset
	if plan.Distinct || plan.Limit <= 0 || k <= 0 || k > 65536 {
		return nil
	}
	selector := &scanTopK{limit: k, descending: plan.IsDesc, allColumns: len(plan.Projections) == 0, needVector: e.scanNeedsVector(plan)}
	if strings.EqualFold(plan.OrderBy, "id") && hasExactProjection(plan, plan.OrderBy) {
		selector.columns = scanProjectionColumns(plan)
		return selector
	}
	rank, ok := scanOrderRank(plan)
	if !ok {
		return nil
	}
	selector.rank = rank
	selector.columns = scanProjectionColumns(plan)
	return selector
}
func hasExactProjection(plan *optimizer.PhysicalPlan, name string) bool {
	for _, projection := range plan.Projections {
		if projection == name {
			return true
		}
	}
	return len(plan.Projections) == 0
}
func (e *Executor) scanNeedsVector(plan *optimizer.PhysicalPlan) bool {
	if len(plan.Projections) == 0 || len(plan.VectorFuncProjections) > 0 || len(plan.VectorOperatorProjections) > 0 {
		return true
	}
	for _, name := range plan.Projections {
		if e.isVectorColumn(plan.CollectionName, projectionSourceName(plan, name)) {
			return true
		}
	}
	return false
}
func scanProjectionColumns(plan *optimizer.PhysicalPlan) []string {
	columns := make([]string, 0, len(plan.Projections)+len(plan.FTSProjections)+len(plan.FTSRankProjections))
	for _, name := range plan.Projections {
		source := projectionSourceName(plan, name)
		if !strings.EqualFold(source, "id") {
			columns = append(columns, source)
		}
	}
	for _, projection := range plan.FTSProjections {
		if projection.Column != "" {
			columns = append(columns, projection.Column)
		}
	}
	for _, projection := range plan.FTSRankProjections {
		columns = append(columns, projection.TextColumn)
	}
	return columns
}
func scanOrderRank(plan *optimizer.PhysicalPlan) (*preparedFTSPredicate, bool) {
	for _, projection := range plan.FTSProjections {
		if projection.Name != plan.OrderBy {
			continue
		}
		if projection.Kind != optimizer.FTSProjectionRank || projection.Normalization != 0 {
			return nil, false
		}
		return simpleScanRank(projection.Column, projection.Query, projection.QueryMode, projection.Config)
	}
	for _, projection := range plan.FTSRankProjections {
		if projection.Name == plan.OrderBy {
			return simpleScanRank(projection.TextColumn, projection.TextQuery, "plain", "simple")
		}
	}
	return nil, false
}
func simpleScanRank(column, query, mode, config string) (*preparedFTSPredicate, bool) {
	if column == "" {
		return nil, false
	}
	prepared := prepareFTSPredicates([]optimizer.FTSPredicate{{Column: column, Query: query, QueryMode: mode, Config: config}})
	if len(prepared[0].terms) == 0 {
		return nil, false
	}
	return &prepared[0], true
}
func (selector *scanTopK) score(rec Record) (float64, error) {
	if selector.rank == nil {
		// The general sorter treats numeric-looking ids as numbers. Mixed numeric
		// and text ids do not define a transitive order; preserve its existing path.
		if numericStringCandidate(strings.TrimSpace(rec.ID)) {
			return 0, errScanTopKFallback
		}
		return 0, nil
	}
	predicate := selector.rank
	text := predicate.text
	if predicate.column != "" {
		value, ok := recordMetadataValue(rec.Metadata, predicate.column)
		if !ok || value == nil {
			return 0, nil
		}
		text = recordMetaToString(value)
	}
	total := 0
	for _, term := range predicate.terms {
		count := countSimpleFTSTerm(text, term)
		if count == 0 {
			return 0, nil
		}
		total += count
	}
	return float64(total) / float64(len(predicate.terms)), nil
}
func (selector *scanTopK) add(rec Record) error {
	score, err := selector.score(rec)
	if err != nil {
		return err
	}
	if len(selector.heap) < selector.limit {
		slot := selector.newSlot()
		selector.replace(slot, rec, score)
		selector.heap = append(selector.heap, slot)
		selector.up(len(selector.heap) - 1)
		return nil
	}
	slot := selector.heap[0]
	if selector.better(rec.ID, score, selector.ids[slot], selector.scores[slot]) {
		selector.replace(slot, rec, score)
		selector.down(0)
	}
	return nil
}
func (selector *scanTopK) newSlot() int {
	slot := len(selector.ids)
	selector.ids = append(selector.ids, "")
	selector.ordinals = append(selector.ordinals, 0)
	selector.versions = append(selector.versions, 0)
	selector.metadata = append(selector.metadata, nil)
	selector.vectors = append(selector.vectors, nil)
	selector.scores = append(selector.scores, 0)
	return slot
}
func (selector *scanTopK) replace(slot int, rec Record, score float64) {
	selector.ids[slot] = rec.ID
	selector.ordinals[slot] = rec.Ordinal
	selector.versions[slot] = rec.Version
	selector.scores[slot] = score
	if selector.needVector {
		selector.vectors[slot] = append(selector.vectors[slot][:0], rec.Vector...)
	}
	selector.copyMetadata(slot, rec.Metadata)
}
func (selector *scanTopK) copyMetadata(slot int, source map[string]interface{}) {
	if selector.metadata[slot] == nil {
		selector.metadata[slot] = make(map[string]interface{}, len(selector.columns))
	}
	target := selector.metadata[slot]
	clear(target)
	if selector.allColumns {
		for name, value := range source {
			target[name] = cloneMetadataValue(value)
		}
		return
	}
	for _, name := range selector.columns {
		if value, ok := recordMetadataValue(source, name); ok {
			target[name] = cloneMetadataValue(value)
		}
	}
}

// better intentionally shares the established sort's descending id tie break.
func (selector *scanTopK) better(left string, leftScore float64, right string, rightScore float64) bool {
	if leftScore != rightScore {
		if selector.descending {
			return leftScore > rightScore
		}
		return leftScore < rightScore
	}
	if selector.descending {
		return left > right
	}
	return left < right
}
func (selector *scanTopK) worse(left, right int) bool {
	return selector.better(selector.ids[right], selector.scores[right], selector.ids[left], selector.scores[left])
}
func (selector *scanTopK) up(index int) {
	for index > 0 {
		parent := (index - 1) / 2
		if !selector.worse(selector.heap[index], selector.heap[parent]) {
			return
		}
		selector.heap[index], selector.heap[parent] = selector.heap[parent], selector.heap[index]
		index = parent
	}
}
func (selector *scanTopK) down(index int) {
	for {
		child := index*2 + 1
		if child >= len(selector.heap) {
			return
		}
		right := child + 1
		if right < len(selector.heap) && selector.worse(selector.heap[right], selector.heap[child]) {
			child = right
		}
		if !selector.worse(selector.heap[child], selector.heap[index]) {
			return
		}
		selector.heap[index], selector.heap[child] = selector.heap[child], selector.heap[index]
		index = child
	}
}
func (selector *scanTopK) records() []*SearchResult {
	results := make([]*SearchResult, len(selector.ids))
	for i, id := range selector.ids {
		results[i] = &SearchResult{ID: id, Ordinal: selector.ordinals[i], Version: selector.versions[i], Vector: selector.vectors[i], Metadata: selector.metadata[i], Score: 1}
	}
	return results
}
func (e *Executor) executeOrderedLiveScan(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan, selector *scanTopK) (*SearchResults, error) {
	predicates := prepareFTSPredicates(plan.FTSPredicates)
	examined := 0
	err := col.scanReadOnly(ctx, func(rec Record) error {
		examined++
		if !planMatchesRecord(plan, rec) || !matchesPreparedFTS(rec, predicates) {
			return nil
		}
		return selector.add(rec)
	})
	trackSQLRowsExamined(ctx, examined)
	if err != nil {
		return nil, err
	}
	return e.projectLiveScan(col, selector.records(), plan), nil
}
