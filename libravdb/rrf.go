package libravdb

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

const defaultRRFK = 60.0

type rrfCandidate struct {
	record   *Record
	metadata map[string]interface{}
	values   []float64
	valid    []bool
	score    float64
}

// executeRRF evaluates true reciprocal-rank fusion over the candidate set
// produced by relational and graph predicates. Each signal receives its own
// deterministic ranking (distance ascending; lexical/centrality descending),
// and the fused score is the sum of 1/(k+rank).
func (e *Executor) executeRRF(ctx context.Context, plan *optimizer.PhysicalPlan, snapshotLSN uint64) (*SearchResults, error) {
	if plan == nil || !plan.HasRRF || len(plan.RRFComponents) < 2 {
		return nil, fmt.Errorf("invalid RRF plan")
	}
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}

	candidateIDs, err := e.rrfCandidateIDs(ctx, col, plan, snapshotLSN)
	if err != nil {
		return nil, err
	}
	if len(candidateIDs) == 0 {
		return &SearchResults{Columns: plan.Projections}, nil
	}
	joinedMetadata, err := e.rrfJoinedMetadata(ctx, col, plan, candidateIDs, snapshotLSN)
	if err != nil {
		return nil, err
	}

	candidates := make([]rrfCandidate, 0, len(candidateIDs))
	for id := range candidateIDs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var rec *Record
		if snapshotLSN != 0 {
			rec, err = col.GetAtLSN(ctx, id, snapshotLSN)
		} else if visible, visibleErr := recordsVisibleInContext(ctx, col); visibleErr == nil {
			for i := range visible {
				if visible[i].ID == id {
					rec = &visible[i]
					break
				}
			}
			if rec == nil {
				err = fmt.Errorf("record %q is not visible", id)
			}
		} else {
			err = visibleErr
		}
		if err != nil || rec == nil {
			continue
		}
		values := make([]float64, len(plan.RRFComponents))
		valid := make([]bool, len(plan.RRFComponents))
		for i := range plan.RRFComponents {
			value, ok, valueErr := e.rrfComponentValue(ctx, col, rec, plan.RRFComponents[i], snapshotLSN)
			if valueErr != nil {
				return nil, valueErr
			}
			values[i], valid[i] = value, ok
		}
		metadata := rec.Metadata
		if joined, ok := joinedMetadata[id]; ok {
			metadata = joined
		}
		candidates = append(candidates, rrfCandidate{record: rec, metadata: metadata, values: values, valid: valid})
	}

	if len(candidates) == 0 {
		return &SearchResults{Columns: plan.Projections}, nil
	}
	k := plan.RRFK
	if k <= 0 || math.IsNaN(k) || math.IsInf(k, 0) {
		k = defaultRRFK
	}
	for componentIndex, component := range plan.RRFComponents {
		indices := make([]int, 0, len(candidates))
		for i := range candidates {
			if candidates[i].valid[componentIndex] {
				indices = append(indices, i)
			}
		}
		sort.SliceStable(indices, func(i, j int) bool {
			left, right := candidates[indices[i]], candidates[indices[j]]
			lv, rv := left.values[componentIndex], right.values[componentIndex]
			if lv != rv {
				if component.Ascending {
					return lv < rv
				}
				return lv > rv
			}
			return left.record.ID < right.record.ID
		})
		for rank, candidateIndex := range indices {
			candidates[candidateIndex].score += 1.0 / (k + float64(rank+1))
		}
	}

	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].score != candidates[j].score {
			return candidates[i].score > candidates[j].score
		}
		return candidates[i].record.ID < candidates[j].record.ID
	})
	if plan.Offset > 0 {
		if plan.Offset >= len(candidates) {
			candidates = nil
		} else {
			candidates = candidates[plan.Offset:]
		}
	}
	if plan.Limit > 0 && plan.Limit < len(candidates) {
		candidates = candidates[:plan.Limit]
	}

	out := &SearchResults{Columns: plan.Projections, Results: make([]*SearchResult, 0, len(candidates))}
	for _, candidate := range candidates {
		metadata := make(map[string]interface{}, len(plan.Projections)+1)
		if len(plan.Projections) == 0 {
			metadata = cloneMetadata(candidate.metadata)
			if metadata == nil {
				metadata = make(map[string]interface{})
			}
		}
		for _, ref := range plan.ProjectionRefs {
			if value, ok := recordMetadataValue(candidate.metadata, ref.SourceName); ok {
				metadata[ref.OutputName] = value
			}
		}
		for _, column := range plan.Projections {
			if strings.EqualFold(column, "id") {
				metadata[column] = candidate.record.ID
				continue
			}
			if strings.EqualFold(column, plan.RRFAlias) {
				metadata[column] = candidate.score
				continue
			}
			if value, ok := recordMetadataValue(candidate.metadata, column); ok {
				metadata[column] = value
			}
		}
		if plan.RRFAlias != "" {
			metadata[plan.RRFAlias] = candidate.score
		}
		out.Results = append(out.Results, &SearchResult{ID: candidate.record.ID, Score: float32(candidate.score), Metadata: metadata})
	}
	out.Total = len(out.Results)
	return out, nil
}

// rrfJoinedMetadata preserves the relational side of a multimodal query. RRF
// ranks the base/graph candidate once, but a SELECT may also project columns
// from ordinary JOIN relations (for example author.name). Reuse the existing
// join row semantics so ON predicates and SQL NULL padding stay consistent.
func (e *Executor) rrfJoinedMetadata(ctx context.Context, left *Collection, plan *optimizer.PhysicalPlan, candidateIDs map[string]struct{}, snapshotLSN uint64) (map[string]map[string]interface{}, error) {
	if len(plan.Joins) == 0 {
		return nil, nil
	}
	leftAlias := plan.CollectionName
	if plan.Joins[0].LeftAlias != "" {
		leftAlias = plan.Joins[0].LeftAlias
	}
	leftRecords := make([]Record, 0, len(candidateIDs))
	for id := range candidateIDs {
		var rec *Record
		var err error
		if snapshotLSN != 0 {
			rec, err = left.GetAtLSN(ctx, id, snapshotLSN)
		} else if visible, visibleErr := recordsVisibleInContext(ctx, left); visibleErr == nil {
			for i := range visible {
				if visible[i].ID == id {
					rec = &visible[i]
					break
				}
			}
			if rec == nil {
				err = fmt.Errorf("record %q is not visible", id)
			}
		} else {
			err = visibleErr
		}
		if err != nil || rec == nil {
			continue
		}
		leftRecords = append(leftRecords, *rec)
	}

	rows := make([]sqlJoinRow, 0, len(leftRecords))
	leftColumns := collectionColumns(left)
	for i := range leftRecords {
		record := &leftRecords[i]
		rows = append(rows, sqlJoinRow{Sources: map[string]*Record{leftAlias: record}, Schemas: map[string][]string{leftAlias: leftColumns}, BaseAlias: leftAlias})
	}
	for _, join := range plan.Joins {
		right, err := e.db.GetCollection(join.CollectionName)
		if err != nil {
			return nil, err
		}
		var rightRecords []Record
		if snapshotLSN != 0 {
			err = right.ListVisibleAtLSN(ctx, snapshotLSN, func(rec *Record) bool {
				rightRecords = append(rightRecords, *rec)
				return true
			})
		} else {
			rightRecords, err = recordsVisibleInContext(ctx, right)
		}
		if err != nil {
			return nil, err
		}
		rows = applyRelationalJoin(rows, rightRecords, collectionColumns(right), join)
	}

	out := make(map[string]map[string]interface{}, len(rows))
	for _, row := range rows {
		result := row.searchResult()
		if result.ID == "" {
			continue
		}
		metadata := cloneMetadata(result.Metadata)
		if metadata == nil {
			metadata = make(map[string]interface{})
		}
		// The base record identity remains the result identity even when the
		// joined relation also has an id column.
		metadata["id"] = result.ID
		out[result.ID] = metadata
	}
	return out, nil
}

func (e *Executor) rrfCandidateIDs(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan, snapshotLSN uint64) (map[string]struct{}, error) {
	var ids map[string]struct{}
	if len(plan.GraphJoins) > 0 {
		var anchors []string
		var err error
		if snapshotLSN != 0 {
			anchors, err = e.multiModalAnchorsAtLSN(ctx, col, plan.Joins, snapshotLSN)
		} else if len(plan.Joins) > 0 {
			anchors, err = e.multiModalAnchors(ctx, col, plan.Joins)
		} else {
			visibleIDs, visibleErr := e.rrfVisibleIDs(ctx, col, snapshotLSN, nil)
			err = visibleErr
			for id := range visibleIDs {
				anchors = append(anchors, id)
			}
		}
		if err != nil {
			return nil, err
		}
		if snapshotLSN != 0 {
			ids, err = e.multiModalGraphCandidatesAtLSN(ctx, col, plan, anchors, snapshotLSN)
		} else {
			ids, err = e.multiModalGraphCandidates(ctx, col, plan.GraphJoins, anchors)
		}
		if err != nil {
			return nil, err
		}
	} else if len(plan.Joins) > 0 {
		anchors, err := e.multiModalAnchors(ctx, col, plan.Joins)
		if err != nil {
			return nil, err
		}
		ids = make(map[string]struct{}, len(anchors))
		for _, id := range anchors {
			ids[id] = struct{}{}
		}
	} else {
		visibleIDs, err := e.rrfVisibleIDs(ctx, col, snapshotLSN, nil)
		if err != nil {
			return nil, err
		}
		ids = visibleIDs
	}

	// Apply source-side WHERE predicates after graph traversal. Terminal
	// predicates are already pushed into the graph traversal and are excluded
	// here by alias.
	filtered := make(map[string]struct{}, len(ids))
	for id := range ids {
		var rec *Record
		var err error
		if snapshotLSN != 0 {
			rec, err = col.GetAtLSN(ctx, id, snapshotLSN)
		} else {
			r, getErr := col.Get(ctx, id)
			if getErr == nil {
				rec = &r
			} else if visible, visibleErr := recordsVisibleInContext(ctx, col); visibleErr == nil {
				for i := range visible {
					if visible[i].ID == id {
						rec = &visible[i]
						break
					}
				}
			} else {
				err = visibleErr
			}
		}
		if err != nil || rec == nil {
			continue
		}
		predicates := plan.Predicates
		if len(plan.GraphJoins) > 0 {
			alias := plan.GraphJoins[0].LeftAlias
			predicates = nil
			for _, predicate := range plan.Predicates {
				if predicate.Alias == "" || strings.EqualFold(predicate.Alias, alias) {
					predicates = append(predicates, predicate)
				}
			}
		}
		if len(predicates) == 0 || recordMatchesPredicates(*rec, predicates) {
			filtered[id] = struct{}{}
		}
	}
	return filtered, nil
}

func (e *Executor) rrfVisibleIDs(ctx context.Context, col *Collection, snapshotLSN uint64, _ []optimizer.RelationalPredicate) (map[string]struct{}, error) {
	ids := make(map[string]struct{})
	if snapshotLSN != 0 {
		err := col.ListVisibleAtLSN(ctx, snapshotLSN, func(rec *Record) bool {
			ids[rec.ID] = struct{}{}
			return true
		})
		return ids, err
	}
	records, err := recordsVisibleInContext(ctx, col)
	if err != nil {
		return nil, err
	}
	for _, rec := range records {
		ids[rec.ID] = struct{}{}
	}
	return ids, nil
}

func (e *Executor) rrfComponentValue(ctx context.Context, col *Collection, rec *Record, component optimizer.RRFComponent, snapshotLSN uint64) (float64, bool, error) {
	switch component.Kind {
	case optimizer.RRFComponentVectorDistance:
		if len(rec.Vector) == 0 || len(rec.Vector) != len(component.Vector) {
			return 0, false, nil
		}
		return float64(computeVectorScore(col, optimizer.VectorFuncProjection{
			IsDistance:  component.Ascending,
			QueryVector: component.Vector,
		}, rec.Vector)), true, nil
	case optimizer.RRFComponentFTSRank:
		text, ok := recordMetadataValue(rec.Metadata, component.TextColumn)
		if !ok || text == nil {
			return 0, false, nil
		}
		rank := ftsRankText(recordMetaToString(text), component.TextQuery, "plain")
		// A zero lexical score is not a member of the lexical result list and
		// therefore contributes no reciprocal-rank term.
		return rank, rank > 0, nil
	case optimizer.RRFComponentGraphCentrality:
		if col.graph == nil {
			return 0, false, fmt.Errorf("GRAPH_CENTRALITY requires a graph-backed collection")
		}
		nodeID, err := e.lookupNodeIDInContext(ctx, col.name, rec.ID)
		if err != nil {
			return 0, false, nil
		}
		if snapshotLSN != 0 {
			return col.graph.CentralityAtLSN(nodeID, snapshotLSN), true, nil
		}
		return col.graph.GraphCentrality(nodeID), true, nil
	default:
		return 0, false, fmt.Errorf("unknown RRF component kind %d", component.Kind)
	}
}

func recordMetadataValue(metadata map[string]interface{}, name string) (interface{}, bool) {
	if value, ok := metadata[name]; ok {
		return value, true
	}
	for key, value := range metadata {
		if strings.EqualFold(key, name) {
			return value, true
		}
	}
	return nil, false
}
