package libravdb

import (
	"context"
	"errors"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

func liveRelationalScan(ctx context.Context, plan *optimizer.PhysicalPlan) bool {
	return plan.SnapshotLSN == 0 && plan.SnapshotTimestamp.IsZero() && epochFromContext(ctx) == nil && transactionFromContext(ctx) == nil
}
func (e *Executor) executeLiveRelationalScan(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	if selector := e.newScanTopK(plan); selector != nil {
		result, err := e.executeOrderedLiveScan(ctx, col, plan, selector)
		if !errors.Is(err, errScanTopKFallback) {
			return result, err
		}
	}
	return e.executeUnorderedLiveScan(ctx, col, plan)
}
func (e *Executor) executeUnorderedLiveScan(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	predicates := prepareFTSPredicates(plan.FTSPredicates)
	results := []*SearchResult{}
	examined := 0
	err := col.scanReadOnly(ctx, func(rec Record) error {
		examined++
		if !planMatchesRecord(plan, rec) || !matchesPreparedFTS(rec, predicates) {
			return nil
		}
		// Only qualifying rows escape the borrowed callback. Their nested metadata
		// and vectors are owned, so projection and callers cannot mutate the store.
		results = append(results, &SearchResult{ID: rec.ID, Score: 1, Ordinal: rec.Ordinal, Version: rec.Version, Vector: cloneVector(rec.Vector), Metadata: cloneMetadata(rec.Metadata)})
		return nil
	})
	trackSQLRowsExamined(ctx, examined)
	if err != nil {
		return nil, err
	}
	return e.projectLiveScan(col, results, plan), nil
}
func (e *Executor) projectLiveScan(col *Collection, results []*SearchResult, plan *optimizer.PhysicalPlan) *SearchResults {
	for _, row := range results {
		rec := Record{ID: row.ID, Ordinal: row.Ordinal, Version: row.Version, Vector: row.Vector, Metadata: row.Metadata}
		e.attachRecordMetadata(col, row, plan, rec)
		row.Vector = nil
		row.Version = 0
	}
	return e.finishSelectResults(col, results, plan)
}
