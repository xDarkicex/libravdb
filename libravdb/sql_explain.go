package libravdb

import (
	"context"
	"fmt"
	"time"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/optimizer"
)

const sqlExplainColumn = "libravdb_explain"

// SQLExplainColumn is the stable output column name for EXPLAIN ANALYZE.
const SQLExplainColumn = sqlExplainColumn

// SQLExplainPlan is the stable JSON value returned by EXPLAIN ANALYZE.
// Estimated rows are intentionally absent: the graph planner does not yet
// have a cardinality estimator, so reporting an estimate would be misleading.
type SQLExplainPlan struct {
	Strategy            string `json:"strategy"`
	Anchor              string `json:"anchor,omitempty"`
	ActualRows          uint64 `json:"actual_rows"`
	GraphExpansions     uint64 `json:"graph_expansions"`
	PredicateRejections uint64 `json:"predicate_rejections"`
	IndexHits           uint64 `json:"index_hits"`
	ExecutionTimeNanos  uint64 `json:"execution_time_ns"`
	PlanReused          bool   `json:"plan_reused"`
}

func (db *Database) executeSQLExplain(ctx context.Context, src []byte, doc *parser.QueryDoc, boundParams *optimizer.ParameterSet, legacyParams QueryParams, sessionConfig *SessionConfig, tracker *sqlQueryTracker) (*SearchResults, error) {
	if doc == nil || !doc.Explain {
		return nil, fmt.Errorf("invalid EXPLAIN query")
	}
	if !doc.ExplainAnalyze {
		return nil, fmt.Errorf("EXPLAIN without ANALYZE is not supported; use EXPLAIN ANALYZE")
	}
	if doc.ExplainQueryStart >= uint32(len(src)) || doc.ExplainQueryEnd <= doc.ExplainQueryStart || doc.ExplainQueryEnd > uint32(len(src)) {
		return nil, fmt.Errorf("invalid EXPLAIN query span")
	}
	strategy, anchor, graph := explainGraphShape(src, doc)
	if !graph {
		return nil, fmt.Errorf("EXPLAIN ANALYZE currently supports graph queries only")
	}

	before := sqlTrackerSnapshot(tracker)
	started := time.Now()
	innerSQL := string(src[doc.ExplainQueryStart:doc.ExplainQueryEnd])
	results, err := db.queryWithBoundParamsAndConfigInternal(ctx, innerSQL, boundParams, legacyParams, sessionConfig, tracker)
	elapsed := time.Since(started)
	if err != nil {
		return nil, err
	}
	after := sqlTrackerSnapshot(tracker)
	actualRows := uint64(0)
	if results != nil {
		actualRows = uint64(results.Total)
		if actualRows < uint64(len(results.Results)) {
			actualRows = uint64(len(results.Results))
		}
	}
	if tracker != nil {
		tracker.rowsReturned = actualRows
		tracker.rowsReturnedOverride = true
	}
	plan := SQLExplainPlan{
		Strategy:            strategy,
		Anchor:              anchor,
		ActualRows:          actualRows,
		GraphExpansions:     after.graphExpansions - before.graphExpansions,
		PredicateRejections: after.predicateRejections - before.predicateRejections,
		IndexHits:           after.indexHits - before.indexHits,
		ExecutionTimeNanos:  uint64(maxDurationNanos(elapsed)),
		PlanReused:          after.planCacheHits > before.planCacheHits,
	}
	return &SearchResults{
		Results:     []*SearchResult{{ID: "1", Score: 1, Metadata: map[string]interface{}{sqlExplainColumn: plan}}},
		Took:        elapsed,
		Total:       1,
		Columns:     []string{sqlExplainColumn},
		ColumnTypes: []uint16{catalog.TypeJSONB},
	}, nil
}

type sqlTrackerSnapshotValue struct {
	planCacheHits       uint64
	graphExpansions     uint64
	predicateRejections uint64
	indexHits           uint64
}

func sqlTrackerSnapshot(tracker *sqlQueryTracker) sqlTrackerSnapshotValue {
	if tracker == nil {
		return sqlTrackerSnapshotValue{}
	}
	return sqlTrackerSnapshotValue{
		planCacheHits:       tracker.planCacheHits,
		graphExpansions:     tracker.graphExpansions,
		predicateRejections: tracker.predicateRejections,
		indexHits:           tracker.indexHits,
	}
}

func maxDurationNanos(duration time.Duration) int64 {
	if duration < 0 {
		return 0
	}
	return duration.Nanoseconds()
}

func explainGraphShape(src []byte, doc *parser.QueryDoc) (strategy, anchor string, graph bool) {
	if doc == nil {
		return "", "", false
	}
	for i := range doc.SelectStmts {
		stmt := &doc.SelectStmts[i]
		for j := range stmt.Joins {
			join := &stmt.Joins[j]
			if join.MatchPath.Kind != parser.NodeKindMatchPath || join.MatchPath.ID < 0 || int(join.MatchPath.ID) >= len(doc.MatchPaths) {
				continue
			}
			path := &doc.MatchPaths[join.MatchPath.ID]
			return "graph_join_match", virtualMatchAnchor(src, doc, path), true
		}
		if stmt.FromTable.Kind == parser.NodeKindGraphTable {
			return "graph_table_match", "", true
		}
	}
	return "", "", false
}
