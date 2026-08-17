package libravdb

import (
	"context"
	"sync/atomic"
	"time"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// SQLQueryStats is a cumulative snapshot of SQL execution activity for one
// Database instance. Duration fields are nanoseconds so callers do not need
// to interpret Go's duration string format across SDK boundaries.
type SQLQueryStats struct {
	Queries             uint64 `json:"queries"`
	Errors              uint64 `json:"errors"`
	PlanCacheHits       uint64 `json:"plan_cache_hits"`
	PlanCacheMisses     uint64 `json:"plan_cache_misses"`
	TotalExecutionNanos uint64 `json:"total_execution_nanos"`
	LastExecutionNanos  uint64 `json:"last_execution_nanos"`
	RowsReturned        uint64 `json:"rows_returned"`
	RowsExamined        uint64 `json:"rows_examined"`
	GraphExpansions     uint64 `json:"graph_expansions"`
	IndexHits           uint64 `json:"index_hits"`
}

type sqlStatsCounters struct {
	queries             atomic.Uint64
	errors              atomic.Uint64
	planCacheHits       atomic.Uint64
	planCacheMisses     atomic.Uint64
	totalExecutionNanos atomic.Uint64
	lastExecutionNanos  atomic.Uint64
	rowsReturned        atomic.Uint64
	rowsExamined        atomic.Uint64
	graphExpansions     atomic.Uint64
	indexHits           atomic.Uint64
}

func newSQLStatsCounters() *sqlStatsCounters {
	return &sqlStatsCounters{}
}

func (s *sqlStatsCounters) snapshot() SQLQueryStats {
	if s == nil {
		return SQLQueryStats{}
	}
	return SQLQueryStats{
		Queries:             s.queries.Load(),
		Errors:              s.errors.Load(),
		PlanCacheHits:       s.planCacheHits.Load(),
		PlanCacheMisses:     s.planCacheMisses.Load(),
		TotalExecutionNanos: s.totalExecutionNanos.Load(),
		LastExecutionNanos:  s.lastExecutionNanos.Load(),
		RowsReturned:        s.rowsReturned.Load(),
		RowsExamined:        s.rowsExamined.Load(),
		GraphExpansions:     s.graphExpansions.Load(),
		IndexHits:           s.indexHits.Load(),
	}
}

func (s *sqlStatsCounters) reset() {
	if s == nil {
		return
	}
	s.queries.Store(0)
	s.errors.Store(0)
	s.planCacheHits.Store(0)
	s.planCacheMisses.Store(0)
	s.totalExecutionNanos.Store(0)
	s.lastExecutionNanos.Store(0)
	s.rowsReturned.Store(0)
	s.rowsExamined.Store(0)
	s.graphExpansions.Store(0)
	s.indexHits.Store(0)
}

// sqlQueryTracker is request-local. Nested query-local evaluators share the
// outer tracker so one public SQL request produces one stats sample.
type sqlQueryTracker struct {
	planCacheHits        uint64
	planCacheMisses      uint64
	rowsExamined         uint64
	graphExpansions      uint64
	indexHits            uint64
	predicateRejections  uint64
	rowsReturned         uint64
	rowsReturnedOverride bool
}

type sqlQueryTrackerContextKey struct{}

func sqlTrackerFromContext(ctx context.Context) *sqlQueryTracker {
	if ctx == nil {
		return nil
	}
	tracker, _ := ctx.Value(sqlQueryTrackerContextKey{}).(*sqlQueryTracker)
	return tracker
}

func trackSQLRowsExamined(ctx context.Context, rows int) {
	if tracker := sqlTrackerFromContext(ctx); tracker != nil && rows > 0 {
		tracker.rowsExamined += uint64(rows)
	}
}

func trackSQLGraphExpansion(ctx context.Context, count int) {
	if tracker := sqlTrackerFromContext(ctx); tracker != nil && count > 0 {
		tracker.graphExpansions += uint64(count)
	}
}

func trackSQLIndexHit(ctx context.Context, count int) {
	if tracker := sqlTrackerFromContext(ctx); tracker != nil && count > 0 {
		tracker.indexHits += uint64(count)
	}
}

func trackSQLPredicateRejection(ctx context.Context, count int) {
	if tracker := sqlTrackerFromContext(ctx); tracker != nil && count > 0 {
		tracker.predicateRejections += uint64(count)
	}
}

func recordMatchesPredicatesTracked(ctx context.Context, record Record, predicates []optimizer.RelationalPredicate) bool {
	if recordMatchesPredicates(record, predicates) {
		return true
	}
	trackSQLPredicateRejection(ctx, 1)
	return false
}

func recordMatchesPredicatesSnapshotTracked(ctx context.Context, record *Record, predicates []optimizer.RelationalPredicate) bool {
	if recordMatchesPredicatesSnapshot(record, predicates) {
		return true
	}
	trackSQLPredicateRejection(ctx, 1)
	return false
}

func graphJoinMatchesAlternativesTracked(ctx context.Context, plan *optimizer.PhysicalPlan, aliases map[string]Record, defaultAlias string) bool {
	if graphJoinMatchesAlternatives(plan, aliases, defaultAlias) {
		return true
	}
	trackSQLPredicateRejection(ctx, 1)
	return false
}

func (db *Database) recordSQLQuery(duration time.Duration, results *SearchResults, err error, tracker *sqlQueryTracker) {
	if db == nil || db.sqlStats == nil {
		return
	}
	durationNanos := duration.Nanoseconds()
	if durationNanos < 0 {
		durationNanos = 0
	}
	db.sqlStats.queries.Add(1)
	if err != nil {
		db.sqlStats.errors.Add(1)
	}
	db.sqlStats.totalExecutionNanos.Add(uint64(durationNanos))
	db.sqlStats.lastExecutionNanos.Store(uint64(durationNanos))
	if results != nil {
		rows := results.Total
		if rows < len(results.Results) {
			rows = len(results.Results)
		}
		if tracker != nil && tracker.rowsReturnedOverride {
			rows = int(tracker.rowsReturned)
		}
		if rows > 0 {
			db.sqlStats.rowsReturned.Add(uint64(rows))
		}
	}
	if tracker != nil {
		db.sqlStats.planCacheHits.Add(tracker.planCacheHits)
		db.sqlStats.planCacheMisses.Add(tracker.planCacheMisses)
		db.sqlStats.rowsExamined.Add(tracker.rowsExamined)
		db.sqlStats.graphExpansions.Add(tracker.graphExpansions)
		db.sqlStats.indexHits.Add(tracker.indexHits)
	}
}

// SQLStats returns a concurrency-safe cumulative SQL metrics snapshot.
func (db *Database) SQLStats() SQLQueryStats {
	if db == nil || db.sqlStats == nil {
		return SQLQueryStats{}
	}
	return db.sqlStats.snapshot()
}

// ResetSQLStats clears cumulative SQL metrics. It intentionally does not
// evict compiled plans; use this to measure reuse without changing behavior.
func (db *Database) ResetSQLStats() {
	if db != nil && db.sqlStats != nil {
		db.sqlStats.reset()
	}
}
