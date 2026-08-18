package libravdb

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

func TestSQLPlanCacheIsCatalogGenerationAware(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql-plan-cache"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	if _, err := db.Query(ctx, `CREATE TABLE cache_rows (id TEXT PRIMARY KEY, name TEXT)`); err != nil {
		t.Fatalf("CREATE TABLE: %v", err)
	}
	if _, err := db.Query(ctx, `INSERT INTO cache_rows (id, name) VALUES ('a', 'Alice'), ('b', 'Bob')`); err != nil {
		t.Fatalf("INSERT: %v", err)
	}
	db.ResetSQLStats()

	query := `SELECT id FROM cache_rows WHERE id = 'a'`
	for i := 0; i < 2; i++ {
		results, err := db.Query(ctx, query)
		if err != nil {
			t.Fatalf("cached SELECT %d: %v", i, err)
		}
		if len(results.Results) != 1 || results.Results[0].ID != "a" {
			t.Fatalf("cached SELECT %d result=%#v", i, results)
		}
	}

	stats := db.SQLStats()
	if stats.Queries != 2 || stats.PlanCacheMisses != 1 || stats.PlanCacheHits != 1 {
		t.Fatalf("reuse stats=%+v, want queries=2 miss=1 hit=1", stats)
	}
	if stats.RowsReturned != 2 || stats.RowsExamined == 0 || stats.LastExecutionNanos == 0 {
		t.Fatalf("execution stats=%+v, want returned=2 examined>0 duration>0", stats)
	}

	// ALTER TABLE republishes the immutable catalog and increments its
	// generation. The next execution must not use a plan bound to the old
	// catalog, even though the SQL text is unchanged.
	if _, err := db.Query(ctx, `ALTER TABLE cache_rows ADD COLUMN status TEXT`); err != nil {
		t.Fatalf("ALTER TABLE: %v", err)
	}
	results, err := db.Query(ctx, query)
	if err != nil {
		t.Fatalf("SELECT after catalog change: %v", err)
	}
	if len(results.Results) != 1 || results.Results[0].ID != "a" {
		t.Fatalf("SELECT after catalog change result=%#v", results)
	}
	stats = db.SQLStats()
	if stats.PlanCacheMisses != 2 || stats.PlanCacheHits != 1 {
		t.Fatalf("post-invalidation stats=%+v, want miss=2 hit=1", stats)
	}
}

func TestSQLStatsScalarFunctionReturnsPublicSnapshot(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql-stats-function"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	if _, err := db.Query(ctx, `CREATE TABLE stats_rows (id TEXT PRIMARY KEY)`); err != nil {
		t.Fatalf("CREATE TABLE: %v", err)
	}
	db.ResetSQLStats()
	if _, err := db.Query(ctx, `SELECT id FROM stats_rows`); err != nil {
		t.Fatalf("ordinary SELECT: %v", err)
	}

	results, err := db.Query(ctx, `SELECT LIBRAVDB_SQL_STATS()`)
	if err != nil {
		t.Fatalf("SQL stats function: %v", err)
	}
	if len(results.Results) != 1 || len(results.Columns) != 1 || results.Columns[0] != sqlStatsColumn {
		t.Fatalf("SQL stats shape=%#v", results)
	}
	value, ok := results.Results[0].Metadata[sqlStatsColumn]
	if !ok {
		t.Fatalf("SQL stats metadata=%#v", results.Results[0].Metadata)
	}
	stats, ok := value.(SQLQueryStats)
	if !ok {
		t.Fatalf("SQL stats value type=%T, want SQLQueryStats", value)
	}
	if stats.Queries != 1 {
		t.Fatalf("SQL stats snapshot queries=%d, want 1 before function call is finalized", stats.Queries)
	}
}

func TestSQLParameterizedPlansReuseScalarSlotsWithoutStaleValues(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql-parameter-plan-safety"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, `CREATE TABLE parameter_rows (id TEXT PRIMARY KEY)`); err != nil {
		t.Fatalf("CREATE TABLE: %v", err)
	}
	if _, err := db.Query(ctx, `INSERT INTO parameter_rows (id) VALUES ('a'), ('b')`); err != nil {
		t.Fatalf("INSERT: %v", err)
	}
	db.ResetSQLStats()

	query := `SELECT id FROM parameter_rows WHERE id = $1`
	for _, want := range []string{"a", "b"} {
		results, err := db.QueryWithParams(ctx, query, QueryParams{"1": want})
		if err != nil {
			t.Fatalf("parameterized SELECT %q: %v", want, err)
		}
		if len(results.Results) != 1 || results.Results[0].ID != want {
			t.Fatalf("parameterized SELECT %q result=%#v", want, results)
		}
	}
	stats := db.SQLStats()
	if stats.PlanCacheHits != 1 || stats.PlanCacheMisses != 1 {
		t.Fatalf("parameterized scalar plan reuse stats=%+v, want hit=1 miss=1", stats)
	}
}

func TestSQLPlanCacheReusesTemporalAndAggregatePlans(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql-plan-cache-temporal"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, `CREATE TABLE plan_events (id TEXT PRIMARY KEY, category TEXT, amount FLOAT)`); err != nil {
		t.Fatalf("CREATE TABLE: %v", err)
	}
	if _, err := db.Query(ctx, `INSERT INTO plan_events (id, category, amount) VALUES ('a', 'alpha', 2), ('b', 'alpha', 3)`); err != nil {
		t.Fatalf("INSERT: %v", err)
	}
	snapshotLSN, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatalf("LatestCommitLSN: %v", err)
	}
	db.ResetSQLStats()

	temporalSQL := `SELECT id FROM plan_events AS OF LSN $snapshot_lsn WHERE id = $id`
	for _, id := range []string{"a", "b"} {
		rows, queryErr := db.QueryWithParams(ctx, temporalSQL, QueryParams{
			"snapshot_lsn": int64(snapshotLSN),
			"id":           id,
		})
		if queryErr != nil || rows.Total != 1 || rows.Results[0].ID != id {
			t.Fatalf("temporal plan id=%q rows=%+v err=%v", id, rows, queryErr)
		}
	}
	aggregateSQL := `SELECT category, SUM(amount) AS total FROM plan_events GROUP BY category`
	for i := 0; i < 2; i++ {
		rows, queryErr := db.Query(ctx, aggregateSQL)
		if queryErr != nil || rows.Total != 1 {
			t.Fatalf("aggregate plan rows=%+v err=%v", rows, queryErr)
		}
	}
	stats := db.SQLStats()
	if stats.PlanCacheMisses != 2 || stats.PlanCacheHits != 2 {
		t.Fatalf("temporal/aggregate plan reuse stats=%+v, want misses=2 hits=2", stats)
	}
}

func TestSQLPlanCacheRebindsTemporalSnapshot(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql-plan-cache-snapshot"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, `CREATE TABLE snapshot_rows (id TEXT PRIMARY KEY)`); err != nil {
		t.Fatalf("CREATE TABLE: %v", err)
	}
	if _, err := db.Query(ctx, `INSERT INTO snapshot_rows (id) VALUES ('old')`); err != nil {
		t.Fatalf("insert old: %v", err)
	}
	oldLSN, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatalf("old LSN: %v", err)
	}
	if _, err := db.Query(ctx, `INSERT INTO snapshot_rows (id) VALUES ('future')`); err != nil {
		t.Fatalf("insert future: %v", err)
	}
	liveLSN, err := db.LatestCommitLSN(ctx)
	if err != nil || liveLSN <= oldLSN {
		t.Fatalf("live LSN=%d old=%d err=%v", liveLSN, oldLSN, err)
	}
	query := `SELECT id FROM snapshot_rows AS OF LSN $snapshot_lsn ORDER BY id`
	oldRows, err := db.QueryWithParams(ctx, query, QueryParams{"snapshot_lsn": int64(oldLSN)})
	if err != nil || oldRows.Total != 1 || oldRows.Results[0].ID != "old" {
		t.Fatalf("old snapshot rows=%+v err=%v", oldRows, err)
	}
	liveRows, err := db.QueryWithParams(ctx, query, QueryParams{"snapshot_lsn": int64(liveLSN)})
	if err != nil || liveRows.Total != 2 {
		t.Fatalf("live snapshot rows=%+v err=%v", liveRows, err)
	}
}

func TestSQLPlanCacheConcurrentReuse(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql-plan-cache-concurrent"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, `CREATE TABLE concurrent_rows (id TEXT PRIMARY KEY)`); err != nil {
		t.Fatalf("CREATE TABLE: %v", err)
	}
	if _, err := db.Query(ctx, `INSERT INTO concurrent_rows (id) VALUES ('a')`); err != nil {
		t.Fatalf("INSERT: %v", err)
	}
	db.ResetSQLStats()

	query := `SELECT id FROM concurrent_rows WHERE id = $1`
	const workers = 8
	const iterations = 4
	errCh := make(chan error, workers)
	var wg sync.WaitGroup
	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				results, queryErr := db.QueryWithParams(ctx, query, QueryParams{"1": "a"})
				if queryErr != nil {
					errCh <- queryErr
					return
				}
				if len(results.Results) != 1 || results.Results[0].ID != "a" {
					errCh <- fmt.Errorf("unexpected concurrent result: %#v", results)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for queryErr := range errCh {
		t.Fatalf("concurrent query: %v", queryErr)
	}
	stats := db.SQLStats()
	if stats.Queries != workers*iterations || stats.Errors != 0 || stats.PlanCacheHits == 0 {
		t.Fatalf("concurrent stats=%+v, want queries=%d errors=0 hits>0", stats, workers*iterations)
	}
}
