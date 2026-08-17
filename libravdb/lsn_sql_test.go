package libravdb

import (
	"context"
	"path/filepath"
	"strconv"
	"testing"
)

func TestLatestCommitLSNSQLMatchesNativeAndSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "lsn-sql.db")

	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	if _, err := db.Query(ctx, `CREATE TABLE docs (id TEXT PRIMARY KEY, value BIGINT)`); err != nil {
		t.Fatalf("CREATE TABLE: %v", err)
	}
	initial, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatalf("LatestCommitLSN after create: %v", err)
	}
	assertLatestCommitLSNSQL(t, db, initial)

	if _, err := db.Query(ctx, `INSERT INTO docs (id, value) VALUES ('one', 1)`); err != nil {
		t.Fatalf("INSERT: %v", err)
	}
	latest, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatalf("LatestCommitLSN after insert: %v", err)
	}
	if latest <= initial {
		t.Fatalf("latest LSN=%d, want > initial %d", latest, initial)
	}
	assertLatestCommitLSNSQL(t, db, latest)

	if _, err := db.Query(ctx, `SELECT LIBRAVDB_LATEST_COMMIT_LSN() AS snapshot_lsn`); err != nil {
		t.Fatalf("aliased LSN query: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	reopened, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer reopened.Close()
	reopenedLatest, err := reopened.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatalf("reopened LatestCommitLSN: %v", err)
	}
	if reopenedLatest != latest {
		t.Fatalf("reopened latest LSN=%d, want %d", reopenedLatest, latest)
	}
	assertLatestCommitLSNSQL(t, reopened, reopenedLatest)
}

func assertLatestCommitLSNSQL(t *testing.T, db *Database, want uint64) {
	t.Helper()
	results, err := db.Query(context.Background(), `SELECT LIBRAVDB_LATEST_COMMIT_LSN()`)
	if err != nil {
		t.Fatalf("SQL LSN query: %v", err)
	}
	if len(results.Results) != 1 || len(results.Columns) != 1 || results.Columns[0] != latestCommitLSNColumn {
		t.Fatalf("unexpected SQL LSN result shape: %#v", results)
	}
	value, ok := results.Results[0].Metadata[latestCommitLSNColumn]
	if !ok {
		t.Fatalf("SQL LSN result missing %q: %#v", latestCommitLSNColumn, results.Results[0].Metadata)
	}
	got, ok := value.(uint64)
	if !ok {
		t.Fatalf("SQL LSN value type=%T, want uint64", value)
	}
	if got != want {
		t.Fatalf("SQL LSN=%d, want %d (serialized %s)", got, want, strconv.FormatUint(want, 10))
	}
}
