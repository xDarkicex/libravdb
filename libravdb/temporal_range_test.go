package libravdb

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/xDarkicex/lexer/parser"
)

func TestTemporalRangeParserShape(t *testing.T) {
	src := []byte("SELECT id FROM VERSIONS OF documents BETWEEN TIMESTAMP '2026-01-01T00:00:00Z' AND TIMESTAMP '2026-02-01T00:00:00Z'")
	doc := &parser.QueryDoc{}
	if err := parser.Parse(src, doc); err != nil {
		t.Fatalf("parse temporal range: %v", err)
	}
	if len(doc.SelectStmts) != 1 || doc.SelectStmts[0].FromTable.Kind != parser.NodeKindTableExpr {
		t.Fatalf("unexpected select/table AST: %+v", doc.SelectStmts)
	}
	table := doc.TableExprs[doc.SelectStmts[0].FromTable.ID]
	if !table.TemporalRange || string(src[table.Start:table.End]) != "documents" {
		t.Fatalf("temporal range table=%+v", table)
	}
	if got := string(src[table.RangeStartStart:table.RangeStartEnd]); got != "'2026-01-01T00:00:00Z'" {
		t.Fatalf("start bound=%q", got)
	}
	if got := string(src[table.RangeEndStart:table.RangeEndEnd]); got != "'2026-02-01T00:00:00Z'" {
		t.Fatalf("end bound=%q", got)
	}
}

func TestTemporalRangeSQLMaterializesVersions(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/versions.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(ctx)
	col, err := db.CreateCollection(ctx, "documents", WithDimension(3), WithMetadataSchema(MetadataSchema{"title": StringField}))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := col.Insert(ctx, "d1", []float32{1, 0, 0}, map[string]interface{}{"title": "first"}); err != nil {
		t.Fatalf("insert: %v", err)
	}
	startSnap, err := db.SnapshotAt(ctx, time.Now().UTC().Add(time.Second))
	if err != nil {
		t.Fatalf("start snapshot: %v", err)
	}
	start := startSnap.Timestamp
	startSnap.Close()
	if err := col.Update(ctx, "d1", []float32{0, 1, 0}, map[string]interface{}{"title": "second"}); err != nil {
		t.Fatalf("update: %v", err)
	}
	endSnap, err := db.SnapshotAt(ctx, time.Now().UTC().Add(time.Second))
	if err != nil {
		t.Fatalf("end snapshot: %v", err)
	}
	end := endSnap.Timestamp
	endSnap.Close()

	query := fmt.Sprintf("SELECT id, version, title, version_start, version_end FROM VERSIONS OF documents BETWEEN TIMESTAMP '%s' AND TIMESTAMP '%s' ORDER BY version", start.Format(time.RFC3339Nano), end.Format(time.RFC3339Nano))
	results, err := db.Query(ctx, query)
	if err != nil {
		t.Fatalf("versions query: %v", err)
	}
	if results.Total != 2 {
		t.Fatalf("versions rows=%d, want 2: %+v", results.Total, results.Results)
	}
	if got := results.Results[0].Metadata["title"]; got != "first" {
		t.Fatalf("first version title=%v", got)
	}
	if got := results.Results[1].Metadata["title"]; got != "second" {
		t.Fatalf("current version title=%v", got)
	}
	if results.Results[0].Metadata["version_end"] == nil {
		t.Fatal("historical version should have an end timestamp")
	}
	if results.Results[1].Metadata["version_end"] != nil {
		t.Fatal("live version should have NULL version_end")
	}
	parameterized, err := db.QueryWithParams(ctx,
		"SELECT id, version FROM VERSIONS OF documents BETWEEN TIMESTAMP $start AND TIMESTAMP $end ORDER BY version",
		QueryParams{"start": start.Format(time.RFC3339Nano), "end": end.Format(time.RFC3339Nano)})
	if err != nil || parameterized.Total != 2 {
		t.Fatalf("parameterized temporal range rows=%v err=%v", parameterized, err)
	}
}

func TestEpochSQLStandardSavepointRoundTrip(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/savepoint.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(ctx)
	if _, err := db.CreateCollection(ctx, "docs", WithDimension(3)); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}
	if _, err := epoch.Query(ctx, "SAVEPOINT before_branch", nil); err != nil {
		t.Fatalf("SAVEPOINT: %v", err)
	}
	if err := epoch.Insert(ctx, "docs", "kept", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("insert kept: %v", err)
	}
	if _, err := epoch.Query(ctx, "SAVEPOINT after_kept", nil); err != nil {
		t.Fatalf("second SAVEPOINT: %v", err)
	}
	if err := epoch.Insert(ctx, "docs", "discarded", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("insert discarded: %v", err)
	}
	if _, err := epoch.Query(ctx, "ROLLBACK TO SAVEPOINT after_kept", nil); err != nil {
		t.Fatalf("ROLLBACK TO SAVEPOINT: %v", err)
	}
	rows, err := epoch.Query(ctx, "SELECT id FROM docs ORDER BY id", nil)
	if err != nil {
		t.Fatalf("query after rollback: %v", err)
	}
	if rows.Total != 1 || rows.Results[0].ID != "kept" {
		t.Fatalf("rows after rollback=%+v, want kept only", rows.Results)
	}
	if _, err := epoch.Query(ctx, "RELEASE SAVEPOINT after_kept", nil); err != nil {
		t.Fatalf("RELEASE SAVEPOINT: %v", err)
	}
	if err := epoch.Rollback(ctx); err != nil {
		t.Fatalf("epoch rollback: %v", err)
	}
	rows, err = db.Query(ctx, "SELECT id FROM docs")
	if err != nil {
		t.Fatalf("post-rollback query: %v", err)
	}
	if rows.Total != 0 {
		t.Fatalf("post-rollback rows=%d, want 0", rows.Total)
	}
}
