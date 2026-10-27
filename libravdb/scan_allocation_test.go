package libravdb

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

func TestPreparedFTSPreservesDictionaryAndQuerySemantics(t *testing.T) {
	documents := []string{"Lease lease RENEWAL", "other text", "ſ S Σ σ ς İ i K k", "lease_renewal lease-renewal", "running runs runner", "", "123 १२३", "lease renewal boundary"}
	queries := []string{"lease", "LEASE renewal", "s", "σ", "i", "k", "lease_renewal", "lease | renewal", "lease & !boundary", "lease:*", "lease <-> renewal", "!lease", "running", "123", ""}
	for _, config := range []string{"simple", "english"} {
		for _, mode := range []string{"plain", "raw", "websearch", "phrase"} {
			for _, query := range queries {
				prepared := prepareFTSPredicates([]optimizer.FTSPredicate{{Text: "", Query: query, QueryMode: mode, Config: config}})[0]
				for _, text := range documents {
					want := ftsRankTextConfig(text, query, mode, config) > 0
					if got := prepared.matches(text); got != want {
						t.Fatalf("%s/%s %q text %q got %v want %v", config, mode, query, text, got, want)
					}
				}
			}
		}
	}
}
func TestPreparedSimpleFTSDoesNotAllocatePerDocument(t *testing.T) {
	predicate := prepareFTSPredicates([]optimizer.FTSPredicate{{Query: "lease renewal", QueryMode: "plain", Config: "simple"}})[0]
	allocs := testing.AllocsPerRun(1000, func() {
		if !predicate.matches("LEASE and renewal preserve fencing") {
			t.Fatal("lost match")
		}
	})
	if allocs != 0 {
		t.Fatalf("FTS matching allocates %.0f objects per document", allocs)
	}
}
func newScanTestCollection(t *testing.T) (*Database, *Collection) {
	t.Helper()
	db, err := Open(WithStoragePath(t.TempDir()+"/scan.libravdb"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := db.Close(); err != nil {
			t.Error(err)
		}
	})
	col, err := db.CreateCollection(context.Background(), "documents", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{"content": StringField, "kind": StringField, "data": JSONField}))
	if err != nil {
		t.Fatal(err)
	}
	return db, col
}
func fillScanTestCollection(t *testing.T, col *Collection) {
	t.Helper()
	for i, text := range []string{"lease", "lease lease renewal", "LEASE renewal", "unrelated", "lease renewal renewal", "lease"} {
		if err := col.Insert(context.Background(), fmt.Sprintf("d-%02d", i), nil, map[string]interface{}{"content": text, "kind": "memory", "data": map[string]interface{}{"nested": []interface{}{"original"}}}); err != nil {
			t.Fatal(err)
		}
	}
}
func TestOrderedScanMatchesUnboundedSelection(t *testing.T) {
	db, col := newScanTestCollection(t)
	fillScanTestCollection(t, col)
	statements := []string{
		"SELECT id, content FROM documents ORDER BY id ASC",
		"SELECT id, content FROM documents ORDER BY id DESC",
		"SELECT id, content, ts_rank(to_tsvector(content), plainto_tsquery('lease')) AS rank FROM documents WHERE to_tsvector(content) @@ plainto_tsquery('lease') ORDER BY rank DESC",
		"SELECT id, content, ts_rank(to_tsvector(content), plainto_tsquery('lease renewal')) AS rank FROM documents WHERE to_tsvector(content) @@ plainto_tsquery('lease renewal') ORDER BY rank ASC",
		"SELECT id, data FROM documents ORDER BY id",
		"SELECT * FROM documents ORDER BY id DESC",
		"SELECT DISTINCT kind FROM documents ORDER BY kind",
	}
	for _, sql := range statements {
		assertBoundedScanMatches(t, db, sql)
	}
}
func assertBoundedScanMatches(t *testing.T, db *Database, sql string) {
	t.Helper()
	all, err := db.Query(context.Background(), sql)
	if err != nil {
		t.Fatal(err)
	}
	limited, err := db.Query(context.Background(), sql+" LIMIT 2 OFFSET 1")
	if err != nil {
		t.Fatal(err)
	}
	want := all.Results[min(1, len(all.Results)):min(3, len(all.Results))]
	if len(want) == 0 {
		want = nil
	}
	if !reflect.DeepEqual(limited.Results, want) {
		t.Fatalf("%s bounded differs\n got %#v\nwant %#v", sql, limited.Results, want)
	}
}
func TestScanResultsNeverAliasStoredNestedMetadata(t *testing.T) {
	db, col := newScanTestCollection(t)
	fillScanTestCollection(t, col)
	result, err := db.Query(context.Background(), "SELECT id, data FROM documents ORDER BY id LIMIT 2")
	if err != nil {
		t.Fatal(err)
	}
	nested := result.Results[0].Metadata["data"].(map[string]interface{})["nested"].([]interface{})
	nested[0] = "corrupt"
	record, err := col.Get(context.Background(), result.Results[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	if got := record.Metadata["data"].(map[string]interface{})["nested"].([]interface{})[0]; got != "original" {
		t.Fatalf("result mutated storage: %v", got)
	}
}
func TestOrderedScanNumericIDFallbackMatchesGeneralSort(t *testing.T) {
	db, col := newScanTestCollection(t)
	for _, id := range []string{"10", "2", "1a", " 3", "nan"} {
		if err := col.Insert(context.Background(), id, nil, map[string]interface{}{"content": "lease"}); err != nil {
			t.Fatal(err)
		}
	}
	assertBoundedScanMatches(t, db, "SELECT id FROM documents ORDER BY id")
}
func TestNumericSortFastRejectionPreservesSpecialValues(t *testing.T) {
	for _, text := range []string{"NaN", "nan", "NAN", "Inf", "inf", "Infinity", "infinity", "-Inf", " +Inf ", " .5 ", "1e2", "d-001", "", "not numeric"} {
		got, ok := toFloat(text)
		want, err := strconv.ParseFloat(strings.TrimSpace(text), 64)
		if ok != (err == nil) {
			t.Fatalf("%q acceptance changed", text)
		}
		if err == nil && got != want && !math.IsNaN(want) {
			t.Fatalf("%q value changed", text)
		}
	}
}
func TestReadOnlyScanCancellationAndErrors(t *testing.T) {
	_, col := newScanTestCollection(t)
	fillScanTestCollection(t, col)
	expected := errors.New("callback failed")
	if err := col.scanReadOnly(context.Background(), func(Record) error { return expected }); !errors.Is(err, expected) {
		t.Fatalf("callback error=%v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := col.scanReadOnly(ctx, func(Record) error { return nil }); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation=%v", err)
	}
}

func TestUnmatchedFTSAllocationsDoNotScaleWithCollection(t *testing.T) {
	db, col := newScanTestCollection(t)
	addUnmatchedRows(t, col, 0, 32)
	before := unmatchedQueryAllocations(t, db)
	addUnmatchedRows(t, col, 32, 512)
	after := unmatchedQueryAllocations(t, db)
	if after > before+10 {
		t.Fatalf("rejected rows allocate: 32 rows %.0f allocs, 512 rows %.0f allocs", before, after)
	}
}
func addUnmatchedRows(t *testing.T, col *Collection, start, end int) {
	t.Helper()
	ctx := context.Background()
	tx, err := col.db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	for i := start; i < end; i++ {
		if err := tx.Insert(ctx, col.name, fmt.Sprintf("d-%04d", i), nil, map[string]interface{}{"content": "gardening notes", "data": map[string]interface{}{"source": "fixture"}}); err != nil {
			t.Fatal(err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
}
func unmatchedQueryAllocations(t *testing.T, db *Database) float64 {
	t.Helper()
	return testing.AllocsPerRun(20, func() {
		result, err := db.QueryWithParams(context.Background(), "SELECT id FROM documents WHERE to_tsvector(content) @@ plainto_tsquery($term) ORDER BY id LIMIT 1", QueryParams{"term": "lease"})
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Results) != 0 {
			t.Fatal("unmatched row returned")
		}
	})
}
