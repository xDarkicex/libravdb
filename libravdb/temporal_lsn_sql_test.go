package libravdb

import (
	"context"
	"fmt"
	"math"
	"testing"

	internalgraph "github.com/xDarkicex/libravdb/internal/graph"
)

func temporalNumericValue(value interface{}) (float64, bool) {
	switch v := value.(type) {
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case uint64:
		return float64(v), true
	case float32:
		return float64(v), true
	case float64:
		return v, true
	default:
		return 0, false
	}
}

func TestTemporalSQLAsOfLSNRecordsCTEVectorAndReopen(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/as-of-lsn.libravdb"
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	col, err := db.CreateCollection(ctx, "docs", WithDimension(3), WithMetadataSchema(MetadataSchema{"title": StringField}))
	if err != nil {
		db.Close()
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := col.Insert(ctx, "old", []float32{1, 0, 0}, map[string]interface{}{"title": "historical"}); err != nil {
		db.Close()
		t.Fatalf("insert old: %v", err)
	}
	baseLSN, err := db.LatestCommitLSN(ctx)
	if err != nil {
		db.Close()
		t.Fatalf("base LSN: %v", err)
	}
	if err := col.Insert(ctx, "future", []float32{0, 1, 0}, map[string]interface{}{"title": "current"}); err != nil {
		db.Close()
		t.Fatalf("insert future: %v", err)
	}
	liveLSN, err := db.LatestCommitLSN(ctx)
	if err != nil || liveLSN <= baseLSN {
		db.Close()
		t.Fatalf("live LSN=%d base=%d err=%v", liveLSN, baseLSN, err)
	}

	params := QueryParams{"snapshot_lsn": int64(baseLSN)}
	historical, err := db.QueryWithParams(ctx,
		"SELECT id, title FROM docs AS OF LSN $snapshot_lsn ORDER BY id", params)
	if err != nil {
		db.Close()
		t.Fatalf("AS OF LSN query: %v", err)
	}
	if got := resultIDs(historical); len(got) != 1 || !got["old"] {
		db.Close()
		t.Fatalf("historical rows=%v, want old only", got)
	}

	// The same SQL text must resolve each parameter to its own snapshot; a
	// cached current plan must never retain a prior snapshot LSN.
	live, err := db.QueryWithParams(ctx,
		"SELECT id FROM docs AS OF LSN $snapshot_lsn ORDER BY id",
		QueryParams{"snapshot_lsn": int64(liveLSN)})
	if err != nil {
		db.Close()
		t.Fatalf("live AS OF LSN query: %v", err)
	}
	if got := resultIDs(live); len(got) != 2 || !got["old"] || !got["future"] {
		db.Close()
		t.Fatalf("live rows=%v, want old and future", got)
	}

	cte, err := db.QueryWithParams(ctx, `
		WITH bounded AS (
			SELECT id, title
			FROM docs AS OF LSN $snapshot_lsn
			ORDER BY id
			LIMIT $input_limit
		)
		SELECT id, title FROM bounded ORDER BY id`,
		QueryParams{"snapshot_lsn": int64(liveLSN), "input_limit": int64(1)})
	if err != nil {
		db.Close()
		t.Fatalf("AS OF LSN CTE query: %v", err)
	}
	if cte.Total != 1 || len(cte.Results) != 1 || cte.Results[0].ID != "future" && cte.Results[0].ID != "old" {
		db.Close()
		t.Fatalf("bounded CTE rows=%+v, want one row", cte.Results)
	}

	vector, err := db.QueryWithParams(ctx,
		"SELECT id, embedding <-> $query_vec AS distance "+
			"FROM docs AS OF LSN $snapshot_lsn d ORDER BY distance LIMIT 1",
		QueryParams{"snapshot_lsn": int64(baseLSN), "query_vec": []float32{1, 0, 0}})
	if err != nil {
		db.Close()
		t.Fatalf("AS OF LSN vector query: %v", err)
	}
	if vector.Total != 1 || len(vector.Results) != 1 || vector.Results[0].ID != "old" {
		db.Close()
		t.Fatalf("historical vector rows=%+v, want old", vector.Results)
	}

	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer db.Close()
	reopened, err := db.QueryWithParams(ctx,
		"SELECT id FROM docs AS OF LSN $snapshot_lsn ORDER BY id",
		QueryParams{"snapshot_lsn": int64(baseLSN)})
	if err != nil {
		t.Fatalf("reopened AS OF LSN query: %v", err)
	}
	if got := resultIDs(reopened); len(got) != 1 || !got["old"] {
		t.Fatalf("reopened historical rows=%v, want old only", got)
	}

	if _, err := db.Query(ctx, fmt.Sprintf("SELECT id FROM docs AS OF LSN %d", baseLSN)); err != nil {
		t.Fatalf("literal AS OF LSN: %v", err)
	}
	if _, err := db.Query(ctx, "SELECT id FROM docs AS OF LSN 0"); err == nil {
		t.Fatal("AS OF LSN 0 should fail")
	}
}

func TestTemporalSQLAsOfLSNGraphVisibility(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir()+"/as-of-lsn-graph.libravdb"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	graph, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer graph.Close()
	col, err := db.CreateCollection(ctx, "nodes", WithMetadataOnly(), WithGraph(graph),
		WithMetadataSchema(MetadataSchema{"category": StringField}))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	const edgeKind uint8 = 241
	const edgeName = "AS_OF_LSN_LINK"
	if !internalgraph.RegisterEdgeKind(edgeName, edgeKind) && internalgraph.ResolveEdgeKind(edgeName) != edgeKind {
		t.Fatalf("edge kind registration conflict")
	}
	if err := col.Insert(ctx, "source", nil, map[string]interface{}{"category": "source"}); err != nil {
		t.Fatalf("insert source: %v", err)
	}
	if err := col.Insert(ctx, "target", nil, map[string]interface{}{"category": "target"}); err != nil {
		t.Fatalf("insert target: %v", err)
	}
	baseLSN, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatalf("base LSN: %v", err)
	}
	source, err := db.GetNodeID(ctx, "nodes", "source")
	if err != nil {
		t.Fatalf("source node: %v", err)
	}
	target, err := db.GetNodeID(ctx, "nodes", "target")
	if err != nil {
		t.Fatalf("target node: %v", err)
	}
	txn := graph.BeginTxn()
	if err := txn.AddEdge(source, target, 1, edgeKind); err != nil {
		t.Fatalf("add edge: %v", err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatalf("commit edge: %v", err)
	}
	edgeSnapshot := snapshotAfterGraphCommit(t, db, graph, source, target)
	edgeLSN := edgeSnapshot.LSN
	edgeSnapshot.Close()

	query := "SELECT s.category FROM nodes s AS OF LSN $snapshot_lsn " +
		"WHERE MATCH (s)-[r:AS_OF_LSN_LINK]->(p)"
	before, err := db.QueryWithParams(ctx, query, QueryParams{"snapshot_lsn": int64(baseLSN)})
	if err != nil {
		t.Fatalf("historical graph query: %v", err)
	}
	if before.Total != 0 {
		t.Fatalf("graph edge leaked before edge LSN: %+v", before.Results)
	}
	after, err := db.QueryWithParams(ctx, query, QueryParams{"snapshot_lsn": int64(edgeLSN)})
	if err != nil {
		t.Fatalf("current graph query: %v", err)
	}
	if after.Total != 1 || len(after.Results) != 1 || after.Results[0].ID != "source" {
		t.Fatalf("graph at edge LSN=%+v, want source", after.Results)
	}
}

func TestTemporalSQLAsOfLSNRelationalOrderOffsetLimit(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/as-of-lsn-order.libravdb"
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	col, err := db.CreateCollection(ctx, "docs", WithDimension(1), WithMetadataSchema(MetadataSchema{"title": StringField}))
	if err != nil {
		db.Close()
		t.Fatalf("CreateCollection: %v", err)
	}
	for _, row := range []struct {
		id    string
		title string
	}{
		{id: "z", title: "zulu"},
		{id: "a", title: "alpha"},
		{id: "m", title: "middle"},
	} {
		if err := col.Insert(ctx, row.id, []float32{0}, map[string]interface{}{"title": row.title}); err != nil {
			db.Close()
			t.Fatalf("insert %s: %v", row.id, err)
		}
	}
	snapshotLSN, err := db.LatestCommitLSN(ctx)
	if err != nil {
		db.Close()
		t.Fatalf("LatestCommitLSN: %v", err)
	}

	query := "SELECT id, title FROM docs AS OF LSN $snapshot_lsn ORDER BY title LIMIT $limit OFFSET $offset"
	params := QueryParams{
		"snapshot_lsn": int64(snapshotLSN),
		"limit":        int64(1),
		"offset":       int64(1),
	}
	assertOrdered := func(label string, current *Database) {
		t.Helper()
		result, queryErr := current.QueryWithParams(ctx, query, params)
		if queryErr != nil {
			t.Fatalf("%s query: %v", label, queryErr)
		}
		if result.Total != 1 || len(result.Results) != 1 || result.Results[0].ID != "m" {
			t.Fatalf("%s rows=%+v, want the second title-ordered row m", label, result.Results)
		}
	}
	assertOrdered("before reopen", db)
	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer db.Close()
	assertOrdered("after reopen", db)
}

func TestTemporalSQLAsOfLSNAggregatesRemainHistoricalAcrossReopen(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/as-of-lsn-aggregate.libravdb"
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	col, err := db.CreateCollection(ctx, "events", WithDimension(1), WithMetadataSchema(MetadataSchema{
		"category": StringField,
		"amount":   IntField,
	}))
	if err != nil {
		db.Close()
		t.Fatalf("CreateCollection: %v", err)
	}
	for _, row := range []struct {
		id       string
		category string
		amount   int
	}{
		{id: "a1", category: "alpha", amount: 2},
		{id: "a2", category: "alpha", amount: 3},
	} {
		if err := col.Insert(ctx, row.id, []float32{0}, map[string]interface{}{
			"category": row.category,
			"amount":   row.amount,
		}); err != nil {
			db.Close()
			t.Fatalf("insert %s: %v", row.id, err)
		}
	}
	baseLSN, err := db.LatestCommitLSN(ctx)
	if err != nil {
		db.Close()
		t.Fatalf("base LSN: %v", err)
	}
	if err := col.Insert(ctx, "b1", []float32{0}, map[string]interface{}{
		"category": "beta",
		"amount":   100,
	}); err != nil {
		db.Close()
		t.Fatalf("future insert: %v", err)
	}

	queryParams := QueryParams{"snapshot_lsn": int64(baseLSN)}
	count, err := db.QueryWithParams(ctx,
		"SELECT COUNT(*) AS total FROM events AS OF LSN $snapshot_lsn", queryParams)
	if err != nil {
		db.Close()
		t.Fatalf("historical count: %v", err)
	}
	countValue, countOK := temporalNumericValue(count.Results[0].Metadata["total"])
	if count.Total != 1 || len(count.Results) != 1 || !countOK || countValue != 2 {
		db.Close()
		t.Fatalf("historical count=%+v, want 2", count.Results)
	}

	grouped, err := db.QueryWithParams(ctx, `
		SELECT category, SUM(amount) AS amount_total
		FROM events AS OF LSN $snapshot_lsn
		GROUP BY category
		ORDER BY category`, queryParams)
	if err != nil {
		db.Close()
		t.Fatalf("historical grouped aggregate: %v", err)
	}
	if grouped.Total != 1 || len(grouped.Results) != 1 {
		db.Close()
		t.Fatalf("historical groups=%+v, want one alpha group", grouped.Results)
	}
	amountValue, amountOK := temporalNumericValue(grouped.Results[0].Metadata["amount_total"])
	if grouped.Results[0].Metadata["category"] != "alpha" || !amountOK || math.Abs(amountValue-5) > 1e-9 {
		db.Close()
		t.Fatalf("historical aggregate row=%+v, want alpha/5", grouped.Results[0].Metadata)
	}

	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer db.Close()
	reopened, err := db.QueryWithParams(ctx, `
		SELECT category, SUM(amount) AS amount_total
		FROM events AS OF LSN $snapshot_lsn
		GROUP BY category`, queryParams)
	if err != nil {
		t.Fatalf("reopened grouped aggregate: %v", err)
	}
	reopenedAmount, reopenedAmountOK := temporalNumericValue(reopened.Results[0].Metadata["amount_total"])
	if reopened.Total != 1 || len(reopened.Results) != 1 || !reopenedAmountOK || math.Abs(reopenedAmount-5) > 1e-9 {
		t.Fatalf("reopened aggregate=%+v, want alpha/5", reopened.Results)
	}
}
