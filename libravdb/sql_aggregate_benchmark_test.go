package libravdb

import (
	"context"
	"fmt"
	"testing"
)

func newSQLAggregateBenchmarkDB(b *testing.B) *Database {
	b.Helper()
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_aggregate_benchmark"), WithMetrics(false))
	if err != nil {
		b.Fatal(err)
	}
	if _, err := db.Query(ctx, `CREATE TABLE aggregate_bench (
		id TEXT PRIMARY KEY,
		category TEXT,
		amount BIGINT,
		payload JSONB
	)`); err != nil {
		db.Close()
		b.Fatal(err)
	}
	col, err := db.GetCollection("aggregate_bench")
	if err != nil {
		db.Close()
		b.Fatal(err)
	}
	for i := 0; i < 512; i++ {
		if err := col.Insert(ctx, fmt.Sprintf("row-%04d", i), nil, map[string]interface{}{
			"category": fmt.Sprintf("category-%02d", i%16),
			"amount":   int64(i + 1),
			"payload":  []interface{}{"go", "sql", i % 4},
		}); err != nil {
			db.Close()
			b.Fatalf("insert benchmark row %d: %v", i, err)
		}
	}
	return db
}

func BenchmarkSQLAggregateCount(b *testing.B) {
	db := newSQLAggregateBenchmarkDB(b)
	defer db.Close()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := db.Query(ctx, "SELECT COUNT(*) FROM aggregate_bench")
		if err != nil || result.Total != 1 {
			b.Fatalf("COUNT result=%#v err=%v", result, err)
		}
	}
}

func BenchmarkSQLAggregateGroupedSum(b *testing.B) {
	db := newSQLAggregateBenchmarkDB(b)
	defer db.Close()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := db.Query(ctx, "SELECT category, SUM(amount) AS total FROM aggregate_bench GROUP BY category ORDER BY category")
		if err != nil || result.Total != 16 {
			b.Fatalf("grouped SUM result=%#v err=%v", result, err)
		}
	}
}

func BenchmarkSQLAggregateGroupedMultiKey(b *testing.B) {
	db := newSQLAggregateBenchmarkDB(b)
	defer db.Close()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := db.Query(ctx, "SELECT category, amount, COUNT(*) FROM aggregate_bench GROUP BY category, amount")
		if err != nil || result.Total != 512 {
			b.Fatalf("multi-key grouped result=%#v err=%v", result, err)
		}
	}
}

func BenchmarkSQLJSONExpansion(b *testing.B) {
	db := newSQLAggregateBenchmarkDB(b)
	defer db.Close()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := db.Query(ctx, "SELECT elem FROM aggregate_bench b CROSS JOIN jsonb_array_elements(b.payload) AS elem")
		if err != nil || result.Total != 1536 {
			b.Fatalf("JSON expansion result rows=%d err=%v", result.Total, err)
		}
	}
}

func BenchmarkSQLJSONMembership(b *testing.B) {
	db := newSQLAggregateBenchmarkDB(b)
	defer db.Close()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := db.Query(ctx, `SELECT COUNT(*) FROM aggregate_bench WHERE payload @> '{"tags":["go"]}'::jsonb`)
		if err != nil || result.Total != 1 {
			b.Fatalf("JSON membership result=%#v err=%v", result, err)
		}
	}
}
