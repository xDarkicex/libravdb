package libravdb

import (
	"context"
	"fmt"
	"math"
	"testing"
)

func TestCosinePublicScoreContract_SearchAndQuerySurfaces(t *testing.T) {
	ctx := context.Background()
	collection := newCosineContractCollection(t, "flat_contract", WithFlat())

	searchResults, err := collection.Search(ctx, []float32{1, 0, 0}, 4)
	if err != nil {
		t.Fatalf("collection search failed: %v", err)
	}

	if got := len(searchResults.Results); got != 4 {
		t.Fatalf("expected 4 search results, got %d", got)
	}

	assertResultOrder(t, searchResults.Results, "exact", "close", "orthogonal", "opposite")
	assertApproxScore(t, searchResults.Results[0].Score, 1.0, 1e-5)
	assertApproxScore(t, searchResults.Results[1].Score, 0.8, 1e-4)
	assertApproxScore(t, searchResults.Results[2].Score, 0.0, 1e-5)
	assertApproxScore(t, searchResults.Results[3].Score, -1.0, 1e-5)

	for i := 1; i < len(searchResults.Results); i++ {
		if searchResults.Results[i-1].Score < searchResults.Results[i].Score {
			t.Fatalf("results not sorted by descending public score: %+v", searchResults.Results)
		}
	}

	queryResults, err := collection.Query(ctx).
		WithVector([]float32{1, 0, 0}).
		Limit(4).
		Execute()
	if err != nil {
		t.Fatalf("query execute failed: %v", err)
	}

	if len(queryResults.Results) != len(searchResults.Results) {
		t.Fatalf("search/execute mismatch: search=%d execute=%d", len(searchResults.Results), len(queryResults.Results))
	}

	for i := range searchResults.Results {
		if searchResults.Results[i].ID != queryResults.Results[i].ID {
			t.Fatalf("search/execute result %d ID mismatch: search=%q execute=%q", i, searchResults.Results[i].ID, queryResults.Results[i].ID)
		}
		assertApproxScore(t, queryResults.Results[i].Score, searchResults.Results[i].Score, 1e-5)
	}
}

func TestCosinePublicScoreContract_ThresholdSemantics(t *testing.T) {
	ctx := context.Background()
	collection := newCosineContractCollection(t, "flat_threshold", WithFlat())

	strong, err := collection.Query(ctx).
		WithVector([]float32{1, 0, 0}).
		WithThreshold(0.8).
		Limit(10).
		Execute()
	if err != nil {
		t.Fatalf("strong threshold execute failed: %v", err)
	}

	assertResultOrder(t, strong.Results, "exact", "close")
	for _, result := range strong.Results {
		if result.Score < 0.8 {
			t.Fatalf("result %q below threshold: %f", result.ID, result.Score)
		}
	}

	nonNegative, err := collection.Query(ctx).
		WithVector([]float32{1, 0, 0}).
		WithThreshold(0.0).
		Limit(10).
		Execute()
	if err != nil {
		t.Fatalf("zero threshold execute failed: %v", err)
	}

	assertResultOrder(t, nonNegative.Results, "exact", "close", "orthogonal")
	for _, result := range nonNegative.Results {
		if result.Score < 0 {
			t.Fatalf("expected threshold 0.0 to exclude negative scores, got %q=%f", result.ID, result.Score)
		}
	}

	listResults, err := collection.Query(ctx).
		WithVector([]float32{1, 0, 0}).
		WithThreshold(0.0).
		Limit(10).
		List()
	if err != nil {
		t.Fatalf("zero threshold list failed: %v", err)
	}

	if len(listResults) != len(nonNegative.Results) {
		t.Fatalf("list/execute length mismatch: list=%d execute=%d", len(listResults), len(nonNegative.Results))
	}

	for i := range listResults {
		if listResults[i].ID != nonNegative.Results[i].ID {
			t.Fatalf("list/execute order mismatch at %d: list=%q execute=%q", i, listResults[i].ID, nonNegative.Results[i].ID)
		}
	}
}

func TestCosinePublicScoreContract_BackendConsistency(t *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		name string
		opts []CollectionOption
	}{
		{
			name: "flat",
			opts: []CollectionOption{WithFlat()},
		},
		{
			name: "hnsw",
			opts: []CollectionOption{WithHNSW(16, 100, 50)},
		},
		{
			name: "ivfpq",
			opts: []CollectionOption{WithIVFPQ(4, 4)},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			collection := newCosineContractCollection(t, tc.name, tc.opts...)

			results, err := collection.Search(ctx, []float32{1, 0, 0}, 4)
			if err != nil {
				t.Fatalf("search failed: %v", err)
			}

			if len(results.Results) == 0 {
				t.Fatal("expected at least one result")
			}

			if results.Results[0].ID != "exact" {
				t.Fatalf("expected exact match to rank first, got %+v", results.Results)
			}
			assertApproxScore(t, results.Results[0].Score, 1.0, 1e-4)

			for i := 1; i < len(results.Results); i++ {
				if results.Results[i-1].Score < results.Results[i].Score {
					t.Fatalf("results not sorted by descending public score: %+v", results.Results)
				}
			}

			for _, result := range results.Results {
				if result.Score < -1.0001 || result.Score > 1.0001 {
					t.Fatalf("cosine public score out of range for %q: %f", result.ID, result.Score)
				}
			}

			strong, err := collection.Query(ctx).
				WithVector([]float32{1, 0, 0}).
				WithThreshold(0.8).
				Limit(10).
				Execute()
			if err != nil {
				t.Fatalf("threshold execute failed: %v", err)
			}

			for _, result := range strong.Results {
				if result.Score < 0.8 {
					t.Fatalf("result %q below threshold: %f", result.ID, result.Score)
				}
			}

			if len(strong.Results) == 0 || strong.Results[0].ID != "exact" {
				t.Fatalf("expected thresholded results to retain the exact match, got %+v", strong.Results)
			}
		})
	}
}

func TestCosinePublicScoreContract_IVFPQPersistence(t *testing.T) {
	ctx := context.Background()
	dbPath := testDBPath(t)

	db, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("create database: %v", err)
	}

	collection, err := db.CreateCollection(ctx, "ivfpq_persist",
		WithDimension(3),
		WithMetric(CosineDistance),
		WithIVFPQ(4, 4),
	)
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	entries := []VectorEntry{
		{ID: "exact", Vector: []float32{1, 0, 0}},
		{ID: "close", Vector: []float32{0.8, 0.6, 0}},
		{ID: "orthogonal", Vector: []float32{0, 1, 0}},
		{ID: "opposite", Vector: []float32{-1, 0, 0}},
	}
	if err := collection.InsertBatch(ctx, entries); err != nil {
		t.Fatalf("insert batch: %v", err)
	}

	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}

	reopened, err := New(WithStoragePath(dbPath))
	if err != nil {
		t.Fatalf("reopen database: %v", err)
	}
	defer reopened.Close()

	reloaded, err := reopened.GetCollection("ivfpq_persist")
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}

	results, err := reloaded.Search(ctx, []float32{1, 0, 0}, 4)
	if err != nil {
		t.Fatalf("search after reopen failed: %v", err)
	}
	assertResultOrder(t, results.Results, "exact", "close", "orthogonal", "opposite")
	assertApproxScore(t, results.Results[0].Score, 1.0, 1e-4)
	assertApproxScore(t, results.Results[1].Score, 0.8, 1e-4)
	assertApproxScore(t, results.Results[2].Score, 0.0, 1e-4)
	assertApproxScore(t, results.Results[3].Score, -1.0, 1e-4)

	thresholded, err := reloaded.Query(ctx).
		WithVector([]float32{1, 0, 0}).
		WithThreshold(0.0).
		Limit(10).
		Execute()
	if err != nil {
		t.Fatalf("threshold query after reopen failed: %v", err)
	}
	assertResultOrder(t, thresholded.Results, "exact", "close", "orthogonal")
}

func TestCosinePublicScoreContract_BackendTopResultAgreement(t *testing.T) {
	ctx := context.Background()

	entries := make([]VectorEntry, 32)
	for i := range entries {
		entries[i] = VectorEntry{
			ID:     "vec_" + fmt.Sprint(i),
			Vector: benchVector(16, i),
		}
	}

	queries := [][]float32{
		benchVector(16, 1),
		benchVector(16, 7),
		benchVector(16, 13),
		benchVector(16, 21),
	}

	flat := newCosineCollectionWithEntries(t, "flat_quality", entries, WithFlat())
	hnsw := newCosineCollectionWithEntries(t, "hnsw_quality", entries, WithHNSW(16, 100, 64))
	ivfpq := newCosineCollectionWithEntries(t, "ivfpq_quality", entries, WithIVFPQ(8, 8))

	for i, query := range queries {
		flatResults, err := flat.Search(ctx, query, 5)
		if err != nil {
			t.Fatalf("flat search %d failed: %v", i, err)
		}
		hnswResults, err := hnsw.Search(ctx, query, 5)
		if err != nil {
			t.Fatalf("hnsw search %d failed: %v", i, err)
		}
		ivfpqResults, err := ivfpq.Search(ctx, query, 5)
		if err != nil {
			t.Fatalf("ivfpq search %d failed: %v", i, err)
		}

		if flatResults.Results[0].ID != hnswResults.Results[0].ID {
			t.Fatalf("query %d top-1 mismatch flat=%q hnsw=%q", i, flatResults.Results[0].ID, hnswResults.Results[0].ID)
		}
		if flatResults.Results[0].ID != ivfpqResults.Results[0].ID {
			t.Fatalf("query %d top-1 mismatch flat=%q ivfpq=%q", i, flatResults.Results[0].ID, ivfpqResults.Results[0].ID)
		}
	}
}

func newCosineContractCollection(t *testing.T, name string, extraOpts ...CollectionOption) *Collection {
	t.Helper()

	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("create database: %v", err)
	}
	t.Cleanup(func() {
		if err := db.Close(); err != nil {
			t.Fatalf("close database: %v", err)
		}
	})

	opts := []CollectionOption{
		WithDimension(3),
		WithMetric(CosineDistance),
	}
	opts = append(opts, extraOpts...)

	collection, err := db.CreateCollection(context.Background(), name, opts...)
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	entries := []VectorEntry{
		{ID: "exact", Vector: []float32{1, 0, 0}},
		{ID: "close", Vector: []float32{0.8, 0.6, 0}},
		{ID: "orthogonal", Vector: []float32{0, 1, 0}},
		{ID: "opposite", Vector: []float32{-1, 0, 0}},
	}

	if err := collection.InsertBatch(context.Background(), entries); err != nil {
		t.Fatalf("insert batch: %v", err)
	}

	return collection
}

func newCosineCollectionWithEntries(t *testing.T, name string, entries []VectorEntry, extraOpts ...CollectionOption) *Collection {
	t.Helper()

	db, err := New(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("create database: %v", err)
	}
	t.Cleanup(func() {
		if err := db.Close(); err != nil {
			t.Fatalf("close database: %v", err)
		}
	})

	opts := []CollectionOption{
		WithDimension(len(entries[0].Vector)),
		WithMetric(CosineDistance),
	}
	opts = append(opts, extraOpts...)

	collection, err := db.CreateCollection(context.Background(), name, opts...)
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.InsertBatch(context.Background(), entries); err != nil {
		t.Fatalf("insert batch: %v", err)
	}
	return collection
}

func assertResultOrder(t *testing.T, results []*SearchResult, expectedIDs ...string) {
	t.Helper()

	if len(results) != len(expectedIDs) {
		t.Fatalf("expected %d results, got %d", len(expectedIDs), len(results))
	}

	for i, expectedID := range expectedIDs {
		if results[i].ID != expectedID {
			t.Fatalf("result %d: expected %q, got %q", i, expectedID, results[i].ID)
		}
	}
}

func assertApproxScore(t *testing.T, got, want, tolerance float32) {
	t.Helper()

	if math.Abs(float64(got-want)) > float64(tolerance) {
		t.Fatalf("expected score %.6f +/- %.6f, got %.6f", want, tolerance, got)
	}
}
