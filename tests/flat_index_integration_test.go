package tests

import (
	"context"
	"fmt"
	"testing"

	"github.com/xDarkicex/libravdb/libravdb"
)

func TestFlatIndexIntegration(t *testing.T) {
	// Create a database
	db, err := libravdb.New(libravdb.WithStoragePath(":memory:basic"))
	if err != nil {
		t.Fatalf("failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create a collection with flat index
	collection, err := db.CreateCollection(ctx, "test_flat",
		libravdb.WithDimension(3),
		libravdb.WithMetric(libravdb.CosineDistance),
		libravdb.WithFlat(),
	)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	// Insert some test vectors
	testVectors := []struct {
		id       string
		vector   []float32
		metadata map[string]interface{}
	}{
		{"v1", []float32{1.0, 0.0, 0.0}, map[string]interface{}{"category": "A"}},
		{"v2", []float32{0.0, 1.0, 0.0}, map[string]interface{}{"category": "B"}},
		{"v3", []float32{0.0, 0.0, 1.0}, map[string]interface{}{"category": "A"}},
		{"v4", []float32{1.0, 1.0, 0.0}, map[string]interface{}{"category": "C"}},
	}

	for _, tv := range testVectors {
		err := collection.Insert(ctx, tv.id, tv.vector, tv.metadata)
		if err != nil {
			t.Fatalf("failed to insert vector %s: %v", tv.id, err)
		}
	}

	// Test search
	query := []float32{1.0, 0.0, 0.0}
	results, err := collection.Search(ctx, query, 2)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results.Results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results.Results))
	}

	// First result should be v1 (exact match)
	if results.Results[0].ID != "v1" {
		t.Errorf("expected first result to be v1, got %s", results.Results[0].ID)
	}

	// Test that we can get stats
	stats := collection.Stats()
	if stats.VectorCount != 4 {
		t.Errorf("expected 4 vectors, got %d", stats.VectorCount)
	}

	if stats.IndexType != "Flat" {
		t.Errorf("expected Flat index type, got %s", stats.IndexType)
	}
}

func TestAutoIndexSelection(t *testing.T) {
	// Create a database with a different path
	db, err := libravdb.New(libravdb.WithStoragePath(":memory:auto"))
	if err != nil {
		t.Fatalf("failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create a collection with auto index selection
	collection, err := db.CreateCollection(ctx, "test_auto",
		libravdb.WithDimension(3),
		libravdb.WithMetric(libravdb.CosineDistance),
		libravdb.WithAutoIndexSelection(true),
	)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	// Initially should use Flat index for small collections
	stats := collection.Stats()
	if stats.IndexType != "Flat" {
		t.Errorf("expected Flat index for small collection, got %s", stats.IndexType)
	}

	// Insert a few vectors
	for i := 0; i < 5; i++ {
		vector := []float32{float32(i), float32(i + 1), float32(i + 2)}
		err := collection.Insert(ctx, fmt.Sprintf("v%d", i), vector, nil)
		if err != nil {
			t.Fatalf("failed to insert vector: %v", err)
		}
	}

	// Should still be Flat
	stats = collection.Stats()
	if stats.IndexType != "Flat" {
		t.Errorf("expected Flat index for small collection, got %s", stats.IndexType)
	}
}

func TestFlatIndexPerformance(t *testing.T) {
	// Create a database with a different path
	db, err := libravdb.New(libravdb.WithStoragePath(":memory:perf"))
	if err != nil {
		t.Fatalf("failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create a collection with flat index
	collection, err := db.CreateCollection(ctx, "test_perf",
		libravdb.WithDimension(128),
		libravdb.WithMetric(libravdb.CosineDistance),
		libravdb.WithFlat(),
	)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	// Insert 1000 vectors
	for i := 0; i < 1000; i++ {
		vector := make([]float32, 128)
		for j := range vector {
			vector[j] = float32(i + j)
		}
		err := collection.Insert(ctx, fmt.Sprintf("v%d", i), vector, nil)
		if err != nil {
			t.Fatalf("failed to insert vector %d: %v", i, err)
		}
	}

	// Test search performance
	query := make([]float32, 128)
	for i := range query {
		query[i] = float32(i)
	}

	results, err := collection.Search(ctx, query, 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results.Results) != 10 {
		t.Errorf("expected 10 results, got %d", len(results.Results))
	}

	// Verify search is exact (flat index should give exact results)
	// Allow for small floating-point precision errors
	if results.Results[0].Score > 1e-6 {
		t.Errorf("expected near-exact match (score ~0.0), got %f", results.Results[0].Score)
	}
}
