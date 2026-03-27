package libravdb

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/xDarkicex/libravdb/internal/filter"
)

func TestQueryBuilderAdvancedFiltering(t *testing.T) {
	// Create temporary directory for test database
	tempDir, err := os.MkdirTemp("", "libravdb_query_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create test database and collection
	db, err := New(WithStoragePath(tempDir))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	collection, err := db.CreateCollection(ctx, "test", WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert test data with various metadata
	testData := []*VectorEntry{
		{
			ID:     "1",
			Vector: []float32{1.0, 0.0, 0.0},
			Metadata: map[string]interface{}{
				"category":   "electronics",
				"price":      100.0,
				"tags":       []string{"phone", "mobile"},
				"in_stock":   true,
				"created_at": "2023-01-01T00:00:00Z",
				"rating":     4.5,
			},
		},
		{
			ID:     "2",
			Vector: []float32{0.0, 1.0, 0.0},
			Metadata: map[string]interface{}{
				"category":   "electronics",
				"price":      200.0,
				"tags":       []string{"laptop", "computer"},
				"in_stock":   false,
				"created_at": "2023-02-01T00:00:00Z",
				"rating":     4.8,
			},
		},
		{
			ID:     "3",
			Vector: []float32{0.0, 0.0, 1.0},
			Metadata: map[string]interface{}{
				"category":   "books",
				"price":      25.0,
				"tags":       []string{"fiction", "novel"},
				"in_stock":   true,
				"created_at": "2023-03-01T00:00:00Z",
				"rating":     3.9,
			},
		},
		{
			ID:     "4",
			Vector: []float32{0.5, 0.5, 0.0},
			Metadata: map[string]interface{}{
				"category":   "electronics",
				"price":      150.0,
				"tags":       []string{"tablet", "mobile"},
				"in_stock":   true,
				"created_at": "2023-04-01T00:00:00Z",
				"rating":     4.2,
			},
		},
		{
			ID:     "5",
			Vector: []float32{0.3, 0.3, 0.4},
			Metadata: map[string]interface{}{
				"category":   "books",
				"price":      15.0,
				"tags":       []string{"non-fiction", "science"},
				"in_stock":   false,
				"created_at": "2023-05-01T00:00:00Z",
				"rating":     4.0,
			},
		},
	}

	for _, entry := range testData {
		err := collection.Insert(ctx, entry.ID, entry.Vector, entry.Metadata)
		if err != nil {
			t.Fatalf("Failed to insert entry %s: %v", entry.ID, err)
		}
	}

	queryVector := []float32{1.0, 0.0, 0.0}

	t.Run("Simple Equality Filter", func(t *testing.T) {
		results, err := collection.Query(ctx).
			WithVector(queryVector).
			Eq("category", "electronics").
			Limit(10).
			Execute()

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		if len(results.Results) != 3 {
			t.Errorf("Expected 3 results, got %d", len(results.Results))
		}

		// Verify all results have electronics category
		for _, result := range results.Results {
			if result.Metadata["category"] != "electronics" {
				t.Errorf("Expected category 'electronics', got %v", result.Metadata["category"])
			}
		}
	})

	t.Run("Range Filter", func(t *testing.T) {
		results, err := collection.Query(ctx).
			WithVector(queryVector).
			Between("price", 50.0, 180.0).
			Limit(10).
			Execute()

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		if len(results.Results) != 2 {
			t.Errorf("Expected 2 results, got %d", len(results.Results))
		}

		// Verify all results have price in range
		for _, result := range results.Results {
			price := result.Metadata["price"].(float64)
			if price < 50.0 || price > 180.0 {
				t.Errorf("Price %f not in range [50, 180]", price)
			}
		}
	})

	t.Run("Containment Filter", func(t *testing.T) {
		results, err := collection.Query(ctx).
			WithVector(queryVector).
			ContainsAny("tags", []interface{}{"mobile", "fiction"}).
			Limit(10).
			Execute()

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		if len(results.Results) != 3 {
			t.Errorf("Expected 3 results, got %d", len(results.Results))
		}
	})

	t.Run("AND Filter Chain", func(t *testing.T) {
		results, err := collection.Query(ctx).
			WithVector(queryVector).
			And().
			Eq("category", "electronics").
			Eq("in_stock", true).
			End().
			Limit(10).
			Execute()

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		if len(results.Results) != 2 {
			t.Errorf("Expected 2 results, got %d", len(results.Results))
		}

		// Verify all results match both conditions
		for _, result := range results.Results {
			if result.Metadata["category"] != "electronics" {
				t.Errorf("Expected category 'electronics', got %v", result.Metadata["category"])
			}
			if result.Metadata["in_stock"] != true {
				t.Errorf("Expected in_stock true, got %v", result.Metadata["in_stock"])
			}
		}
	})

	t.Run("OR Filter Chain", func(t *testing.T) {
		results, err := collection.Query(ctx).
			WithVector(queryVector).
			Or().
			Eq("category", "books").
			Lt("price", 30.0).
			End().
			Limit(10).
			Execute()

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		if len(results.Results) != 2 {
			t.Errorf("Expected 2 results, got %d", len(results.Results))
		}
	})

	t.Run("Complex Nested Filters", func(t *testing.T) {
		// (category = "electronics" AND in_stock = true) OR (category = "books" AND price < 20)
		electronicsFilter := filter.NewAndFilter(
			filter.NewEqualityFilter("category", "electronics"),
			filter.NewEqualityFilter("in_stock", true),
		)

		booksFilter := filter.NewAndFilter(
			filter.NewEqualityFilter("category", "books"),
			filter.NewLessThanFilter("price", 20.0),
		)

		complexFilter := filter.NewOrFilter(electronicsFilter, booksFilter)

		results, err := collection.Query(ctx).
			WithVector(queryVector).
			WithFilter(complexFilter).
			Limit(10).
			Execute()

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		if len(results.Results) != 3 {
			t.Errorf("Expected 3 results, got %d", len(results.Results))
		}
	})

	t.Run("NOT Filter", func(t *testing.T) {
		results, err := collection.Query(ctx).
			WithVector(queryVector).
			NotEq("category", "electronics").
			Limit(10).
			Execute()

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		if len(results.Results) != 2 {
			t.Errorf("Expected 2 results, got %d", len(results.Results))
		}

		// Verify no results have electronics category
		for _, result := range results.Results {
			if result.Metadata["category"] == "electronics" {
				t.Errorf("Found electronics category in NOT filter results")
			}
		}
	})

	t.Run("Multiple Filter Types Combined", func(t *testing.T) {
		results, err := collection.Query(ctx).
			WithVector(queryVector).
			And().
			Eq("category", "electronics").
			Gt("price", 120.0).
			ContainsAny("tags", []interface{}{"laptop", "tablet"}).
			End().
			Limit(10).
			Execute()

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		if len(results.Results) != 2 {
			t.Errorf("Expected 2 results, got %d", len(results.Results))
		}
	})

	t.Run("Filter Optimization", func(t *testing.T) {
		// Create a query with filters of different selectivities
		qb := collection.Query(ctx).
			WithVector(queryVector).
			Eq("category", "electronics").                // Medium selectivity
			Gt("rating", 4.0).                            // High selectivity
			ContainsAny("tags", []interface{}{"mobile"}). // Low selectivity
			Limit(10)

		// Test that optimization doesn't break functionality
		results, err := qb.Execute()
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		if len(results.Results) != 2 {
			t.Errorf("Expected 2 results, got %d", len(results.Results))
		}
	})

	t.Run("Threshold Filtering", func(t *testing.T) {
		results, err := collection.Query(ctx).
			WithVector(queryVector).
			WithThreshold(0.8).
			Limit(10).
			Execute()

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		// Verify all results meet threshold
		for _, result := range results.Results {
			if result.Score < 0.8 {
				t.Errorf("Result score %f below threshold 0.8", result.Score)
			}
		}
	})

	t.Run("Empty Filter Chain", func(t *testing.T) {
		results, err := collection.Query(ctx).
			WithVector(queryVector).
			And().
			End().
			Limit(10).
			Execute()

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		// Should return all results since no filters applied
		if len(results.Results) != 5 {
			t.Errorf("Expected 5 results, got %d", len(results.Results))
		}
	})
}

func TestQueryBuilderErrorHandling(t *testing.T) {
	// Create temporary directory for test database
	tempDir, err := os.MkdirTemp("", "libravdb_query_error_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create test database and collection
	db, err := New(WithStoragePath(tempDir))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	collection, err := db.CreateCollection(ctx, "test", WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	t.Run("Missing Query Vector", func(t *testing.T) {
		_, err := collection.Query(ctx).
			Eq("category", "electronics").
			Limit(10).
			Execute()

		if err == nil {
			t.Error("Expected error for missing query vector")
		}
	})

	t.Run("Invalid Limit", func(t *testing.T) {
		_, err := collection.Query(ctx).
			WithVector([]float32{1.0, 0.0, 0.0}).
			Limit(0).
			Execute()

		if err == nil {
			t.Error("Expected error for invalid limit")
		}
	})

	t.Run("Invalid Filter", func(t *testing.T) {
		invalidFilter := filter.NewEqualityFilter("", nil) // Invalid field and value

		_, err := collection.Query(ctx).
			WithVector([]float32{1.0, 0.0, 0.0}).
			WithFilter(invalidFilter).
			Limit(10).
			Execute()

		if err == nil {
			t.Error("Expected error for invalid filter")
		}
	})
}

func TestFilterSelectivityEstimation(t *testing.T) {
	tests := []struct {
		name                string
		filter              filter.Filter
		expectedSelectivity float64
		tolerance           float64
	}{
		{
			name:                "Equality Filter",
			filter:              filter.NewEqualityFilter("category", "electronics"),
			expectedSelectivity: 0.1,
			tolerance:           0.01,
		},
		{
			name:                "Range Filter (Both Bounds)",
			filter:              filter.NewBetweenFilter("price", 50.0, 150.0),
			expectedSelectivity: 0.3,
			tolerance:           0.01,
		},
		{
			name:                "Range Filter (Single Bound)",
			filter:              filter.NewGreaterThanFilter("price", 100.0),
			expectedSelectivity: 0.5,
			tolerance:           0.01,
		},
		{
			name:                "Containment Filter (Any)",
			filter:              filter.NewContainsAnyFilter("tags", []interface{}{"mobile", "laptop"}),
			expectedSelectivity: 0.4,
			tolerance:           0.01,
		},
		{
			name:                "Containment Filter (All)",
			filter:              filter.NewContainsAllFilter("tags", []interface{}{"mobile", "phone"}),
			expectedSelectivity: 0.2,
			tolerance:           0.01,
		},
		{
			name: "AND Filter",
			filter: filter.NewAndFilter(
				filter.NewEqualityFilter("category", "electronics"),
				filter.NewGreaterThanFilter("price", 100.0),
			),
			expectedSelectivity: 0.05, // 0.1 * 0.5
			tolerance:           0.01,
		},
		{
			name: "OR Filter",
			filter: filter.NewOrFilter(
				filter.NewEqualityFilter("category", "electronics"),
				filter.NewEqualityFilter("category", "books"),
			),
			expectedSelectivity: 0.19, // 1 - (0.9 * 0.9)
			tolerance:           0.01,
		},
		{
			name:                "NOT Filter",
			filter:              filter.NewNotFilter(filter.NewEqualityFilter("category", "electronics")),
			expectedSelectivity: 0.9, // 1 - 0.1
			tolerance:           0.01,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			selectivity := tt.filter.EstimateSelectivity()
			if selectivity < tt.expectedSelectivity-tt.tolerance || selectivity > tt.expectedSelectivity+tt.tolerance {
				t.Errorf("Expected selectivity %f ± %f, got %f", tt.expectedSelectivity, tt.tolerance, selectivity)
			}
		})
	}
}

func BenchmarkQueryBuilderFiltering(b *testing.B) {
	// Create temporary directory for benchmark database
	tempDir, err := os.MkdirTemp("", "libravdb_query_bench")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create test database and collection
	db, err := New(WithStoragePath(tempDir))
	if err != nil {
		b.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	collection, err := db.CreateCollection(ctx, "test", WithDimension(128), WithMetric(CosineDistance))
	if err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}

	// Insert benchmark data
	numEntries := 1000

	for i := 0; i < numEntries; i++ {
		vector := make([]float32, 128)
		for j := range vector {
			vector[j] = float32(i%10) / 10.0
		}

		entry := &VectorEntry{
			ID:     fmt.Sprintf("entry_%d", i),
			Vector: vector,
			Metadata: map[string]interface{}{
				"category": []string{"electronics", "books", "clothing"}[i%3],
				"price":    float64(10 + (i % 200)),
				"rating":   4.0 + float64(i%10)/10.0,
				"in_stock": i%2 == 0,
			},
		}

		err := collection.Insert(ctx, entry.ID, entry.Vector, entry.Metadata)
		if err != nil {
			b.Fatalf("Failed to insert entry: %v", err)
		}
	}

	queryVector := make([]float32, 128)
	for i := range queryVector {
		queryVector[i] = 0.5
	}

	b.Run("No Filters", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := collection.Query(ctx).
				WithVector(queryVector).
				Limit(50).
				Execute()
			if err != nil {
				b.Fatalf("Query failed: %v", err)
			}
		}
	})

	b.Run("Single Filter", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := collection.Query(ctx).
				WithVector(queryVector).
				Eq("category", "electronics").
				Limit(50).
				Execute()
			if err != nil {
				b.Fatalf("Query failed: %v", err)
			}
		}
	})

	b.Run("Multiple AND Filters", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := collection.Query(ctx).
				WithVector(queryVector).
				And().
				Eq("category", "electronics").
				Gt("price", 50.0).
				Eq("in_stock", true).
				End().
				Limit(50).
				Execute()
			if err != nil {
				b.Fatalf("Query failed: %v", err)
			}
		}
	})

	b.Run("Complex OR Filters", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := collection.Query(ctx).
				WithVector(queryVector).
				Or().
				Eq("category", "electronics").
				Lt("price", 30.0).
				Gt("rating", 4.8).
				End().
				Limit(50).
				Execute()
			if err != nil {
				b.Fatalf("Query failed: %v", err)
			}
		}
	})
}
