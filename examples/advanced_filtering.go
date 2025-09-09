package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/xDarkicex/libravdb/internal/filter"
	"github.com/xDarkicex/libravdb/libravdb"
)

func main() {
	// Create temporary directory for demo
	tempDir, err := os.MkdirTemp("", "libravdb_advanced_filtering_demo")
	if err != nil {
		log.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create database and collection
	db, err := libravdb.New(libravdb.WithStoragePath(tempDir))
	if err != nil {
		log.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	collection, err := db.CreateCollection(ctx, "products",
		libravdb.WithDimension(3),
		libravdb.WithMetric(libravdb.CosineDistance))
	if err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}

	// Insert sample product data
	products := []struct {
		id       string
		vector   []float32
		category string
		price    float64
		tags     []string
		inStock  bool
		rating   float64
	}{
		{"phone1", []float32{1.0, 0.0, 0.0}, "electronics", 699.99, []string{"smartphone", "mobile"}, true, 4.5},
		{"laptop1", []float32{0.0, 1.0, 0.0}, "electronics", 1299.99, []string{"laptop", "computer"}, true, 4.8},
		{"book1", []float32{0.0, 0.0, 1.0}, "books", 24.99, []string{"fiction", "novel"}, true, 4.2},
		{"tablet1", []float32{0.5, 0.5, 0.0}, "electronics", 499.99, []string{"tablet", "mobile"}, false, 4.3},
		{"book2", []float32{0.3, 0.3, 0.4}, "books", 19.99, []string{"non-fiction", "science"}, true, 4.0},
	}

	for _, product := range products {
		metadata := map[string]interface{}{
			"category": product.category,
			"price":    product.price,
			"tags":     product.tags,
			"in_stock": product.inStock,
			"rating":   product.rating,
		}

		err := collection.Insert(ctx, product.id, product.vector, metadata)
		if err != nil {
			log.Fatalf("Failed to insert product %s: %v", product.id, err)
		}
	}

	queryVector := []float32{1.0, 0.0, 0.0}

	fmt.Println("=== Advanced Filtering Examples ===\n")

	// Example 1: Simple equality filter
	fmt.Println("1. Simple Equality Filter (category = 'electronics'):")
	results, err := collection.Query(ctx).
		WithVector(queryVector).
		Eq("category", "electronics").
		Limit(10).
		Execute()
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}
	printResults(results)

	// Example 2: Range filter
	fmt.Println("2. Range Filter (price between $400 and $800):")
	results, err = collection.Query(ctx).
		WithVector(queryVector).
		Between("price", 400.0, 800.0).
		Limit(10).
		Execute()
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}
	printResults(results)

	// Example 3: Containment filter
	fmt.Println("3. Containment Filter (tags contain 'mobile' or 'fiction'):")
	results, err = collection.Query(ctx).
		WithVector(queryVector).
		ContainsAny("tags", []interface{}{"mobile", "fiction"}).
		Limit(10).
		Execute()
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}
	printResults(results)

	// Example 4: AND filter chain
	fmt.Println("4. AND Filter Chain (electronics AND in_stock AND rating > 4.0):")
	results, err = collection.Query(ctx).
		WithVector(queryVector).
		And().
		Eq("category", "electronics").
		Eq("in_stock", true).
		Gt("rating", 4.0).
		End().
		Limit(10).
		Execute()
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}
	printResults(results)

	// Example 5: OR filter chain
	fmt.Println("5. OR Filter Chain (books OR price < $30):")
	results, err = collection.Query(ctx).
		WithVector(queryVector).
		Or().
		Eq("category", "books").
		Lt("price", 30.0).
		End().
		Limit(10).
		Execute()
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}
	printResults(results)

	// Example 6: Complex nested filters
	fmt.Println("6. Complex Nested Filters ((electronics AND in_stock) OR (books AND rating > 4.1)):")
	electronicsFilter := filter.NewAndFilter(
		filter.NewEqualityFilter("category", "electronics"),
		filter.NewEqualityFilter("in_stock", true),
	)
	booksFilter := filter.NewAndFilter(
		filter.NewEqualityFilter("category", "books"),
		filter.NewGreaterThanFilter("rating", 4.1),
	)
	complexFilter := filter.NewOrFilter(electronicsFilter, booksFilter)

	results, err = collection.Query(ctx).
		WithVector(queryVector).
		WithFilter(complexFilter).
		Limit(10).
		Execute()
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}
	printResults(results)

	// Example 7: NOT filter
	fmt.Println("7. NOT Filter (NOT electronics):")
	results, err = collection.Query(ctx).
		WithVector(queryVector).
		NotEq("category", "electronics").
		Limit(10).
		Execute()
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}
	printResults(results)

	// Example 8: Threshold filtering
	fmt.Println("8. Threshold Filtering (similarity > 0.8):")
	results, err = collection.Query(ctx).
		WithVector(queryVector).
		WithThreshold(0.8).
		Limit(10).
		Execute()
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}
	printResults(results)
}

func printResults(results *libravdb.SearchResults) {
	fmt.Printf("Found %d results (took %v):\n", len(results.Results), results.Took)
	for i, result := range results.Results {
		fmt.Printf("  %d. ID: %s, Score: %.3f, Category: %v, Price: $%.2f\n",
			i+1, result.ID, result.Score,
			result.Metadata["category"], result.Metadata["price"])
	}
	fmt.Println()
}
