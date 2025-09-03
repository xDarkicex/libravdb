package tests

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/xDarkicex/libravdb/libravdb"
)

func TestCoreFunctionality(t *testing.T) {
	ctx := context.Background()

	// Clean up any existing test data
	os.RemoveAll("./test_data")
	defer os.RemoveAll("./test_data")

	// Create database
	db, err := libravdb.New(
		libravdb.WithStoragePath("./test_data"),
		libravdb.WithMetrics(true),
	)
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Create collection with 3D vectors
	collection, err := db.CreateCollection(ctx, "test_vectors",
		libravdb.WithDimension(3),
		libravdb.WithMetric(libravdb.CosineDistance),
	)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert test vectors
	testVectors := map[string][]float32{
		"x_axis": {1.0, 0.0, 0.0},
		"y_axis": {0.0, 1.0, 0.0},
		"z_axis": {0.0, 0.0, 1.0},
	}

	fmt.Println("🔄 Inserting vectors...")
	for id, vector := range testVectors {
		err = collection.Insert(ctx, id, vector, nil)
		if err != nil {
			t.Fatalf("Failed to insert vector '%s': %v", id, err)
		}
		fmt.Printf("  ✅ Inserted: %s %v\n", id, vector)
	}

	// Test search
	fmt.Println("\n🔍 Testing search...")
	queryVector := []float32{1.0, 0.0, 0.0}
	results, err := collection.Search(ctx, queryVector, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if results.Total == 0 {
		t.Fatal("Expected at least 1 search result, got 0")
	}

	bestMatch := results.Results[0]
	if bestMatch.ID != "x_axis" {
		t.Errorf("Expected nearest neighbor to be 'x_axis', got '%s'", bestMatch.ID)
	}

	fmt.Printf("  ✅ Best match: %s (score: %.4f)\n", bestMatch.ID, bestMatch.Score)
	fmt.Println("✅ All core functionality tests passed!")
}
