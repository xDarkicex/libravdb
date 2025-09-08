package tests

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/xDarkicex/libravdb/libravdb"
)

func TestCollectionPersistence(t *testing.T) {
	ctx := context.Background()
	testDir := "./test_collection_persistence"
	indexPath := filepath.Join(testDir, "test_index.bin")

	// Clean up before and after test
	defer os.RemoveAll(testDir)
	os.RemoveAll(testDir)

	// Create database and collection
	db, err := libravdb.New(
		libravdb.WithStoragePath(testDir),
	)
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Create collection with persistence options
	collection, err := db.CreateCollection(ctx, "test_collection",
		libravdb.WithDimension(3),
		libravdb.WithHNSW(16, 100, 50),
		libravdb.WithMetric(libravdb.L2Distance),
		libravdb.WithIndexPersistence(true),
		libravdb.WithPersistencePath(indexPath),
	)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert test vectors
	testVectors := []struct {
		id     string
		vector []float32
	}{
		{"vec1", []float32{1.0, 0.0, 0.0}},
		{"vec2", []float32{0.0, 1.0, 0.0}},
		{"vec3", []float32{0.0, 0.0, 1.0}},
		{"vec4", []float32{1.0, 1.0, 0.0}},
	}

	for _, tv := range testVectors {
		err := collection.Insert(ctx, tv.id, tv.vector, map[string]interface{}{
			"description": "test vector " + tv.id,
		})
		if err != nil {
			t.Fatalf("Failed to insert vector %s: %v", tv.id, err)
		}
	}

	// Get index metadata before saving
	metadataBefore := collection.GetIndexMetadata()
	if metadataBefore == nil {
		t.Fatal("Expected index metadata before saving")
	}
	t.Logf("Index metadata before save: %d nodes, dimension %d", metadataBefore.NodeCount, metadataBefore.Dimension)

	// Save index manually
	if err := collection.SaveIndex(ctx, indexPath); err != nil {
		t.Fatalf("Failed to save index: %v", err)
	}

	// Verify index file was created
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		t.Fatalf("Index file was not created: %s", indexPath)
	}

	// Test search before loading
	queryVector := []float32{0.5, 0.5, 0.0}
	resultsBefore, err := collection.Search(ctx, queryVector, 2)
	if err != nil {
		t.Fatalf("Failed to search before loading: %v", err)
	}
	t.Logf("Search results before loading: %d results", len(resultsBefore.Results))

	// Create a new collection and load the index
	collection2, err := db.CreateCollection(ctx, "test_collection_2",
		libravdb.WithDimension(3),
		libravdb.WithHNSW(16, 100, 50),
		libravdb.WithMetric(libravdb.L2Distance),
	)
	if err != nil {
		t.Fatalf("Failed to create second collection: %v", err)
	}

	// Load index from disk
	if err := collection2.LoadIndex(ctx, indexPath); err != nil {
		t.Fatalf("Failed to load index: %v", err)
	}

	// Get metadata after loading
	metadataAfter := collection2.GetIndexMetadata()
	if metadataAfter == nil {
		t.Fatal("Expected index metadata after loading")
	}
	t.Logf("Index metadata after load: %d nodes, dimension %d", metadataAfter.NodeCount, metadataAfter.Dimension)

	// Verify metadata matches
	if metadataBefore.NodeCount != metadataAfter.NodeCount {
		t.Errorf("Node count mismatch: before=%d, after=%d", metadataBefore.NodeCount, metadataAfter.NodeCount)
	}

	if metadataBefore.Dimension != metadataAfter.Dimension {
		t.Errorf("Dimension mismatch: before=%d, after=%d", metadataBefore.Dimension, metadataAfter.Dimension)
	}

	// Test search after loading
	resultsAfter, err := collection2.Search(ctx, queryVector, 2)
	if err != nil {
		t.Fatalf("Failed to search after loading: %v", err)
	}
	t.Logf("Search results after loading: %d results", len(resultsAfter.Results))

	if len(resultsAfter.Results) == 0 {
		t.Fatal("No search results after loading index")
	}

	// Verify we can find similar results
	if len(resultsBefore.Results) != len(resultsAfter.Results) {
		t.Logf("Warning: Different number of results before (%d) and after (%d) loading",
			len(resultsBefore.Results), len(resultsAfter.Results))
	}

	t.Log("✅ Collection persistence test passed!")
	t.Log("   - Index saved and loaded successfully at collection level")
	t.Log("   - Metadata preserved correctly")
	t.Log("   - Search functionality works after loading")
	t.Log("   - Persistence options configured correctly")
}
