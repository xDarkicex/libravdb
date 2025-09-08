package tests

import (
	"context"
	"os"
	"testing"

	"github.com/xDarkicex/libravdb/internal/index/hnsw"
	"github.com/xDarkicex/libravdb/internal/util"
)

func TestHNSWPersistence(t *testing.T) {
	ctx := context.Background()
	testFile := "./test_hnsw_index.bin"

	// Clean up
	defer os.Remove(testFile)
	os.Remove(testFile)

	// Create HNSW index
	config := &hnsw.Config{
		Dimension:      4,
		M:              16,
		EfConstruction: 200,
		EfSearch:       50,
		ML:             1.0 / 2.303,
		Metric:         util.L2Distance,
		RandomSeed:     42,
	}

	index, err := hnsw.NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}

	// Insert some test vectors
	vectors := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 3.0, 4.0, 5.0},
		{3.0, 4.0, 5.0, 6.0},
		{4.0, 5.0, 6.0, 7.0},
	}

	for i, vector := range vectors {
		entry := &hnsw.VectorEntry{
			ID:       string(rune('A' + i)),
			Vector:   vector,
			Metadata: map[string]interface{}{"index": i},
		}
		if err := index.Insert(ctx, entry); err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}
	}

	// Get metadata before saving
	metadataBefore := index.GetPersistenceMetadata()
	if metadataBefore == nil {
		t.Fatal("Expected metadata before saving")
	}

	t.Logf("Index before save: %d nodes, dimension %d", metadataBefore.NodeCount, metadataBefore.Dimension)

	// Save to disk
	if err := index.SaveToDisk(ctx, testFile); err != nil {
		t.Fatalf("Failed to save index to disk: %v", err)
	}

	// Verify file was created
	if _, err := os.Stat(testFile); os.IsNotExist(err) {
		t.Fatalf("Index file was not created: %s", testFile)
	}

	// Create new index and load from disk
	index2, err := hnsw.NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create second HNSW index: %v", err)
	}
	if err := index2.LoadFromDisk(ctx, testFile); err != nil {
		t.Fatalf("Failed to load index from disk: %v", err)
	}

	// Get metadata after loading
	metadataAfter := index2.GetPersistenceMetadata()
	if metadataAfter == nil {
		t.Log("Metadata is nil after loading - checking if index is empty")
		// Try a simple search to see if data was loaded
		testResults, searchErr := index2.Search(ctx, vectors[0], 1)
		if searchErr != nil {
			t.Fatalf("Search failed on loaded index: %v", searchErr)
		}
		t.Logf("Search returned %d results", len(testResults))
		t.Fatal("Expected metadata after loading")
	}

	t.Logf("Index after load: %d nodes, dimension %d", metadataAfter.NodeCount, metadataAfter.Dimension)

	// Verify metadata matches
	if metadataBefore.NodeCount != metadataAfter.NodeCount {
		t.Errorf("Node count mismatch: before=%d, after=%d", metadataBefore.NodeCount, metadataAfter.NodeCount)
	}

	if metadataBefore.Dimension != metadataAfter.Dimension {
		t.Errorf("Dimension mismatch: before=%d, after=%d", metadataBefore.Dimension, metadataAfter.Dimension)
	}

	// Test search functionality on loaded index
	queryVector := []float32{1.5, 2.5, 3.5, 4.5}
	results, err := index2.Search(ctx, queryVector, 2)
	if err != nil {
		t.Fatalf("Failed to search loaded index: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("No search results from loaded index")
	}

	t.Logf("Search found %d results", len(results))
	for i, result := range results {
		t.Logf("  Result %d: ID=%s, Score=%.4f", i, result.ID, result.Score)
	}

	t.Log("✅ HNSW persistence test passed!")
	t.Log("   - Index saved and loaded successfully")
	t.Log("   - Metadata preserved correctly")
	t.Log("   - Search functionality works after loading")
}
