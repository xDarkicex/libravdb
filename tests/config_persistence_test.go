package tests

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/xDarkicex/libravdb/libravdb"
)

func TestConfigPersistence(t *testing.T) {
	ctx := context.Background()
	testDir := "./test_config_persistence"

	// Clean up before and after test
	defer os.RemoveAll(testDir)
	os.RemoveAll(testDir)

	// Create database with custom configuration
	db, err := libravdb.New(
		libravdb.WithStoragePath(testDir),
	)
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}

	// Create collection with custom configuration
	collection, err := db.CreateCollection(ctx, "test_config",
		libravdb.WithDimension(128),    // Custom dimension
		libravdb.WithHNSW(16, 100, 40), // Custom HNSW params: M=16, EfConstruction=100, EfSearch=40
		libravdb.WithMetric(libravdb.L2Distance),
	)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert test data
	testVector := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
		11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
		21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
		31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
		41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
		51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
		61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
		71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
		81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
		91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
		101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
		111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
		121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0} // 128 dimensions

	err = collection.Insert(ctx, "test_vector", testVector, map[string]interface{}{
		"description": "test vector with custom config",
	})
	if err != nil {
		t.Fatalf("Failed to insert vector: %v", err)
	}

	// Verify config file was created
	configPath := filepath.Join(testDir, "test_config", "config.json")
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		t.Fatalf("Config file was not created at %s", configPath)
	}

	// Close database
	if err := db.Close(); err != nil {
		t.Fatalf("Failed to close database: %v", err)
	}

	// Reopen database
	db2, err := libravdb.New(
		libravdb.WithStoragePath(testDir),
	)
	if err != nil {
		t.Fatalf("Failed to reopen database: %v", err)
	}
	defer db2.Close()

	// Get collection (should load from storage with preserved config)
	collection2, err := db2.GetCollection("test_config")
	if err != nil {
		t.Fatalf("Failed to get collection after reopening: %v", err)
	}

	// Verify the data is still there
	results, err := collection2.Search(ctx, testVector[:128], 1)
	if err != nil {
		t.Fatalf("Failed to search in reopened collection: %v", err)
	}

	if len(results.Results) == 0 {
		t.Fatal("No results found in reopened collection")
	}

	if results.Results[0].ID != "test_vector" {
		t.Errorf("Expected ID 'test_vector', got '%s'", results.Results[0].ID)
	}

	// Verify the configuration was preserved by checking if we can insert vectors with the correct dimension
	// If dimension wasn't preserved correctly, this would fail
	testVector2 := make([]float32, 128) // Should match the stored dimension
	for i := range testVector2 {
		testVector2[i] = float32(i + 200)
	}

	err = collection2.Insert(ctx, "test_vector_2", testVector2, map[string]interface{}{
		"description": "second test vector",
	})
	if err != nil {
		t.Fatalf("Failed to insert second vector (config not preserved): %v", err)
	}

	// Search should now return 2 results
	results2, err := collection2.Search(ctx, testVector2, 2)
	if err != nil {
		t.Fatalf("Failed to search after second insert: %v", err)
	}

	if len(results2.Results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results2.Results))
	}

	t.Log("✅ Configuration persistence test passed!")
	t.Logf("   - Config file created at: %s", configPath)
	t.Logf("   - Collection recovered with preserved configuration")
	t.Logf("   - Data persisted correctly across database restarts")
	t.Logf("   - Custom dimension (128) preserved and working")
}
