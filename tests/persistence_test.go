package tests

import (
	"context"
	"os"
	"testing"

	"github.com/xDarkicex/libravdb/internal/obs"
	"github.com/xDarkicex/libravdb/libravdb"
)

func TestDataPersistence(t *testing.T) {
	// Clean up metrics before test
	obs.ResetForTesting()

	// Clean up test directory
	os.RemoveAll("./persist_test")
	defer func() {
		os.RemoveAll("./persist_test")
		obs.ResetForTesting() // Clean up metrics after test
	}()

	ctx := context.Background()

	// Insert data, close DB, reopen, verify data exists
	db1, err := libravdb.New(libravdb.WithStoragePath("./persist_test"))
	if err != nil {
		t.Fatalf("Failed to create initial database: %v", err)
	}

	collection, err := db1.CreateCollection(ctx, "test", libravdb.WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert test data
	err = collection.Insert(ctx, "test1", []float32{1, 0, 0}, nil)
	if err != nil {
		t.Fatalf("Failed to insert data: %v", err)
	}

	db1.Close()

	// Reopen database - should recover data
	db2, err := libravdb.New(libravdb.WithStoragePath("./persist_test"))
	if err != nil {
		t.Fatalf("Failed to reopen database: %v", err)
	}
	defer db2.Close()

	collection2, err := db2.GetCollection("test")
	if err != nil {
		t.Fatalf("Failed to get collection: %v", err)
	}

	// Search should find recovered data
	results, err := collection2.Search(ctx, []float32{1, 0, 0}, 1)
	if err != nil {
		t.Fatalf("Failed to search collection: %v", err)
	}

	// Verify results
	if len(results.Results) == 0 {
		t.Fatalf("Expected 1 result, got 0")
	}

	if results.Results[0].ID != "test1" {
		t.Fatalf("Expected ID 'test1', got '%s'", results.Results[0].ID)
	}

	t.Log("✅ Data persistence test passed!")
}
