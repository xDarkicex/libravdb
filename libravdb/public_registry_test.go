package libravdb

import (
	"context"
	"path/filepath"
	"testing"
)

func TestPublicRegistryAPI(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "test.db")
	db, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatalf("Failed to create db: %v", err)
	}
	defer db.Close()

	col, err := db.CreateCollection(ctx, "testcol", WithDimension(4))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Insert a record
	err = col.Insert(ctx, "rec1", []float32{1, 2, 3, 4}, nil)
	if err != nil {
		t.Fatalf("Failed to insert: %v", err)
	}

	// Lookup via Collection
	gnid, err := col.LookupNodeID(ctx, "rec1")
	if err != nil {
		t.Fatalf("Failed to lookup node ID: %v", err)
	}
	if gnid == 0 {
		t.Fatalf("Expected non-zero GraphNodeID")
	}

	// Resolve via Database
	colName, recID, err := db.ResolveNodeID(ctx, gnid)
	if err != nil {
		t.Fatalf("Failed to resolve node ID: %v", err)
	}
	if colName != "testcol" {
		t.Errorf("Expected collection name 'testcol', got %q", colName)
	}
	if recID != "rec1" {
		t.Errorf("Expected record ID 'rec1', got %q", recID)
	}

	// Test non-existent record
	_, err = col.LookupNodeID(ctx, "nonexistent")
	if err == nil {
		t.Errorf("Expected error looking up non-existent record")
	}

	// Test non-existent node ID
	_, _, err = db.ResolveNodeID(ctx, 999999)
	if err == nil {
		t.Errorf("Expected error resolving non-existent node ID")
	}
}

// TestPublicRegistryAPISharded verifies that LookupNodeID and ResolveNodeID
// work correctly on sharded collections. Resolution must return the logical
// collection name, not an internal shard name.
func TestPublicRegistryAPISharded(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "test_sharded.db")
	db, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatalf("Failed to create sharded db: %v", err)
	}
	defer db.Close()

	col, err := db.CreateCollection(ctx, "sharded", WithDimension(4))
	if err != nil {
		t.Fatalf("Failed to create sharded collection: %v", err)
	}

	// Insert records
	if err := col.Insert(ctx, "rec_a", []float32{1, 2, 3, 4}, nil); err != nil {
		t.Fatalf("insert rec_a: %v", err)
	}
	if err := col.Insert(ctx, "rec_b", []float32{5, 6, 7, 8}, nil); err != nil {
		t.Fatalf("insert rec_b: %v", err)
	}

	// Lookup via Collection
	gnidA, err := col.LookupNodeID(ctx, "rec_a")
	if err != nil {
		t.Fatalf("LookupNodeID rec_a: %v", err)
	}
	if gnidA == 0 {
		t.Errorf("Expected non-zero GraphNodeID for rec_a")
	}

	gnidB, err := col.LookupNodeID(ctx, "rec_b")
	if err != nil {
		t.Fatalf("LookupNodeID rec_b: %v", err)
	}
	if gnidB == 0 {
		t.Errorf("Expected non-zero GraphNodeID for rec_b")
	}

	// Different records must have different GraphNodeIDs
	if gnidA == gnidB {
		t.Errorf("Sharded records aliased GraphNodeID: both got %d", gnidA)
	}

	// Resolve must return the logical collection name, not __shard__
	colNameA, recIDA, err := db.ResolveNodeID(ctx, gnidA)
	if err != nil {
		t.Fatalf("ResolveNodeID %d: %v", gnidA, err)
	}
	if colNameA != "sharded" {
		t.Errorf("Expected logical collection 'sharded', got %q", colNameA)
	}
	if recIDA != "rec_a" {
		t.Errorf("Expected record 'rec_a', got %q", recIDA)
	}

	colNameB, recIDB, err := db.ResolveNodeID(ctx, gnidB)
	if err != nil {
		t.Fatalf("ResolveNodeID %d: %v", gnidB, err)
	}
	if colNameB != "sharded" {
		t.Errorf("Expected logical collection 'sharded', got %q", colNameB)
	}
	if recIDB != "rec_b" {
		t.Errorf("Expected record 'rec_b', got %q", recIDB)
	}
}
