package libravdb

import (
	"context"
	"testing"
)

func TestMetadataOnlyUpdate(t *testing.T) {
	db := openTempDB(t, "meta_upd")
	defer db.Close()

	// Create a metadata-only collection
	exec(t, db, "CREATE TABLE items (id TEXT PRIMARY KEY, name TEXT)")
	coll := getColl(t, db, "items")

	// Insert
	insertRecord(t, coll, "item-1", "name", "old")

	// Update (should work)
	err := coll.Update(context.Background(), "item-1", nil, map[string]interface{}{"name": "new"})
	if err != nil {
		t.Fatalf("Update: %v", err)
	}
}
