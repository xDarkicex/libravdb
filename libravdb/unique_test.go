package libravdb

import (
	"context"
	"strings"
	"testing"
)

func TestUniqueEnforcement(t *testing.T) {
	t.Run("UNIQUE column rejects duplicate", func(t *testing.T) {
		db := openTempDB(t, "uniq_reject")
		defer db.Close()
		exec(t, db, "CREATE TABLE users (id TEXT, email TEXT UNIQUE)")
		coll := getColl(t, db, "users")

		insertRecord(t, coll, "u1", "email", "alice@ex.com")

		err := coll.Insert(context.Background(), "u2", nil,
			map[string]interface{}{"email": "alice@ex.com"})
		if err == nil {
			t.Fatal("expected UNIQUE violation, got nil")
		}
		if !strings.Contains(err.Error(), "UNIQUE") {
			t.Errorf("got: %v", err)
		}
	})

	t.Run("UNIQUE column allows different values", func(t *testing.T) {
		db := openTempDB(t, "uniq_allow")
		defer db.Close()
		exec(t, db, "CREATE TABLE users (id TEXT, email TEXT UNIQUE)")
		coll := getColl(t, db, "users")

		insertRecord(t, coll, "u1", "email", "alice@ex.com")

		err := coll.Insert(context.Background(), "u2", nil,
			map[string]interface{}{"email": "bob@ex.com"})
		if err != nil {
			t.Errorf("different UNIQUE value failed: %v", err)
		}
	})

	t.Run("UNIQUE update to duplicate rejected", func(t *testing.T) {
		db := openTempDB(t, "uniq_update")
		defer db.Close()
		exec(t, db, "CREATE TABLE users (id TEXT, email TEXT UNIQUE)")
		coll := getColl(t, db, "users")

		insertRecord(t, coll, "u1", "email", "alice@ex.com")
		insertRecord(t, coll, "u2", "email", "bob@ex.com")

		err := coll.Update(context.Background(), "u2", nil,
			map[string]interface{}{"email": "alice@ex.com"})
		if err == nil {
			t.Fatal("expected UNIQUE violation on update, got nil")
		}
		if !strings.Contains(err.Error(), "UNIQUE") {
			t.Errorf("got: %v", err)
		}
	})
}
