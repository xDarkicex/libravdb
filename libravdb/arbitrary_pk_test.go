package libravdb

import (
	"context"
	"testing"
)

// TestArbitraryPK verifies that non-id PRIMARY KEY columns derive the
// internal record key from the declared PK value.
func TestArbitraryPK(t *testing.T) {
	t.Run("single non-id PK via SQL", func(t *testing.T) {
		db := openTempDB(t, "arb_pk_single")
		defer db.Close()

		// Declare email as the PRIMARY KEY — no explicit id column.
		_, err := db.Query(context.Background(),
			"CREATE TABLE users (email TEXT PRIMARY KEY, name TEXT)")
		if err != nil {
			t.Fatalf("CREATE TABLE: %v", err)
		}

		// Insert via SQL using only the declared PK column.
		_, err = db.Query(context.Background(),
			"INSERT INTO users (email, name) VALUES ('alice@example.com', 'Alice')")
		if err != nil {
			t.Fatalf("INSERT: %v", err)
		}

		// Verify the record exists.
		coll := getColl(t, db, "users")
		records, err := coll.ListAll(context.Background())
		if err != nil {
			t.Fatalf("ListAll: %v", err)
		}
		if len(records) != 1 {
			t.Fatalf("expected 1 record, got %d", len(records))
		}
		if records[0].Metadata["name"] != "Alice" {
			t.Errorf("name: want Alice, got %v", records[0].Metadata["name"])
		}
	})

	t.Run("single non-id PK via SQL INSERT", func(t *testing.T) {
		db := openTempDB(t, "arb_pk_sql2")
		defer db.Close()

		_, err := db.Query(context.Background(),
			"CREATE TABLE users (email TEXT PRIMARY KEY, name TEXT)")
		if err != nil {
			t.Fatalf("CREATE TABLE: %v", err)
		}

		// Insert via SQL — key derived from email by the executor.
		_, err = db.Query(context.Background(),
			"INSERT INTO users (email, name) VALUES ('bob@example.com', 'Bob')")
		if err != nil {
			t.Fatalf("INSERT: %v", err)
		}

		// Verify the record exists and has the right metadata.
		records, err := getColl(t, db, "users").ListAll(context.Background())
		if err != nil {
			t.Fatalf("ListAll: %v", err)
		}
		if len(records) != 1 {
			t.Fatalf("expected 1 record, got %d", len(records))
		}
		if records[0].Metadata["name"] != "Bob" {
			t.Errorf("name: want Bob, got %v", records[0].Metadata["name"])
		}
	})

	t.Run("table-level PK via SQL", func(t *testing.T) {
		db := openTempDB(t, "arb_pk_table")
		defer db.Close()

		_, err := db.Query(context.Background(),
			"CREATE TABLE items (sku TEXT, quantity INT, PRIMARY KEY (sku))")
		if err != nil {
			t.Fatalf("CREATE TABLE: %v", err)
		}

		_, err = db.Query(context.Background(),
			"INSERT INTO items (sku, quantity) VALUES ('SKU-001', '42')")
		if err != nil {
			t.Fatalf("INSERT: %v", err)
		}

		records, err := getColl(t, db, "items").ListAll(context.Background())
		if err != nil {
			t.Fatalf("ListAll: %v", err)
		}
		if len(records) != 1 {
			t.Fatalf("expected 1 record, got %d", len(records))
		}
	})
}

// TestEncodeCompositePrimaryKey verifies key encoding format.
func TestEncodeCompositePrimaryKey(t *testing.T) {
	tests := []struct {
		name    string
		columns []string
		values  map[string]string
		want    string
		wantErr bool
	}{
		{
			name:    "single column",
			columns: []string{"email"},
			values:  map[string]string{"email": "alice@ex.com"},
			// Format: <nameLen>:<name><valueLen>:<value>|
			want: "__pk:5:email12:alice@ex.com|",
		},
		{
			name:    "missing value",
			columns: []string{"email"},
			values:  map[string]string{},
			wantErr: true,
		},
		{
			name:    "composite sorted",
			columns: []string{"last_name", "first_name"},
			values:  map[string]string{"first_name": "Alice", "last_name": "Smith"},
			want:    "__pk:10:first_name5:Alice|9:last_name5:Smith|",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := encodeCompositePrimaryKey(tt.columns, tt.values)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("got  %q\nwant %q", got, tt.want)
			}
		})
	}
}
