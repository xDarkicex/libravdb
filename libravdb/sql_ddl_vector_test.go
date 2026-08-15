package libravdb

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// TestSQL_DDLCreateTableVector verifies that CREATE TABLE with VECTOR(n)
// preserves the dimension through the full parse → plan → execute → catalog
// pipeline.
func TestSQL_DDLCreateTableVector(t *testing.T) {
	tests := []struct {
		name        string
		ddl         string
		tableName   string
		wantDim     int
		wantErr     bool
		errContains string
	}{
		{
			name:      "VECTOR(3)",
			ddl:       "CREATE TABLE test_vec3 (v VECTOR(3))",
			tableName: "test_vec3",
			wantDim:   3,
		},
		{
			name:      "VECTOR(768)",
			ddl:       "CREATE TABLE test_vec768 (v VECTOR(768))",
			tableName: "test_vec768",
			wantDim:   768,
		},
		{
			name:      "VECTOR(1536)",
			ddl:       "CREATE TABLE test_vec1536 (v VECTOR(1536))",
			tableName: "test_vec1536",
			wantDim:   1536,
		},
		{
			name:      "VECTOR(4096)",
			ddl:       "CREATE TABLE test_vec4096 (v VECTOR(4096))",
			tableName: "test_vec4096",
			wantDim:   4096,
		},
		{
			name:      "case insensitive type vector(128)",
			ddl:       "CREATE TABLE test_veclower (v vector(128))",
			tableName: "test_veclower",
			wantDim:   128,
		},
		{
			name:      "vector column with metadata columns",
			ddl:       "CREATE TABLE test_vecmeta (id TEXT, v VECTOR(64), name TEXT)",
			tableName: "test_vecmeta",
			wantDim:   64,
		},
		{
			name:        "VECTOR(0) rejected",
			ddl:         "CREATE TABLE test_vec0 (v VECTOR(0))",
			tableName:   "test_vec0",
			wantErr:     true,
			errContains: "positive",
		},
		{
			name:        "bare VECTOR rejected",
			ddl:         "CREATE TABLE test_bare (v VECTOR)",
			tableName:   "test_bare",
			wantErr:     true,
			errContains: "requires a dimension",
		},
		{
			name:        "multiple vector columns rejected",
			ddl:         "CREATE TABLE test_multi (a VECTOR(3), b VECTOR(768))",
			tableName:   "test_multi",
			wantErr:     true,
			errContains: "only one vector column",
		},
		{
			name:      "metadata-only table (no vector)",
			ddl:       "CREATE TABLE test_meta (id TEXT, score INT)",
			tableName: "test_meta",
			wantDim:   0, // metadata-only, dimension=0
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := ":memory:" + tt.tableName
			db, err := Open(WithStoragePath(path), WithMetrics(false))
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			defer os.RemoveAll(path)
			defer db.Close()

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			results, err := db.Query(ctx, tt.ddl)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.errContains)
				}
				if tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
					t.Fatalf("expected error containing %q, got %q", tt.errContains, err.Error())
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if results == nil {
				t.Fatal("db.Query returned nil results")
			}

			// Verify the collection was created with the correct dimension.
			coll, err := db.GetCollection(tt.tableName)
			if err != nil {
				t.Fatalf("GetCollection(%q): %v", tt.tableName, err)
			}
			gotDim := coll.Dimension()
			if gotDim != tt.wantDim {
				t.Errorf("dimension: want %d, got %d", tt.wantDim, gotDim)
			}

			// Verify insert-time dimension validation works with the correct
			// dimension from the DDL.
			if tt.wantDim > 0 {
				correctVec := make([]float32, tt.wantDim)
				for i := range correctVec {
					correctVec[i] = float32(i) / float32(tt.wantDim)
				}
				if err := coll.Insert(ctx, "test-1", correctVec, nil); err != nil {
					t.Errorf("insert with correct dimension %d: %v", tt.wantDim, err)
				}

				// Verify wrong dimension is rejected.
				wrongDim := tt.wantDim - 1
				if wrongDim < 1 {
					wrongDim = tt.wantDim + 1
				}
				wrongVec := make([]float32, wrongDim)
				err = coll.Insert(ctx, "test-2", wrongVec, nil)
				if err == nil {
					t.Errorf("insert with wrong dimension %d should have failed", wrongDim)
				}
			}
		})
	}
}
