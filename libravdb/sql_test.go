package libravdb

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"testing"
	"time"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/catalog"
)

func TestSQLExecutionE2E(t *testing.T) {
	// 1. Setup a real embedded DB
	path := ":memory:sql_e2e"
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("Failed to open DB: %v", err)
	}
	defer os.RemoveAll(path)
	defer db.Close()

	// 2. Create a collection (which mimics a Table in the catalog)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	_, err = db.CreateCollection(ctx, "test_table", WithDimension(3))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Wait, we need a mock catalog data block since singlefile.Engine doesn't automatically 
	// generate the MMap catalog section yet (that's Phase 3 integration into singlefile).
	// But we can overwrite db.catalog manually for the test.
	
	// Manually inject a mock catalog that binder can resolve "test_table" against.
	mockCatalogBytes := buildMockCatalogBytes("test_table")
	mockCat, err := catalog.Load(mockCatalogBytes, db.quantRegistry)
	if err != nil {
		t.Fatalf("Failed to load mock catalog: %v", err)
	}
	
	db.mu.Lock()
	db.catalog = mockCat
	db.mu.Unlock()

	// 3. Insert a mock vector to force query execution to evaluate distance/thresholds
	col, err := db.GetCollection("test_table")
	if err != nil {
		t.Fatalf("Failed to get collection: %v", err)
	}
	err = col.Insert(ctx, "item-1", []float32{0.1, 0.2, 0.3}, nil)
	if err != nil {
		t.Fatalf("Failed to insert item-1: %v", err)
	}
	
	// Insert a second vector that is far away (cosine similarity < 0)
	err = col.Insert(ctx, "item-2", []float32{-0.9, -0.9, -0.9}, nil)
	if err != nil {
		t.Fatalf("Failed to insert item-2: %v", err)
	}

	// Wait for async index to process the insert (flush)
	// (Since we are hitting query execution directly, the WAL/MMap may need it flushed)
	col.FlushIndex(ctx)

	// 4. Issue a SQL query
	sql := `SELECT id FROM test_table WHERE SIMILARITY(vector, '[0.1, 0.2, 0.3]') > 0.5 LIMIT 10`
	
	results, err := db.Query(ctx, sql)
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}

	// 4. Verify we got a SearchResults object
	if results == nil {
		t.Fatalf("Expected results, got nil")
	}
	
	// We inserted one identical vector, so it should have 1.0 similarity (> 0.5).
	if len(results.Results) != 1 {
		t.Errorf("Expected 1 hit, got %d", len(results.Results))
	}
}

// hashIdentifier computes a case-insensitive FNV-1a hash
func hashIdentifier(src string) uint64 {
	var hash uint64 = 14695981039346656037
	for i := 0; i < len(src); i++ {
		c := src[i]
		if c >= 'A' && c <= 'Z' {
			c += 32
		}
		hash ^= uint64(c)
		hash *= 1099511628211
	}
	return hash
}

func buildMockCatalogBytes(tableName string) []byte {
	buf := new(bytes.Buffer)

	hdr := catalog.Header{
		Magic:         catalog.CatalogMagic,
		Version:       catalog.CatalogVersion,
		TablesCount:   1,
		TablesOffset:  uint32(unsafe.Sizeof(catalog.Header{})),
		VectorsCount:  1,
		VectorsOffset: uint32(unsafe.Sizeof(catalog.Header{}) + unsafe.Sizeof(catalog.TableDef{}) + 2*unsafe.Sizeof(catalog.ColumnDef{})),
	}
	_ = binary.Write(buf, binary.LittleEndian, hdr)

	tbl := catalog.TableDef{
		OID:           100,
		NameHash:      hashIdentifier(tableName),
		ColumnsOffset: uint32(unsafe.Sizeof(catalog.Header{}) + unsafe.Sizeof(catalog.TableDef{})),
		ColumnsCount:  2,
	}
	_ = binary.Write(buf, binary.LittleEndian, tbl)

	col1 := catalog.ColumnDef{
		OID:      200,
		NameHash: hashIdentifier("id"),
		Type:     catalog.TypeInt,
	}
	_ = binary.Write(buf, binary.LittleEndian, col1)

	col2 := catalog.ColumnDef{
		OID:      201,
		NameHash: hashIdentifier("vector"),
		Type:     catalog.TypeString,
	}
	_ = binary.Write(buf, binary.LittleEndian, col2)

	vec := catalog.VectorIndexDef{
		OID:      300,
		NameHash: hashIdentifier("vector"),
		Dims:     3,
		Metric:   catalog.MetricCosine,
	}
	_ = binary.Write(buf, binary.LittleEndian, vec)

	return buf.Bytes()
}
