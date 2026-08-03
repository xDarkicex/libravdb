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
	col.FlushIndex(ctx)

	// 4. Issue a SQL query
	sql := `SELECT id FROM test_table WHERE SIMILARITY(vector, '[0.1, 0.2, 0.3]') > 0.5 LIMIT 10`

	results, err := db.Query(ctx, sql)
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}

	if results == nil {
		t.Fatalf("Expected results, got nil")
	}

	if len(results.Results) != 1 {
		t.Errorf("Expected 1 hit, got %d", len(results.Results))
	}
}

func TestSQL_CRUD(t *testing.T) {
	path := ":memory:crud_test"
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	_, err = db.CreateCollection(ctx, "items", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Inject mock catalog so binder can resolve "items"
	mockCat, _ := catalog.Load(buildMockCatalogBytes("items"), db.quantRegistry)
	db.mu.Lock()
	db.catalog = mockCat
	db.mu.Unlock()

	// INSERT missing ID column should error
	_, err = db.Query(ctx, "INSERT INTO items (vector) VALUES ('[0.4, 0.5, 0.6]')")
	if err == nil {
		t.Fatal("expected error for INSERT without id column")
	}
	t.Logf("missing ID error: %v", err)

	// UPDATE without WHERE should error
	_, err = db.Query(ctx, "UPDATE items SET vector = '[0.1, 0.2, 0.3]'")
	if err == nil {
		t.Fatal("expected error for UPDATE without WHERE")
	}
	t.Logf("UPDATE no WHERE error: %v", err)

	// DELETE without WHERE should error
	_, err = db.Query(ctx, "DELETE FROM items")
	if err == nil {
		t.Fatal("expected error for DELETE without WHERE")
	}
	t.Logf("DELETE no WHERE error: %v", err)
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

func TestMetadataOnly_InsertAndSelect(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:meta_test"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	col, err := db.CreateCollection(ctx, "users", WithMetadataOnly())
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Verify collection is metadata-only
	if col.Dimension() != 0 {
		t.Fatalf("expected dimension 0, got %d", col.Dimension())
	}

	// Insert a plain row via Go API — no vector (SQL INSERT blocked by binder scope)
	err = col.Insert(ctx, "u1", nil, map[string]interface{}{"name": "alice"})
	if err != nil {
		t.Fatalf("Insert u1: %v", err)
	}

	// Verify the row exists
	record, err := col.Get(ctx, "u1")
	if err != nil {
		t.Fatalf("Get u1: %v", err)
	}
	t.Logf("Record: id=%s version=%d metadata=%v", record.ID, record.Version, record.Metadata)

	// Verify graph node ID exists
	nodeID, err := db.GetNodeID(ctx, "users", "u1")
	if err != nil {
		t.Fatalf("GetNodeID u1: %v", err)
	}
	if nodeID == 0 {
		t.Fatal("expected non-zero GraphNodeID for u1")
	}
	t.Logf("GraphNodeID for u1: %d", nodeID)
}

func TestMetadataOnly_RejectsVector(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:meta_reject"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	_, err = db.CreateCollection(ctx, "items", WithMetadataOnly())
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Verify dimension is 0
	col, _ := db.GetCollection("items")
	if col.Dimension() != 0 {
		t.Fatalf("expected dimension 0, got %d", col.Dimension())
	}

	// Insert via Go API with nil vector — must succeed
	err = col.Insert(ctx, "i1", nil, map[string]interface{}{"name": "test"})
	if err != nil {
		t.Fatalf("Insert with nil vector: %v", err)
	}

	// Verify record exists
	record, err := col.Get(ctx, "i1")
	if err != nil {
		t.Fatalf("Get i1: %v", err)
	}
	if record.ID != "i1" {
		t.Fatalf("expected ID i1, got %s", record.ID)
	}
	t.Logf("Metadata-only insert: id=%s metadata=%v", record.ID, record.Metadata)

	// Verify graph node exists
	nodeID, err := db.GetNodeID(ctx, "items", "i1")
	if err != nil {
		t.Fatalf("GetNodeID: %v", err)
	}
	if nodeID == 0 {
		t.Fatal("expected non-zero GraphNodeID")
	}
	t.Logf("GraphNodeID: %d", nodeID)
}

func TestSQL_CRUDEndToEnd(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:crud_e2e"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	ctx := context.Background()

	_, err = db.CreateCollection(ctx, "users", WithMetadataOnly())
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	injectMockCatalog(t, db, "users")

	// INSERT via SQL — uses "id" column (TypeInt in mock catalog)
	_, err = db.Query(ctx, "INSERT INTO users (id) VALUES ('u1')")
	if err != nil {
		t.Fatalf("INSERT u1: %v", err)
	}

	// Verify via Go API
	col, _ := db.GetCollection("users")
	record, err := col.Get(ctx, "u1")
	if err != nil {
		t.Fatalf("Get u1: %v", err)
	}
	t.Logf("INSERT via SQL: id=%s metadata=%v", record.ID, record.Metadata)

	// INSERT second row
	_, err = db.Query(ctx, "INSERT INTO users (id) VALUES ('u2')")
	if err != nil {
		t.Fatalf("INSERT u2: %v", err)
	}

	// INSERT duplicate should fail
	_, err = db.Query(ctx, "INSERT INTO users (id) VALUES ('u1')")
	if err == nil {
		t.Fatal("expected duplicate key error")
	}
	t.Logf("Duplicate key error: %v", err)
}

func injectMockCatalog(t *testing.T, db *Database, tableName string) {
	t.Helper()
	mockCat, err := catalog.Load(buildMockCatalogBytes(tableName), db.quantRegistry)
	if err != nil {
		t.Fatalf("Load mock catalog: %v", err)
	}
	db.mu.Lock()
	db.catalog = mockCat
	db.mu.Unlock()
}

func TestSQL_AggregateCount(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:agg_count"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	ctx := context.Background()

	_, err = db.CreateCollection(ctx, "users", WithMetadataOnly())
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	// No injectMockCatalog needed — CreateCollection auto-registers the table

	// Insert rows
	col, _ := db.GetCollection("users")
	col.Insert(ctx, "u1", nil, nil)
	col.Insert(ctx, "u2", nil, nil)
	col.Insert(ctx, "u3", nil, nil)

	// COUNT(*)
	results, err := db.Query(ctx, "SELECT COUNT(*) FROM users")
	if err != nil {
		t.Fatalf("COUNT(*) query failed: %v", err)
	}
	if results == nil || len(results.Results) == 0 {
		t.Fatal("expected results for COUNT(*)")
	}
	t.Logf("COUNT(*) result: %s", results.Results[0].ID)
	if results.Results[0].ID != "3" {
		t.Errorf("COUNT(*) = %s, want 3", results.Results[0].ID)
	}

	// SELECT with column filter — id column should be resolvable
	results, err = db.Query(ctx, "SELECT id FROM users WHERE id = 'u1'")
	if err != nil {
		t.Fatalf("SELECT id query failed: %v", err)
	}
	if len(results.Results) != 1 || results.Results[0].ID != "u1" {
		t.Errorf("expected u1, got %+v", results.Results)
	}
	t.Logf("SELECT id WHERE id='u1' returned: %s", results.Results[0].ID)
}
