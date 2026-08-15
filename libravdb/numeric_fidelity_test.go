package libravdb

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/catalog"
)

func TestSQLBigIntPreservesDistinctFieldType(t *testing.T) {
	got, ok := sqlTypeToFieldType("BIGINT")
	if !ok {
		t.Fatal("BIGINT should be a supported SQL field type")
	}
	if got != BigIntField {
		t.Fatalf("BIGINT mapped to %v, want BigIntField", got)
	}
	if intField, _ := sqlTypeToFieldType("INTEGER"); intField != IntField {
		t.Fatalf("INTEGER mapped to %v, want IntField", intField)
	}
	if metadataFieldToCatalogType(BigIntField) != catalog.TypeBigInt {
		t.Fatalf("BigIntField did not map to catalog TypeBigInt")
	}
}

func TestGraphNodesCatalogIDIsBigInt(t *testing.T) {
	table, ok := catalog.ResolveSystemTable("GRAPH_NODES")
	if !ok {
		t.Fatal("GRAPH_NODES system table not registered")
	}
	col, err := catalog.ResolveSystemColumn(table.OID, catalog.HashIdentifier("id"))
	if err != nil {
		t.Fatalf("resolve GRAPH_NODES.id: %v", err)
	}
	if col.Type != catalog.TypeBigInt {
		t.Fatalf("GRAPH_NODES.id catalog type = %d, want TypeBigInt (%d)", col.Type, catalog.TypeBigInt)
	}
}

func TestSQLDeclaredBigIntIDCatalogType(t *testing.T) {
	db := openTempDB(t, "numeric_bigint_id")
	defer db.Close()

	if _, err := db.Query(context.Background(), "CREATE TABLE numbers (id BIGINT PRIMARY KEY, value BIGINT)"); err != nil {
		t.Fatalf("create BIGINT table: %v", err)
	}
	db.mu.RLock()
	cat := db.catalog
	db.mu.RUnlock()
	table, err := cat.GetTable(catalog.HashIdentifier("numbers"))
	if err != nil {
		t.Fatalf("resolve numbers table: %v", err)
	}
	col, err := cat.GetColumn(table, catalog.HashIdentifier("id"))
	if err != nil {
		t.Fatalf("resolve numbers.id: %v", err)
	}
	if col.Type != catalog.TypeBigInt {
		t.Fatalf("declared BIGINT id catalog type = %d, want TypeBigInt (%d)", col.Type, catalog.TypeBigInt)
	}
}
