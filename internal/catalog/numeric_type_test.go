package catalog

import "testing"

func TestBigIntColumnRoundTripsThroughCatalog(t *testing.T) {
	b := NewBuilder()
	b.AddTable("numbers", []ColumnInfo{{Name: "value", Type: TypeBigInt}})
	cat, err := Load(b.Build(), nil)
	if err != nil {
		t.Fatalf("load catalog: %v", err)
	}
	table, err := cat.GetTable(HashIdentifier("numbers"))
	if err != nil {
		t.Fatalf("resolve numbers table: %v", err)
	}
	col, err := cat.GetColumn(table, HashIdentifier("value"))
	if err != nil {
		t.Fatalf("resolve numbers.value: %v", err)
	}
	if col.Type != TypeBigInt {
		t.Fatalf("round-tripped type = %d, want TypeBigInt (%d)", col.Type, TypeBigInt)
	}
}
