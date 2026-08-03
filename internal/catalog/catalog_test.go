package catalog

import (
	"bytes"
	"encoding/binary"
	"testing"
	"unsafe"

	"github.com/xDarkicex/lexer/parser"
)

func buildMockCatalogBytes() []byte {
	// Allocate enough for Header + 1 Table + 2 Columns + 1 Vector Index + 1 Graph Label
	buf := new(bytes.Buffer)

	// Build Header
	hdr := Header{
		Magic:         CatalogMagic,
		Version:       CatalogVersion,
		TablesCount:   1,
		TablesOffset:  uint32(unsafe.Sizeof(Header{})),
		VectorsCount:  1,
		VectorsOffset: uint32(unsafe.Sizeof(Header{}) + unsafe.Sizeof(TableDef{}) + 2*unsafe.Sizeof(ColumnDef{})),
		GraphsCount:   1,
		GraphsOffset:  uint32(unsafe.Sizeof(Header{}) + unsafe.Sizeof(TableDef{}) + 2*unsafe.Sizeof(ColumnDef{}) + unsafe.Sizeof(VectorIndexDef{})),
	}
	_ = binary.Write(buf, binary.LittleEndian, hdr)

	// Build Table: "users"
	usersHash := hashIdentifier([]byte("users"), 0, 5)
	tbl := TableDef{
		OID:           100,
		NameHash:      usersHash,
		ColumnsOffset: uint32(unsafe.Sizeof(Header{}) + unsafe.Sizeof(TableDef{})),
		ColumnsCount:  2,
	}
	_ = binary.Write(buf, binary.LittleEndian, tbl)

	// Build Columns: "id", "name"
	idHash := hashIdentifier([]byte("id"), 0, 2)
	col1 := ColumnDef{
		OID:      200,
		NameHash: idHash,
		Type:     TypeInt,
		Flags:    1,
	}
	_ = binary.Write(buf, binary.LittleEndian, col1)

	nameHash := hashIdentifier([]byte("name"), 0, 4)
	col2 := ColumnDef{
		OID:      201,
		NameHash: nameHash,
		Type:     TypeString,
		Flags:    0,
	}
	_ = binary.Write(buf, binary.LittleEndian, col2)

	// Build Vector: "vec"
	vecHash := hashIdentifier([]byte("vec"), 0, 3)
	vec := VectorIndexDef{
		OID:      300,
		NameHash: vecHash,
		Dims:     1536,
		Metric:   MetricCosine,
	}
	_ = binary.Write(buf, binary.LittleEndian, vec)

	// Build Graph Label: "person"
	personHash := hashIdentifier([]byte("person"), 0, 6)
	graph := GraphLabelDef{
		OID:       400,
		NameHash:  personHash,
		LabelType: GraphLabelVertex,
	}
	_ = binary.Write(buf, binary.LittleEndian, graph)

	return buf.Bytes()
}

func TestCatalogLoad(t *testing.T) {
	data := buildMockCatalogBytes()
	cat, err := Load(data, nil)
	if err != nil {
		t.Fatalf("Failed to load catalog: %v", err)
	}

	usersHash := hashIdentifier([]byte("users"), 0, 5)
	tbl, err := cat.GetTable(usersHash)
	if err != nil || tbl.OID != 100 {
		t.Fatalf("Expected table OID 100, got %v", tbl)
	}

	idHash := hashIdentifier([]byte("id"), 0, 2)
	col, err := cat.GetColumn(tbl, idHash)
	if err != nil || col.OID != 200 {
		t.Fatalf("Expected column OID 200, got %v", col)
	}

	vecHash := hashIdentifier([]byte("vec"), 0, 3)
	vec, err := cat.GetVectorIndex(vecHash)
	if err != nil || vec.OID != 300 {
		t.Fatalf("Expected vector index OID 300, got %v", vec)
	}

	personHash := hashIdentifier([]byte("person"), 0, 6)
	graph, err := cat.GetGraphLabel(personHash)
	if err != nil || graph.OID != 400 {
		t.Fatalf("Expected graph label OID 400, got %v", graph)
	}
}

func TestBinder(t *testing.T) {
	data := buildMockCatalogBytes()
	cat, _ := Load(data, nil)

	// Simulate "SELECT id, name FROM users"
	src := []byte("SELECT id, name FROM users")
	doc := &parser.QueryDoc{
		TableExprs: []parser.TableExpr{
			{ID: 0, Start: 21, End: 26}, // "users"
		},
		Identifiers: []parser.Identifier{
			{ID: 0, Start: 7, End: 9},   // "id"
			{ID: 1, Start: 11, End: 15}, // "name"
		},
	}

	binder := NewBinder(cat, src)
	err := binder.Bind(doc)
	if err != nil {
		t.Fatalf("Bind failed: %v", err)
	}

	// Validate OIDs
	if doc.Identifiers[0].ColumnOID != 200 {
		t.Errorf("Expected 'id' to bind to ColumnOID 200, got %d", doc.Identifiers[0].ColumnOID)
	}
	if doc.Identifiers[1].ColumnOID != 201 {
		t.Errorf("Expected 'name' to bind to ColumnOID 201, got %d", doc.Identifiers[1].ColumnOID)
	}
	if doc.TableExprs[0].TableOID != 100 {
		t.Errorf("Expected 'users' to bind to TableOID 100, got %d", doc.TableExprs[0].TableOID)
	}
}

func BenchmarkBinder_Bind(b *testing.B) {
	data := buildMockCatalogBytes()
	cat, _ := Load(data, nil)

	src := []byte("SELECT id, name FROM users")
	doc := &parser.QueryDoc{
		TableExprs: []parser.TableExpr{
			{ID: 0, Start: 21, End: 26},
		},
		Identifiers: []parser.Identifier{
			{ID: 0, Start: 7, End: 9},
			{ID: 1, Start: 11, End: 15},
		},
	}

	binder := NewBinder(cat, src)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = binder.Bind(doc)
	}
}
