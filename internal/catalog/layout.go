package catalog

// This file defines the on-disk and in-memory binary layout of the catalog.
// It is designed to be cast directly over an mmap-backed []byte using unsafe.Pointer.

const (
	CatalogMagic   = 0x4341544C49425241 // "CATLIBRA"
	CatalogVersion = 1

	TypeInt    = 1
	TypeFloat  = 2
	TypeString = 3
	TypeVector = 4

	MetricL2      = 1
	MetricCosine  = 2
	MetricIP      = 3

	GraphLabelVertex = 1
	GraphLabelEdge   = 2
)

// Header is the 64-byte block at the start of the catalog file.
type Header struct {
	Magic         uint64
	Version       uint32
	Padding       uint32
	TablesCount   uint32
	TablesOffset  uint32
	VectorsCount  uint32
	VectorsOffset uint32
	GraphsCount   uint32
	GraphsOffset  uint32
	Reserved      [32]byte
}

// TableDef defines a relational table.
// Strings (like names) are FNV-1a hashes in the catalog to enforce stringless lookup.
type TableDef struct {
	OID           uint32
	Padding       uint32
	NameHash      uint64
	ColumnsOffset uint32
	ColumnsCount  uint32
}

// ColumnDef defines a column inside a table.
type ColumnDef struct {
	OID      uint32
	Padding  uint32
	NameHash uint64
	Type     uint16 // TypeInt, TypeFloat, etc.
	Flags    uint16 // Nullable, Primary Key
	Padding2 uint32
}

// VectorIndexDef defines a vector index and its metadata.
type VectorIndexDef struct {
	OID      uint32
	Padding  uint32
	NameHash uint64
	Dims     uint32
	Metric   uint8 // MetricL2, MetricCosine, etc.
	Padding2 [3]byte
}

// GraphLabelDef defines a vertex or edge in the property graph.
type GraphLabelDef struct {
	OID        uint32
	Padding    uint32
	NameHash   uint64
	LabelType  uint8 // GraphLabelVertex or GraphLabelEdge
	Padding2   [3]byte
	// For edges, Source and Target constraints. 0 means any.
	SourceOID  uint32 
	TargetOID  uint32
	Padding3   uint32
}
