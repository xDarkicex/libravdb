package catalog

// This file defines the on-disk and in-memory binary layout of the catalog.
// It is designed to be cast directly over an mmap-backed []byte using unsafe.Pointer.

const (
	CatalogMagic   = 0x4341544C49425241 // "CATLIBRA"
	CatalogVersion = 2                  // v2: adds FK section + column flags

	TypeInt    = 1
	TypeFloat  = 2
	TypeString = 3
	TypeVector = 4

	MetricL2     = 1
	MetricCosine = 2
	MetricIP     = 3

	GraphLabelVertex = 1
	GraphLabelEdge   = 2

	// Column constraint flags — stored in ColumnDef.Flags.
	ColFlagNotNull    uint16 = 1 << 0
	ColFlagPrimaryKey uint16 = 1 << 1
	ColFlagUnique     uint16 = 1 << 2

	// OnDeleteAction values for ForeignKeyDef.
	// Must match parser.OnDeleteAction ordering.
	OnDeleteNoAction uint8 = 0
	OnDeleteCascade  uint8 = 1
	OnDeleteRestrict uint8 = 2
)

// Header is the 72-byte block at the start of the catalog file.
// v2 repurposes the first 8 bytes of Reserved for FK metadata.
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
	FKsCount      uint32   // v2: foreign key constraint count
	FKsOffset     uint32   // v2: byte offset to ForeignKeyDef array
	Reserved      [24]byte // v2: shrunk from 32 to make room for FK fields
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
	OID       uint32
	Padding   uint32
	NameHash  uint64
	LabelType uint8 // GraphLabelVertex or GraphLabelEdge
	Padding2  [3]byte
	// For edges, Source and Target constraints. 0 means any.
	SourceOID uint32
	TargetOID uint32
	Padding3  uint32
}

// ForeignKeyDef defines a foreign key constraint between two tables.
// v2 catalog only. Name hashes are case-insensitive FNV-1a.
type ForeignKeyDef struct {
	OID             uint32
	Padding         uint32
	NameHash        uint64 // FNV-1a hash of constraint name (may be auto-generated)
	SourceTableHash uint64 // FNV-1a hash of source (child) table name
	TargetTableHash uint64 // FNV-1a hash of target (parent) table name
	SourceColHash   uint64 // FNV-1a hash of source column name
	TargetColHash   uint64 // FNV-1a hash of target column name
	OnDelete        uint8  // OnDeleteAction constant
	OnUpdate        uint8  // OnDeleteAction constant (reused enum)
	Padding2        [6]byte
}
