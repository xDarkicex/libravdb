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
	// TypeBigInt is a signed 64-bit integer. Keep this value append-only:
	// catalog type numbers are persisted on disk and the older values are
	// already in use by existing catalogs.
	TypeBigInt = 5
	// PostgreSQL-specific types used by the built-in pg_catalog relations.
	// These are append-only for the same on-disk compatibility reason.
	TypeOID      = 6
	TypeName     = 7
	TypeChar     = 8
	TypeSmallInt = 9
	TypeBool     = 10
	TypeFloat4   = 11
	// JSON and JSONB are append-only catalog types for PostgreSQL-compatible
	// metadata columns.
	TypeJSON  = 12
	TypeJSONB = 13
	// TypeUUID is used for expression result metadata and UUID-aware catalog
	// columns. It is append-only so existing catalog type numbers remain stable.
	TypeUUID = 14
	// TypeTimestamp is used for expression results carrying a time.Time.
	TypeTimestamp = 15

	MetricL2     = 1
	MetricCosine = 2
	MetricIP     = 3

	GraphLabelVertex = 1
	GraphLabelEdge   = 2

	// Column constraint flags — stored in ColumnDef.Flags.
	ColFlagNotNull    uint16 = 1 << 0
	ColFlagPrimaryKey uint16 = 1 << 1
	ColFlagUnique     uint16 = 1 << 2
	ColFlagHasDefault uint16 = 1 << 3

	// OnDeleteAction values for ForeignKeyDef.
	// Must match parser.OnDeleteAction ordering.
	OnDeleteNoAction   uint8 = 0
	OnDeleteCascade    uint8 = 1
	OnDeleteRestrict   uint8 = 2
	OnDeleteSetNull    uint8 = 3
	OnDeleteSetDefault uint8 = 4
)

// Header is the 72-byte block at the start of the catalog file.
type Header struct {
	Magic          uint64
	Version        uint32
	Padding        uint32
	TablesCount    uint32
	TablesOffset   uint32
	VectorsCount   uint32
	VectorsOffset  uint32
	GraphsCount    uint32
	GraphsOffset   uint32
	FKsCount       uint32 // foreign key constraint count
	FKsOffset      uint32 // byte offset to ForeignKeyDef array
	ChecksCount    uint32 // CHECK constraint count
	ChecksOffset   uint32 // byte offset to CheckConstraintDef array
	DefaultsCount  uint32 // DEFAULT value count
	DefaultsOffset uint32 // byte offset to DefaultValueDef array
	// The final eight bytes were reserved in catalog v2. They now carry the
	// count/offset for durable JSON expression-index definitions; zero values
	// preserve compatibility with older v2 catalogs.
	JSONIndexesCount  uint32
	JSONIndexesOffset uint32
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
	Flags    uint16 // ColFlag*
	NameOff  uint16 // byte offset of column name in trailing data section
	NameLen  uint16 // byte length of column name
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
// Name hashes are case-insensitive FNV-1a.
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

// CheckConstraintDef defines a CHECK constraint. The expression text is stored
// in the trailing variable-length data section at offset ExprOff with length ExprLen.
type CheckConstraintDef struct {
	OID       uint32
	Padding   uint32
	NameHash  uint64 // FNV-1a hash of constraint name
	TableHash uint64 // FNV-1a hash of owning table
	ColHash   uint64 // FNV-1a hash of column (0 for table-level CHECK)
	ExprOff   uint32 // byte offset into trailing data section
	ExprLen   uint32 // byte length of expression text
	Padding2  [4]byte
}

// DefaultValueDef defines a column DEFAULT value. The value text is stored in
// the trailing variable-length data section at offset DataOff with length DataLen.
type DefaultValueDef struct {
	OID       uint32
	Padding   uint32
	TableHash uint64 // FNV-1a hash of owning table
	ColHash   uint64 // FNV-1a hash of column
	DataOff   uint32 // byte offset into trailing data section
	DataLen   uint32 // byte length of default value text
	Padding2  [4]byte
}

// JSONIndexDef describes a durable inverted index over one JSON/JSONB path.
// Path text is stored in the catalog trailing-data section at PathOff/PathLen.
type JSONIndexDef struct {
	OID        uint32
	Padding    uint32
	NameHash   uint64
	TableHash  uint64
	ColumnHash uint64
	NameOff    uint32
	NameLen    uint32
	PathOff    uint32
	PathLen    uint32
	TextResult uint8 // 1 for #>>, 0 for #>
	Padding2   [3]byte
}
