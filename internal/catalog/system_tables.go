package catalog

// System table registry — hardcoded table/column definitions for pg_catalog tables
// that the SQL engine handles natively.
//
// Design rationale: System tables are NOT registered in the catalog binary blob.
// Catalog NewBuilderFrom can't recover table names from hashes (see builder.go:58),
// system tables shouldn't persist to disk, and DDL-generated catalog pages shouldn't
// need to know about them. Instead, the binder resolves them via this registry as a
// fallback when GetTable returns ErrTableNotFound.
//
// System tables use reserved OIDs 1-99. User tables start at 100.

const (
	SystemTableOIDMax = 99 // reserved OID ceiling; user tables start at 100

	// System table OIDs
	sysOIDPgClass      = 1
	sysOIDGraphNodes   = 2
	sysOIDPgAttribute  = 3
	sysOIDPgType       = 4
	sysOIDPgNamespace  = 5
	sysOIDPgRange      = 6
	sysOIDPgProc       = 7
	sysOIDPgConstraint = 8
	sysOIDPgIndex      = 9
	sysOIDPgAttrdef    = 10
	sysOIDPgIndexes    = 11

	// pg_class column OIDs
	sysColOIDOID          = 10
	sysColOIDRelname      = 11
	sysColOIDRelnamespace = 12
	sysColOIDRelkind      = 13
	sysColOIDReltuples    = 14

	// pg_attribute column OIDs
	sysColOIDAttrelid   = 30
	sysColOIDAttname    = 31
	sysColOIDAtttypid   = 32
	sysColOIDAttnum     = 33
	sysColOIDAttnotnull = 34

	// pg_type column OIDs
	sysColOIDTypOID  = 40
	sysColOIDTypname = 41
	sysColOIDTyplen  = 42

	// pg_namespace column OIDs
	sysColOIDNspOID   = 50
	sysColOIDNspname  = 51
	sysColOIDNspowner = 52

	// pg_indexes view column OIDs
	sysColOIDIdxSchema     = 60
	sysColOIDIdxTable      = 61
	sysColOIDIdxName       = 62
	sysColOIDIdxTablespace = 63
	sysColOIDIdxDef        = 64

	// GRAPH_NODES column OIDs
	sysColOIDGNID         = 20
	sysColOIDGNCollection = 21
	sysColOIDGNRecordID   = 22
)

// SystemTableInfo holds the pre-built catalog definitions for a system table.
type SystemTableInfo struct {
	Table   TableDef
	Columns map[uint64]*ColumnDef // keyed by FNV-1a hash of column name
}

// systemTables maps system table OID → definition.
// Add entries here for each system table the engine natively supports.
var systemTables = func() map[uint32]*SystemTableInfo {
	m := make(map[uint32]*SystemTableInfo)

	// pg_class: one row per real user table
	pgClass := &SystemTableInfo{
		Table: TableDef{
			OID:          sysOIDPgClass,
			NameHash:     hashString("pg_class"),
			ColumnsCount: 5,
		},
		Columns: make(map[uint64]*ColumnDef),
	}
	pgClass.Columns[hashString("oid")] = &ColumnDef{
		OID: sysColOIDOID, NameHash: hashString("oid"), Type: TypeOID,
	}
	pgClass.Columns[hashString("relname")] = &ColumnDef{
		OID: sysColOIDRelname, NameHash: hashString("relname"), Type: TypeName,
	}
	pgClass.Columns[hashString("relnamespace")] = &ColumnDef{
		OID: sysColOIDRelnamespace, NameHash: hashString("relnamespace"), Type: TypeOID,
	}
	pgClass.Columns[hashString("relkind")] = &ColumnDef{
		OID: sysColOIDRelkind, NameHash: hashString("relkind"), Type: TypeChar,
	}
	pgClass.Columns[hashString("reltuples")] = &ColumnDef{
		OID: sysColOIDReltuples, NameHash: hashString("reltuples"), Type: TypeFloat4,
	}
	m[sysOIDPgClass] = pgClass

	// GRAPH_NODES: virtual table exposing graph node identity.
	// Node IDs are durable 64-bit graph identities and therefore use TypeBigInt.
	graphNodes := &SystemTableInfo{
		Table: TableDef{
			OID:          sysOIDGraphNodes,
			NameHash:     hashString("GRAPH_NODES"),
			ColumnsCount: 3,
		},
		Columns: make(map[uint64]*ColumnDef),
	}
	graphNodes.Columns[hashString("id")] = &ColumnDef{
		OID: sysColOIDGNID, NameHash: hashString("id"), Type: TypeBigInt,
	}
	graphNodes.Columns[hashString("collection")] = &ColumnDef{
		OID: sysColOIDGNCollection, NameHash: hashString("collection"), Type: TypeString,
	}
	graphNodes.Columns[hashString("record_id")] = &ColumnDef{
		OID: sysColOIDGNRecordID, NameHash: hashString("record_id"), Type: TypeString,
	}
	m[sysOIDGraphNodes] = graphNodes

	// pg_attribute: one row per column across all user tables
	pgAttr := &SystemTableInfo{
		Table: TableDef{
			OID:          sysOIDPgAttribute,
			NameHash:     hashString("pg_attribute"),
			ColumnsCount: 5,
		},
		Columns: make(map[uint64]*ColumnDef),
	}
	pgAttr.Columns[hashString("attrelid")] = &ColumnDef{
		OID: sysColOIDAttrelid, NameHash: hashString("attrelid"), Type: TypeOID,
	}
	pgAttr.Columns[hashString("attname")] = &ColumnDef{
		OID: sysColOIDAttname, NameHash: hashString("attname"), Type: TypeName,
	}
	pgAttr.Columns[hashString("atttypid")] = &ColumnDef{
		OID: sysColOIDAtttypid, NameHash: hashString("atttypid"), Type: TypeOID,
	}
	pgAttr.Columns[hashString("attnum")] = &ColumnDef{
		OID: sysColOIDAttnum, NameHash: hashString("attnum"), Type: TypeSmallInt,
	}
	pgAttr.Columns[hashString("attnotnull")] = &ColumnDef{
		OID: sysColOIDAttnotnull, NameHash: hashString("attnotnull"), Type: TypeBool,
	}
	m[sysOIDPgAttribute] = pgAttr

	// pg_type: known PostgreSQL type mappings
	pgType := &SystemTableInfo{
		Table: TableDef{
			OID:          sysOIDPgType,
			NameHash:     hashString("pg_type"),
			ColumnsCount: 3,
		},
		Columns: make(map[uint64]*ColumnDef),
	}
	pgType.Columns[hashString("oid")] = &ColumnDef{
		OID: sysColOIDTypOID, NameHash: hashString("oid"), Type: TypeOID,
	}
	pgType.Columns[hashString("typname")] = &ColumnDef{
		OID: sysColOIDTypname, NameHash: hashString("typname"), Type: TypeName,
	}
	pgType.Columns[hashString("typlen")] = &ColumnDef{
		OID: sysColOIDTyplen, NameHash: hashString("typlen"), Type: TypeSmallInt,
	}
	m[sysOIDPgType] = pgType

	// pg_namespace: known schemas
	pgNs := &SystemTableInfo{
		Table: TableDef{
			OID:          sysOIDPgNamespace,
			NameHash:     hashString("pg_namespace"),
			ColumnsCount: 3,
		},
		Columns: make(map[uint64]*ColumnDef),
	}
	pgNs.Columns[hashString("oid")] = &ColumnDef{
		OID: sysColOIDNspOID, NameHash: hashString("oid"), Type: TypeOID,
	}
	pgNs.Columns[hashString("nspname")] = &ColumnDef{
		OID: sysColOIDNspname, NameHash: hashString("nspname"), Type: TypeName,
	}
	pgNs.Columns[hashString("nspowner")] = &ColumnDef{
		OID: sysColOIDNspowner, NameHash: hashString("nspowner"), Type: TypeOID,
	}
	m[sysOIDPgNamespace] = pgNs

	// These relations are intentionally present even when empty. Python
	// drivers probe pg_range/pg_proc during type setup, while SQLAlchemy uses
	// pg_constraint/pg_index/pg_attrdef during reflection. Their live rows are
	// populated by the catalog compatibility layer as support is needed; an
	// empty relation is still the correct answer for unsupported PostgreSQL
	// features such as ranges, procedures, and generated defaults.
	for oid, name := range map[uint32]string{
		sysOIDPgRange:      "pg_range",
		sysOIDPgProc:       "pg_proc",
		sysOIDPgConstraint: "pg_constraint",
		sysOIDPgIndex:      "pg_index",
		sysOIDPgAttrdef:    "pg_attrdef",
	} {
		m[oid] = &SystemTableInfo{Table: TableDef{OID: oid, NameHash: hashString(name)}, Columns: make(map[uint64]*ColumnDef)}
	}

	// pg_indexes is a read-only view over durable SQL index declarations. Its
	// rows are materialized from live collection configuration by the SQL
	// executor rather than persisted as a separate catalog relation.
	pgIndexes := &SystemTableInfo{
		Table: TableDef{
			OID:          sysOIDPgIndexes,
			NameHash:     hashString("pg_indexes"),
			ColumnsCount: 5,
		},
		Columns: make(map[uint64]*ColumnDef),
	}
	pgIndexes.Columns[hashString("schemaname")] = &ColumnDef{
		OID: sysColOIDIdxSchema, NameHash: hashString("schemaname"), Type: TypeName,
	}
	pgIndexes.Columns[hashString("tablename")] = &ColumnDef{
		OID: sysColOIDIdxTable, NameHash: hashString("tablename"), Type: TypeName,
	}
	pgIndexes.Columns[hashString("indexname")] = &ColumnDef{
		OID: sysColOIDIdxName, NameHash: hashString("indexname"), Type: TypeName,
	}
	pgIndexes.Columns[hashString("tablespace")] = &ColumnDef{
		OID: sysColOIDIdxTablespace, NameHash: hashString("tablespace"), Type: TypeName,
	}
	pgIndexes.Columns[hashString("indexdef")] = &ColumnDef{
		OID: sysColOIDIdxDef, NameHash: hashString("indexdef"), Type: TypeString,
	}
	m[sysOIDPgIndexes] = pgIndexes

	return m
}()

// systemTableByName maps table name hash → OID for fast lookup during binder
// table resolution (when the catalog doesn't have the table).
var systemTableByName = func() map[uint64]uint32 {
	m := make(map[uint64]uint32, len(systemTables))
	m[hashString("pg_class")] = sysOIDPgClass
	m[hashString("pg_attribute")] = sysOIDPgAttribute
	m[hashString("pg_type")] = sysOIDPgType
	m[hashString("pg_namespace")] = sysOIDPgNamespace
	m[hashString("pg_range")] = sysOIDPgRange
	m[hashString("pg_proc")] = sysOIDPgProc
	m[hashString("pg_constraint")] = sysOIDPgConstraint
	m[hashString("pg_index")] = sysOIDPgIndex
	m[hashString("pg_attrdef")] = sysOIDPgAttrdef
	m[hashString("pg_indexes")] = sysOIDPgIndexes
	m[hashString("graph_nodes")] = sysOIDGraphNodes
	return m
}()

// IsSystemTableOID reports whether oid falls in the reserved system table range.
func IsSystemTableOID(oid uint32) bool {
	return oid > 0 && oid <= SystemTableOIDMax
}

// ResolveSystemTable looks up a system table by name (case-insensitive FNV-1a hash).
// Returns the synthetic TableDef and true if found.
func ResolveSystemTable(name string) (*TableDef, bool) {
	oid, ok := systemTableByName[hashString(name)]
	if !ok {
		return nil, false
	}
	info, ok := systemTables[oid]
	if !ok {
		return nil, false
	}
	return &info.Table, true
}

// ResolveSystemColumn looks up a column in a system table by name hash.
// Returns the synthetic ColumnDef or ErrColumnNotFound.
func ResolveSystemColumn(tableOID uint32, colNameHash uint64) (*ColumnDef, error) {
	info, ok := systemTables[tableOID]
	if !ok {
		return nil, ErrColumnNotFound
	}
	col, ok := info.Columns[colNameHash]
	if !ok {
		return nil, ErrColumnNotFound
	}
	return col, nil
}
