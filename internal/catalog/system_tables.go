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
	sysOIDPgClass = 1

	// pg_class column OIDs
	sysColOIDOID          = 10
	sysColOIDRelname      = 11
	sysColOIDRelnamespace = 12
	sysColOIDRelkind      = 13
	sysColOIDReltuples    = 14
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
			OID:           sysOIDPgClass,
			NameHash:      hashString("pg_class"),
			ColumnsCount:  5,
		},
		Columns: make(map[uint64]*ColumnDef),
	}
	pgClass.Columns[hashString("oid")] = &ColumnDef{
		OID: sysColOIDOID, NameHash: hashString("oid"), Type: TypeInt,
	}
	pgClass.Columns[hashString("relname")] = &ColumnDef{
		OID: sysColOIDRelname, NameHash: hashString("relname"), Type: TypeString,
	}
	pgClass.Columns[hashString("relnamespace")] = &ColumnDef{
		OID: sysColOIDRelnamespace, NameHash: hashString("relnamespace"), Type: TypeInt,
	}
	pgClass.Columns[hashString("relkind")] = &ColumnDef{
		OID: sysColOIDRelkind, NameHash: hashString("relkind"), Type: TypeString,
	}
	pgClass.Columns[hashString("reltuples")] = &ColumnDef{
		OID: sysColOIDReltuples, NameHash: hashString("reltuples"), Type: TypeFloat,
	}
	m[sysOIDPgClass] = pgClass

	return m
}()

// systemTableByName maps table name hash → OID for fast lookup during binder
// table resolution (when the catalog doesn't have the table).
var systemTableByName = func() map[uint64]uint32 {
	m := make(map[uint64]uint32, len(systemTables))
	m[hashString("pg_class")] = sysOIDPgClass
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
