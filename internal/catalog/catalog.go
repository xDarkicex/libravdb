package catalog

import (
	"errors"
	"sort"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/quant"
)

// HashIdentifier returns the case-insensitive catalog hash used for SQL
// identifiers. It is exported for execution paths that need to bind a
// runtime column name to catalog metadata without storing strings in the
// catalog itself.
func HashIdentifier(name string) uint64 {
	var hash uint64 = 14695981039346656037
	for i := 0; i < len(name); i++ {
		c := name[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		hash ^= uint64(c)
		hash *= 1099511628211
	}
	return hash
}

var (
	ErrTableNotFound  = errors.New("table not found in catalog")
	ErrColumnNotFound = errors.New("column not found in catalog")
	ErrVectorNotFound = errors.New("vector index not found in catalog")
	ErrGraphNotFound  = errors.New("graph label not found in catalog")
	ErrInvalidCatalog = errors.New("invalid catalog layout")
)

// Catalog provides a zero-copy read-only view over the mmap-backed catalog memory.
//
// HARD INVARIANT FOR DDL (Copy-on-Write):
// A Catalog instance is immutable per generation. When a DDL schema change occurs
// (e.g. CREATE INDEX, ALTER TABLE), a new catalog page must be written and swapped
// in the database superblock. The system must NEVER mutate a catalog page in place.
// Pointers returned by a Catalog (e.g., *TableDef) point directly into the mmap'd
// data. A stale Catalog instance will hold dangling pointers if the underlying page
// is unmapped. Therefore, execution paths must bind strings to OIDs (which are ints
// and immortal) immediately and NEVER hold a *TableDef across a catalog generation swap.
type Catalog struct {
	data []byte

	// Fast SWAR hashmaps mapping FNV-1a hash of identifier -> Definition struct
	// These are built exactly once at Load() time.
	tables  map[uint64]*TableDef
	vectors map[uint64]*VectorIndexDef
	graphs  map[uint64]*GraphLabelDef
	// Foreign keys are stored as a slice because one constraint can contain
	// multiple column pairs and several unnamed constraints can legitimately
	// share the same name hash (zero/empty names in older catalogs).  The OID
	// is the durable identity; NameHash is only the logical constraint-group
	// key used by grouped accessors.
	foreignKeys []*ForeignKeyDef

	registry *quant.Registry
}

// Load casts the raw bytes to the Catalog layout.
// Note on Portability: This cast assumes native endianness matches the writer.
// If cross-machine portability of the .libravdb file is required, a byte-swap pass
// must be budgeted here (e.g. SQLite-style fixed-endian on-disk + conversion on load).
func Load(data []byte, reg *quant.Registry) (*Catalog, error) {
	if len(data) < int(unsafe.Sizeof(Header{})) {
		return nil, ErrInvalidCatalog
	}

	hdr := (*Header)(unsafe.Pointer(&data[0]))
	if hdr.Magic != CatalogMagic || hdr.Version != CatalogVersion {
		return nil, ErrInvalidCatalog
	}

	c := &Catalog{
		data:        data,
		tables:      make(map[uint64]*TableDef, hdr.TablesCount),
		vectors:     make(map[uint64]*VectorIndexDef, hdr.VectorsCount),
		graphs:      make(map[uint64]*GraphLabelDef, hdr.GraphsCount),
		foreignKeys: make([]*ForeignKeyDef, 0, hdr.FKsCount),
		registry:    reg,
	}

	// Load Tables
	tableSize := uint32(unsafe.Sizeof(TableDef{}))
	for i := uint32(0); i < hdr.TablesCount; i++ {
		offset := hdr.TablesOffset + (i * tableSize)
		if int(offset+tableSize) > len(data) {
			return nil, ErrInvalidCatalog
		}
		t := (*TableDef)(unsafe.Pointer(&data[offset]))
		c.tables[t.NameHash] = t
	}

	// Load Vectors
	vecSize := uint32(unsafe.Sizeof(VectorIndexDef{}))
	for i := uint32(0); i < hdr.VectorsCount; i++ {
		offset := hdr.VectorsOffset + (i * vecSize)
		if int(offset+vecSize) > len(data) {
			return nil, ErrInvalidCatalog
		}
		v := (*VectorIndexDef)(unsafe.Pointer(&data[offset]))
		c.vectors[v.NameHash] = v
	}

	// Load Graphs
	graphSize := uint32(unsafe.Sizeof(GraphLabelDef{}))
	for i := uint32(0); i < hdr.GraphsCount; i++ {
		offset := hdr.GraphsOffset + (i * graphSize)
		if int(offset+graphSize) > len(data) {
			return nil, ErrInvalidCatalog
		}
		g := (*GraphLabelDef)(unsafe.Pointer(&data[offset]))
		c.graphs[g.NameHash] = g
	}

	// Load Foreign Keys (v2+ only)
	if hdr.Version >= 2 && hdr.FKsCount > 0 {
		fkSize := uint32(unsafe.Sizeof(ForeignKeyDef{}))
		for i := uint32(0); i < hdr.FKsCount; i++ {
			offset := hdr.FKsOffset + (i * fkSize)
			if int(offset+fkSize) > len(data) {
				return nil, ErrInvalidCatalog
			}
			fk := (*ForeignKeyDef)(unsafe.Pointer(&data[offset]))
			c.foreignKeys = append(c.foreignKeys, fk)
		}
	}

	return c, nil
}

// Data returns the raw catalog bytes backing this instance.
func (c *Catalog) Data() []byte {
	return c.data
}

func (c *Catalog) GetTable(nameHash uint64) (*TableDef, error) {
	if t, ok := c.tables[nameHash]; ok {
		return t, nil
	}
	return nil, ErrTableNotFound
}

func (c *Catalog) GetColumn(table *TableDef, colNameHash uint64) (*ColumnDef, error) {
	colSize := uint32(unsafe.Sizeof(ColumnDef{}))
	for i := uint32(0); i < table.ColumnsCount; i++ {
		offset := table.ColumnsOffset + (i * colSize)
		if int(offset+colSize) > len(c.data) {
			return nil, ErrInvalidCatalog
		}
		col := (*ColumnDef)(unsafe.Pointer(&c.data[offset]))
		if col.NameHash == colNameHash {
			return col, nil
		}
	}
	return nil, ErrColumnNotFound
}

// PrimaryKeyColumnHashes returns the primary-key columns for a table in the
// table's catalog column order. The catalog stores flags on each ColumnDef;
// this method lets runtime execution recover the constraint after reopen.
func (c *Catalog) PrimaryKeyColumnHashes(tableNameHash uint64) ([]uint64, error) {
	table, err := c.GetTable(tableNameHash)
	if err != nil {
		return nil, err
	}
	colSize := uint32(unsafe.Sizeof(ColumnDef{}))
	result := make([]uint64, 0, table.ColumnsCount)
	for i := uint32(0); i < table.ColumnsCount; i++ {
		offset := table.ColumnsOffset + i*colSize
		if int(offset+colSize) > len(c.data) {
			return nil, ErrInvalidCatalog
		}
		col := (*ColumnDef)(unsafe.Pointer(&c.data[offset]))
		if col.Flags&ColFlagPrimaryKey != 0 {
			result = append(result, col.NameHash)
		}
	}
	// Every LibraVDB collection has an internal id key. When additional
	// declared PK columns are present, that implicit physical key must not be
	// mistaken for part of the user's composite SQL key after reopen.
	if len(result) > 1 {
		idHash := HashIdentifier("id")
		filtered := result[:0]
		for _, hash := range result {
			if hash != idHash {
				filtered = append(filtered, hash)
			}
		}
		result = filtered
	}
	return result, nil
}

func (c *Catalog) GetVectorIndex(nameHash uint64) (*VectorIndexDef, error) {
	if v, ok := c.vectors[nameHash]; ok {
		return v, nil
	}
	return nil, ErrVectorNotFound
}

func (c *Catalog) GetGraphLabel(nameHash uint64) (*GraphLabelDef, error) {
	if g, ok := c.graphs[nameHash]; ok {
		return g, nil
	}
	return nil, ErrGraphNotFound
}

// CentroidDistance calculates the distance between two centroids using the configured codebook metric.
// In the full engine, this resolves directly against the `quant.Registry` codebook.
func (c *Catalog) CentroidDistance(metric int, a, b uint32) float32 {
	if c.registry == nil {
		// Fallback for tests if no registry is wired
		return 0.0
	}
	// For ECQO integration, this will eventually fetch the codebook for the specific metric
	// and invoke the SIMD-accelerated distance. For now, it delegates to the quant package
	// to verify structural pipeline integration.
	return quant.CentroidDistance(metric, a, b)
}

// GetForeignKey returns the foreign key constraint with the given name hash.
func (c *Catalog) GetForeignKey(nameHash uint64) (*ForeignKeyDef, error) {
	for _, fk := range c.foreignKeys {
		if fk.NameHash == nameHash {
			return fk, nil
		}
	}
	return nil, ErrVectorNotFound // reuse sentinel for "not found"
}

// ForeignKeysForTable returns all FK constraints where the given table
// (identified by name hash) is the source (child) table.
func (c *Catalog) ForeignKeysForTable(sourceTableHash uint64) []*ForeignKeyDef {
	var result []*ForeignKeyDef
	for _, fk := range c.foreignKeys {
		if fk.SourceTableHash == sourceTableHash {
			result = append(result, fk)
		}
	}
	return result
}

// AllForeignKeys returns every FK constraint in the catalog.
func (c *Catalog) AllForeignKeys() []*ForeignKeyDef {
	result := append([]*ForeignKeyDef(nil), c.foreignKeys...)
	return result
}

// ForeignKeyGroup is the ordered set of column pairs belonging to one
// logical constraint. NameHash identifies the group; OID order preserves the
// pair order from the DDL.
type ForeignKeyGroup struct {
	NameHash        uint64
	SourceTableHash uint64
	TargetTableHash uint64
	OnDelete        uint8
	OnUpdate        uint8
	Pairs           []*ForeignKeyDef
}

func groupForeignKeys(fks []*ForeignKeyDef) []*ForeignKeyGroup {
	groups := make([]*ForeignKeyGroup, 0)
	byKey := make(map[[5]uint64]*ForeignKeyGroup)
	for _, fk := range fks {
		key := [5]uint64{fk.NameHash, fk.SourceTableHash, fk.TargetTableHash, uint64(fk.OnDelete), uint64(fk.OnUpdate)}
		g := byKey[key]
		if g == nil {
			g = &ForeignKeyGroup{NameHash: fk.NameHash, SourceTableHash: fk.SourceTableHash, TargetTableHash: fk.TargetTableHash, OnDelete: fk.OnDelete, OnUpdate: fk.OnUpdate}
			byKey[key] = g
			groups = append(groups, g)
		}
		g.Pairs = append(g.Pairs, fk)
	}
	for _, g := range groups {
		// Catalog records are emitted in pair order, but make this explicit for
		// catalogs rebuilt by older tooling.
		sort.SliceStable(g.Pairs, func(i, j int) bool { return g.Pairs[i].OID < g.Pairs[j].OID })
	}
	return groups
}

// ForeignKeyGroupsForTable returns logical FK constraints for a child table.
func (c *Catalog) ForeignKeyGroupsForTable(sourceTableHash uint64) []*ForeignKeyGroup {
	var pairs []*ForeignKeyDef
	for _, fk := range c.foreignKeys {
		if fk.SourceTableHash == sourceTableHash {
			pairs = append(pairs, fk)
		}
	}
	return groupForeignKeys(pairs)
}

// ForeignKeyGroupsToTable returns logical FK constraints referencing a parent.
func (c *Catalog) ForeignKeyGroupsToTable(targetTableHash uint64) []*ForeignKeyGroup {
	var pairs []*ForeignKeyDef
	for _, fk := range c.foreignKeys {
		if fk.TargetTableHash == targetTableHash {
			pairs = append(pairs, fk)
		}
	}
	return groupForeignKeys(pairs)
}

// ForeignKeysToTable returns all FK constraints where the given table
// (identified by name hash) is the target (parent) table.
func (c *Catalog) ForeignKeysToTable(targetTableHash uint64) []*ForeignKeyDef {
	var result []*ForeignKeyDef
	for _, fk := range c.foreignKeys {
		if fk.TargetTableHash == targetTableHash {
			result = append(result, fk)
		}
	}
	return result
}
