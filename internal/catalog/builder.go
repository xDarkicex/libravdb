package catalog

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"unsafe"
)

// ColumnInfo describes a column for the catalog builder.
type ColumnInfo struct {
	Name     string
	Type     uint16 // TypeInt, TypeBigInt, TypeFloat, TypeString, TypeVector, or pg_catalog types
	Flags    uint16 // ColFlagNotNull, ColFlagPrimaryKey, ColFlagUnique
	nameHash uint64
}

// ForeignKeyInfo describes a foreign key constraint for the catalog builder.
type ForeignKeyInfo struct {
	Name         string // constraint name (empty = auto-generated)
	SourceTable  string // child table
	SourceColumn string // child column
	TargetTable  string // parent table
	TargetColumn string // parent column
	OnDelete     uint8  // OnDeleteAction constant
	OnUpdate     uint8  // OnDeleteAction constant (reused enum)
}

// JSONIndexInfo describes a JSON expression index for catalog persistence.
type JSONIndexInfo struct {
	Name       string
	Table      string
	Column     string
	Path       string
	TextResult bool
}

// Builder constructs a catalog binary from a set of table, vector, and graph definitions.
// The resulting []byte is compatible with Load().
type Builder struct {
	tables      []tableEntry
	vectors     []vectorEntry
	graphs      []graphEntry
	foreignKeys []fkEntry
	checks      []checkEntry
	defaults    []defaultEntry
	jsonIndexes []jsonIndexEntry
	nextOID     uint32
}

type tableEntry struct {
	name     string
	nameHash uint64
	columns  []ColumnInfo
	oid      uint32
}

type vectorEntry struct {
	name     string
	nameHash uint64
	dims     uint32
	metric   uint8
	oid      uint32
}

type graphEntry struct {
	name      string
	nameHash  uint64
	labelType uint8
	oid       uint32
}

type fkEntry struct {
	name            string
	nameHash        uint64
	sourceTable     string
	sourceTableHash uint64
	targetTable     string
	targetTableHash uint64
	sourceCol       string
	sourceColHash   uint64
	targetCol       string
	targetColHash   uint64
	onDelete        uint8
	onUpdate        uint8
	oid             uint32
}

type checkEntry struct {
	name      string
	nameHash  uint64
	tableName string
	tableHash uint64
	colName   string
	colHash   uint64
	expr      string // expression text
	oid       uint32
}

type defaultEntry struct {
	tableName string
	tableHash uint64
	colName   string
	colHash   uint64
	value     string // literal value text
	oid       uint32
}

type jsonIndexEntry struct {
	name       string
	nameHash   uint64
	table      string
	tableHash  uint64
	column     string
	columnHash uint64
	path       string
	textResult bool
	oid        uint32
}

// NewBuilder creates an empty catalog builder with OIDs starting at 100.
func NewBuilder() *Builder {
	return &Builder{nextOID: 100}
}

// NewBuilderFrom creates a builder pre-populated with the entries from an existing catalog.
// Returns an empty builder if cat is nil.
func NewBuilderFrom(cat *Catalog) *Builder {
	b := NewBuilder()
	if cat == nil {
		return b
	}
	for _, t := range cat.tables {
		// We can't recover the string name from the hash, so we store the hash
		// as a sentinel name. The hashes are preserved during Build via
		// FNV-1a hashing.
		te := tableEntry{oid: t.OID, nameHash: t.NameHash}
		colSize := uint32(unsafe.Sizeof(ColumnDef{}))
		for i := uint32(0); i < t.ColumnsCount; i++ {
			offset := t.ColumnsOffset + (i * colSize)
			if int(offset+colSize) > len(cat.data) {
				break
			}
			col := (*ColumnDef)(unsafe.Pointer(&cat.data[offset]))
			te.columns = append(te.columns, ColumnInfo{
				Type:     col.Type,
				Flags:    col.Flags,
				nameHash: col.NameHash,
			})
		}
		b.tables = append(b.tables, te)
		if t.OID >= b.nextOID {
			b.nextOID = t.OID + 1
		}
	}
	for _, v := range cat.vectors {
		b.vectors = append(b.vectors, vectorEntry{
			nameHash: v.NameHash,
			dims:     v.Dims,
			metric:   v.Metric,
			oid:      v.OID,
		})
		if v.OID >= b.nextOID {
			b.nextOID = v.OID + 1
		}
	}
	for _, g := range cat.graphs {
		b.graphs = append(b.graphs, graphEntry{
			nameHash:  g.NameHash,
			labelType: g.LabelType,
			oid:       g.OID,
		})
		if g.OID >= b.nextOID {
			b.nextOID = g.OID + 1
		}
	}
	for _, fk := range cat.foreignKeys {
		b.foreignKeys = append(b.foreignKeys, fkEntry{
			nameHash:        fk.NameHash,
			sourceTableHash: fk.SourceTableHash,
			targetTableHash: fk.TargetTableHash,
			sourceColHash:   fk.SourceColHash,
			targetColHash:   fk.TargetColHash,
			onDelete:        fk.OnDelete,
			onUpdate:        fk.OnUpdate,
			oid:             fk.OID,
		})
		if fk.OID >= b.nextOID {
			b.nextOID = fk.OID + 1
		}
	}
	for _, chk := range cat.checkConstraints {
		b.checks = append(b.checks, checkEntry{
			nameHash:  chk.NameHash,
			tableHash: chk.TableHash,
			colHash:   chk.ColHash,
			expr:      cat.CheckExpr(chk),
			oid:       chk.OID,
		})
		if chk.OID >= b.nextOID {
			b.nextOID = chk.OID + 1
		}
	}
	for _, def := range cat.defaultValues {
		val := ""
		if def.DataLen > 0 && int(def.DataOff)+int(def.DataLen) <= len(cat.data) {
			val = string(cat.data[def.DataOff : def.DataOff+def.DataLen])
		}
		b.defaults = append(b.defaults, defaultEntry{
			tableHash: def.TableHash,
			colHash:   def.ColHash,
			value:     val,
			oid:       def.OID,
		})
		if def.OID >= b.nextOID {
			b.nextOID = def.OID + 1
		}
	}
	for _, idx := range cat.jsonIndexes {
		b.jsonIndexes = append(b.jsonIndexes, jsonIndexEntry{
			name: cat.JSONIndexName(idx), nameHash: idx.NameHash, tableHash: idx.TableHash, columnHash: idx.ColumnHash,
			path: cat.JSONIndexPath(idx), textResult: idx.TextResult != 0, oid: idx.OID,
		})
		if idx.OID >= b.nextOID {
			b.nextOID = idx.OID + 1
		}
	}
	return b
}

// AddTable registers a table with its columns. Returns the assigned OID.
func (b *Builder) AddTable(name string, columns []ColumnInfo) uint32 {
	oid := b.nextOID
	b.nextOID++
	b.tables = append(b.tables, tableEntry{
		name:    name,
		columns: columns,
		oid:     oid,
	})
	return oid
}

// AddVectorIndex registers a vector index. Returns the assigned OID.
func (b *Builder) AddVectorIndex(name string, dims uint32, metric uint8) uint32 {
	oid := b.nextOID
	b.nextOID++
	b.vectors = append(b.vectors, vectorEntry{
		name:   name,
		dims:   dims,
		metric: metric,
		oid:    oid,
	})
	return oid
}

// AddForeignKey registers a foreign key constraint. Returns the assigned OID.
func (b *Builder) AddForeignKey(info ForeignKeyInfo) uint32 {
	oid := b.nextOID
	b.nextOID++
	name := info.Name
	if name == "" {
		name = fmt.Sprintf("__fk_%d", oid)
	}
	b.foreignKeys = append(b.foreignKeys, fkEntry{
		name:        name,
		sourceTable: info.SourceTable,
		targetTable: info.TargetTable,
		sourceCol:   info.SourceColumn,
		targetCol:   info.TargetColumn,
		onDelete:    info.OnDelete,
		onUpdate:    info.OnUpdate,
		oid:         oid,
	})
	return oid
}

// AddCheckConstraint registers a CHECK constraint. Returns the assigned OID.
func (b *Builder) AddCheckConstraint(tableName, expr, colName string) uint32 {
	oid := b.nextOID
	b.nextOID++
	name := fmt.Sprintf("__chk_%d", oid)
	b.checks = append(b.checks, checkEntry{
		name:      name,
		tableName: tableName,
		colName:   colName,
		expr:      expr,
		oid:       oid,
	})
	return oid
}

// AddDefaultValue registers a column DEFAULT value. Returns the assigned OID.
func (b *Builder) AddDefaultValue(tableName, colName, value string) uint32 {
	oid := b.nextOID
	b.nextOID++
	b.defaults = append(b.defaults, defaultEntry{
		tableName: tableName,
		colName:   colName,
		value:     value,
		oid:       oid,
	})
	return oid
}

// AddJSONIndex registers a durable JSON expression index definition.
func (b *Builder) AddJSONIndex(info JSONIndexInfo) uint32 {
	tableHash := hashString(info.Table)
	columnHash := hashString(info.Column)
	for _, existing := range b.jsonIndexes {
		if hashOr(existing.tableHash, hashString(existing.table)) == tableHash &&
			hashOr(existing.columnHash, hashString(existing.column)) == columnHash &&
			existing.path == info.Path && existing.textResult == info.TextResult {
			return existing.oid
		}
	}
	oid := b.nextOID
	b.nextOID++
	b.jsonIndexes = append(b.jsonIndexes, jsonIndexEntry{
		name: info.Name, table: info.Table, column: info.Column, path: info.Path,
		textResult: info.TextResult, oid: oid,
	})
	return oid
}

// ReplaceJSONIndexesForTable replaces one table's JSON index definitions in
// the copy-on-write catalog builder. This lets DROP INDEX remove stale
// definitions while preserving all other tables and constraints.
func (b *Builder) ReplaceJSONIndexesForTable(table string, indexes []JSONIndexInfo) {
	tableHash := hashString(table)
	kept := b.jsonIndexes[:0]
	for _, existing := range b.jsonIndexes {
		if hashOr(existing.tableHash, hashString(existing.table)) != tableHash {
			kept = append(kept, existing)
		}
	}
	b.jsonIndexes = kept
	for _, index := range indexes {
		b.AddJSONIndex(index)
	}
}

// AddGraphLabel registers a graph label. Returns the assigned OID.
func (b *Builder) AddGraphLabel(name string, labelType uint8) uint32 {
	oid := b.nextOID
	b.nextOID++
	b.graphs = append(b.graphs, graphEntry{
		name:      name,
		labelType: labelType,
		oid:       oid,
	})
	return oid
}

// Build serializes the catalog into a binary blob compatible with Load().
//
// Layout (all offsets are absolute byte positions from start of buffer):
//
//	[Header] [TableDef * N] [ColumnDef blocks] [VectorIndexDef * K] [GraphLabelDef * L]
//	[ForeignKeyDef * M] [CheckConstraintDef * P] [DefaultValueDef * Q] [JSONIndexDef * R] [variable-length data]
func (b *Builder) Build() []byte {
	headerSize := uint32(unsafe.Sizeof(Header{}))
	tableDefSize := uint32(unsafe.Sizeof(TableDef{}))
	colDefSize := uint32(unsafe.Sizeof(ColumnDef{}))
	vecDefSize := uint32(unsafe.Sizeof(VectorIndexDef{}))
	graphDefSize := uint32(unsafe.Sizeof(GraphLabelDef{}))
	fkDefSize := uint32(unsafe.Sizeof(ForeignKeyDef{}))
	chkDefSize := uint32(unsafe.Sizeof(CheckConstraintDef{}))
	defDefSize := uint32(unsafe.Sizeof(DefaultValueDef{}))
	jsonIndexDefSize := uint32(unsafe.Sizeof(JSONIndexDef{}))

	numTables := uint32(len(b.tables))
	numVectors := uint32(len(b.vectors))
	numGraphs := uint32(len(b.graphs))
	numFKs := uint32(len(b.foreignKeys))
	numChecks := uint32(len(b.checks))
	numDefaults := uint32(len(b.defaults))
	numJSONIndexes := uint32(len(b.jsonIndexes))

	// Calculate offsets
	tablesOffset := headerSize
	columnsOffset := tablesOffset + numTables*tableDefSize

	vectorsOffset := columnsOffset
	for i := uint32(0); i < numTables; i++ {
		vectorsOffset += uint32(len(b.tables[i].columns)) * colDefSize
	}

	graphsOffset := vectorsOffset + numVectors*vecDefSize
	fksOffset := graphsOffset + numGraphs*graphDefSize
	checksOffset := fksOffset + numFKs*fkDefSize
	defaultsOffset := checksOffset + numChecks*chkDefSize
	jsonIndexesOffset := defaultsOffset + numDefaults*defDefSize

	// Calculate variable-length data section size and offsets.
	// Order: column names, CHECK expressions, DEFAULT values.
	var dataOff uint32
	// Column name offsets
	type nameSlot struct{ off, ln uint32 }
	colNameOffsets := make([][]nameSlot, numTables)
	for ti, t := range b.tables {
		colNameOffsets[ti] = make([]nameSlot, len(t.columns))
		for ci, col := range t.columns {
			nm := col.Name
			if nm == "" {
				continue
			}
			colNameOffsets[ti][ci] = nameSlot{off: dataOff, ln: uint32(len(nm))}
			dataOff += uint32(len(nm))
		}
	}
	// CHECK expression offsets
	exprOffsets := make([]uint32, numChecks)
	for i, chk := range b.checks {
		exprOffsets[i] = dataOff
		dataOff += uint32(len(chk.expr))
	}
	// DEFAULT value offsets
	defOffsets := make([]uint32, numDefaults)
	for i, def := range b.defaults {
		defOffsets[i] = dataOff
		dataOff += uint32(len(def.value))
	}
	jsonNameOffsets := make([]uint32, numJSONIndexes)
	jsonPathOffsets := make([]uint32, numJSONIndexes)
	for i, idx := range b.jsonIndexes {
		jsonNameOffsets[i] = dataOff
		dataOff += uint32(len(idx.name))
		jsonPathOffsets[i] = dataOff
		dataOff += uint32(len(idx.path))
	}

	dataSectionOffset := jsonIndexesOffset + numJSONIndexes*jsonIndexDefSize
	totalSize := dataSectionOffset + dataOff
	buf := make([]byte, totalSize)

	// Write Header
	writeUint64(buf, 0, CatalogMagic)
	writeUint32(buf, 8, CatalogVersion)
	writeUint32(buf, 12, 0) // padding
	writeUint32(buf, 16, numTables)
	writeUint32(buf, 20, tablesOffset)
	writeUint32(buf, 24, numVectors)
	writeUint32(buf, 28, vectorsOffset)
	writeUint32(buf, 32, numGraphs)
	writeUint32(buf, 36, graphsOffset)
	writeUint32(buf, 40, numFKs)
	writeUint32(buf, 44, fksOffset)
	writeUint32(buf, 48, numChecks)
	writeUint32(buf, 52, checksOffset)
	writeUint32(buf, 56, numDefaults)
	writeUint32(buf, 60, defaultsOffset)
	writeUint32(buf, 64, numJSONIndexes)
	writeUint32(buf, 68, jsonIndexesOffset)

	// Write TableDef entries
	colBlockOffset := columnsOffset
	for i, t := range b.tables {
		off := tablesOffset + uint32(i)*tableDefSize
		nameHash := t.nameHash
		if nameHash == 0 {
			nameHash = hashString(t.name)
		}
		writeUint32(buf, off, t.oid)
		writeUint32(buf, off+4, 0) // padding
		writeUint64(buf, off+8, nameHash)
		writeUint32(buf, off+16, colBlockOffset)
		writeUint32(buf, off+20, uint32(len(t.columns)))
		for j, col := range t.columns {
			colOff := colBlockOffset + uint32(j)*colDefSize
			colNameHash := col.nameHash
			if colNameHash == 0 {
				colNameHash = hashString(col.Name)
			}
			writeUint32(buf, colOff, 0)
			writeUint32(buf, colOff+4, 0)
			writeUint64(buf, colOff+8, colNameHash)
			writeUint16(buf, colOff+16, col.Type)
			writeUint16(buf, colOff+18, col.Flags)
			// Write name offset/length in trailing data section.
			ns := colNameOffsets[i][j]
			writeUint16(buf, colOff+20, uint16(dataSectionOffset+ns.off))
			writeUint16(buf, colOff+22, uint16(ns.ln))
			if ns.ln > 0 {
				copy(buf[dataSectionOffset+ns.off:], col.Name)
			}
		}
		colBlockOffset += uint32(len(t.columns)) * colDefSize
	}

	// Write VectorIndexDef entries
	for i, v := range b.vectors {
		off := vectorsOffset + uint32(i)*vecDefSize
		nameHash := v.nameHash
		if nameHash == 0 {
			nameHash = hashString(v.name)
		}
		writeUint32(buf, off, v.oid)
		writeUint32(buf, off+4, 0)
		writeUint64(buf, off+8, nameHash)
		writeUint32(buf, off+16, v.dims)
		buf[off+20] = v.metric
	}

	// Write GraphLabelDef entries
	for i, g := range b.graphs {
		off := graphsOffset + uint32(i)*graphDefSize
		nameHash := g.nameHash
		if nameHash == 0 {
			nameHash = hashString(g.name)
		}
		writeUint32(buf, off, g.oid)
		writeUint32(buf, off+4, 0)
		writeUint64(buf, off+8, nameHash)
		buf[off+16] = g.labelType
	}

	// Write ForeignKeyDef entries
	for i, fk := range b.foreignKeys {
		off := fksOffset + uint32(i)*fkDefSize
		nameHash := fk.nameHash
		if nameHash == 0 {
			nameHash = hashString(fk.name)
		}
		writeUint32(buf, off, fk.oid)
		writeUint32(buf, off+4, 0)
		writeUint64(buf, off+8, nameHash)
		writeUint64(buf, off+16, hashOr(fk.sourceTableHash, hashString(fk.sourceTable)))
		writeUint64(buf, off+24, hashOr(fk.targetTableHash, hashString(fk.targetTable)))
		writeUint64(buf, off+32, hashOr(fk.sourceColHash, hashString(fk.sourceCol)))
		writeUint64(buf, off+40, hashOr(fk.targetColHash, hashString(fk.targetCol)))
		buf[off+48] = fk.onDelete
		buf[off+49] = fk.onUpdate
	}

	// Write CheckConstraintDef entries
	for i, chk := range b.checks {
		off := checksOffset + uint32(i)*chkDefSize
		writeUint32(buf, off, chk.oid)
		writeUint32(buf, off+4, 0)
		writeUint64(buf, off+8, hashOr(chk.nameHash, hashString(chk.name)))
		writeUint64(buf, off+16, hashOr(chk.tableHash, hashString(chk.tableName)))
		writeUint64(buf, off+24, hashOr(chk.colHash, hashString(chk.colName)))
		writeUint32(buf, off+32, dataSectionOffset+exprOffsets[i])
		writeUint32(buf, off+36, uint32(len(chk.expr)))
		// copy expression text into data section
		copy(buf[dataSectionOffset+exprOffsets[i]:], chk.expr)
	}

	// Write DefaultValueDef entries
	for i, def := range b.defaults {
		off := defaultsOffset + uint32(i)*defDefSize
		writeUint32(buf, off, def.oid)
		writeUint32(buf, off+4, 0)
		writeUint64(buf, off+8, hashOr(def.tableHash, hashString(def.tableName)))
		writeUint64(buf, off+16, hashOr(def.colHash, hashString(def.colName)))
		writeUint32(buf, off+24, dataSectionOffset+defOffsets[i])
		writeUint32(buf, off+28, uint32(len(def.value)))
		copy(buf[dataSectionOffset+defOffsets[i]:], []byte(def.value))
	}

	// Write durable JSON expression-index definitions. Path strings share the
	// trailing data section, keeping catalog reads zero-copy after mmap load.
	for i, idx := range b.jsonIndexes {
		off := jsonIndexesOffset + uint32(i)*jsonIndexDefSize
		writeUint32(buf, off, idx.oid)
		writeUint32(buf, off+4, 0)
		writeUint64(buf, off+8, hashOr(idx.nameHash, hashString(idx.name)))
		writeUint64(buf, off+16, hashOr(idx.tableHash, hashString(idx.table)))
		writeUint64(buf, off+24, hashOr(idx.columnHash, hashString(idx.column)))
		writeUint32(buf, off+32, dataSectionOffset+jsonNameOffsets[i])
		writeUint32(buf, off+36, uint32(len(idx.name)))
		writeUint32(buf, off+40, dataSectionOffset+jsonPathOffsets[i])
		writeUint32(buf, off+44, uint32(len(idx.path)))
		if idx.textResult {
			buf[off+48] = 1
		}
		copy(buf[dataSectionOffset+jsonNameOffsets[i]:], idx.name)
		copy(buf[dataSectionOffset+jsonPathOffsets[i]:], idx.path)
	}

	return buf
}

func hashOr(hash uint64, fallback uint64) uint64 {
	if hash != 0 {
		return hash
	}
	return fallback
}

// MustBuild is like Build but panics on error (for use in tests and initialization).
func (b *Builder) MustBuild() []byte {
	return b.Build()
}

// hashString computes the case-insensitive FNV-1a hash used by the catalog and binder.
func hashString(s string) uint64 {
	var hash uint64 = 14695981039346656037
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 32
		}
		hash ^= uint64(c)
		hash *= 1099511628211
	}
	return hash
}

// Helper write functions to avoid importing encoding/binary everywhere.
func writeUint64(buf []byte, off uint32, v uint64) {
	binary.LittleEndian.PutUint64(buf[off:off+8], v)
}

func writeUint32(buf []byte, off uint32, v uint32) {
	binary.LittleEndian.PutUint32(buf[off:off+4], v)
}

func writeUint16(buf []byte, off uint32, v uint16) {
	binary.LittleEndian.PutUint16(buf[off:off+2], v)
}

// Ensure bytes import is used.
var _ = bytes.NewBuffer
