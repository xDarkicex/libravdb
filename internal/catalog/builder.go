package catalog

import (
	"bytes"
	"encoding/binary"
	"unsafe"
)

// ColumnInfo describes a column for the catalog builder.
type ColumnInfo struct {
	Name string
	Type uint16 // TypeInt, TypeFloat, TypeString, TypeVector
}

// Builder constructs a catalog binary from a set of table, vector, and graph definitions.
// The resulting []byte is compatible with Load().
type Builder struct {
	tables  []tableEntry
	vectors []vectorEntry
	graphs  []graphEntry
	nextOID uint32
}

type tableEntry struct {
	name    string
	columns []ColumnInfo
	oid     uint32
}

type vectorEntry struct {
	name   string
	dims   uint32
	metric uint8
	oid    uint32
}

type graphEntry struct {
	name      string
	labelType uint8
	oid       uint32
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
		te := tableEntry{oid: t.OID}
		colSize := uint32(unsafe.Sizeof(ColumnDef{}))
		for i := uint32(0); i < t.ColumnsCount; i++ {
			offset := t.ColumnsOffset + (i * colSize)
			if int(offset+colSize) > len(cat.data) {
				break
			}
			col := (*ColumnDef)(unsafe.Pointer(&cat.data[offset]))
			te.columns = append(te.columns, ColumnInfo{
				Type: col.Type,
			})
		}
		b.tables = append(b.tables, te)
		if t.OID >= b.nextOID {
			b.nextOID = t.OID + 1
		}
	}
	for _, v := range cat.vectors {
		b.vectors = append(b.vectors, vectorEntry{
			dims:   v.Dims,
			metric: v.Metric,
			oid:    v.OID,
		})
		if v.OID >= b.nextOID {
			b.nextOID = v.OID + 1
		}
	}
	for _, g := range cat.graphs {
		b.graphs = append(b.graphs, graphEntry{
			labelType: g.LabelType,
			oid:       g.OID,
		})
		if g.OID >= b.nextOID {
			b.nextOID = g.OID + 1
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
//	[Header] [TableDef * N] [ColumnDef blocks per table] [VectorIndexDef * K] [GraphLabelDef * L]
func (b *Builder) Build() []byte {
	headerSize := uint32(unsafe.Sizeof(Header{}))
	tableDefSize := uint32(unsafe.Sizeof(TableDef{}))
	colDefSize := uint32(unsafe.Sizeof(ColumnDef{}))
	vecDefSize := uint32(unsafe.Sizeof(VectorIndexDef{}))
	graphDefSize := uint32(unsafe.Sizeof(GraphLabelDef{}))

	numTables := uint32(len(b.tables))
	numVectors := uint32(len(b.vectors))
	numGraphs := uint32(len(b.graphs))

	// Calculate offsets
	tablesOffset := headerSize

	// Column blocks immediately follow the TableDef array
	columnsOffset := tablesOffset + numTables*tableDefSize

	vectorsOffset := columnsOffset
	for i := uint32(0); i < numTables; i++ {
		vectorsOffset += uint32(len(b.tables[i].columns)) * colDefSize
	}

	graphsOffset := vectorsOffset + numVectors*vecDefSize

	totalSize := graphsOffset + numGraphs*graphDefSize
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
	// Reserved [32]byte at offset 40-71 is already zero

	// Write TableDef entries
	colBlockOffset := columnsOffset
	for i, t := range b.tables {
		off := tablesOffset + uint32(i)*tableDefSize
		nameHash := hashString(t.name)
		writeUint32(buf, off, t.oid)
		writeUint32(buf, off+4, 0) // padding
		writeUint64(buf, off+8, nameHash)
		writeUint32(buf, off+16, colBlockOffset)
		writeUint32(buf, off+20, uint32(len(t.columns)))

		// Write columns for this table
		for j, col := range t.columns {
			colOff := colBlockOffset + uint32(j)*colDefSize
			colNameHash := hashString(col.Name)
			writeUint32(buf, colOff, 0) // OID — 0 for now (not used by binder)
			writeUint32(buf, colOff+4, 0) // padding
			writeUint64(buf, colOff+8, colNameHash)
			writeUint16(buf, colOff+16, col.Type)
			writeUint16(buf, colOff+18, 0) // flags
			writeUint32(buf, colOff+20, 0) // padding2
		}
		colBlockOffset += uint32(len(t.columns)) * colDefSize
	}

	// Write VectorIndexDef entries
	for i, v := range b.vectors {
		off := vectorsOffset + uint32(i)*vecDefSize
		nameHash := hashString(v.name)
		writeUint32(buf, off, v.oid)
		writeUint32(buf, off+4, 0) // padding
		writeUint64(buf, off+8, nameHash)
		writeUint32(buf, off+16, v.dims)
		buf[off+20] = v.metric
		// Padding2 [3]byte already zero
	}

	// Write GraphLabelDef entries
	for i, g := range b.graphs {
		off := graphsOffset + uint32(i)*graphDefSize
		nameHash := hashString(g.name)
		writeUint32(buf, off, g.oid)
		writeUint32(buf, off+4, 0) // padding
		writeUint64(buf, off+8, nameHash)
		buf[off+16] = g.labelType
		// Padding2 [3]byte, SourceOID, TargetOID, Padding3 already zero
	}

	return buf
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
