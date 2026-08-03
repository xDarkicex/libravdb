package catalog

import (
	"errors"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/quant"
)

var (
	ErrTableNotFound   = errors.New("table not found in catalog")
	ErrColumnNotFound  = errors.New("column not found in catalog")
	ErrVectorNotFound  = errors.New("vector index not found in catalog")
	ErrGraphNotFound   = errors.New("graph label not found in catalog")
	ErrInvalidCatalog  = errors.New("invalid catalog layout")
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
		data:     data,
		tables:   make(map[uint64]*TableDef, hdr.TablesCount),
		vectors:  make(map[uint64]*VectorIndexDef, hdr.VectorsCount),
		graphs:   make(map[uint64]*GraphLabelDef, hdr.GraphsCount),
		registry: reg,
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
	
	return c, nil
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
