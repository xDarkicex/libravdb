package hnsw

import (
	"time"
)

// Binary format constants
const (
	// Magic number for HNSW index files: "HNSWVIDX"
	IndexFileMagic = "HNSWVIDX"

	// Current binary format version
	FormatVersion = uint32(1)

	// Maximum supported format version (for forward compatibility)
	MaxSupportedVersion = uint32(1)

	// Section alignment (for performance)
	SectionAlignment = 64

	// Chunk size for streaming operations (64KB)
	StreamChunkSize = 64 * 1024
)

// IndexFileHeader defines the binary file format header
// Total size: 128 bytes (cache-line friendly)
type IndexFileHeader struct {
	Magic     [8]byte // "HNSWVIDX" magic identifier
	Version   uint32  // Binary format version
	NodeCount uint32  // Total number of nodes in index
	Dimension uint32  // Vector dimension
	MaxLevel  int32   // Maximum graph level

	// Section sizes (for efficient seeking)
	ConfigSize uint32 // Config section size in bytes
	NodesSize  uint64 // Nodes section size in bytes
	LinksSize  uint64 // Links section size in bytes
	MetaSize   uint32 // Metadata section size in bytes

	// Integrity and performance
	ChecksumCRC uint32 // Header + all sections CRC32
	Compressed  uint32 // Compression flags (reserved for future)

	// Reserved space for future extensions
	Reserved [32]byte // Must be zero-filled
}

// NodeEntry represents a single node in the binary format
type NodeEntry struct {
	IDLength    uint32 // Length of ID string
	Level       int32  // Maximum level for this node
	VectorBytes uint32 // Size of vector data (dimension * 4)
	MetaBytes   uint32 // Size of metadata (JSON serialized)

	// Variable-length data follows:
	// - ID string (IDLength bytes)
	// - Vector data (VectorBytes bytes, float32 little-endian)
	// - Metadata (MetaBytes bytes, JSON)
}

// LinkEntry represents connections for a single level of a node
type LinkEntry struct {
	NodeIndex uint32 // Index of the node this entry belongs to
	Level     int32  // Graph level
	LinkCount uint32 // Number of connections at this level

	// Variable-length data follows:
	// - Link indices (LinkCount * 4 bytes, uint32 little-endian)
}

// ConfigEntry holds the HNSW configuration used to build the index
type ConfigEntry struct {
	Dimension      uint32  // Vector dimension
	M              uint32  // Maximum connections per node
	EfConstruction uint32  // Construction-time candidate list size
	EfSearch       uint32  // Search-time candidate list size
	ML             float64 // Level generation factor
	Metric         uint32  // Distance metric (0=L2, 1=Inner, 2=Cosine)
	RandomSeed     int64   // Random seed used during construction
}

// MetadataEntry holds index-wide metadata
type MetadataEntry struct {
	CreationTime    int64  // Unix timestamp of index creation
	BuildDuration   int64  // Time taken to build index (nanoseconds)
	EntryPointIndex uint32 // Index of the entry point node
	IndexTypeID     uint32 // Type identifier (1=HNSW)

	// Statistics
	TotalInsertions   uint64  // Total vectors inserted
	TotalDeletions    uint64  // Total vectors deleted
	GraphConnectivity float32 // Average connections per node

	Reserved [16]byte // Future metadata
}

// HNSWPersistenceMetadata holds metadata about persisted HNSW index
type HNSWPersistenceMetadata struct {
	Version       uint32    `json:"version"`
	NodeCount     int       `json:"node_count"`
	Dimension     int       `json:"dimension"`
	MaxLevel      int       `json:"max_level"`
	CreatedAt     time.Time `json:"created_at"`
	ChecksumCRC32 uint32    `json:"checksum_crc32"`
	FileSize      int64     `json:"file_size"`
}

// File layout specification:
// ┌─────────────────────┐
// │ IndexFileHeader     │ 128 bytes
// ├─────────────────────┤
// │ ConfigEntry         │ Variable size
// ├─────────────────────┤
// │ Node Section        │ NodesSize bytes
// │ ├─ NodeEntry[0]     │   ├─ ID + Vector + Meta
// │ ├─ NodeEntry[1]     │   ├─ ID + Vector + Meta
// │ └─ ...              │   └─ ...
// ├─────────────────────┤
// │ Links Section       │ LinksSize bytes
// │ ├─ LinkEntry[0][0]  │   ├─ Level 0 links for node 0
// │ ├─ LinkEntry[0][1]  │   ├─ Level 1 links for node 0
// │ ├─ LinkEntry[1][0]  │   ├─ Level 0 links for node 1
// │ └─ ...              │   └─ ...
// ├─────────────────────┤
// │ MetadataEntry       │ MetaSize bytes
// └─────────────────────┘
//
// Benefits of this layout:
// 1. **Sequential access** - nodes and links stored contiguously
// 2. **Alignment-friendly** - 64-byte aligned sections for cache efficiency
// 3. **Streamable** - can load/save in chunks for memory efficiency
// 4. **Extensible** - reserved fields allow format evolution
// 5. **Integrity-checked** - CRC32 prevents corruption
