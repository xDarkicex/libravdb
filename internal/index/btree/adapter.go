package btree

import (
	"context"
	"encoding/binary"
	"time"
)

// Value encoding: 16 bytes = ordinal(4) + version(4) + graphNodeID(8).
const valueEncodedLen = 16

// EncodeValue packs an ordinal, version, and graph node ID into a value.
func EncodeValue(ordinal, version uint32, graphNodeID uint64) []byte {
	buf := make([]byte, valueEncodedLen)
	binary.LittleEndian.PutUint32(buf[0:4], ordinal)
	binary.LittleEndian.PutUint32(buf[4:8], version)
	binary.LittleEndian.PutUint64(buf[8:16], graphNodeID)
	return buf
}

// DecodeValue unpacks a value into its components.
func DecodeValue(val []byte) (ordinal, version uint32, graphNodeID uint64) {
	if len(val) < valueEncodedLen {
		return 0, 0, 0
	}
	ordinal = binary.LittleEndian.Uint32(val[0:4])
	version = binary.LittleEndian.Uint32(val[4:8])
	graphNodeID = binary.LittleEndian.Uint64(val[8:16])
	return
}

// ErrKeyExists is re-exported for callers handling duplicate keys.
var ErrKeyExists = errKeyExists

// Index wraps a BTree with typed Insert/Get/Delete using string keys and encoded values.
type Index struct {
	Tree   *BTree
	config Config
}

// NewIndex creates a B-tree-backed index.
func NewIndex(cfg Config) (*Index, error) {
	tree, err := New(cfg)
	if err != nil {
		return nil, err
	}
	return &Index{Tree: tree, config: cfg}, nil
}

// Insert adds an entry by string ID with encoded ordinal/version/nodeID.
func (idx *Index) Insert(ctx context.Context, id string, ordinal, version uint32, graphNodeID uint64) error {
	return idx.Tree.Insert(ctx, []byte(id), EncodeValue(ordinal, version, graphNodeID))
}

// Get retrieves an entry by exact ID.
func (idx *Index) Get(ctx context.Context, id string) (ordinal, version uint32, graphNodeID uint64, err error) {
	val, err := idx.Tree.Search(ctx, []byte(id))
	if err != nil {
		return 0, 0, 0, err
	}
	ordinal, version, graphNodeID = DecodeValue(val)
	return
}

// Delete removes an entry by ID.
func (idx *Index) Delete(ctx context.Context, id string) error {
	return idx.Tree.Delete(ctx, []byte(id))
}

// Len returns the number of entries.
func (idx *Index) Len() int { return idx.Tree.Len() }

// Close releases all resources.
func (idx *Index) Close() error { return idx.Tree.Close() }

// SaveToDisk writes the index to a file.
func (idx *Index) SaveToDisk(ctx context.Context, path string) error {
	return idx.Tree.SaveToDisk(ctx, path)
}

// LoadFromDisk reads the index from a file.
func (idx *Index) LoadFromDisk(ctx context.Context, path string) error {
	return idx.Tree.LoadFromDisk(ctx, path)
}

// SerializeToBytes encodes the index as bytes.
func (idx *Index) SerializeToBytes() ([]byte, error) {
	return idx.Tree.SerializeToBytes()
}

// DeserializeFromBytes reconstructs the index from bytes.
func (idx *Index) DeserializeFromBytes(ctx context.Context, data []byte) error {
	return idx.Tree.DeserializeFromBytes(ctx, data)
}

// PersistenceMeta holds B-tree persistence metadata.
type PersistenceMeta struct {
	CreatedAt time.Time
	IndexType string
	NodeCount int
	Version   uint32
}

// GetPersistenceMetadata returns index metadata.
func (idx *Index) GetPersistenceMetadata() *PersistenceMeta {
	return &PersistenceMeta{
		Version:   1,
		NodeCount: idx.Tree.Len(),
		IndexType: "BTree",
		CreatedAt: time.Now(),
	}
}
