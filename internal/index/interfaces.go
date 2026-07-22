package index

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"time"

	"unsafe"

	"github.com/xDarkicex/libravdb/internal/index/flat"
	"github.com/xDarkicex/libravdb/internal/index/hnsw"
	"github.com/xDarkicex/libravdb/internal/index/ivfpq"
	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/record"
	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/memory"
)

// GraphFilter is an interface used to filter search candidates based on a graph bitset.
type GraphFilter interface {
	Test(idx uint64) bool
}

// Index defines the interface for all vector index implementations
type Index interface {
	// Insert adds a vector to the index
	Insert(ctx context.Context, entry *VectorEntry) error
	BatchInsert(ctx context.Context, entries []*VectorEntry) error // NEW: Optimized batch insertion
	// Search finds the k nearest neighbors
	Search(ctx context.Context, query []float32, k int, filter GraphFilter) ([]*SearchResult, error)
	Delete(ctx context.Context, id string) error
	Size() int
	MemoryUsage() int64
	Close() error

	// NEW: Index Persistence Methods
	SaveToDisk(ctx context.Context, path string) error
	LoadFromDisk(ctx context.Context, path string) error
	SerializeToBytes() ([]byte, error)
	DeserializeFromBytes(ctx context.Context, data []byte) error
	GetPersistenceMetadata() *PersistenceMetadata
}

// PreparedMutation is a fully materialized index delta. Commit must not
// allocate or search; it only publishes state after the storage transaction
// is durable. Abort releases an unpublished candidate generation.
type PreparedMutation interface {
	Commit() error
	Abort() error
}

// DeltaIndex is implemented by indexes whose transaction path can publish a
// copy-on-write delta instead of rebuilding every record in the collection.
type DeltaIndex interface {
	PrepareMutations(ctx context.Context, puts []*VectorEntry, deletes []string) (PreparedMutation, error)
}

// VectorEntry represents a vector entry (avoid circular imports)
type VectorEntry struct {
	Metadata map[string]interface{}
	ID       string
	Vector   []float32
	Version  uint64
	Ordinal  uint32
}

// SearchResult represents a search result (avoid circular imports)
type SearchResult struct {
	Metadata map[string]interface{}
	ID       string
	Vector   []float32
	Version  uint64
	Ordinal  uint32
	Score    float32
}

// NEW: PersistenceMetadata holds metadata about persisted index
type PersistenceMetadata struct {
	CreatedAt     time.Time `json:"created_at"`
	IndexType     string    `json:"index_type"`
	NodeCount     int       `json:"node_count"`
	Dimension     int       `json:"dimension"`
	MaxLevel      int       `json:"max_level"`
	FileSize      int64     `json:"file_size"`
	Version       uint32    `json:"version"`
	ChecksumCRC32 uint32    `json:"checksum_crc32"`
}

// IndexType represents different index algorithms
type IndexType int

const (
	IndexTypeHNSW IndexType = iota
	IndexTypeIVFPQ
	IndexTypeFlat
)

// String returns the string representation of the index type
func (it IndexType) String() string {
	switch it {
	case IndexTypeHNSW:
		return "HNSW"
	case IndexTypeIVFPQ:
		return "IVF-PQ"
	case IndexTypeFlat:
		return "Flat"
	default:
		return "Unknown"
	}
}

// HNSWConfig holds configuration for HNSW index
type HNSWConfig struct {
	Provider             hnsw.VectorProvider
	Quantization         *quant.QuantizationConfig
	RawVectorStore       string
	Dimension            int
	M                    int
	EfConstruction       int
	EfSearch             int
	ML                   float64
	Metric               util.DistanceMetric
	PruneAlpha           float32
	Level0LinkMultiplier float64
	RepairEnabled        bool
	RepairQueueSize      int
	RepairBatchSize      int
	RawStoreCap          int
	IDMapCapacity        int
}

// IVFPQConfig holds configuration for IVF-PQ index
type IVFPQConfig struct {
	Quantization  *quant.QuantizationConfig
	Dimension     int
	NClusters     int
	NProbes       int
	Metric        util.DistanceMetric
	MaxIterations int
	Tolerance     float64
	RandomSeed    int64
}

// FlatConfig holds configuration for Flat index
type FlatConfig struct {
	Quantization *quant.QuantizationConfig
	Dimension    int
	Metric       util.DistanceMetric
}

// hnswWrapper wraps the HNSW index to adapt between interface types
type hnswWrapper struct {
	index *hnsw.Index
	sfl   *memory.ShardedFreeList
}

func (w *hnswWrapper) RawVectorStoreProfile() map[string]any {
	return w.index.RawVectorStoreProfile()
}

// Insert adapts the interface VectorEntry to HNSW VectorEntry
func (w *hnswWrapper) Insert(ctx context.Context, entry *VectorEntry) error {
	hnswEntry := &hnsw.VectorEntry{
		ID:       entry.ID,
		Ordinal:  entry.Ordinal,
		Vector:   entry.Vector,
		Metadata: entry.Metadata,
		Version:  entry.Version,
	}
	return w.index.Insert(ctx, hnswEntry)
}

// BatchInsert adapts the interface VectorEntry slice to HNSW VectorEntry slice
func (w *hnswWrapper) BatchInsert(ctx context.Context, entries []*VectorEntry) error {
	hnswEntries := make([]*hnsw.VectorEntry, len(entries))
	slots := make([][]byte, len(entries))
	for i, entry := range entries {
		slot, err := w.sfl.Allocate()
		if err != nil {
			for j := 0; j < i; j++ {
				_ = w.sfl.Deallocate(slots[j])
			}
			return err
		}
		slots[i] = slot
		hnswEntry := (*hnsw.VectorEntry)(unsafe.Pointer(&slot[48]))
		hnswEntry.ID = entry.ID
		hnswEntry.Ordinal = entry.Ordinal
		hnswEntry.Vector = entry.Vector
		hnswEntry.Metadata = entry.Metadata
		hnswEntry.Version = entry.Version
		hnswEntries[i] = hnswEntry
	}
	err := w.index.BatchInsert(ctx, hnswEntries)
	for _, slot := range slots {
		_ = w.sfl.Deallocate(slot)
	}
	return err
}

// Search adapts the search results from HNSW to interface types
func (w *hnswWrapper) Search(ctx context.Context, query []float32, k int, filter GraphFilter) ([]*SearchResult, error) {
	hnswResults, err := w.index.Search(ctx, query, k, filter)
	if err != nil {
		return nil, err
	}

	results := make([]*SearchResult, len(hnswResults))
	for i, r := range hnswResults {
		results[i] = &SearchResult{
			Ordinal:  r.Ordinal,
			ID:       r.ID,
			Score:    r.Score,
			Vector:   r.Vector,
			Metadata: r.Metadata,
			Version:  r.Version,
		}
	}
	return results, nil
}

// Delete delegates to the wrapped index
func (w *hnswWrapper) Delete(ctx context.Context, id string) error {
	return w.index.Delete(ctx, id)
}

// Size delegates to the wrapped index
func (w *hnswWrapper) Size() int {
	return w.index.Size()
}

// MemoryUsage delegates to the wrapped index
func (w *hnswWrapper) MemoryUsage() int64 {
	return w.index.MemoryUsage()
}

// Close delegates to the wrapped index
func (w *hnswWrapper) Close() error {
	if w.sfl != nil {
		_ = w.sfl.Free()
	}
	return w.index.Close()
}

// NEW: SaveToDisk delegates persistence to the wrapped index
func (w *hnswWrapper) SaveToDisk(ctx context.Context, path string) error {
	return w.index.SaveToDisk(ctx, path)
}

// NEW: LoadFromDisk delegates loading to the wrapped index
func (w *hnswWrapper) LoadFromDisk(ctx context.Context, path string) error {
	return w.index.LoadFromDisk(ctx, path)
}

// SerializeToBytes delegates to the wrapped HNSW index
func (w *hnswWrapper) SerializeToBytes() ([]byte, error) {
	return w.index.SerializeToBytes()
}

// DeserializeFromBytes delegates to the wrapped HNSW index
func (w *hnswWrapper) DeserializeFromBytes(ctx context.Context, data []byte) error {
	return w.index.DeserializeFromBytes(ctx, data)
}

// NEW: GetPersistenceMetadata delegates to the wrapped index
func (w *hnswWrapper) GetPersistenceMetadata() *PersistenceMetadata {
	hnswMeta := w.index.GetPersistenceMetadata()
	if hnswMeta == nil {
		return nil
	}

	// Convert from HNSW metadata to interface metadata
	return &PersistenceMetadata{
		Version:       hnswMeta.Version,
		NodeCount:     hnswMeta.NodeCount,
		Dimension:     hnswMeta.Dimension,
		MaxLevel:      hnswMeta.MaxLevel,
		IndexType:     "HNSW",
		CreatedAt:     hnswMeta.CreatedAt,
		ChecksumCRC32: hnswMeta.ChecksumCRC32,
		FileSize:      hnswMeta.FileSize,
	}
}

// NewHNSW creates a new HNSW index
func NewHNSW(config *HNSWConfig) (Index, error) {
	// Convert to internal HNSW config
	hnswConfig := &hnsw.Config{
		Dimension:            config.Dimension,
		M:                    config.M,
		EfConstruction:       config.EfConstruction,
		EfSearch:             config.EfSearch,
		ML:                   config.ML,
		Metric:               config.Metric,
		PruneAlpha:           config.PruneAlpha,
		Level0LinkMultiplier: config.Level0LinkMultiplier,
		RepairEnabled:        config.RepairEnabled,
		RepairQueueSize:      config.RepairQueueSize,
		RepairBatchSize:      config.RepairBatchSize,
		Provider:             config.Provider,
		RandomSeed:           0, // Default seed for Phase 1
		RawVectorStore:       config.RawVectorStore,
		RawStoreCap:          config.RawStoreCap,
		IDMapCapacity:        config.IDMapCapacity,
		Quantization:         config.Quantization,
	}

	hnswIndex, err := hnsw.NewHNSW(hnswConfig)
	if err != nil {
		return nil, err
	}

	slotSize := 48 + uint64(unsafe.Sizeof(hnsw.VectorEntry{}))
	slotSize = (slotSize + 7) &^ 7
	if slotSize < 32 {
		slotSize = 32
	}
	sfl, err := memory.NewShardedFreeList(memory.FreeListConfig{
		SlotSize:  slotSize,
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 4,
	}, 64, 16)
	if err != nil {
		hnswIndex.Close()
		return nil, err
	}

	return &hnswWrapper{index: hnswIndex, sfl: sfl}, nil
}

// ivfpqWrapper wraps the IVF-PQ index to adapt between interface types
type ivfpqWrapper struct {
	index *ivfpq.Index
	sfl   *memory.ShardedFreeList
}

func (w *ivfpqWrapper) Train(ctx context.Context, vectors [][]float32) error {
	return w.index.Train(ctx, vectors)
}

func (w *ivfpqWrapper) IsTrained() bool {
	return w.index.IsTrained()
}

// HasDeserializedMeta reports whether deserialized entry metadata is pending
// population. Used by the collection layer to decide between two-phase
// deserialization (PopulateEntriesFromStorage) and a fully-populated index.
func (w *ivfpqWrapper) HasDeserializedMeta() bool {
	return w.index.HasDeserializedMeta()
}

// PopulateEntriesFromStorage delegates to the wrapped IVFPQ index so the
// collection layer can wire deserialized centroids/codebooks to storage entries.
// Uses anonymous interface to match the type assertion in the collection layer.
func (w *ivfpqWrapper) PopulateEntriesFromStorage(provider interface {
	IterateEntries(fn func(id string, ordinal uint32, vector []float32, metadata map[string]interface{}) error) error
}) error {
	return w.index.PopulateEntriesFromStorage(provider)
}

// Insert adapts the interface VectorEntry to IVF-PQ VectorEntry
func (w *ivfpqWrapper) Insert(ctx context.Context, entry *VectorEntry) error {
	ivfpqEntry := &ivfpq.VectorEntry{
		ID:       entry.ID,
		Ordinal:  entry.Ordinal,
		Vector:   entry.Vector,
		Metadata: entry.Metadata,
		Version:  entry.Version,
	}
	return w.index.Insert(ctx, ivfpqEntry)
}

// BatchInsert delegates batch insertion to the underlying IVF-PQ index
func (w *ivfpqWrapper) BatchInsert(ctx context.Context, entries []*VectorEntry) error {
	ivfpqEntries := make([]*ivfpq.VectorEntry, len(entries))
	slots := make([][]byte, len(entries))
	for i, entry := range entries {
		slot, err := w.sfl.Allocate()
		if err != nil {
			for j := 0; j < i; j++ {
				_ = w.sfl.Deallocate(slots[j])
			}
			return err
		}
		slots[i] = slot
		ivfpqEntry := (*ivfpq.VectorEntry)(unsafe.Pointer(&slot[48]))
		ivfpqEntry.ID = entry.ID
		ivfpqEntry.Ordinal = entry.Ordinal
		ivfpqEntry.Vector = entry.Vector
		ivfpqEntry.Metadata = entry.Metadata
		ivfpqEntry.Version = entry.Version
		ivfpqEntries[i] = ivfpqEntry
	}
	err := w.index.BatchInsert(ctx, ivfpqEntries)
	for _, slot := range slots {
		_ = w.sfl.Deallocate(slot)
	}
	return err
}

// Search adapts the search results from IVF-PQ to interface types
func (w *ivfpqWrapper) Search(ctx context.Context, query []float32, k int, filter GraphFilter) ([]*SearchResult, error) {
	ivfpqResults, err := w.index.Search(ctx, query, k, filter)
	if err != nil {
		return nil, err
	}

	results := make([]*SearchResult, len(ivfpqResults))
	for i, r := range ivfpqResults {
		results[i] = &SearchResult{
			ID:       r.ID,
			Score:    r.Score,
			Vector:   r.Vector,
			Metadata: r.Metadata,
			Version:  r.Version,
		}
	}
	return results, nil
}

// Delete delegates to the wrapped index
func (w *ivfpqWrapper) Delete(ctx context.Context, id string) error {
	return w.index.Delete(ctx, id)
}

// Size delegates to the wrapped index
func (w *ivfpqWrapper) Size() int {
	return w.index.Size()
}

// MemoryUsage delegates to the wrapped index
func (w *ivfpqWrapper) MemoryUsage() int64 {
	return w.index.MemoryUsage()
}

// Close delegates to the wrapped index
func (w *ivfpqWrapper) Close() error {
	if w.sfl != nil {
		_ = w.sfl.Free()
	}
	return w.index.Close()
}

// SaveToDisk delegates persistence to the wrapped index
func (w *ivfpqWrapper) SaveToDisk(ctx context.Context, path string) error {
	return w.index.SaveToDisk(ctx, path)
}

// LoadFromDisk delegates loading to the wrapped index
func (w *ivfpqWrapper) LoadFromDisk(ctx context.Context, path string) error {
	return w.index.LoadFromDisk(ctx, path)
}

// SerializeToBytes delegates to the wrapped IVFPQ index
func (w *ivfpqWrapper) SerializeToBytes() ([]byte, error) {
	return w.index.SerializeToBytes()
}

// DeserializeFromBytes delegates to the wrapped IVFPQ index
func (w *ivfpqWrapper) DeserializeFromBytes(ctx context.Context, data []byte) error {
	return w.index.DeserializeFromBytes(ctx, data)
}

// GetPersistenceMetadata delegates to the wrapped index
func (w *ivfpqWrapper) GetPersistenceMetadata() *PersistenceMetadata {
	ivfpqMeta := w.index.GetPersistenceMetadata()
	if ivfpqMeta == nil {
		return nil
	}

	// FileSize is the total bytes of compressed vectors (the on-disk payload
	// for an IVF-PQ index). ChecksumCRC32 stays 0 here because the in-memory
	// representation has no canonical serialized form without calling
	// SerializeToBytes (which would allocate the full buffer). Callers that
	// need a real CRC should persist via SaveToDisk and read the on-disk
	// header instead.
	return &PersistenceMetadata{
		Version:       1,
		NodeCount:     w.index.Size(),
		Dimension:     w.index.GetConfig().Dimension,
		MaxLevel:      0, // Not applicable for IVF-PQ
		IndexType:     "IVF-PQ",
		CreatedAt:     time.Now(),
		ChecksumCRC32: 0, // 0 for in-memory; populated by on-disk format header
		FileSize:      ivfpqMeta.CompressedSize,
	}
}

// NewIVFPQ creates a new IVF-PQ index
func NewIVFPQ(config *IVFPQConfig) (Index, error) {
	// Convert to internal IVF-PQ config
	ivfpqConfig := &ivfpq.Config{
		Dimension:     config.Dimension,
		NClusters:     config.NClusters,
		NProbes:       config.NProbes,
		Metric:        config.Metric,
		Quantization:  config.Quantization,
		MaxIterations: config.MaxIterations,
		Tolerance:     config.Tolerance,
		RandomSeed:    config.RandomSeed,
	}

	ivfpqIndex, err := ivfpq.NewIVFPQ(ivfpqConfig)
	if err != nil {
		return nil, err
	}

	slotSize := 48 + uint64(unsafe.Sizeof(ivfpq.VectorEntry{}))
	slotSize = (slotSize + 7) &^ 7
	if slotSize < 32 {
		slotSize = 32
	}
	sfl, err := memory.NewShardedFreeList(memory.FreeListConfig{
		SlotSize:  slotSize,
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 4,
	}, 64, 16)
	if err != nil {
		ivfpqIndex.Close()
		return nil, err
	}

	return &ivfpqWrapper{index: ivfpqIndex, sfl: sfl}, nil
}

// flatWrapper wraps the Flat index to adapt between interface types
type flatWrapper struct {
	core *flat.Core
}

const flatCoreFormatVersion uint32 = 2

func newFlatWrapper(core *flat.Core) (*flatWrapper, error) {
	if core == nil {
		return nil, fmt.Errorf("flat core is nil")
	}
	return &flatWrapper{core: core}, nil
}

func flatDeltaBytes(entries []*VectorEntry) uint64 {
	bytes := uint64(16 << 10)
	for _, entry := range entries {
		if entry == nil {
			continue
		}
		bytes += uint64(len(entry.ID)) + uint64(len(entry.Vector))*4 + 256
	}
	return bytes
}

// Insert publishes one immutable off-heap generation. Metadata remains owned
// by the authoritative storage layer and is hydrated at the public boundary;
// Flat therefore never retains a Go metadata map.
func (w *flatWrapper) Insert(ctx context.Context, entry *VectorEntry) error {
	return w.BatchInsert(ctx, []*VectorEntry{entry})
}

// BatchInsert stages all writes in one bounded delta and publishes exactly one
// new generation. The core copies every retained byte into its arena before
// publication, so caller-owned slices and maps cannot leak into the index.
func (w *flatWrapper) BatchInsert(ctx context.Context, entries []*VectorEntry) error {
	prepared, err := w.PrepareMutations(ctx, entries, nil)
	if err != nil {
		return err
	}
	defer prepared.Abort()
	return prepared.Commit()
}

type flatPreparedMutation struct{ prepared *flat.PreparedDelta }

func (p *flatPreparedMutation) Commit() error {
	if p == nil || p.prepared == nil {
		return fmt.Errorf("flat prepared mutation is closed")
	}
	err := p.prepared.Commit()
	if err == nil {
		p.prepared = nil
	}
	return err
}

func (p *flatPreparedMutation) Abort() error {
	if p == nil || p.prepared == nil {
		return nil
	}
	err := p.prepared.Abort()
	p.prepared = nil
	return err
}

// PrepareMutations stages one transaction-owned delta against the current
// immutable generation. It does not publish anything; callers commit it only
// after the matching storage/WAL transaction succeeds.
func (w *flatWrapper) PrepareMutations(ctx context.Context, entries []*VectorEntry, deletes []string) (PreparedMutation, error) {
	if len(entries) == 0 && len(deletes) == 0 {
		return nil, fmt.Errorf("flat mutation is empty")
	}
	delta, err := w.core.NewDelta(flatDeltaBytes(entries)+uint64(len(deletes))*128, uint32(len(entries)+len(deletes)), uint64(len(entries))*128+uint64(len(deletes))*128+4096)
	if err != nil {
		return nil, err
	}
	prepared := false
	defer func() {
		if !prepared {
			_ = delta.Close()
		}
	}()
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if entry == nil || entry.ID == "" {
			return nil, fmt.Errorf("flat entry ID cannot be empty")
		}
		if len(entry.Vector) != w.core.Config().Dimension {
			return nil, fmt.Errorf("vector dimension %d does not match index dimension %d", len(entry.Vector), w.core.Config().Dimension)
		}
		id := record.BorrowBytes([]byte(entry.ID))
		before, found := w.core.CurrentRecord(id)
		if !found {
			before = record.RecordRef{}
		}
		_, _, err = delta.StagePut(record.MutationUpsert, before, record.RecordBuilder{
			ID:      id,
			Vector:  record.BorrowVector(entry.Vector),
			Version: entry.Version,
			Ordinal: entry.Ordinal,
		}, 0, false)
		if err != nil {
			return nil, err
		}
	}
	for _, id := range deletes {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if id == "" {
			return nil, fmt.Errorf("flat entry ID cannot be empty")
		}
		key := record.BorrowBytes([]byte(id))
		before, found := w.core.CurrentRecord(key)
		if !found {
			continue
		}
		if _, _, err := delta.StageDelete(before, key, 0, false); err != nil {
			return nil, err
		}
	}
	candidate, err := w.core.PrepareDelta(delta)
	if err != nil {
		return nil, err
	}
	prepared = true
	return &flatPreparedMutation{prepared: candidate}, nil
}

// Search adapts the search results from Flat to interface types
func (w *flatWrapper) Search(ctx context.Context, query []float32, k int, filter GraphFilter) ([]*SearchResult, error) {
	set, err := w.core.SearchBorrowed(ctx, record.BorrowVector(query), k, filter)
	if err != nil {
		return nil, err
	}
	defer set.Close()

	results := make([]*SearchResult, set.Len())
	for i := 0; i < set.Len(); i++ {
		ref, score := set.At(i)
		results[i] = &SearchResult{
			ID:      string(ref.ID().Bytes()),
			Score:   score,
			Version: ref.Version(),
			Ordinal: ref.Ordinal(),
		}
	}
	return results, nil
}

// Delete delegates to the wrapped index
func (w *flatWrapper) Delete(ctx context.Context, id string) error {
	prepared, err := w.PrepareMutations(ctx, nil, []string{id})
	if err != nil {
		return err
	}
	defer prepared.Abort()
	return prepared.Commit()
}

// Size delegates to the wrapped index
func (w *flatWrapper) Size() int {
	return w.core.Size()
}

// MemoryUsage delegates to the wrapped index
func (w *flatWrapper) MemoryUsage() int64 {
	return w.core.MemoryUsage()
}

// Close delegates to the wrapped index
func (w *flatWrapper) Close() error {
	return w.core.Close()
}

// SaveToDisk writes a generation snapshot atomically. Metadata is persisted by
// the storage engine, which is its authoritative owner; Flat snapshots retain
// only the routing fields needed to restore vector search.
func (w *flatWrapper) SaveToDisk(ctx context.Context, path string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	data, err := w.SerializeToBytes()
	if err != nil {
		return err
	}
	temporary := path + ".tmp"
	if err := os.WriteFile(temporary, data, 0o644); err != nil {
		return err
	}
	if err := os.Rename(temporary, path); err != nil {
		_ = os.Remove(temporary)
		return err
	}
	return nil
}

func (w *flatWrapper) LoadFromDisk(ctx context.Context, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return w.DeserializeFromBytes(ctx, data)
}

func (w *flatWrapper) SerializeToBytes() ([]byte, error) {
	config := w.core.Config()
	enc := &util.BinaryEncoder{}
	enc.WriteUint32(uint32(config.Dimension))
	enc.WriteUint32(uint32(config.Metric))
	hasQuantization := config.Quantization != nil
	enc.WriteBool(hasQuantization)
	if hasQuantization {
		enc.WriteUint32(uint32(config.Quantization.Type))
		enc.WriteUint32(uint32(config.Quantization.Codebooks))
		enc.WriteUint32(uint32(config.Quantization.Bits))
		enc.WriteFloat64(config.Quantization.TrainRatio)
		enc.WriteUint32(uint32(config.Quantization.CacheSize))
	}
	enc.WriteUint32(uint32(w.core.Size()))
	if err := w.core.VisitVisible(func(ref record.RecordRef) error {
		enc.WriteString(string(ref.ID().Bytes()))
		enc.WriteUint32(ref.Ordinal())
		enc.WriteUint64(ref.Version())
		enc.WriteVector(ref.Vector().Float32s())
		return nil
	}); err != nil {
		return nil, err
	}
	body := enc.DetachBytes()
	result := make([]byte, 16+len(body))
	copy(result[:8], flat.FlatMagicBytes)
	binary.LittleEndian.PutUint32(result[8:12], flatCoreFormatVersion)
	binary.LittleEndian.PutUint32(result[12:16], crc32.ChecksumIEEE(body))
	copy(result[16:], body)
	return result, nil
}

func (w *flatWrapper) DeserializeFromBytes(ctx context.Context, data []byte) error {
	if len(data) < 16 || string(data[:8]) != string(flat.FlatMagicBytes) {
		return fmt.Errorf("invalid flat core snapshot")
	}
	if binary.LittleEndian.Uint32(data[8:12]) != flatCoreFormatVersion {
		return fmt.Errorf("unsupported flat core snapshot version %d", binary.LittleEndian.Uint32(data[8:12]))
	}
	body := data[16:]
	if crc32.ChecksumIEEE(body) != binary.LittleEndian.Uint32(data[12:16]) {
		return fmt.Errorf("flat core snapshot checksum mismatch")
	}
	dec := &util.BinaryDecoder{Data: body}
	dimension, err := dec.ReadUint32()
	if err != nil {
		return err
	}
	metric, err := dec.ReadUint32()
	if err != nil {
		return err
	}
	hasQuantization, err := dec.ReadBool()
	if err != nil {
		return err
	}
	config := &flat.Config{Dimension: int(dimension), Metric: util.DistanceMetric(metric)}
	if hasQuantization {
		kind, err := dec.ReadUint32()
		if err != nil {
			return err
		}
		codebooks, err := dec.ReadUint32()
		if err != nil {
			return err
		}
		bits, err := dec.ReadUint32()
		if err != nil {
			return err
		}
		ratio, err := dec.ReadFloat64()
		if err != nil {
			return err
		}
		cacheSize, err := dec.ReadUint32()
		if err != nil {
			return err
		}
		config.Quantization = &quant.QuantizationConfig{Type: quant.QuantizationType(kind), Codebooks: int(codebooks), Bits: int(bits), TrainRatio: ratio, CacheSize: int(cacheSize)}
	}
	count, err := dec.ReadUint32()
	if err != nil {
		return err
	}
	core, err := flat.NewCore(config)
	if err != nil {
		return err
	}
	if count > 0 {
		delta, err := core.NewDelta(uint64(len(body))+16<<10, count, uint64(len(body))+4096)
		if err != nil {
			_ = core.Close()
			return err
		}
		for i := uint32(0); i < count; i++ {
			if err := ctx.Err(); err != nil {
				_ = delta.Close()
				_ = core.Close()
				return err
			}
			id, err := dec.ReadString()
			if err != nil {
				_ = delta.Close()
				_ = core.Close()
				return err
			}
			ordinal, err := dec.ReadUint32()
			if err != nil {
				_ = delta.Close()
				_ = core.Close()
				return err
			}
			version, err := dec.ReadUint64()
			if err != nil {
				_ = delta.Close()
				_ = core.Close()
				return err
			}
			vector, err := dec.ReadVector()
			if err != nil {
				_ = delta.Close()
				_ = core.Close()
				return err
			}
			if _, _, err := delta.StagePut(record.MutationInsert, record.RecordRef{}, record.RecordBuilder{ID: record.BorrowBytes([]byte(id)), Vector: record.BorrowVector(vector), Ordinal: ordinal, Version: version}, 0, false); err != nil {
				_ = delta.Close()
				_ = core.Close()
				return err
			}
		}
		if err := core.CommitDelta(delta); err != nil {
			_ = core.Close()
			return err
		}
	}
	previous := w.core
	w.core = core
	return previous.Close()
}

// GetPersistenceMetadata delegates to the wrapped index
func (w *flatWrapper) GetPersistenceMetadata() *PersistenceMetadata {
	return &PersistenceMetadata{
		Version:   flatCoreFormatVersion,
		NodeCount: w.core.Size(),
		Dimension: w.core.Config().Dimension,
		IndexType: "Flat",
	}
}

// NewFlat creates a new Flat index
func NewFlat(config *FlatConfig) (Index, error) {
	// Convert to internal Flat config
	flatConfig := &flat.Config{
		Dimension:    config.Dimension,
		Metric:       config.Metric,
		Quantization: config.Quantization,
	}

	flatIndex, err := flat.NewCore(flatConfig)
	if err != nil {
		return nil, err
	}

	return newFlatWrapper(flatIndex)
}
