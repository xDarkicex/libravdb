package flat

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"sync"
	"time"
	"unsafe"

	"github.com/xDarkicex/memory"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

const (
	FlatFormatVersion = 1
)

var FlatMagicBytes = []byte("LIBRAFLT")

// VectorEntry represents a vector entry in the flat index
type VectorEntry struct {
	Metadata     map[string]interface{} `json:"metadata"`
	ID           string                 `json:"id"`
	Vector       []float32              `json:"vector"`
	Version      uint64                 `json:"version"`
	metadataSize int64
}

// SearchResult represents a search result from the flat index
type SearchResult struct {
	Metadata map[string]interface{} `json:"metadata"`
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Version  uint64                 `json:"version"`
	Score    float32                `json:"score"`
}

// Config holds configuration for the flat index
type Config struct {
	Quantization *quant.QuantizationConfig `json:"quantization,omitempty"`
	Dimension    int                       `json:"dimension"`
	Metric       util.DistanceMetric       `json:"metric"`
}

// PersistenceMetadata holds metadata about persisted flat index
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

// Index implements a flat (brute-force) vector index
type Index struct {
	quantizer   quant.Quantizer
	config      *Config
	idToIndex   map[string]int
	vectorSFL   *memory.ShardedFreeList
	scratchPool *sync.Pool
	vectors     []*VectorEntry
	mu          sync.RWMutex
	queryTiers  [4]poolTier
}

// NewFlat creates a new flat index
func NewFlat(config *Config) (*Index, error) {
	if config.Dimension <= 0 {
		return nil, fmt.Errorf("dimension must be positive, got %d", config.Dimension)
	}

	sfl, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  256 * 1024 * 1024,
		SlotSize:  uint64(48 + config.Dimension*4),
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 8,
		Prealloc:  false,
	}, 64)
	if err != nil {
		return nil, fmt.Errorf("failed to create memory pool for vectors: %w", err)
	}

	index := &Index{
		config:    config,
		vectors:   make([]*VectorEntry, 0),
		idToIndex: make(map[string]int),
		vectorSFL: sfl,
		scratchPool: &sync.Pool{
			New: func() any {
				a, err := memory.NewArena(1024 * 1024)
				if err != nil {
					return nil
				}
				return a
			},
		},
		queryTiers: [4]poolTier{
			{maxK: 16},
			{maxK: 128},
			{maxK: 1024},
			{maxK: 4096},
		},
	}

	// Initialize quantizer if configured
	if config.Quantization != nil {
		var err error
		index.quantizer, err = quant.Create(config.Quantization)
		if err != nil {
			return nil, fmt.Errorf("failed to create quantizer: %w", err)
		}
	}

	return index, nil
}

// Insert adds a vector to the index.
// The index retains a reference to entry.Metadata. Callers that mutate the map after insertion will observe the mutation in subsequent Search results.
func (idx *Index) Insert(ctx context.Context, entry *VectorEntry) error {
	if len(entry.Vector) == 0 {
		return fmt.Errorf("vector cannot be empty")
	}
	if len(entry.Vector) != idx.config.Dimension {
		return fmt.Errorf("dimension mismatch: expected %d, got %d: %w",
			idx.config.Dimension, len(entry.Vector), util.ErrDimension)
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	return idx.insertLocked(entry)
}

// BatchInsert adds multiple vectors to the index under a single lock.
// The index retains a reference to entry.Metadata. Callers that mutate the map after insertion will observe the mutation in subsequent Search results.
func (idx *Index) BatchInsert(ctx context.Context, entries []*VectorEntry) error {
	if len(entries) == 0 {
		return nil
	}

	for i, entry := range entries {
		if entry == nil {
			return fmt.Errorf("entry at index %d is nil", i)
		}
		if len(entry.Vector) == 0 {
			return fmt.Errorf("vector at index %d cannot be empty", i)
		}
		if len(entry.Vector) != idx.config.Dimension {
			return fmt.Errorf("dimension mismatch: expected %d, got %d: %w",
				idx.config.Dimension, len(entry.Vector), util.ErrDimension)
		}
	}

	newEntries := make([]*VectorEntry, len(entries))
	for i, entry := range entries {
		slot, err := idx.vectorSFL.Allocate()
		if err != nil {
			// Rollback allocated slots on error
			for j := 0; j < i; j++ {
				idx.deallocateVector(newEntries[j].Vector)
			}
			return err
		}

		vecPtr := unsafe.Pointer(&slot[48])
		offHeapVec := unsafe.Slice((*float32)(vecPtr), idx.config.Dimension)
		copy(offHeapVec, entry.Vector)

		newEntries[i] = &VectorEntry{
			ID:           entry.ID,
			Vector:       offHeapVec,
			Metadata:     cloneMetadata(entry.Metadata),
			Version:      entry.Version,
			metadataSize: estimateMetadataSize(entry.Metadata),
		}
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	if cap(idx.vectors) < len(idx.vectors)+len(entries) {
		newVecs := make([]*VectorEntry, len(idx.vectors), len(idx.vectors)+len(entries)*2)
		copy(newVecs, idx.vectors)
		idx.vectors = newVecs
	}

	for _, newEntry := range newEntries {
		if existingIndex, exists := idx.idToIndex[newEntry.ID]; exists {
			idx.deallocateVector(idx.vectors[existingIndex].Vector)
			idx.vectors[existingIndex] = newEntry
		} else {
			idx.idToIndex[newEntry.ID] = len(idx.vectors)
			idx.vectors = append(idx.vectors, newEntry)
		}
	}

	return nil
}

// insertLocked inserts or updates an entry. The caller must hold idx.mu.
func (idx *Index) insertLocked(entry *VectorEntry) error {
	// Add new entry
	slot, err := idx.vectorSFL.Allocate()
	if err != nil {
		return err
	}
	vecPtr := unsafe.Pointer(&slot[48])
	offHeapVec := unsafe.Slice((*float32)(vecPtr), idx.config.Dimension)
	copy(offHeapVec, entry.Vector)

	mdSize := estimateMetadataSize(entry.Metadata)
	newEntry := &VectorEntry{
		ID:           entry.ID,
		Vector:       offHeapVec,
		Metadata:     cloneMetadata(entry.Metadata),
		Version:      entry.Version,
		metadataSize: mdSize,
	}

	if existingIndex, exists := idx.idToIndex[entry.ID]; exists {
		idx.deallocateVector(idx.vectors[existingIndex].Vector)
		idx.vectors[existingIndex] = newEntry
		return nil
	}

	idx.idToIndex[entry.ID] = len(idx.vectors)
	idx.vectors = append(idx.vectors, newEntry)

	return nil
}

// heapElement is a max-heap node storing a vector index and its distance.
// Sized at 16 bytes (int=8 + float32=4 + padding) — fits two per cache line.
type heapElement struct {
	vecIdx int
	score  float32
}

// upHeap bubbles the element at i up to restore max-heap property.
func upHeap(h []heapElement, i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h[parent].score >= h[i].score {
			break
		}
		h[parent], h[i] = h[i], h[parent]
		i = parent
	}
}

// downHeap sifts the element at i down to restore max-heap property.
func downHeap(h []heapElement, i, n int) {
	for {
		largest := i
		left := 2*i + 1
		right := 2*i + 2
		if left < n && h[left].score > h[largest].score {
			largest = left
		}
		if right < n && h[right].score > h[largest].score {
			largest = right
		}
		if largest == i {
			break
		}
		h[i], h[largest] = h[largest], h[i]
		i = largest
	}
}

// userDataOffset is the byte offset within a ShardedFreeList slot where user
// data begins. The memory package's SFL metadata occupies offsets 0–43
// (Hyaline chain at 0/8/16/24/32, structIdx+shardIdx at 40); 8-byte aligned to 48.
const userDataOffset = 48

// heapSlot binds an off-heap slot to its originating pool so that free()
// routes to the correct tier by construction — no runtime lookup, no ignored
// error from a mismatched Deallocate.
type heapSlot struct {
	pool *memory.ShardedFreeList
	slot []byte
}

func (hs *heapSlot) free() { hs.pool.Deallocate(hs.slot) }

type poolTier struct {
	pool *memory.ShardedFreeList
	maxK int
	once sync.Once
}

// acquireHeapSlot returns a heapSlot paired with a []heapElement buffer backed
// by the appropriate off-heap tier. The buffer has len=k and cap=tierMaxK.
// Returns nil, nil if k exceeds the largest tier — caller must fall back to
// Go heap allocation.
func (idx *Index) acquireHeapSlot(k int) (*heapSlot, []heapElement) {
	for i := range idx.queryTiers {
		if k > idx.queryTiers[i].maxK {
			continue
		}
		tier := &idx.queryTiers[i]
		tier.once.Do(func() {
			slotSize := uint64(userDataOffset + tier.maxK*16)
			pool, err := memory.NewShardedFreeList(memory.FreeListConfig{
				PoolSize:  16 * 1024 * 1024,
				SlotSize:  slotSize,
				SlabSize:  1 * 1024 * 1024,
				SlabCount: 16,
				Prealloc:  true,
			}, 64)
			if err != nil {
				panic("flat: failed to create query pool tier: " + err.Error())
			}
			tier.pool = pool
		})
		slot, err := tier.pool.Allocate()
		if err != nil {
			return nil, nil
		}
		ptr := unsafe.Add(unsafe.Pointer(unsafe.SliceData(slot)), userDataOffset)
		// Create the buffer with the tier's full capacity, then re-slice to k
		// so len(heapBuf) is the authority for heap bounds.
		heapBuf := unsafe.Slice((*heapElement)(ptr), tier.maxK)[:k]
		return &heapSlot{slot: slot, pool: tier.pool}, heapBuf
	}
	return nil, nil
}

// Search finds the k nearest neighbors using exhaustive search
func (idx *Index) Search(ctx context.Context, query []float32, k int, filter interface {
	Test(idx uint64) bool
}) ([]*SearchResult, error) {
	if len(query) != idx.config.Dimension {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d",
			idx.config.Dimension, len(query))
	}

	if k <= 0 {
		return []*SearchResult{}, nil
	}
	if k > 4096 {
		return nil, fmt.Errorf("k %d exceeds maximum allowed search result limit of 4096", k)
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(idx.vectors) == 0 {
		return []*SearchResult{}, nil
	}

	if k > len(idx.vectors) {
		k = len(idx.vectors)
	}

	limit := k

	// Acquire an arena for search-scoped scratch, released on return.
	arena := idx.scratchPool.Get().(*memory.Arena)
	if arena == nil {
		a, err := memory.NewArena(1024 * 1024)
		if err != nil {
			return nil, fmt.Errorf("arena allocate for search: %w", err)
		}
		arena = a
	}
	defer func() {
		arena.Reset()
		idx.scratchPool.Put(arena)
	}()

	// Acquire off-heap buffer for the heap. Gracefully degrades to arena
	// allocation if k exceeds the largest tier or the SFL pool is exhausted.
	var heapBuf []heapElement
	hs, buf := idx.acquireHeapSlot(k)
	if hs != nil {
		defer hs.free()
		heapBuf = buf
	} else {
		var err error
		heapBuf, err = memory.ArenaSlice[heapElement](arena, k)
		if err != nil {
			return nil, fmt.Errorf("arena allocate heap buf: %w", err)
		}
		heapBuf = heapBuf[:k]
	}

	count := 0

	for i := 0; i < len(idx.vectors); i++ {
		select {
		case <-ctx.Done():
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		default:
		}

		if filter != nil && !filter.Test(uint64(i)) {
			continue
		}

		entry := idx.vectors[i]
		distance, err := idx.computeDistance(query, entry.Vector)
		if err != nil {
			return nil, fmt.Errorf("failed to compute distance: %w", err)
		}

		if count < limit {
			heapBuf[count] = heapElement{vecIdx: i, score: distance}
			upHeap(heapBuf, count)
			count++
		} else if distance < heapBuf[0].score {
			heapBuf[0] = heapElement{vecIdx: i, score: distance}
			downHeap(heapBuf, 0, count)
		}
	}

	// Extract results in ascending distance order (smallest first).
	results := make([]*SearchResult, count)
	for i := count - 1; i >= 0; i-- {
		// Pop root (largest distance)
		elem := heapBuf[0]
		count--
		heapBuf[0] = heapBuf[count]
		downHeap(heapBuf, 0, count)

		entry := idx.vectors[elem.vecIdx]
		vec := make([]float32, len(entry.Vector))
		copy(vec, entry.Vector)
		results[i] = &SearchResult{
			ID:       entry.ID,
			Score:    elem.score,
			Vector:   vec,
			Metadata: cloneMetadata(entry.Metadata),
			Version:  entry.Version,
		}
	}

	return results, nil
}

// Delete removes a vector from the index
func (idx *Index) Delete(ctx context.Context, id string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	index, exists := idx.idToIndex[id]
	if !exists {
		return fmt.Errorf("vector with ID %s: %w", id, util.ErrNotFound)
	}

	// Deallocate the off-heap vector before dropping the reference
	idx.deallocateVector(idx.vectors[index].Vector)

	// Remove from vectors slice
	idx.vectors = append(idx.vectors[:index], idx.vectors[index+1:]...)

	// Update idToIndex map
	delete(idx.idToIndex, id)
	for i := index; i < len(idx.vectors); i++ {
		idx.idToIndex[idx.vectors[i].ID] = i
	}

	return nil
}

// Size returns the number of vectors in the index
func (idx *Index) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vectors)
}

// MemoryUsage estimates the memory usage of the index
func (idx *Index) MemoryUsage() int64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	var usage int64

	// Vector storage
	usage += int64(len(idx.vectors)) * int64(idx.config.Dimension) * 4 // 4 bytes per float32

	// ID storage (estimate 20 bytes per ID on average)
	usage += int64(len(idx.vectors)) * 20

	// Index map overhead (estimate 32 bytes per entry)
	usage += int64(len(idx.idToIndex)) * 32

	// Metadata storage (cached per-entry, computed at insert)
	for _, entry := range idx.vectors {
		usage += entry.metadataSize
	}

	return usage
}

// Close cleans up the index resources
func (idx *Index) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.vectors = nil
	idx.idToIndex = nil
	if idx.quantizer != nil {
		idx.quantizer.Close()
		idx.quantizer = nil
	}
	if idx.vectorSFL != nil {
		idx.vectorSFL.Free()
		idx.vectorSFL = nil
	}

	for i := range idx.queryTiers {
		if idx.queryTiers[i].pool != nil {
			idx.queryTiers[i].pool.Free()
			idx.queryTiers[i].pool = nil
		}
	}

	return nil
}

// SaveToDisk persists the index to disk
func (idx *Index) SaveToDisk(ctx context.Context, path string) error {
	data, err := idx.SerializeToBytes()
	if err != nil {
		return fmt.Errorf("failed to serialize index: %w", err)
	}

	tempPath := path + ".tmp"
	if err := os.WriteFile(tempPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write temp file: %w", err)
	}
	if err := os.Rename(tempPath, path); err != nil {
		os.Remove(tempPath)
		return fmt.Errorf("failed to commit file: %w", err)
	}
	return nil
}

// LoadFromDisk loads the index from disk
func (idx *Index) LoadFromDisk(ctx context.Context, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return idx.DeserializeFromBytes(ctx, data)
}

// GetPersistenceMetadata returns metadata about the persisted index
func (idx *Index) GetPersistenceMetadata() *PersistenceMetadata {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	return &PersistenceMetadata{
		Version:   FlatFormatVersion,
		NodeCount: len(idx.vectors),
		Dimension: idx.config.Dimension,
		MaxLevel:  0, // Flat index has no levels
		IndexType: "Flat",
		CreatedAt: time.Now(),
	}
}

// SerializeToBytes serializes the index to an in-memory byte slice using binary encoding.
func (idx *Index) SerializeToBytes() ([]byte, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	enc := util.AcquireBinaryEncoder(256 + len(idx.vectors)*(4+idx.config.Dimension*4+256))
	defer util.ReleaseBinaryEncoder(enc)

	// Config
	enc.WriteUint32(uint32(idx.config.Dimension))
	enc.WriteUint32(uint32(idx.config.Metric))
	hasQ := idx.config.Quantization != nil
	enc.WriteBool(hasQ)
	if hasQ {
		enc.WriteUint32(uint32(idx.config.Quantization.Type))
		enc.WriteUint32(uint32(idx.config.Quantization.Codebooks))
		enc.WriteUint32(uint32(idx.config.Quantization.Bits))
		enc.WriteFloat64(idx.config.Quantization.TrainRatio)
		enc.WriteUint32(uint32(idx.config.Quantization.CacheSize))
	}

	// Vectors
	enc.WriteUint32(uint32(len(idx.vectors)))
	for _, entry := range idx.vectors {
		enc.WriteString(entry.ID)
		enc.WriteUint64(entry.Version)
		enc.WriteVector(entry.Vector)
		if err := enc.WriteMetadata(entry.Metadata); err != nil {
			return nil, fmt.Errorf("serialize entry %s metadata: %w", entry.ID, err)
		}
	}

	// Persistence metadata
	enc.WriteUint32(FlatFormatVersion)
	enc.WriteUint64(uint64(time.Now().UnixNano()))

	body := enc.DetachBytes()
	crc := crc32.ChecksumIEEE(body)
	buf := make([]byte, 16+len(body))
	copy(buf[0:8], FlatMagicBytes)
	binary.LittleEndian.PutUint32(buf[8:12], FlatFormatVersion)
	binary.LittleEndian.PutUint32(buf[12:16], crc)
	copy(buf[16:], body)

	return buf, nil
}

// DeserializeFromBytes restores the index from an in-memory byte slice.
func (idx *Index) DeserializeFromBytes(ctx context.Context, data []byte) error {
	if len(data) < 16 {
		return fmt.Errorf("invalid file format: file too short")
	}

	magic := data[0:8]
	if string(magic) != string(FlatMagicBytes) {
		return fmt.Errorf("invalid file format: incorrect magic bytes, use libravdb-migrate-flat for legacy files")
	}

	version := binary.LittleEndian.Uint32(data[8:12])
	if version != FlatFormatVersion {
		return fmt.Errorf("unsupported format version: %d", version)
	}

	expectedCRC := binary.LittleEndian.Uint32(data[12:16])
	body := data[16:]

	if crc32.ChecksumIEEE(body) != expectedCRC {
		return fmt.Errorf("data corruption: CRC mismatch")
	}

	dec := &util.BinaryDecoder{Data: body}

	// Config
	dim, err := dec.ReadUint32()
	if err != nil {
		return fmt.Errorf("read dimension: %w", err)
	}
	metric, err := dec.ReadUint32()
	if err != nil {
		return fmt.Errorf("read metric: %w", err)
	}
	hasQ, err := dec.ReadBool()
	if err != nil {
		return fmt.Errorf("read hasQuantization: %w", err)
	}
	cfg := &Config{
		Dimension: int(dim),
		Metric:    util.DistanceMetric(metric),
	}
	if hasQ {
		qType, err := dec.ReadUint32()
		if err != nil {
			return fmt.Errorf("read quant type: %w", err)
		}
		qCodebooks, err := dec.ReadUint32()
		if err != nil {
			return fmt.Errorf("read quant codebooks: %w", err)
		}
		qBits, err := dec.ReadUint32()
		if err != nil {
			return fmt.Errorf("read quant bits: %w", err)
		}
		qTrainRatio, err := dec.ReadFloat64()
		if err != nil {
			return fmt.Errorf("read quant train ratio: %w", err)
		}
		qCacheSize, err := dec.ReadUint32()
		if err != nil {
			return fmt.Errorf("read quant cache size: %w", err)
		}
		cfg.Quantization = &quant.QuantizationConfig{
			Type:       quant.QuantizationType(qType),
			Codebooks:  int(qCodebooks),
			Bits:       int(qBits),
			TrainRatio: qTrainRatio,
			CacheSize:  int(qCacheSize),
		}
	}

	// Vectors
	vecCount, err := dec.ReadUint32()
	if err != nil {
		return fmt.Errorf("read vector count: %w", err)
	}
	vectors := make([]*VectorEntry, int(vecCount))
	for i := range vectors {
		id, err := dec.ReadString()
		if err != nil {
			return fmt.Errorf("read entry %d id: %w", i, err)
		}
		ver, err := dec.ReadUint64()
		if err != nil {
			return fmt.Errorf("read entry %d version: %w", i, err)
		}
		vec, err := dec.ReadVector()
		if err != nil {
			return fmt.Errorf("read entry %d vector: %w", i, err)
		}
		metadata, err := dec.ReadMetadata()
		if err != nil {
			return fmt.Errorf("read entry %d metadata: %w", i, err)
		}
		vectors[i] = &VectorEntry{
			ID:           id,
			Version:      ver,
			Vector:       vec,
			Metadata:     metadata,
			metadataSize: estimateMetadataSize(metadata),
		}
	}

	// Move deserialized vectors off-heap before mutating the live index.
	// If any allocation fails, the index is untouched.
	for _, entry := range vectors {
		slot, err := idx.vectorSFL.Allocate()
		if err != nil {
			return err
		}
		vecPtr := unsafe.Pointer(&slot[48])
		offHeapVec := unsafe.Slice((*float32)(vecPtr), idx.config.Dimension)
		copy(offHeapVec, entry.Vector)
		entry.Vector = offHeapVec
	}
	newIDToIndex := make(map[string]int, len(vectors))
	for i, entry := range vectors {
		newIDToIndex[entry.ID] = i
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.config = cfg
	idx.vectors = vectors
	idx.idToIndex = newIDToIndex

	if cfg.Quantization != nil {
		var err error
		idx.quantizer, err = quant.Create(cfg.Quantization)
		if err != nil {
			return fmt.Errorf("failed to recreate quantizer: %w", err)
		}
	}

	return nil
}

// GetConfig returns the index configuration
func (idx *Index) GetConfig() *Config {
	return idx.config
}

// computeDistance computes the distance between two vectors
func (idx *Index) computeDistance(v1, v2 []float32) (float32, error) {
	switch idx.config.Metric {
	case util.CosineDistance:
		return util.CosineDistance_func(v1, v2), nil
	case util.L2Distance:
		return util.L2Distance_func(v1, v2), nil
	case util.InnerProduct:
		return util.InnerProduct_func(v1, v2), nil
	default:
		return 0, fmt.Errorf("unsupported distance metric: %v", idx.config.Metric)
	}
}

func estimateMetadataSize(md map[string]interface{}) int64 {
	var size int64
	for k, v := range md {
		size += int64(len(k)) + estimateValueSize(v)
	}
	return size
}

// estimateValueSize estimates the memory size of a metadata value
func estimateValueSize(v interface{}) int64 {
	switch val := v.(type) {
	case string:
		return int64(len(val))
	case int, int32, int64, float32, float64:
		return 8
	case bool:
		return 1
	case []interface{}:
		size := int64(0)
		for _, item := range val {
			size += estimateValueSize(item)
		}
		return size
	case map[string]interface{}:
		size := int64(0)
		for k, val := range val {
			size += int64(len(k)) + estimateValueSize(val)
		}
		return size
	default:
		return 16 // Default estimate
	}
}

// cloneMetadata returns a deep copy of the metadata map so callers
// cannot mutate the index's internal state through a stored reference.
func cloneMetadata(src map[string]interface{}) map[string]interface{} {
	if src == nil {
		return nil
	}
	dst := make(map[string]interface{}, len(src))
	for k, v := range src {
		dst[k] = deepCloneValue(v)
	}
	return dst
}

func deepCloneValue(v interface{}) interface{} {
	switch typed := v.(type) {
	case map[string]interface{}:
		return cloneMetadata(typed)
	case []interface{}:
		clone := make([]interface{}, len(typed))
		for i, item := range typed {
			clone[i] = deepCloneValue(item)
		}
		return clone
	case []string:
		clone := make([]string, len(typed))
		copy(clone, typed)
		return clone
	default:
		return v
	}
}

// deallocateVector returns the off-heap slice back to the ShardedFreeList.
func (idx *Index) deallocateVector(v []float32) {
	ptr := unsafe.Pointer(unsafe.SliceData(v))
	basePtr := unsafe.Pointer(uintptr(ptr) - 48)
	slot := unsafe.Slice((*byte)(basePtr), int(48+idx.config.Dimension*4))
	_ = idx.vectorSFL.Deallocate(slot)
}
