package hnsw

import (
	"fmt"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/memory"
	mem "github.com/xDarkicex/memory"
)

const (
	RawVectorStoreMmap = "mmap"
)

// MmapRawVectorStore is a read-only vector store backed by a memory-mapped file.
// Vectors are stored sequentially by Ordinal (Slot).
type MmapRawVectorStore struct {
	mmap     *memory.MemoryMap
	copyPool *mem.Pool
	dim      int
	active   int
}

func NewMmapRawVectorStore(dim int, active int, mmap *memory.MemoryMap) *MmapRawVectorStore {
	pool, err := mem.NewPool(mem.AllocatorConfig{
		PoolSize:  16 * 1024 * 1024,
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 8,
		Prealloc:  false,
	}, 64)
	if err != nil {
		// Fallback: nil pool means Get() will use make().
		// This should not happen in practice.
		return &MmapRawVectorStore{dim: dim, mmap: mmap, active: active}
	}
	return &MmapRawVectorStore{
		dim:      dim,
		mmap:     mmap,
		active:   active,
		copyPool: pool,
	}
}

func (s *MmapRawVectorStore) Put(vec []float32) (VectorRef, error) {
	return VectorRef{}, fmt.Errorf("MmapRawVectorStore is read-only")
}

func (s *MmapRawVectorStore) Get(ref VectorRef) ([]float32, error) {
	if s == nil || s.mmap == nil {
		return nil, fmt.Errorf("MmapRawVectorStore is closed or nil")
	}
	if !ref.Valid || ref.Kind != VectorEncodingRaw {
		return nil, fmt.Errorf("invalid raw vector reference")
	}

	bytesPerVector := s.dim * 4
	offset := int64(ref.Slot) * int64(bytesPerVector)

	if offset+int64(bytesPerVector) > s.mmap.Size() {
		return nil, fmt.Errorf("vector offset out of bounds")
	}

	data := s.mmap.Data()
	if offset+int64(bytesPerVector) > int64(len(data)) {
		return nil, fmt.Errorf("vector offset out of mmap data bounds")
	}

	// Copy into off-heap memory so the returned slice is independent of the
	// mmap region (which may be unmapped concurrently via DisableMemoryMapping).
	// Use Pool for off-heap allocation; pool lifetime matches the store so
	// returned vectors remain valid until the store is closed.
	if s.copyPool != nil {
		buf, err := s.copyPool.Allocate(uint64(s.dim * 4))
		if err != nil {
			return nil, fmt.Errorf("pool allocate vector copy: %w", err)
		}
		vec := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), s.dim)
		copy(vec, unsafe.Slice((*float32)(unsafe.Pointer(&data[offset])), s.dim))
		return vec, nil
	}
	// Fallback if pool creation failed during construction.
	vec := make([]float32, s.dim)
	copy(vec, unsafe.Slice((*float32)(unsafe.Pointer(&data[offset])), s.dim))
	return vec, nil
}

func (s *MmapRawVectorStore) Delete(ref VectorRef) error {
	return fmt.Errorf("MmapRawVectorStore is read-only")
}

func (s *MmapRawVectorStore) Reset() error {
	return nil // Memory map lifecycle is managed externally
}

func (s *MmapRawVectorStore) Close() error {
	if s.copyPool != nil {
		s.copyPool.Free()
		s.copyPool = nil
	}
	return nil
}

func (s *MmapRawVectorStore) MemoryUsage() int64 {
	if s == nil || s.mmap == nil {
		return 0
	}
	// Off-heap memory mapped size
	return s.mmap.Size()
}

func (s *MmapRawVectorStore) Profile() RawVectorStoreProfile {
	bytes := int64(0)
	if s.mmap != nil {
		bytes = s.mmap.Size()
	}
	return RawVectorStoreProfile{
		Backend:             RawVectorStoreMmap,
		VectorCount:         s.active,
		Dimension:           s.dim,
		BytesPerVector:      s.dim * 4,
		MemoryUsage:         bytes,
		ReservedBytes:       bytes,
		ReservedDataBytes:   bytes,
		ReservedMetaBytes:   0,
		ReservedGuardBytes:  0,
		LiveBytes:           bytes,
		FreeBytes:           0,
		CapacityUtilization: 1.0,
	}
}
