package hnsw

import (
	"fmt"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/memory"
)

const (
	RawVectorStoreMmap = "mmap"
)

// MmapRawVectorStore is a read-only vector store backed by a memory-mapped file.
// Vectors are stored sequentially by Ordinal (Slot).
type MmapRawVectorStore struct {
	dim    int
	mmap   *memory.MemoryMap
	active int
}

func NewMmapRawVectorStore(dim int, active int, mmap *memory.MemoryMap) *MmapRawVectorStore {
	return &MmapRawVectorStore{
		dim:    dim,
		mmap:   mmap,
		active: active,
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

	// Use unsafe to cast the byte slice to float32 slice directly from mmap
	// Note: We copy bytesAsFloat32View logic to avoid depending on slabby specifics
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[offset])), s.dim), nil
}

func (s *MmapRawVectorStore) Delete(ref VectorRef) error {
	return fmt.Errorf("MmapRawVectorStore is read-only")
}

func (s *MmapRawVectorStore) Reset() error {
	return nil // Memory map lifecycle is managed externally
}

func (s *MmapRawVectorStore) Close() error {
	return nil // Let garbage collection / manager handle it
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
