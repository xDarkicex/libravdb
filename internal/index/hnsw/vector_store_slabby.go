package hnsw

import (
	"fmt"
	"sync/atomic"
	"unsafe"

	"github.com/xDarkicex/memory"
)

const (
	RawVectorStoreMemory = "memory"
	RawVectorStoreSlabby = "slabby"

	defaultSlabbySegmentCapacity = 4096
	userDataOffset               = 64
)

type SlabbyRawVectorStore struct {
	sfl             *memory.ShardedFreeList
	metadataPool    *memory.Pool
	slots           rawSlotArray[byte]
	dim             int
	bytesPerVector  int
	slotSize        int
	segmentCapacity int
	activeCount     atomic.Int32
	nextSlot        atomic.Uint32
}

func NewSlabbyRawVectorStore(dim, segmentCapacity int) (*SlabbyRawVectorStore, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("invalid vector dimension: %d", dim)
	}
	if segmentCapacity <= 0 {
		segmentCapacity = defaultSlabbySegmentCapacity
	}

	bytesPerVector := dim * 4
	slotSize := uint64(bytesPerVector + userDataOffset)
	slotSize = (slotSize + 7) &^ 7

	sfl, err := memory.NewShardedFreeList(memory.FreeListConfig{
		SlotSize:  slotSize,
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 16,
	}, 64, 16)
	if err != nil {
		return nil, fmt.Errorf("failed to create memory pool for slabby store: %w", err)
	}
	metadataPool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize: 256 * 1024 * 1024,
		SlabSize: 1024 * 1024,
	}, 64)
	if err != nil {
		_ = sfl.Free()
		return nil, fmt.Errorf("failed to create slabby metadata pool: %w", err)
	}

	store := &SlabbyRawVectorStore{
		dim:             dim,
		bytesPerVector:  bytesPerVector,
		slotSize:        int(slotSize),
		segmentCapacity: segmentCapacity,
		sfl:             sfl,
		metadataPool:    metadataPool,
	}
	if err := store.slots.Init(metadataPool); err != nil {
		metadataPool.Free()
		_ = sfl.Free()
		return nil, fmt.Errorf("failed to initialize slabby slots: %w", err)
	}
	prewarmed, err := sfl.Allocate()
	if err != nil {
		_ = store.Close()
		return nil, fmt.Errorf("prewarm slabby vector allocator: %w", err)
	}
	if err := sfl.Deallocate(prewarmed); err != nil {
		_ = store.Close()
		return nil, fmt.Errorf("return prewarmed slabby vector slot: %w", err)
	}
	return store, nil
}

func (s *SlabbyRawVectorStore) Put(vec []float32) (VectorRef, error) {
	if len(vec) != s.dim {
		return VectorRef{}, fmt.Errorf("vector dimension mismatch: expected %d, got %d", s.dim, len(vec))
	}

	slot, err := s.sfl.Allocate()
	if err != nil {
		return VectorRef{}, fmt.Errorf("failed to allocate vector slot: %w", err)
	}

	writeVectorBytes(slot[userDataOffset:], vec)
	slotIndex := s.nextSlot.Add(1) - 1
	if err := s.slots.Store(slotIndex, &slot[0]); err != nil {
		_ = s.sfl.Retire(slot)
		return VectorRef{}, err
	}
	s.activeCount.Add(1)

	return VectorRef{
		Kind:  VectorEncodingRaw,
		Slot:  slotIndex,
		Bytes: uint32(s.bytesPerVector),
		Valid: true,
	}, nil
}

func (s *SlabbyRawVectorStore) Get(ref VectorRef) ([]float32, error) {
	if !ref.Valid || ref.Kind != VectorEncodingRaw {
		return nil, fmt.Errorf("invalid raw vector reference")
	}
	ptr := s.slots.Load(ref.Slot)
	if ptr == nil {
		return nil, fmt.Errorf("raw vector slot out of range: %d", ref.Slot)
	}
	slot := unsafe.Slice(ptr, s.slotSize)
	return bytesAsFloat32View(slot[userDataOffset:], s.dim), nil
}

func (s *SlabbyRawVectorStore) Delete(ref VectorRef) error {
	if !ref.Valid || ref.Kind != VectorEncodingRaw {
		return nil
	}
	for ptr := s.slots.Load(ref.Slot); ptr != nil; ptr = s.slots.Load(ref.Slot) {
		if s.slots.CompareAndSwap(ref.Slot, ptr, nil) {
			// Do not retire the slab here. Lock-free readers may already hold a
			// slice view into this slot; safe reuse requires epoch reclamation.
			s.activeCount.Add(-1)
			break
		}
	}
	return nil
}

func (s *SlabbyRawVectorStore) Reset() error {
	s.slots.Reset()
	if s.sfl != nil {
		s.sfl.Reset()
	}
	if s.metadataPool != nil {
		s.metadataPool.Reset()
		if err := s.slots.Init(s.metadataPool); err != nil {
			return err
		}
	}
	s.activeCount.Store(0)
	s.nextSlot.Store(0)
	return nil
}

func (s *SlabbyRawVectorStore) Close() error {
	s.slots.Detach()
	var firstErr error
	if s.sfl != nil {
		firstErr = s.sfl.Free()
	}
	if s.metadataPool != nil {
		s.metadataPool.Free()
	}
	return firstErr
}

func (s *SlabbyRawVectorStore) MemoryUsage() int64 {
	if s == nil || s.sfl == nil {
		return 0
	}
	stats := s.sfl.Stats()
	return int64(stats.Reserved)
}

func (s *SlabbyRawVectorStore) Profile() RawVectorStoreProfile {
	profile := RawVectorStoreProfile{
		Backend:        RawVectorStoreSlabby,
		VectorCount:    int(s.activeCount.Load()),
		Dimension:      s.dim,
		BytesPerVector: s.bytesPerVector,
	}
	if s.sfl != nil {
		stats := s.sfl.Stats()
		profile.ReservedBytes = int64(stats.Reserved)
		profile.LiveBytes = int64(stats.Allocated)
		profile.FreeBytes = int64(stats.Reserved - stats.Allocated)
		profile.MemoryUsage = profile.ReservedBytes
	}
	if s.metadataPool != nil {
		stats := s.metadataPool.Stats()
		profile.ReservedMetaBytes = int64(stats.Reserved)
		profile.ReservedBytes += int64(stats.Reserved)
		profile.MemoryUsage = profile.ReservedBytes
	}
	return profile
}

func writeVectorBytes(dst []byte, vec []float32) {
	if len(vec) == 0 {
		return
	}
	src := unsafe.Slice((*byte)(unsafe.Pointer(&vec[0])), len(vec)*4)
	copy(dst, src)
}

func bytesAsFloat32View(data []byte, dim int) []float32 {
	if len(data) == 0 || dim == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), dim)
}
