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
	recycler        *vectorRecycler
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
	poolSize := uint64(64 * 1024 * 1024)
	if required := uint64(segmentCapacity) * slotSize; required > poolSize {
		poolSize = (required + 2*1024*1024 - 1) &^ (2*1024*1024 - 1)
	}

	sfl, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  poolSize,
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
	recycler, err := newVectorRecycler(int(poolSize / slotSize))
	if err != nil {
		metadataPool.Free()
		_ = sfl.Free()
		return nil, fmt.Errorf("failed to create slabby logical slot recycler: %w", err)
	}

	store := &SlabbyRawVectorStore{
		dim:             dim,
		bytesPerVector:  bytesPerVector,
		slotSize:        int(slotSize),
		segmentCapacity: segmentCapacity,
		sfl:             sfl,
		metadataPool:    metadataPool,
		recycler:        recycler,
	}
	if err := store.slots.Init(metadataPool); err != nil {
		recycler.close()
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

	_, slotIndex, recycled := s.recycler.take()
	if !recycled {
		slotIndex = s.nextSlot.Add(1) - 1
	}
	writeVectorBytes(slot[userDataOffset:], vec)
	if err := s.slots.Store(slotIndex, &slot[0]); err != nil {
		_ = s.sfl.Deallocate(slot)
		_ = s.recycler.put(nil, slotIndex)
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
	s.detachPointer(ref)
	return nil
}

func (s *SlabbyRawVectorStore) detachPointer(ref VectorRef) unsafe.Pointer {
	if !ref.Valid || ref.Kind != VectorEncodingRaw {
		return nil
	}
	for ptr := s.slots.Load(ref.Slot); ptr != nil; ptr = s.slots.Load(ref.Slot) {
		if s.slots.CompareAndSwap(ref.Slot, ptr, nil) {
			s.activeCount.Add(-1)
			return unsafe.Pointer(ptr)
		}
	}
	return nil
}

func (s *SlabbyRawVectorStore) reclaimPointer(ptr unsafe.Pointer, logical uint32) error {
	if s == nil || s.sfl == nil || ptr == nil {
		return nil
	}
	if err := s.sfl.Deallocate(unsafe.Slice((*byte)(ptr), s.slotSize)); err != nil {
		return err
	}
	if s.recycler == nil || !s.recycler.put(nil, logical) {
		return fmt.Errorf("slabby logical slot recycler is full")
	}
	return nil
}

func (s *SlabbyRawVectorStore) release(ref VectorRef) error {
	return s.reclaimPointer(s.detachPointer(ref), ref.Slot)
}

func (s *SlabbyRawVectorStore) Reset() error {
	s.slots.Reset()
	if s.recycler != nil {
		s.recycler.reset()
	}
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
	if s.recycler != nil {
		s.recycler.close()
	}
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
	if s == nil {
		return 0
	}
	return int64(s.activeCount.Load()) * int64(s.slotSize)
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
	if s.recycler != nil {
		recyclerBytes := int64(len(s.recycler.slots)) * int64(unsafe.Sizeof(recycledVectorSlot{}))
		profile.ReservedMetaBytes += recyclerBytes
		profile.ReservedBytes += recyclerBytes
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
