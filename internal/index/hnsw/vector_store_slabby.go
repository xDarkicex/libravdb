package hnsw

import (
	"errors"
	"fmt"
	"sync"
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

type slabbyVectorSegment struct {
	sfl *memory.ShardedFreeList
}

type SlabbyRawVectorStore struct {
	// sfl remains an alias for the first segment for internal diagnostics and
	// compatibility with existing allocator instrumentation.
	sfl             *memory.ShardedFreeList
	metadataPool    *memory.Pool
	recycler        *vectorRecycler
	slots           rawSlotArray[byte]
	owners          rawSlotArray[slabbyVectorSegment]
	segmentsMu      sync.RWMutex
	segments        []*slabbyVectorSegment
	activeSegment   atomic.Pointer[slabbyVectorSegment]
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
	slotSize = (slotSize + 63) &^ 63
	firstSegment, err := newSlabbyVectorSegment(slotSize, segmentCapacity)
	if err != nil {
		return nil, fmt.Errorf("failed to create memory pool for slabby store: %w", err)
	}
	metadataPool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize: 256 * 1024 * 1024,
		SlabSize: 1024 * 1024,
	}, 64)
	if err != nil {
		_ = firstSegment.sfl.Free()
		return nil, fmt.Errorf("failed to create slabby metadata pool: %w", err)
	}
	recycler, err := newVectorRecycler(segmentCapacity)
	if err != nil {
		metadataPool.Free()
		_ = firstSegment.sfl.Free()
		return nil, fmt.Errorf("failed to create slabby logical slot recycler: %w", err)
	}

	store := &SlabbyRawVectorStore{
		dim:             dim,
		bytesPerVector:  bytesPerVector,
		slotSize:        int(slotSize),
		segmentCapacity: segmentCapacity,
		sfl:             firstSegment.sfl,
		metadataPool:    metadataPool,
		recycler:        recycler,
		segments:        []*slabbyVectorSegment{firstSegment},
	}
	store.activeSegment.Store(firstSegment)
	if err := store.slots.Init(metadataPool); err != nil {
		recycler.close()
		metadataPool.Free()
		_ = firstSegment.sfl.Free()
		return nil, fmt.Errorf("failed to initialize slabby slots: %w", err)
	}
	if err := store.owners.Init(metadataPool); err != nil {
		_ = store.Close()
		return nil, fmt.Errorf("failed to initialize slabby slot owners: %w", err)
	}
	return store, nil
}

func newSlabbyVectorSegment(slotSize uint64, capacity int) (*slabbyVectorSegment, error) {
	alignedSlotSize := (slotSize + 63) &^ 63
	segmentBytes := uint64(capacity) * alignedSlotSize
	if segmentBytes < alignedSlotSize {
		segmentBytes = alignedSlotSize
	}
	sfl, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  segmentBytes,
		SlotSize:  slotSize,
		SlabSize:  segmentBytes,
		SlabCount: 1,
	}, 64, 16)
	if err != nil {
		return nil, err
	}
	prewarmed, err := sfl.Allocate()
	if err != nil {
		_ = sfl.Free()
		return nil, fmt.Errorf("prewarm allocator: %w", err)
	}
	if err := sfl.Deallocate(prewarmed); err != nil {
		_ = sfl.Free()
		return nil, fmt.Errorf("return prewarmed slot: %w", err)
	}
	return &slabbyVectorSegment{sfl: sfl}, nil
}

func (s *SlabbyRawVectorStore) allocateSlot() (*slabbyVectorSegment, []byte, error) {
	if segment := s.activeSegment.Load(); segment != nil {
		if slot, err := segment.sfl.Allocate(); err == nil {
			return segment, slot, nil
		} else if !errors.Is(err, memory.ErrFreelistExhausted) {
			return nil, nil, err
		}
	}

	s.segmentsMu.Lock()
	defer s.segmentsMu.Unlock()
	// A prior segment may have gained a reclaimed slot, or another writer may
	// have grown the list while this writer waited for the lock.
	for i := len(s.segments) - 1; i >= 0; i-- {
		segment := s.segments[i]
		slot, err := segment.sfl.Allocate()
		if err == nil {
			s.activeSegment.Store(segment)
			return segment, slot, nil
		}
		if !errors.Is(err, memory.ErrFreelistExhausted) {
			return nil, nil, err
		}
	}

	segment, err := newSlabbyVectorSegment(uint64(s.slotSize), s.segmentCapacity)
	if err != nil {
		return nil, nil, err
	}
	slot, err := segment.sfl.Allocate()
	if err != nil {
		_ = segment.sfl.Free()
		return nil, nil, err
	}
	s.segments = append(s.segments, segment)
	s.activeSegment.Store(segment)
	return segment, slot, nil
}

func (s *SlabbyRawVectorStore) Put(vec []float32) (VectorRef, error) {
	if len(vec) != s.dim {
		return VectorRef{}, fmt.Errorf("vector dimension mismatch: expected %d, got %d", s.dim, len(vec))
	}

	segment, slot, err := s.allocateSlot()
	if err != nil {
		return VectorRef{}, fmt.Errorf("failed to allocate vector slot: %w", err)
	}

	_, slotIndex, recycled := s.recycler.take()
	if !recycled {
		slotIndex = s.nextSlot.Add(1) - 1
	}
	writeVectorBytes(slot[userDataOffset:], vec)
	if err := s.owners.Store(slotIndex, segment); err != nil {
		_ = segment.sfl.Deallocate(slot)
		_ = s.recycler.put(nil, slotIndex)
		return VectorRef{}, err
	}
	if err := s.slots.Store(slotIndex, &slot[0]); err != nil {
		_ = segment.sfl.Deallocate(slot)
		_ = s.owners.CompareAndSwap(slotIndex, segment, nil)
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
	if s == nil || ptr == nil {
		return nil
	}
	segment := s.owners.Load(logical)
	if segment == nil || segment.sfl == nil {
		return fmt.Errorf("slabby vector slot %d has no owning segment", logical)
	}
	if err := segment.sfl.Deallocate(unsafe.Slice((*byte)(ptr), s.slotSize)); err != nil {
		return err
	}
	s.owners.CompareAndSwap(logical, segment, nil)
	// A full logical recycler only means this ordinal hole will not be reused;
	// the physical vector slot has still been returned to its segment.
	if s.recycler != nil {
		_ = s.recycler.put(nil, logical)
	}
	return nil
}

func (s *SlabbyRawVectorStore) release(ref VectorRef) error {
	return s.reclaimPointer(s.detachPointer(ref), ref.Slot)
}

func (s *SlabbyRawVectorStore) Reset() error {
	s.slots.Reset()
	s.owners.Reset()
	if s.recycler != nil {
		s.recycler.reset()
	}
	s.segmentsMu.Lock()
	for _, segment := range s.segments {
		if segment != nil && segment.sfl != nil {
			segment.sfl.Reset()
		}
	}
	if len(s.segments) > 0 {
		s.activeSegment.Store(s.segments[0])
	} else {
		s.activeSegment.Store(nil)
	}
	s.segmentsMu.Unlock()
	if s.metadataPool != nil {
		s.metadataPool.Reset()
		if err := s.slots.Init(s.metadataPool); err != nil {
			return err
		}
		if err := s.owners.Init(s.metadataPool); err != nil {
			return err
		}
	}
	s.activeCount.Store(0)
	s.nextSlot.Store(0)
	return nil
}

func (s *SlabbyRawVectorStore) Close() error {
	s.slots.Detach()
	s.owners.Detach()
	if s.recycler != nil {
		s.recycler.close()
	}
	var firstErr error
	s.activeSegment.Store(nil)
	s.segmentsMu.Lock()
	for _, segment := range s.segments {
		if segment == nil || segment.sfl == nil {
			continue
		}
		if err := segment.sfl.Free(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	s.segments = nil
	s.sfl = nil
	s.segmentsMu.Unlock()
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
	s.segmentsMu.RLock()
	for _, segment := range s.segments {
		if segment == nil || segment.sfl == nil {
			continue
		}
		stats := segment.sfl.Stats()
		profile.ReservedBytes += int64(stats.Reserved)
		profile.LiveBytes += int64(stats.Allocated)
		profile.FreeBytes += int64(stats.Reserved - stats.Allocated)
	}
	s.segmentsMu.RUnlock()
	profile.ReservedDataBytes = profile.ReservedBytes
	profile.MemoryUsage = profile.ReservedBytes
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
	profile.CapacityUtilization = float64(profile.LiveBytes) / float64(max(int64(1), profile.ReservedDataBytes))
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
