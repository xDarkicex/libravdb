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
	userDataOffset               = 48
)

type slabbyRawVectorSlot struct {
	slot   []byte
	active atomic.Bool
}

type SlabbyRawVectorStore struct {
	sfl             *memory.ShardedFreeList
	slots           rawSlotArray[slabbyRawVectorSlot]
	dim             int
	bytesPerVector  int
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

	store := &SlabbyRawVectorStore{
		dim:             dim,
		bytesPerVector:  bytesPerVector,
		segmentCapacity: segmentCapacity,
		sfl:             sfl,
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
	descriptor := &slabbyRawVectorSlot{slot: slot}
	descriptor.active.Store(true)
	if err := s.slots.Store(slotIndex, descriptor); err != nil {
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
	slot := s.slots.Load(ref.Slot)
	if slot == nil {
		return nil, fmt.Errorf("raw vector slot out of range: %d", ref.Slot)
	}
	if !slot.active.Load() || slot.slot == nil {
		return nil, fmt.Errorf("raw vector slot %d is inactive", ref.Slot)
	}
	return bytesAsFloat32View(slot.slot[userDataOffset:], s.dim), nil
}

func (s *SlabbyRawVectorStore) Delete(ref VectorRef) error {
	if !ref.Valid || ref.Kind != VectorEncodingRaw {
		return nil
	}
	slot := s.slots.Load(ref.Slot)
	if slot == nil {
		return nil
	}
	if slot.active.CompareAndSwap(true, false) {
		// Do not retire the slab here. Lock-free readers may already hold a
		// slice view into this slot; safe reuse requires epoch reclamation.
		s.activeCount.Add(-1)
	}
	return nil
}

func (s *SlabbyRawVectorStore) Reset() error {
	if s.sfl != nil {
		s.sfl.Reset()
	}
	s.slots.Reset()
	s.activeCount.Store(0)
	s.nextSlot.Store(0)
	return nil
}

func (s *SlabbyRawVectorStore) Close() error {
	if s.sfl != nil {
		return s.sfl.Free()
	}
	return nil
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
