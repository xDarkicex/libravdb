package hnsw

import (
	"fmt"
	"unsafe"

	"github.com/xDarkicex/slabby"
)

const (
	RawVectorStoreMemory = "memory"
	RawVectorStoreSlabby = "slabby"

	defaultSlabbySegmentCapacity = 4096
)

type slabbySlot struct {
	segment int
	slot    slabby.Slot
	active  bool
}

type SlabbyRawVectorStore struct {
	dim             int
	bytesPerVector  int
	segmentCapacity int
	allocators      []*slabby.SlotArena
	slots           []slabbySlot
	activeCount     int
}

func NewSlabbyRawVectorStore(dim, segmentCapacity int) (*SlabbyRawVectorStore, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("invalid vector dimension: %d", dim)
	}
	if segmentCapacity <= 0 {
		segmentCapacity = defaultSlabbySegmentCapacity
	}

	store := &SlabbyRawVectorStore{
		dim:             dim,
		bytesPerVector:  dim * 4,
		segmentCapacity: segmentCapacity,
		allocators:      make([]*slabby.SlotArena, 0, 1),
		slots:           make([]slabbySlot, 0),
	}
	if err := store.grow(); err != nil {
		return nil, err
	}
	return store, nil
}

func (s *SlabbyRawVectorStore) grow() error {
	alloc, err := slabby.NewSlotArena(s.bytesPerVector, s.segmentCapacity, slabby.WithPCPUCache(true))
	if err != nil {
		return fmt.Errorf("failed to create slabby slot arena: %w", err)
	}
	s.allocators = append(s.allocators, alloc)
	return nil
}

func (s *SlabbyRawVectorStore) Put(vec []float32) (VectorRef, error) {
	if len(vec) != s.dim {
		return VectorRef{}, fmt.Errorf("vector dimension mismatch: expected %d, got %d", s.dim, len(vec))
	}
	if len(s.allocators) == 0 {
		if err := s.grow(); err != nil {
			return VectorRef{}, err
		}
	}

	var (
		slot slabby.Slot
		data []byte
		err  error
		seg  = len(s.allocators) - 1
	)
	slot, data, err = s.allocators[seg].AllocateSlot()
	if err != nil {
		if err := s.grow(); err != nil {
			return VectorRef{}, err
		}
		seg = len(s.allocators) - 1
		slot, data, err = s.allocators[seg].AllocateSlot()
		if err != nil {
			return VectorRef{}, fmt.Errorf("failed to allocate slabby vector slot: %w", err)
		}
	}

	writeVectorBytes(data, vec)
	s.slots = append(s.slots, slabbySlot{
		segment: seg,
		slot:    slot,
		active:  true,
	})
	s.activeCount++
	slotIndex := len(s.slots) - 1

	return VectorRef{
		Kind:  VectorEncodingRaw,
		Slot:  uint32(slotIndex),
		Bytes: uint32(s.bytesPerVector),
		Valid: true,
	}, nil
}

func (s *SlabbyRawVectorStore) Get(ref VectorRef) ([]float32, error) {
	if !ref.Valid || ref.Kind != VectorEncodingRaw {
		return nil, fmt.Errorf("invalid raw vector reference")
	}
	if int(ref.Slot) >= len(s.slots) {
		return nil, fmt.Errorf("raw vector slot out of range: %d", ref.Slot)
	}
	slot := s.slots[ref.Slot]
	if !slot.active {
		return nil, fmt.Errorf("raw vector slot %d is inactive", ref.Slot)
	}
	data := s.allocators[slot.segment].BytesForAllocatedSlot(slot.slot)
	return bytesAsFloat32View(data, s.dim), nil
}

func (s *SlabbyRawVectorStore) Delete(ref VectorRef) error {
	if !ref.Valid || ref.Kind != VectorEncodingRaw {
		return nil
	}
	if int(ref.Slot) >= len(s.slots) {
		return nil
	}
	slot := &s.slots[ref.Slot]
	if !slot.active {
		return nil
	}
	if err := s.allocators[slot.segment].FreeSlot(slot.slot); err != nil {
		return fmt.Errorf("failed to free slabby vector slot: %w", err)
	}
	slot.active = false
	s.activeCount--
	return nil
}

func (s *SlabbyRawVectorStore) Reset() error {
	for i, alloc := range s.allocators {
		if alloc != nil {
			if err := alloc.Close(); err != nil {
				return fmt.Errorf("failed to close slabby segment %d: %w", i, err)
			}
		}
	}
	s.allocators = nil
	s.slots = nil
	s.activeCount = 0
	return nil
}

func (s *SlabbyRawVectorStore) Close() error {
	return s.Reset()
}

func (s *SlabbyRawVectorStore) MemoryUsage() int64 {
	if s == nil {
		return 0
	}
	return int64(len(s.allocators) * s.segmentCapacity * s.bytesPerVector)
}

func (s *SlabbyRawVectorStore) Profile() RawVectorStoreProfile {
	profile := RawVectorStoreProfile{
		Backend:        RawVectorStoreSlabby,
		VectorCount:    s.activeCount,
		Dimension:      s.dim,
		BytesPerVector: s.bytesPerVector,
	}
	var totalSlots int
	var usedSlots int
	for _, alloc := range s.allocators {
		if alloc == nil {
			continue
		}
		stats := alloc.Stats()
		if stats == nil {
			continue
		}
		profile.ReservedBytes += stats.ReservedBytes
		profile.ReservedDataBytes += stats.ReservedDataBytes
		profile.ReservedMetaBytes += stats.ReservedMetaBytes
		profile.ReservedGuardBytes += stats.ReservedGuardBytes
		profile.LiveBytes += stats.LiveBytes
		profile.FreeBytes += stats.FreeBytes
		totalSlots += stats.TotalSlabs
		usedSlots += stats.UsedSlabs
	}
	profile.MemoryUsage = profile.ReservedBytes
	if totalSlots > 0 {
		profile.CapacityUtilization = float64(usedSlots) / float64(totalSlots)
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
