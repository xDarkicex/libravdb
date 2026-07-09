package hnsw

import (
	"fmt"
	"sync/atomic"

	"github.com/xDarkicex/memory"
)

type VectorEncoding uint8

const (
	VectorEncodingNone VectorEncoding = iota
	VectorEncodingRaw
	VectorEncodingCompressed
)

type VectorRef struct {
	Slot  uint32
	Bytes uint32
	Kind  VectorEncoding
	Valid bool
}

type RawVectorStoreProfile struct {
	Backend             string
	VectorCount         int
	Dimension           int
	BytesPerVector      int
	MemoryUsage         int64
	ReservedBytes       int64
	ReservedDataBytes   int64
	ReservedMetaBytes   int64
	ReservedGuardBytes  int64
	LiveBytes           int64
	FreeBytes           int64
	CapacityUtilization float64
}

type RawVectorStore interface {
	Put(vec []float32) (VectorRef, error)
	Get(ref VectorRef) ([]float32, error)
	Delete(ref VectorRef) error
	Reset() error
	Close() error
	MemoryUsage() int64
	Profile() RawVectorStoreProfile
}

type InMemoryRawVectorStore struct {
	pool     *memory.Pool
	slots    rawSlotArray[inMemoryRawVectorSlot]
	dim      int
	bytes    atomic.Int64
	active   atomic.Int32
	nextSlot atomic.Uint32
}

type inMemoryRawVectorSlot struct {
	vec    []float32
	active atomic.Bool
}

func NewInMemoryRawVectorStore(dim int) *InMemoryRawVectorStore {
	pool, err := memory.NewPool(memory.AllocatorConfig{}, 64)
	if err != nil {
		panic(fmt.Sprintf("failed to create memory pool for vector store: %v", err))
	}
	return &InMemoryRawVectorStore{
		pool: pool,
		dim:  dim,
	}
}

func (s *InMemoryRawVectorStore) Put(vec []float32) (VectorRef, error) {
	if s == nil {
		return VectorRef{}, fmt.Errorf("raw vector store is nil")
	}
	if s.dim > 0 && len(vec) != s.dim {
		return VectorRef{}, fmt.Errorf("vector dimension mismatch: expected %d, got %d", s.dim, len(vec))
	}

	storedSlice, err := memory.PoolSlice[float32](s.pool, len(vec))
	if err != nil {
		return VectorRef{}, fmt.Errorf("failed to allocate aligned vector: %w", err)
	}
	stored := storedSlice[:len(vec)]
	copy(stored, vec)

	slotIndex := s.nextSlot.Add(1) - 1
	slot := &inMemoryRawVectorSlot{vec: stored}
	slot.active.Store(true)
	if err := s.slots.Store(slotIndex, slot); err != nil {
		return VectorRef{}, err
	}
	s.bytes.Add(int64(len(stored) * 4))
	s.active.Add(1)

	return VectorRef{
		Kind:  VectorEncodingRaw,
		Slot:  slotIndex,
		Bytes: uint32(len(stored) * 4),
		Valid: true,
	}, nil
}

func (s *InMemoryRawVectorStore) Get(ref VectorRef) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("raw vector store is nil")
	}
	if !ref.Valid || ref.Kind != VectorEncodingRaw {
		return nil, fmt.Errorf("invalid raw vector reference")
	}
	slot := s.slots.Load(ref.Slot)
	if slot == nil {
		return nil, fmt.Errorf("raw vector slot out of range: %d", ref.Slot)
	}
	if !slot.active.Load() || slot.vec == nil {
		return nil, fmt.Errorf("raw vector slot %d is empty", ref.Slot)
	}
	return slot.vec, nil
}

func (s *InMemoryRawVectorStore) Delete(ref VectorRef) error {
	if s == nil || !ref.Valid || ref.Kind != VectorEncodingRaw {
		return nil
	}
	slot := s.slots.Load(ref.Slot)
	if slot == nil {
		return nil
	}
	if slot.active.CompareAndSwap(true, false) {
		s.bytes.Add(-int64(len(slot.vec) * 4))
		s.active.Add(-1)
	}
	return nil
}

func (s *InMemoryRawVectorStore) Reset() error {
	if s == nil {
		return nil
	}
	s.slots.Reset()
	s.bytes.Store(0)
	s.active.Store(0)
	s.nextSlot.Store(0)
	return nil
}

func (s *InMemoryRawVectorStore) MemoryUsage() int64 {
	if s == nil {
		return 0
	}
	return s.bytes.Load()
}

func (s *InMemoryRawVectorStore) Close() error {
	if err := s.Reset(); err != nil {
		return err
	}
	if s.pool != nil {
		s.pool.Free()
	}
	return nil
}

func (s *InMemoryRawVectorStore) Profile() RawVectorStoreProfile {
	bytes := s.bytes.Load()
	return RawVectorStoreProfile{
		Backend:             RawVectorStoreMemory,
		VectorCount:         int(s.active.Load()),
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
