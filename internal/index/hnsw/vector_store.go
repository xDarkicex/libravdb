package hnsw

import (
	"fmt"
	"sync/atomic"
	"unsafe"

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
	slots    rawSlotArray[float32]
	dim      int
	bytes    atomic.Int64
	active   atomic.Int32
	nextSlot atomic.Uint32
}

func NewInMemoryRawVectorStore(dim int) *InMemoryRawVectorStore {
	return NewInMemoryRawVectorStoreWithCapacity(dim, 0)
}

func NewInMemoryRawVectorStoreWithCapacity(dim, capacity int) *InMemoryRawVectorStore {
	const minimumPoolSize = uint64(64 * 1024 * 1024)
	poolSize := minimumPoolSize
	if capacity > 0 {
		vectorStride := uint64((dim*4 + 63) &^ 63)
		chunkCount := uint64((capacity + rawSlotChunkSize - 1) / rawSlotChunkSize)
		required := uint64(capacity)*vectorStride + chunkCount*uint64(unsafe.Sizeof(rawSlotChunk[float32]{})) + uint64(unsafe.Sizeof(rawSlotDirectory[float32]{}))
		if required > poolSize {
			poolSize = (required + 2*1024*1024 - 1) &^ (2*1024*1024 - 1)
		}
	}
	pool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  poolSize,
		SlabSize:  poolSize,
		SlabCount: 1,
		Prealloc:  true,
	}, 64)
	if err != nil {
		panic(fmt.Sprintf("failed to create memory pool for vector store: %v", err))
	}
	store := &InMemoryRawVectorStore{
		pool: pool,
		dim:  dim,
	}
	if err := store.slots.Init(pool); err != nil {
		pool.Free()
		panic(fmt.Sprintf("failed to initialize raw vector slots: %v", err))
	}
	return store
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
	if err := s.slots.Store(slotIndex, &stored[0]); err != nil {
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
	ptr := s.slots.Load(ref.Slot)
	if ptr == nil {
		return nil, fmt.Errorf("raw vector slot out of range: %d", ref.Slot)
	}
	return unsafe.Slice(ptr, s.dim), nil
}

func (s *InMemoryRawVectorStore) Delete(ref VectorRef) error {
	if s == nil || !ref.Valid || ref.Kind != VectorEncodingRaw {
		return nil
	}
	for ptr := s.slots.Load(ref.Slot); ptr != nil; ptr = s.slots.Load(ref.Slot) {
		if s.slots.CompareAndSwap(ref.Slot, ptr, nil) {
			s.bytes.Add(-int64(s.dim * 4))
			s.active.Add(-1)
			break
		}
	}
	return nil
}

func (s *InMemoryRawVectorStore) Reset() error {
	if s == nil {
		return nil
	}
	s.slots.Reset()
	if s.pool != nil {
		s.pool.Reset()
		if err := s.slots.Init(s.pool); err != nil {
			return err
		}
	}
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
	s.slots.Detach()
	if s.pool != nil {
		s.pool.Free()
	}
	return nil
}

func (s *InMemoryRawVectorStore) Profile() RawVectorStoreProfile {
	bytes := s.bytes.Load()
	stats := s.pool.Stats()
	return RawVectorStoreProfile{
		Backend:             RawVectorStoreMemory,
		VectorCount:         int(s.active.Load()),
		Dimension:           s.dim,
		BytesPerVector:      s.dim * 4,
		MemoryUsage:         bytes,
		ReservedBytes:       int64(stats.Reserved),
		ReservedDataBytes:   int64(stats.Reserved),
		ReservedMetaBytes:   0,
		ReservedGuardBytes:  0,
		LiveBytes:           bytes,
		FreeBytes:           int64(stats.Reserved) - bytes,
		CapacityUtilization: float64(bytes) / float64(max(uint64(1), stats.Reserved)),
	}
}
