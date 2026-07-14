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

type recycledVectorSlot struct {
	sequence atomic.Uint64
	ptr      uintptr
	logical  uint32
	_        uint32
}

type vectorRecycler struct {
	enqueuePos atomic.Uint64
	dequeuePos atomic.Uint64
	arena      *memory.Arena
	slots      []recycledVectorSlot
	mask       uint64
	capacity   uint64
}

func newVectorRecycler(capacity int) (*vectorRecycler, error) {
	if capacity < 1 {
		capacity = 1
	}
	queueCapacity := 1
	for queueCapacity < capacity {
		queueCapacity <<= 1
	}
	arena, err := memory.NewArena(uint64(queueCapacity)*uint64(unsafe.Sizeof(recycledVectorSlot{}))+64, 64)
	if err != nil {
		return nil, err
	}
	slots, err := memory.ArenaSlice[recycledVectorSlot](arena, queueCapacity)
	if err != nil {
		_ = arena.Free()
		return nil, err
	}
	slots = slots[:queueCapacity]
	recycler := &vectorRecycler{
		arena:    arena,
		slots:    slots,
		mask:     uint64(queueCapacity - 1),
		capacity: uint64(queueCapacity),
	}
	recycler.reset()
	return recycler, nil
}

func (r *vectorRecycler) put(ptr unsafe.Pointer, logical uint32) bool {
	if r == nil {
		return false
	}
	for {
		pos := r.enqueuePos.Load()
		slot := &r.slots[pos&r.mask]
		delta := int64(slot.sequence.Load()) - int64(pos)
		switch {
		case delta == 0:
			if !r.enqueuePos.CompareAndSwap(pos, pos+1) {
				continue
			}
			slot.ptr = uintptr(ptr)
			slot.logical = logical
			slot.sequence.Store(pos + 1)
			return true
		case delta < 0:
			return false
		}
	}
}

func (r *vectorRecycler) take() (unsafe.Pointer, uint32, bool) {
	if r == nil {
		return nil, 0, false
	}
	for {
		pos := r.dequeuePos.Load()
		slot := &r.slots[pos&r.mask]
		delta := int64(slot.sequence.Load()) - int64(pos+1)
		switch {
		case delta == 0:
			if !r.dequeuePos.CompareAndSwap(pos, pos+1) {
				continue
			}
			ptr := unsafe.Pointer(slot.ptr)
			logical := slot.logical
			slot.ptr = 0
			slot.logical = 0
			slot.sequence.Store(pos + r.capacity)
			return ptr, logical, true
		case delta < 0:
			return nil, 0, false
		}
	}
}

func (r *vectorRecycler) reset() {
	if r == nil {
		return
	}
	r.enqueuePos.Store(0)
	r.dequeuePos.Store(0)
	for i := range r.slots {
		r.slots[i].ptr = 0
		r.slots[i].logical = 0
		r.slots[i].sequence.Store(uint64(i))
	}
}

func (r *vectorRecycler) close() {
	if r == nil || r.arena == nil {
		return
	}
	_ = r.arena.Free()
	r.arena = nil
	r.slots = nil
}

type InMemoryRawVectorStore struct {
	pool     *memory.Pool
	recycler *vectorRecycler
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
	vectorStride := max(1, (dim*4+63)&^63)
	recycler, err := newVectorRecycler(int(poolSize) / vectorStride)
	if err != nil {
		pool.Free()
		panic(fmt.Sprintf("failed to create vector recycler: %v", err))
	}
	store := &InMemoryRawVectorStore{
		pool:     pool,
		recycler: recycler,
		dim:      dim,
	}
	if err := store.slots.Init(pool); err != nil {
		recycler.close()
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

	var stored []float32
	var slotIndex uint32
	if ptr, recycledSlot, ok := s.recycler.take(); ok {
		if ptr == nil {
			return VectorRef{}, fmt.Errorf("recycled raw vector pointer is nil")
		}
		stored = unsafe.Slice((*float32)(ptr), len(vec))
		slotIndex = recycledSlot
	} else {
		storedSlice, err := memory.PoolSlice[float32](s.pool, len(vec))
		if err != nil {
			return VectorRef{}, fmt.Errorf("failed to allocate aligned vector: %w", err)
		}
		stored = storedSlice[:len(vec)]
		slotIndex = s.nextSlot.Add(1) - 1
	}
	copy(stored, vec)

	if err := s.slots.Store(slotIndex, &stored[0]); err != nil {
		_ = s.recycler.put(unsafe.Pointer(&stored[0]), slotIndex)
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
	s.detachPointer(ref)
	return nil
}

func (s *InMemoryRawVectorStore) detachPointer(ref VectorRef) unsafe.Pointer {
	if s == nil || !ref.Valid || ref.Kind != VectorEncodingRaw {
		return nil
	}
	for ptr := s.slots.Load(ref.Slot); ptr != nil; ptr = s.slots.Load(ref.Slot) {
		if s.slots.CompareAndSwap(ref.Slot, ptr, nil) {
			s.bytes.Add(-int64(s.dim * 4))
			s.active.Add(-1)
			return unsafe.Pointer(ptr)
		}
	}
	return nil
}

func (s *InMemoryRawVectorStore) reclaimPointer(ptr unsafe.Pointer, logical uint32) error {
	if ptr == nil {
		return nil
	}
	if s == nil || s.recycler == nil || !s.recycler.put(ptr, logical) {
		return fmt.Errorf("raw vector recycler is full")
	}
	return nil
}

func (s *InMemoryRawVectorStore) release(ref VectorRef) error {
	return s.reclaimPointer(s.detachPointer(ref), ref.Slot)
}

func (s *InMemoryRawVectorStore) Reset() error {
	if s == nil {
		return nil
	}
	s.slots.Reset()
	if s.recycler != nil {
		s.recycler.reset()
	}
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
	if s.recycler != nil {
		s.recycler.close()
	}
	if s.pool != nil {
		s.pool.Free()
	}
	return nil
}

func (s *InMemoryRawVectorStore) Profile() RawVectorStoreProfile {
	bytes := s.bytes.Load()
	stats := s.pool.Stats()
	recyclerBytes := int64(0)
	if s.recycler != nil {
		recyclerBytes = int64(len(s.recycler.slots)) * int64(unsafe.Sizeof(recycledVectorSlot{}))
	}
	return RawVectorStoreProfile{
		Backend:             RawVectorStoreMemory,
		VectorCount:         int(s.active.Load()),
		Dimension:           s.dim,
		BytesPerVector:      s.dim * 4,
		MemoryUsage:         bytes,
		ReservedBytes:       int64(stats.Reserved) + recyclerBytes,
		ReservedDataBytes:   int64(stats.Reserved),
		ReservedMetaBytes:   recyclerBytes,
		ReservedGuardBytes:  0,
		LiveBytes:           bytes,
		FreeBytes:           int64(stats.Reserved) - bytes,
		CapacityUtilization: float64(bytes) / float64(max(uint64(1), stats.Reserved)),
	}
}
