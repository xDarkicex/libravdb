package hnsw

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"unsafe"

	"github.com/xDarkicex/memory"
)

const (
	reclamationReaderSlots      = 64
	minRetiredQueueCapacity     = 1 << 10
	maxRetiredQueueCapacity     = 1 << 14
	defaultRetiredQueueCapacity = 1 << 12
)

type retiredAllocationKind uint32

const (
	retiredNode retiredAllocationKind = iota + 1
	retiredUpperLink
	retiredLevel0Link
	retiredRawVector
)

type reclamationReaderSlot struct {
	epoch atomic.Uint64
	_     [56]byte
}

type retiredAllocationSlot struct {
	sequence atomic.Uint64
	epoch    uint64
	ptr      uintptr
	kind     retiredAllocationKind
	logical  uint32
	_        [32]byte
}

type reclamationDomain struct {
	global     atomic.Uint64
	enqueuePos atomic.Uint64
	dequeuePos atomic.Uint64
	reclaiming atomic.Uint32
	arena      *memory.Arena
	readers    []reclamationReaderSlot
	retired    []retiredAllocationSlot
	queueMask  uint64
	queueCap   uint64
}

func reclamationQueueCapacity(nodeCapacity int) int {
	if nodeCapacity <= 0 {
		return defaultRetiredQueueCapacity
	}
	target := max(minRetiredQueueCapacity, min(maxRetiredQueueCapacity, nodeCapacity*2))
	capacity := 1
	for capacity < target {
		capacity <<= 1
	}
	return capacity
}

func newReclamationDomain(capacity int) (*reclamationDomain, error) {
	capacity = reclamationQueueCapacity(capacity)
	bytes := uint64(reclamationReaderSlots)*uint64(unsafe.Sizeof(reclamationReaderSlot{})) +
		uint64(capacity)*uint64(unsafe.Sizeof(retiredAllocationSlot{})) + 128
	arena, err := memory.NewArena(bytes, 64)
	if err != nil {
		return nil, fmt.Errorf("allocate reclamation arena: %w", err)
	}
	readers, err := memory.ArenaSlice[reclamationReaderSlot](arena, reclamationReaderSlots)
	if err != nil {
		_ = arena.Free()
		return nil, fmt.Errorf("allocate reclamation readers: %w", err)
	}
	readers = readers[:reclamationReaderSlots]
	retired, err := memory.ArenaSlice[retiredAllocationSlot](arena, capacity)
	if err != nil {
		_ = arena.Free()
		return nil, fmt.Errorf("allocate retired queue: %w", err)
	}
	retired = retired[:capacity]
	domain := &reclamationDomain{
		arena:     arena,
		readers:   readers,
		retired:   retired,
		queueMask: uint64(capacity - 1),
		queueCap:  uint64(capacity),
	}
	domain.global.Store(1)
	for i := range retired {
		retired[i].sequence.Store(uint64(i))
	}
	return domain, nil
}

func (d *reclamationDomain) enter(slot uint8) {
	if d == nil {
		return
	}
	reader := &d.readers[int(slot)&(reclamationReaderSlots-1)]
	for {
		epoch := d.global.Load()
		reader.epoch.Store(epoch)
		if d.global.Load() == epoch {
			return
		}
	}
}

func (d *reclamationDomain) leave(slot uint8) {
	if d != nil {
		d.readers[int(slot)&(reclamationReaderSlots-1)].epoch.Store(0)
	}
}

func (d *reclamationDomain) nextRetireEpoch() uint64 {
	if d == nil {
		return 0
	}
	return d.global.Add(1)
}

func (h *Index) retireAllocationAt(epoch uint64, kind retiredAllocationKind, ptr unsafe.Pointer) {
	h.retireAllocationWithLogicalAt(epoch, kind, ptr, 0)
}

func (h *Index) retireRawVectorAt(epoch uint64, ptr unsafe.Pointer, logical uint32) {
	h.retireAllocationWithLogicalAt(epoch, retiredRawVector, ptr, logical)
}

func (h *Index) retireAllocationWithLogicalAt(epoch uint64, kind retiredAllocationKind, ptr unsafe.Pointer, logical uint32) {
	if h == nil || h.reclamation == nil || ptr == nil {
		return
	}
	record := retiredAllocationSlot{
		epoch:   epoch,
		ptr:     uintptr(ptr),
		kind:    kind,
		logical: logical,
	}
	for !h.reclamation.enqueue(record) {
		h.reclamation.tryReclaim(h)
		runtime.Gosched()
	}
}

func (d *reclamationDomain) enqueue(record retiredAllocationSlot) bool {
	for {
		pos := d.enqueuePos.Load()
		slot := &d.retired[pos&d.queueMask]
		sequence := slot.sequence.Load()
		delta := int64(sequence) - int64(pos)
		switch {
		case delta == 0:
			if !d.enqueuePos.CompareAndSwap(pos, pos+1) {
				continue
			}
			slot.epoch = record.epoch
			slot.ptr = record.ptr
			slot.kind = record.kind
			slot.logical = record.logical
			slot.sequence.Store(pos + 1)
			return true
		case delta < 0:
			return false
		default:
			runtime.Gosched()
		}
	}
}

func (d *reclamationDomain) safe(epoch uint64) bool {
	for i := range d.readers {
		active := d.readers[i].epoch.Load()
		if active != 0 && active < epoch {
			return false
		}
	}
	return true
}

func (d *reclamationDomain) tryReclaim(index *Index) int {
	if d == nil || index == nil || !d.reclaiming.CompareAndSwap(0, 1) {
		return 0
	}
	defer d.reclaiming.Store(0)

	reclaimed := 0
	for {
		pos := d.dequeuePos.Load()
		slot := &d.retired[pos&d.queueMask]
		if slot.sequence.Load() != pos+1 || !d.safe(slot.epoch) {
			return reclaimed
		}
		record := retiredAllocationSlot{
			epoch:   slot.epoch,
			ptr:     slot.ptr,
			kind:    slot.kind,
			logical: slot.logical,
		}
		index.reclaimAllocation(record)
		slot.ptr = 0
		slot.kind = 0
		slot.logical = 0
		slot.epoch = 0
		slot.sequence.Store(pos + d.queueCap)
		d.dequeuePos.Store(pos + 1)
		reclaimed++
	}
}

func (d *reclamationDomain) drain(index *Index) {
	if d == nil || index == nil {
		return
	}
	for i := range d.readers {
		d.readers[i].epoch.Store(0)
	}
	d.tryReclaim(index)
}

func (d *reclamationDomain) close() {
	if d == nil || d.arena == nil {
		return
	}
	_ = d.arena.Free()
	d.arena = nil
	d.readers = nil
	d.retired = nil
}

func (h *Index) reclaimAllocation(record retiredAllocationSlot) {
	ptr := unsafe.Pointer(record.ptr)
	switch record.kind {
	case retiredNode:
		if h.nodeSFL != nil {
			_ = h.nodeSFL.Deallocate(h.nodeSlotFromBase(ptr))
		}
	case retiredUpperLink:
		if h.linkSFL != nil {
			_ = h.linkSFL.Deallocate(h.linkSlotFromBase(ptr, 1))
		}
	case retiredLevel0Link:
		if h.link0SFL != nil {
			_ = h.link0SFL.Deallocate(h.linkSlotFromBase(ptr, 0))
		}
	case retiredRawVector:
		switch store := h.rawVectorStore.(type) {
		case *InMemoryRawVectorStore:
			_ = store.reclaimPointer(ptr, record.logical)
		case *SlabbyRawVectorStore:
			_ = store.reclaimPointer(ptr, record.logical)
		}
	}
}

func (h *Index) nodeSlotFromBase(base unsafe.Pointer) []byte {
	slotSize := int(uint64(SFLMetadataOverhead) + inlineNodeSlotPayloadSize(h.config.M))
	return unsafe.Slice((*byte)(base), slotSize)
}

func (h *Index) linkSlotFromBase(base unsafe.Pointer, level int) []byte {
	return unsafe.Slice((*byte)(base), int(SFLMetadataOverhead)+linkArrayCapacity(h.config.M, level)*4)
}
