package singlefile

import (
	"fmt"
	"math/bits"
	"runtime"
	"sync/atomic"

	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/memory"
)

const (
	walRequestCapacity = 4096
	walRequestWords    = walRequestCapacity / 64

	walRequestFree uint32 = iota
	walRequestPending
	walRequestComplete
)

type walRequestRecord struct {
	firstLSN   uint64
	commitLSN  uint64
	state      uint32
	generation uint32
	entryCount uint32
	_          [36]byte
}

type walRequestHandle struct {
	index      uint32
	generation uint32
}

type walRequestError struct {
	err error
}

type walRequestWaiter struct {
	ready chan struct{}
}

type walRequestPool struct {
	arena     *memory.Arena
	slots     []walRequestRecord
	errors    []atomic.Pointer[walRequestError]
	waiters   []atomic.Pointer[walRequestWaiter]
	free      [walRequestWords]atomic.Uint64
	available chan struct{}
}

func newWALRequestPool() (*walRequestPool, error) {
	arena, err := memory.NewArena(walRequestCapacity*64, 64)
	if err != nil {
		return nil, err
	}
	slots, err := memory.ArenaSlice[walRequestRecord](arena, walRequestCapacity)
	if err != nil {
		_ = arena.Free()
		return nil, err
	}
	pool := &walRequestPool{
		arena:     arena,
		slots:     slots[:walRequestCapacity],
		errors:    make([]atomic.Pointer[walRequestError], walRequestCapacity),
		waiters:   make([]atomic.Pointer[walRequestWaiter], walRequestCapacity),
		available: make(chan struct{}, walRequestCapacity),
	}
	for i := range pool.free {
		pool.free[i].Store(^uint64(0))
	}
	return pool, nil
}

func (p *walRequestPool) acquire(entryCount int) walRequestHandle {
	for {
		for word := range p.free {
			available := p.free[word].Load()
			for available != 0 {
				bit := bits.TrailingZeros64(available)
				mask := uint64(1) << bit
				if !p.free[word].CompareAndSwap(available, available&^mask) {
					available = p.free[word].Load()
					continue
				}
				index := uint32(word*64 + bit)
				record := &p.slots[index]
				generation := atomic.AddUint32(&record.generation, 1)
				atomic.StoreUint64(&record.firstLSN, 0)
				atomic.StoreUint64(&record.commitLSN, 0)
				atomic.StoreUint32(&record.entryCount, uint32(entryCount))
				p.errors[index].Store(nil)
				waiter := p.waiter(index)
				select {
				case <-waiter.ready:
				default:
				}
				atomic.StoreUint32(&record.state, walRequestPending)
				return walRequestHandle{index: index, generation: generation}
			}
		}

		<-p.available
	}
}

func (p *walRequestPool) waiter(index uint32) *walRequestWaiter {
	if waiter := p.waiters[index].Load(); waiter != nil {
		return waiter
	}
	waiter := &walRequestWaiter{ready: make(chan struct{}, 1)}
	if p.waiters[index].CompareAndSwap(nil, waiter) {
		return waiter
	}
	return p.waiters[index].Load()
}

func (p *walRequestPool) complete(handle walRequestHandle, durable storage.DurableRange, err error) {
	record := &p.slots[handle.index]
	if atomic.LoadUint32(&record.generation) != handle.generation {
		panic("singlefile: completing stale WAL request")
	}
	if err != nil {
		p.errors[handle.index].Store(&walRequestError{err: err})
	}
	atomic.StoreUint64(&record.firstLSN, durable.FirstLSN)
	atomic.StoreUint64(&record.commitLSN, durable.CommitLSN)
	atomic.StoreUint32(&record.state, walRequestComplete)
	p.waiter(handle.index).ready <- struct{}{}
}

func (p *walRequestPool) waitFor(handle walRequestHandle) (storage.DurableRange, error) {
	record := &p.slots[handle.index]
	for atomic.LoadUint32(&record.generation) == handle.generation &&
		atomic.LoadUint32(&record.state) != walRequestComplete {
		<-p.waiter(handle.index).ready
	}

	if atomic.LoadUint32(&record.generation) != handle.generation {
		return storage.DurableRange{}, fmt.Errorf("stale WAL request completion")
	}
	durable := storage.DurableRange{
		FirstLSN:  atomic.LoadUint64(&record.firstLSN),
		CommitLSN: atomic.LoadUint64(&record.commitLSN),
	}
	var err error
	if result := p.errors[handle.index].Swap(nil); result != nil {
		err = result.err
	}
	p.release(handle)
	runtime.Gosched()
	return durable, err
}

func (p *walRequestPool) cancel(handle walRequestHandle) {
	record := &p.slots[handle.index]
	if atomic.LoadUint32(&record.generation) != handle.generation ||
		atomic.LoadUint32(&record.state) != walRequestPending {
		panic("singlefile: canceling invalid WAL request")
	}
	p.release(handle)
}

func (p *walRequestPool) release(handle walRequestHandle) {
	record := &p.slots[handle.index]
	atomic.StoreUint32(&record.state, walRequestFree)
	word := int(handle.index / 64)
	mask := uint64(1) << (handle.index & 63)
	for {
		free := p.free[word].Load()
		if free&mask != 0 {
			panic("singlefile: WAL request released twice")
		}
		if p.free[word].CompareAndSwap(free, free|mask) {
			break
		}
	}
	select {
	case p.available <- struct{}{}:
	default:
	}
}

func (p *walRequestPool) close() error {
	if p == nil || p.arena == nil {
		return nil
	}
	err := p.arena.Free()
	p.arena = nil
	p.slots = nil
	p.errors = nil
	p.waiters = nil
	return err
}
