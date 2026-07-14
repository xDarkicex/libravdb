package hnsw

import (
	"fmt"
	"sync/atomic"

	"github.com/xDarkicex/memory"
)

const (
	rawSlotChunkBits = 12
	rawSlotChunkSize = 1 << rawSlotChunkBits
	rawSlotChunkMask = rawSlotChunkSize - 1
	rawSlotMaxChunks = 4096
)

type rawSlotChunk[T any] [rawSlotChunkSize]atomic.Pointer[T]
type rawSlotDirectory[T any] [rawSlotMaxChunks]atomic.Pointer[rawSlotChunk[T]]

type rawSlotArray[T any] struct {
	directory *rawSlotDirectory[T]
	pool      *memory.Pool
}

func (a *rawSlotArray[T]) Init(pool *memory.Pool) error {
	directory, err := memory.PoolAlloc[rawSlotDirectory[T]](pool)
	if err != nil {
		return fmt.Errorf("allocate raw slot directory: %w", err)
	}
	a.pool = pool
	a.directory = directory
	return nil
}

func (a *rawSlotArray[T]) Load(id uint32) *T {
	chunkIdx := id >> rawSlotChunkBits
	if chunkIdx >= rawSlotMaxChunks {
		return nil
	}
	if a.directory == nil {
		return nil
	}
	chunk := a.directory[chunkIdx].Load()
	if chunk == nil {
		return nil
	}
	return chunk[id&rawSlotChunkMask].Load()
}

func (a *rawSlotArray[T]) Store(id uint32, value *T) error {
	chunkIdx := id >> rawSlotChunkBits
	if chunkIdx >= rawSlotMaxChunks {
		return fmt.Errorf("raw vector slot capacity exceeded: %d", id)
	}
	if a.directory == nil {
		return fmt.Errorf("raw slot array is not initialized")
	}
	chunk := a.directory[chunkIdx].Load()
	if chunk == nil {
		if a.pool == nil {
			return fmt.Errorf("raw slot array has no off-heap chunk pool")
		}
		newChunk, err := memory.PoolAlloc[rawSlotChunk[T]](a.pool)
		if err != nil {
			return fmt.Errorf("allocate raw slot chunk: %w", err)
		}
		if a.directory[chunkIdx].CompareAndSwap(nil, newChunk) {
			chunk = newChunk
		} else {
			chunk = a.directory[chunkIdx].Load()
		}
	}
	chunk[id&rawSlotChunkMask].Store(value)
	return nil
}

func (a *rawSlotArray[T]) CompareAndSwap(id uint32, old, new *T) bool {
	chunkIdx := id >> rawSlotChunkBits
	if chunkIdx >= rawSlotMaxChunks {
		return false
	}
	if a.directory == nil {
		return false
	}
	chunk := a.directory[chunkIdx].Load()
	if chunk == nil {
		return false
	}
	return chunk[id&rawSlotChunkMask].CompareAndSwap(old, new)
}

func (a *rawSlotArray[T]) Reset() {
	if a.directory == nil {
		return
	}
	for i := range a.directory {
		a.directory[i].Store(nil)
	}
}

func (a *rawSlotArray[T]) Detach() {
	a.directory = nil
	a.pool = nil
}
