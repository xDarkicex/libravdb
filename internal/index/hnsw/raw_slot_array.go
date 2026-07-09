package hnsw

import (
	"fmt"
	"sync/atomic"
)

const (
	rawSlotChunkBits = 12
	rawSlotChunkSize = 1 << rawSlotChunkBits
	rawSlotChunkMask = rawSlotChunkSize - 1
	rawSlotMaxChunks = 4096
)

type rawSlotChunk[T any] [rawSlotChunkSize]atomic.Pointer[T]

type rawSlotArray[T any] struct {
	chunks [rawSlotMaxChunks]atomic.Pointer[rawSlotChunk[T]]
}

func (a *rawSlotArray[T]) Load(id uint32) *T {
	chunkIdx := id >> rawSlotChunkBits
	if chunkIdx >= rawSlotMaxChunks {
		return nil
	}
	chunk := a.chunks[chunkIdx].Load()
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
	chunk := a.chunks[chunkIdx].Load()
	if chunk == nil {
		newChunk := new(rawSlotChunk[T])
		if a.chunks[chunkIdx].CompareAndSwap(nil, newChunk) {
			chunk = newChunk
		} else {
			chunk = a.chunks[chunkIdx].Load()
		}
	}
	chunk[id&rawSlotChunkMask].Store(value)
	return nil
}

func (a *rawSlotArray[T]) Reset() {
	for i := range a.chunks {
		a.chunks[i].Store(nil)
	}
}
