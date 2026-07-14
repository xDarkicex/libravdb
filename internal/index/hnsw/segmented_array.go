package hnsw

import (
	"fmt"
	"sync/atomic"
	"unsafe"

	"github.com/xDarkicex/memory"
)

const (
	chunkSizeBits   = 16
	chunkSize       = 1 << chunkSizeBits // 65536
	chunkMask       = chunkSize - 1
	maxChunks       = 4096 // Supports up to 268,435,456 nodes
	maxNodeCapacity = maxChunks * chunkSize

	segmentedPoolSize = 8 * 1024 * 1024 * 1024
	segmentedSlabSize = 2 * 1024 * 1024
)

type segmentedNodeChunk [chunkSize]atomic.Pointer[Node]
type segmentedNodeDirectory [maxChunks]atomic.Pointer[segmentedNodeChunk]

// segmentedNodeArray is a lock-free, wait-free append-only array for HNSW
// nodes. Both directory and data chunks live off-heap; only this small control
// object is visible to the Go collector.
type segmentedNodeArray struct {
	directory *segmentedNodeDirectory
	pool      *memory.Pool
	length    atomic.Uint32
	ownedPool bool
}

func newSegmentedArrayPool() (*memory.Pool, error) {
	return memory.NewPool(memory.AllocatorConfig{
		PoolSize: segmentedPoolSize,
		SlabSize: segmentedSlabSize,
	}, 64)
}

func newSegmentedNodeArray() *segmentedNodeArray {
	pool, err := newSegmentedArrayPool()
	if err != nil {
		panic(fmt.Sprintf("create segmented node pool: %v", err))
	}
	array, err := newSegmentedNodeArrayWithPool(pool)
	if err != nil {
		pool.Free()
		panic(fmt.Sprintf("create segmented node array: %v", err))
	}
	array.ownedPool = true
	return array
}

func newSegmentedNodeArrayWithPool(pool *memory.Pool) (*segmentedNodeArray, error) {
	directory, err := memory.PoolAlloc[segmentedNodeDirectory](pool)
	if err != nil {
		return nil, err
	}
	return &segmentedNodeArray{directory: directory, pool: pool}, nil
}

func (s *segmentedNodeArray) Get(id uint32) *Node {
	chunkIdx := id >> chunkSizeBits
	if s == nil || s.directory == nil || chunkIdx >= maxChunks {
		return nil
	}
	chunk := s.directory[chunkIdx].Load()
	if chunk == nil {
		return nil
	}
	return chunk[id&chunkMask].Load()
}

func (s *segmentedNodeArray) Set(id uint32, node *Node) {
	chunkIdx := id >> chunkSizeBits
	if s == nil || s.directory == nil || chunkIdx >= maxChunks {
		panic("segmentedNodeArray: maximum capacity exceeded or closed")
	}

	chunk := s.directory[chunkIdx].Load()
	if chunk == nil {
		newChunk, err := memory.PoolAlloc[segmentedNodeChunk](s.pool)
		if err != nil {
			panic(fmt.Sprintf("segmentedNodeArray: allocate chunk: %v", err))
		}
		if s.directory[chunkIdx].CompareAndSwap(nil, newChunk) {
			chunk = newChunk
		} else {
			chunk = s.directory[chunkIdx].Load()
		}
	}

	chunk[id&chunkMask].Store(node)
	for {
		oldLen := s.length.Load()
		if id < oldLen || s.length.CompareAndSwap(oldLen, id+1) {
			break
		}
	}
}

func (s *segmentedNodeArray) Len() int {
	if s == nil {
		return 0
	}
	return int(s.length.Load())
}

func (s *segmentedNodeArray) Close() error {
	if s == nil {
		return nil
	}
	s.directory = nil
	if s.ownedPool && s.pool != nil {
		s.pool.Free()
	}
	return nil
}

type offHeapStringRef struct {
	address atomic.Uintptr
	length  atomic.Uint32
	_       uint32
}

type segmentedStringChunk [chunkSize]offHeapStringRef
type segmentedStringDirectory [maxChunks]atomic.Pointer[segmentedStringChunk]

type segmentedStringArray struct {
	directory *segmentedStringDirectory
	pool      *memory.Pool
	ownedPool bool
}

func newSegmentedStringArray() *segmentedStringArray {
	pool, err := newSegmentedArrayPool()
	if err != nil {
		panic(fmt.Sprintf("create segmented string pool: %v", err))
	}
	array, err := newSegmentedStringArrayWithPool(pool)
	if err != nil {
		pool.Free()
		panic(fmt.Sprintf("create segmented string array: %v", err))
	}
	array.ownedPool = true
	return array
}

func newSegmentedStringArrayWithPool(pool *memory.Pool) (*segmentedStringArray, error) {
	directory, err := memory.PoolAlloc[segmentedStringDirectory](pool)
	if err != nil {
		return nil, err
	}
	return &segmentedStringArray{directory: directory, pool: pool}, nil
}

func (s *segmentedStringArray) Get(id uint32) string {
	chunkIdx := id >> chunkSizeBits
	if s == nil || s.directory == nil || chunkIdx >= maxChunks {
		return ""
	}
	chunk := s.directory[chunkIdx].Load()
	if chunk == nil {
		return ""
	}
	ref := &chunk[id&chunkMask]
	address := ref.address.Load()
	if address == 0 {
		return ""
	}
	return unsafe.String((*byte)(unsafe.Pointer(address)), int(ref.length.Load()))
}

func (s *segmentedStringArray) Set(id uint32, value string) {
	chunkIdx := id >> chunkSizeBits
	if s == nil || s.directory == nil || chunkIdx >= maxChunks {
		panic("segmentedStringArray: maximum capacity exceeded or closed")
	}

	chunk := s.directory[chunkIdx].Load()
	if chunk == nil {
		newChunk, err := memory.PoolAlloc[segmentedStringChunk](s.pool)
		if err != nil {
			panic(fmt.Sprintf("segmentedStringArray: allocate chunk: %v", err))
		}
		if s.directory[chunkIdx].CompareAndSwap(nil, newChunk) {
			chunk = newChunk
		} else {
			chunk = s.directory[chunkIdx].Load()
		}
	}

	ref := &chunk[id&chunkMask]
	if value == "" {
		ref.address.Store(0)
		ref.length.Store(0)
		return
	}
	data, err := s.pool.Allocate(uint64(len(value)))
	if err != nil {
		panic(fmt.Sprintf("segmentedStringArray: allocate value: %v", err))
	}
	copy(data, value)
	ref.length.Store(uint32(len(value)))
	ref.address.Store(uintptr(unsafe.Pointer(&data[0])))
}

func (s *segmentedStringArray) Close() error {
	if s == nil {
		return nil
	}
	s.directory = nil
	if s.ownedPool && s.pool != nil {
		s.pool.Free()
	}
	return nil
}
