package hnsw

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	internalmemory "github.com/xDarkicex/libravdb/internal/memory"
	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/memory"
)

const inFlightRegistrySize = 65536 // Power of 2 for fast modulo
const inFlightSnapshotLimit = 2048
const (
	defaultIDMapCapacity = 8192
	// ID keys remain stable for lock-free readers until the index closes. Keep
	// a separate lifetime budget so table growth beyond the initial capacity
	// does not prematurely exhaust copied-key storage.
	defaultIDMapKeyBytes = 4 * 1024 * 1024
)
const defaultRepairQueueSize = 65536
const defaultRepairBatchSize = 64

type inFlightRegistry struct {
	idx    atomic.Uint64
	active atomic.Int32
	nodes  []uint32
}

func newInFlightRegistry(arena *memory.Arena) *inFlightRegistry {
	var slice []uint32
	if arena != nil {
		slice, _ = memory.ArenaSlice[uint32](arena, inFlightRegistrySize)
	}
	if slice == nil {
		// Fallback to on-heap if arena allocation fails during initialization
		slice = make([]uint32, inFlightRegistrySize)
	} else {
		slice = slice[:inFlightRegistrySize]
	}

	// Initialize with Sentinels
	for i := range slice {
		slice[i] = SentinelNodeID
	}

	return &inFlightRegistry{
		nodes: slice,
	}
}

func (r *inFlightRegistry) Add(id uint32) {
	if r == nil || r.nodes == nil {
		return
	}
	r.active.Add(1)
	i := r.idx.Add(1) - 1
	atomic.StoreUint32(&r.nodes[i&(inFlightRegistrySize-1)], id)
}

func (r *inFlightRegistry) Remove(id uint32) {
	// Ring buffer implicitly removes items by overwriting them.
	// The InFlight flag on the Node struct is cleared instead.
	if r == nil || r.nodes == nil {
		return
	}
	r.active.Add(-1)
}

func (r *inFlightRegistry) Active() int32 {
	if r == nil || r.nodes == nil {
		return 0
	}
	return r.active.Load()
}

func (r *inFlightRegistry) GetSnapshot(buf []uint32) []uint32 {
	if r == nil || r.nodes == nil {
		return buf
	}
	buf = buf[:0]
	end := r.idx.Load()
	start := uint64(0)

	// Bound snapshot work so construction coordination cannot grow with the
	// lifetime append position of the registry.
	scanLimit := uint64(inFlightSnapshotLimit)
	if end > scanLimit {
		start = end - scanLimit
	}

	for i := start; i < end; i++ {
		id := atomic.LoadUint32(&r.nodes[i&(inFlightRegistrySize-1)])
		if id != SentinelNodeID {
			buf = append(buf, id)
		}
	}
	return buf
}

// VectorEntry represents a vector entry for HNSW indexing
type VectorEntry struct {
	Metadata map[string]interface{}
	ID       string
	Vector   []float32
	Version  uint64
	Ordinal  uint32
}

// SearchResult represents a search result from HNSW
type SearchResult struct {
	Metadata map[string]interface{}
	ID       string
	Vector   []float32
	Version  uint64
	Ordinal  uint32
	Score    float32
}

type VectorProvider interface {
	GetByOrdinal(ordinal uint32) ([]float32, error)
	Distance(query []float32, ordinal uint32) (float32, error)
}

// Index implements the HNSW algorithm for approximate nearest neighbor search
type Index struct {
	searchScratchFree     atomic.Uint64
	searchScratches       []searchScratch
	candidateMode         atomic.Uint32
	provider              VectorProvider
	quantizer             quant.Quantizer
	rawVectorStore        RawVectorStore
	idToIndex             *memory.TypedIDMap[Node]
	globalState           atomic.Uint64 // Packs entryPoint ID (32 bits) and maxLevel (32 bits)
	distance              util.DistanceFunc
	vecMmap               *internalmemory.MemoryMap
	config                *Config
	pqMmap                *internalmemory.MemoryMap
	link0SFL              *memory.ShardedFreeList
	ordinalToID           *segmentedStringArray
	linkSFL               *memory.ShardedFreeList
	neighborSelector      *NeighborSelector
	useHeuristicPredicate bool
	registryPool          *memory.Pool
	scratchPool           *sync.Pool
	nodeSFL               *memory.ShardedFreeList
	inFlightNodes         *inFlightRegistry // registry for concurrent insertions
	repairCh              chan uint32
	repairStop            chan struct{}
	repairDone            chan struct{}
	mmapPath              string
	trainingVectors       [][]float32
	trainingCount         atomic.Int32
	nodes                 *segmentedNodeArray
	size                  atomic.Int32
	mmapSize              int64
	originalMemUsage      int64
	nextOrdinal           atomic.Uint32
	quantizationTrained   atomic.Bool
	repairOverflow        atomic.Bool
	reclamation           *reclamationDomain
	memoryMapped          bool
}

// Config holds HNSW configuration parameters
type Config struct {
	Provider             VectorProvider
	Quantization         *quant.QuantizationConfig
	RawVectorStore       string
	Dimension            int
	M                    int
	EfConstruction       int
	EfSearch             int
	ML                   float64
	Metric               util.DistanceMetric
	PruneAlpha           float32
	Level0LinkMultiplier float64
	RepairEnabled        bool
	RepairQueueSize      int
	RepairBatchSize      int
	RandomSeed           int64
	RawStoreCap          int
	IDMapCapacity        int
}

func (c *Config) idMapCapacity() uint64 {
	if c.IDMapCapacity > 0 {
		return uint64(c.IDMapCapacity)
	}
	return defaultIDMapCapacity
}

func (c *Config) idMapKeyBytes() uint64 {
	if c.IDMapCapacity <= 0 {
		return defaultIDMapKeyBytes
	}
	return uint64(c.IDMapCapacity) * 64
}

func (c *Config) level0LinkMultiplier() float64 {
	if c == nil || c.Level0LinkMultiplier <= 0 {
		return level0LinkMultiplier
	}
	return c.Level0LinkMultiplier
}

func (c *Config) repairQueueSize() int {
	if c == nil {
		return 0
	}
	if c.RepairQueueSize > 0 {
		return c.RepairQueueSize
	}
	if c.RepairEnabled {
		return defaultRepairQueueSize
	}
	return 0
}

func (c *Config) repairBatchSize() int {
	if c == nil || c.RepairBatchSize <= 0 {
		return defaultRepairBatchSize
	}
	return c.RepairBatchSize
}

// NewHNSW creates a new HNSW index
func NewHNSW(config *Config) (*Index, error) {
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid HNSW config: %w", err)
	}

	distanceFunc, err := util.GetDistanceFunc(config.Metric)
	if err != nil {
		return nil, fmt.Errorf("unsupported distance metric: %w", err)
	}

	slack := config.M / 4
	if slack < 4 {
		slack = 4
	}
	linkSFL, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  512 * 1024 * 1024,
		SlotSize:  uint64(SFLMetadataOverhead + (config.M+slack)*4),
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 8,
		Prealloc:  false,
	}, 64, 64)
	if err != nil {
		return nil, fmt.Errorf("failed to create linkSFL: %w", err)
	}

	slack0 := (config.M * 2) / 4
	if slack0 < 4 {
		slack0 = 4
	}
	link0SFL, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  512 * 1024 * 1024,
		SlotSize:  uint64(SFLMetadataOverhead + (config.M*2+slack0)*4),
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 8,
		Prealloc:  false,
	}, 64, 64)
	if err != nil {
		linkSFL.Free()
		return nil, fmt.Errorf("failed to create link0SFL: %w", err)
	}

	nodeSFL, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  512 * 1024 * 1024,
		SlotSize:  uint64(SFLMetadataOverhead) + inlineNodeSlotPayloadSize(config.M),
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 8,
		Prealloc:  false,
	}, 64, 64)
	if err != nil {
		linkSFL.Free()
		link0SFL.Free()
		return nil, fmt.Errorf("failed to create nodeSFL: %w", err)
	}

	inFlight := newInFlightRegistry(nil)

	scratchPool := &sync.Pool{
		New: func() any {
			a, _ := memory.NewArena(1024*1024, 64)
			return a
		},
	}

	idToIndexMap, err := memory.NewTypedIDMap[Node](memory.IDMapConfig{
		Capacity:  config.idMapCapacity(),
		KeyBytes:  config.idMapKeyBytes(),
		Alignment: 128,
	})
	if err != nil {
		linkSFL.Free()
		link0SFL.Free()
		nodeSFL.Free()
		return nil, fmt.Errorf("failed to create idToIndex map: %w", err)
	}
	registryPool, err := newSegmentedArrayPool()
	if err != nil {
		_ = idToIndexMap.Free()
		linkSFL.Free()
		link0SFL.Free()
		nodeSFL.Free()
		return nil, fmt.Errorf("failed to create off-heap registry pool: %w", err)
	}
	nodes, err := newSegmentedNodeArrayWithPool(registryPool)
	if err != nil {
		_ = idToIndexMap.Free()
		registryPool.Free()
		linkSFL.Free()
		link0SFL.Free()
		nodeSFL.Free()
		return nil, fmt.Errorf("failed to create off-heap node registry: %w", err)
	}
	ordinalToID, err := newSegmentedStringArrayWithPool(registryPool)
	if err != nil {
		_ = idToIndexMap.Free()
		registryPool.Free()
		linkSFL.Free()
		link0SFL.Free()
		nodeSFL.Free()
		return nil, fmt.Errorf("failed to create off-heap ordinal registry: %w", err)
	}
	reclamation, err := newReclamationDomain(config.RawStoreCap)
	if err != nil {
		_ = idToIndexMap.Free()
		registryPool.Free()
		linkSFL.Free()
		link0SFL.Free()
		nodeSFL.Free()
		return nil, err
	}

	index := &Index{
		config:                config,
		nodes:                 nodes,
		neighborSelector:      NewNeighborSelector(config.M, config.level0LinkMultiplier()),
		useHeuristicPredicate: true,
		distance:              distanceFunc,
		provider:              config.Provider,
		idToIndex:             idToIndexMap,
		ordinalToID:           ordinalToID,
		trainingVectors:       nil,
		linkSFL:               linkSFL,
		link0SFL:              link0SFL,
		nodeSFL:               nodeSFL,
		registryPool:          registryPool,
		inFlightNodes:         inFlight,
		scratchPool:           scratchPool,
		reclamation:           reclamation,
	}
	if repairQueueSize := config.repairQueueSize(); repairQueueSize > 0 {
		index.repairCh = make(chan uint32, repairQueueSize)
		if config.RepairEnabled {
			index.startRepairWorker()
		}
	}
	switch config.RawVectorStore {
	case "", RawVectorStoreMemory:
		index.rawVectorStore = NewInMemoryRawVectorStoreWithCapacity(config.Dimension, config.RawStoreCap)
	case RawVectorStoreSlabby:
		store, err := NewSlabbyRawVectorStore(config.Dimension, config.RawStoreCap)
		if err != nil {
			_ = index.Close()
			return nil, fmt.Errorf("failed to create slabby raw vector store: %w", err)
		}
		index.rawVectorStore = store
	default:
		_ = index.Close()
		return nil, fmt.Errorf("unsupported raw vector store backend: %s", config.RawVectorStore)
	}

	// Initialize quantizer if quantization is configured
	if config.Quantization != nil {
		quantizer, err := quant.Create(config.Quantization)
		if err != nil {
			_ = index.Close()
			return nil, fmt.Errorf("failed to create quantizer: %w", err)
		}
		index.quantizer = quantizer
	}

	// Pre-allocate training vectors array if quantization is enabled
	if config.Quantization != nil {
		index.trainingVectors = make([][]float32, index.getTrainingThreshold())
	}

	// Force allocator growth and initialize the reusable search contexts before
	// the index is published. Inserts and searches must not pay first-use mmap
	// wrappers or scratch allocation on their latency-critical paths.
	allocators := [...]struct {
		name  string
		value *memory.ShardedFreeList
	}{
		{name: "node", value: index.nodeSFL},
		{name: "link", value: index.linkSFL},
		{name: "link0", value: index.link0SFL},
	}
	for _, allocator := range allocators {
		slot, err := allocator.value.Allocate()
		if err != nil {
			_ = index.Close()
			return nil, fmt.Errorf("prewarm %s allocator: %w", allocator.name, err)
		}
		if err := allocator.value.Deallocate(slot); err != nil {
			_ = index.Close()
			return nil, fmt.Errorf("return prewarmed %s slot: %w", allocator.name, err)
		}
	}
	scratchCount := min(64, max(32, runtime.GOMAXPROCS(0)*4))
	index.searchScratches = make([]searchScratch, scratchCount)
	scratchNodeCapacity := config.RawStoreCap
	if scratchNodeCapacity <= 0 {
		scratchNodeCapacity = int(config.idMapCapacity())
	}
	scratchEF := max(config.EfConstruction*2, config.EfSearch)
	for i := range index.searchScratches {
		scratch := &index.searchScratches[i]
		scratch.slot = uint8(i)
		index.prepareSearchScratch(scratch, scratchNodeCapacity, scratchEF)
	}
	if scratchCount == 64 {
		index.searchScratchFree.Store(^uint64(0))
	} else {
		index.searchScratchFree.Store((uint64(1) << scratchCount) - 1)
	}

	return index, nil
}

func (h *Index) Insert(ctx context.Context, entry *VectorEntry) error {
	if h.reclamation != nil {
		h.reclamation.tryReclaim(h)
	}
	// 1. Metadata Setup (Write Lock)
	// 1. Lock-Free Metadata Allocation (slices protected by metaMu)
	node, err := h.insertSingleMetadata(ctx, entry)
	if err != nil {
		return err
	}
	// If first node, it returns nil, nil
	if node == nil {
		return nil
	}
	if h.inFlightNodes != nil {
		atomic.StoreUint32(&node.InFlight, 1)
		h.inFlightNodes.Add(uint32(node.Ordinal))
	}

	// 2. Lock-Free Graph Traversal & Mutually Unaware Edge Construction
	err = h.insertNode(ctx, node, node.Ordinal, entry.Vector)

	// 3. Remove from in-flight registry
	if h.inFlightNodes != nil {
		atomic.StoreUint32(&node.InFlight, 0)
		h.inFlightNodes.Remove(uint32(node.Ordinal))
	}

	if err != nil {
		// Unpublish before releasing owned vector and link storage. The node slot
		// itself remains retired until epoch reclamation is available for readers
		// that may already have captured its address.
		h.retireNodeStorage(node.Ordinal, node)
		if entry.ID != "" {
			h.idToIndex.DeleteString(entry.ID)
			h.ordinalToID.Set(node.Ordinal, "")
		}
		h.size.Add(-1)
	} else {
		// Update entry point atomically if necessary
		h.updateEntryPointCAS(node)
	}
	if h.reclamation != nil {
		h.reclamation.tryReclaim(h)
	}

	return err
}

// insertSingleMetadata handles single vector metadata initialization
func (h *Index) insertSingleMetadata(ctx context.Context, entry *VectorEntry) (*Node, error) {
	if entry == nil {
		return nil, fmt.Errorf("entry cannot be nil")
	}
	if len(entry.Vector) == 0 {
		return nil, fmt.Errorf("vector cannot be empty")
	}
	if len(entry.Vector) != h.config.Dimension {
		return nil, fmt.Errorf("vector dimension mismatch: expected %d, got %d", h.config.Dimension, len(entry.Vector))
	}

	// Handle quantization training collection
	if h.quantizer != nil && !h.quantizationTrained.Load() {
		// Collect vectors for training lock-free
		vectorCopy := make([]float32, len(entry.Vector))
		copy(vectorCopy, entry.Vector)

		threshold := h.getTrainingThreshold()
		count := h.trainingCount.Add(1)
		if int(count) <= threshold {
			h.trainingVectors[count-1] = vectorCopy
			if int(count) == threshold {
				if err := h.trainQuantizer(ctx); err != nil {
					return nil, fmt.Errorf("failed to train quantizer: %w", err)
				}
				h.quantizationTrained.Store(true)
			}
		}
	}

	// Create new node with optimized memory allocation
	var ordinal uint32
	if h.provider == nil {
		ordinal = h.nextOrdinal.Add(1) - 1
	} else {
		ordinal = entry.Ordinal
	}
	if ordinal >= maxNodeCapacity {
		return nil, fmt.Errorf("node ordinal %d exceeds registry capacity %d", ordinal, maxNodeCapacity)
	}
	level := h.generateLevel(ordinal)
	if int(ordinal) < h.nodes.Len() && h.nodes.Get(ordinal) != nil {
		return nil, fmt.Errorf("node with ordinal %d already exists", ordinal)
	}
	nodeSlot, err := h.nodeSFL.Allocate()
	if err != nil {
		return nil, fmt.Errorf("failed to allocate node from SFL: %w", err)
	}
	node := (*Node)(unsafe.Pointer(&nodeSlot[SFLMetadataOverhead]))
	*node = Node{
		Ordinal: ordinal,
		Level:   level,
		Slot:    SentinelNodeID,
	}
	h.initNodeArrays(node, level, h.config.M)

	if h.rawVectorStore != nil {
		ref, err := h.rawVectorStore.Put(entry.Vector)
		if err != nil {
			h.releaseUnpublishedNode(node)
			return nil, fmt.Errorf("failed to store raw vector: %w", err)
		}
		node.Slot = ref.Slot
		if vec, err := h.rawVectorStore.Get(ref); err == nil {
			node.setVector(vec)
		}
	}

	// Handle vector compression when quantization is active. Keep the raw
	// off-heap vector reference as well so final candidate reranking can use
	// exact distances instead of returning approximate PQ/SQ order.
	if h.quantizer != nil && h.quantizationTrained.Load() {
		compressed, err := h.quantizer.Compress(entry.Vector)
		if err != nil {
			h.releaseUnpublishedNode(node)
			return nil, fmt.Errorf("failed to compress vector: %w", err)
		}
		node.CompressedVector = compressed
	}

	nodeID := node.Ordinal

	if entry.ID != "" {
		if _, inserted, err := h.idToIndex.PutStringIfAbsent(entry.ID, node); err != nil {
			h.releaseUnpublishedNode(node)
			return nil, fmt.Errorf("register vector ID %s: %w", entry.ID, err)
		} else if !inserted {
			h.releaseUnpublishedNode(node)
			return nil, fmt.Errorf("vector with ID %s already exists", entry.ID)
		}
	}

	h.nodes.Set(nodeID, node)

	if entry.ID != "" {
		h.ordinalToID.Set(nodeID, entry.ID)
	}

	// No entry point candidate list needed anymore, we fall back to O(N) scan.

	// Only the node that changes the empty state from zero may skip graph
	// insertion. A higher-level concurrent node must still connect normally.
	if h.initializeEntryPointCAS(node) {
		h.size.Add(1)
		return nil, nil
	}

	h.size.Add(1)
	return node, nil
}

const SentinelNodeID uint32 = 0xFFFFFFFF

func (h *Index) newNodeArrays(level int, baseM int) (links, backlinks [MaxLevel]*uint32) {
	for i := 0; i <= level; i++ {
		capacity := levelMaxLinks(baseM, i)
		slack := capacity / 4
		if slack < 4 {
			slack = 4
		}
		maxCapacity := capacity + slack

		// Allocate Links
		var slotL, slotB []byte
		var err error
		if i == 0 {
			slotL, err = h.link0SFL.Allocate()
			if err == nil {
				slotB, err = h.link0SFL.Allocate()
			}
		} else {
			slotL, err = h.linkSFL.Allocate()
			if err == nil {
				slotB, err = h.linkSFL.Allocate()
			}
		}
		if err != nil {
			panic(fmt.Sprintf("hnsw: failed to allocate links/backlinks: %v", err))
		}

		ptrL := (*uint32)(unsafe.Pointer(&slotL[SFLMetadataOverhead]))
		ptrB := (*uint32)(unsafe.Pointer(&slotB[SFLMetadataOverhead]))

		// Initialize with SentinelNodeID
		sliceL := unsafe.Slice(ptrL, maxCapacity)
		sliceB := unsafe.Slice(ptrB, maxCapacity)
		for j := 0; j < maxCapacity; j++ {
			sliceL[j] = SentinelNodeID
			sliceB[j] = SentinelNodeID
		}

		links[i] = ptrL
		backlinks[i] = ptrB
	}
	return links, backlinks
}

func inlineNodeLinkOffset() uintptr {
	const align = uintptr(8)
	size := uintptr(unsafe.Sizeof(Node{}))
	return (size + align - 1) &^ (align - 1)
}

func inlineNodeSlotPayloadSize(baseM int) uint64 {
	linkCap := uintptr(linkArrayCapacity(baseM, 0))
	return uint64(inlineNodeLinkOffset() + linkCap*4*2)
}

func (h *Index) initNodeArrays(node *Node, level int, baseM int) {
	if node == nil {
		return
	}
	h.initInlineLevel0Arrays(node, baseM)
	for i := 1; i <= level; i++ {
		slotL, err := h.linkSFL.Allocate()
		if err != nil {
			panic(fmt.Sprintf("hnsw: failed to allocate upper links: %v", err))
		}
		slotB, err := h.linkSFL.Allocate()
		if err != nil {
			panic(fmt.Sprintf("hnsw: failed to allocate upper backlinks: %v", err))
		}
		ptrL := (*uint32)(unsafe.Pointer(&slotL[SFLMetadataOverhead]))
		ptrB := (*uint32)(unsafe.Pointer(&slotB[SFLMetadataOverhead]))
		maxCapacity := linkArrayCapacity(baseM, i)
		sliceL := unsafe.Slice(ptrL, maxCapacity)
		sliceB := unsafe.Slice(ptrB, maxCapacity)
		for j := 0; j < maxCapacity; j++ {
			sliceL[j] = SentinelNodeID
			sliceB[j] = SentinelNodeID
		}
		node.Links[i] = ptrL
		node.Backlinks[i] = ptrB
	}
}

func (h *Index) initInlineLevel0Arrays(node *Node, baseM int) {
	linkCap := linkArrayCapacity(baseM, 0)
	base := uintptr(unsafe.Pointer(node)) + inlineNodeLinkOffset()
	links := (*uint32)(unsafe.Pointer(base))
	backlinks := (*uint32)(unsafe.Pointer(base + uintptr(linkCap*4)))
	linkSlice := unsafe.Slice(links, linkCap)
	backlinkSlice := unsafe.Slice(backlinks, linkCap)
	for i := 0; i < linkCap; i++ {
		linkSlice[i] = SentinelNodeID
		backlinkSlice[i] = SentinelNodeID
	}
	node.Links[0] = links
	node.Backlinks[0] = backlinks
}

func isInlineLevel0LinkPtr(node *Node, baseM int, ptr *uint32) bool {
	if node == nil || ptr == nil {
		return false
	}
	linkCap := linkArrayCapacity(baseM, 0)
	base := uintptr(unsafe.Pointer(node)) + inlineNodeLinkOffset()
	links := (*uint32)(unsafe.Pointer(base))
	backlinks := (*uint32)(unsafe.Pointer(base + uintptr(linkCap*4)))
	return ptr == links || ptr == backlinks
}

func initialNodeLinkCapacity(baseM int, level int) int {
	maxLinks := levelMaxLinks(baseM, level)
	if level == 0 {
		return maxLinks + max(8, maxLinks/2)
	}
	return maxLinks
}

func linkArrayCapacity(baseM int, level int) int {
	capacity := levelMaxLinks(baseM, level)
	slack := capacity / 4
	if slack < 4 {
		slack = 4
	}
	return capacity + slack
}

func getArraySliceWithCount(ptr *uint32, baseM int, level int, count uint32) []uint32 {
	if ptr == nil {
		return nil
	}
	maxCapacity := linkArrayCapacity(baseM, level)
	n := int(count)
	if n > maxCapacity {
		n = maxCapacity
	}
	return unsafe.Slice(ptr, maxCapacity)[:n]
}

func (h *Index) getNodeLinks(node *Node, level int) []uint32 {
	if node == nil || level < 0 || level >= MaxLevel || level > node.Level {
		return nil
	}
	return getArraySliceWithCount(node.Links[level], h.config.M, level, atomic.LoadUint32(&node.LinkCounts[level]))
}

func (h *Index) getNodeBacklinks(node *Node, level int) []uint32 {
	if node == nil || level < 0 || level >= MaxLevel || level > node.Level {
		return nil
	}
	return getArraySliceWithCount(node.Backlinks[level], h.config.M, level, atomic.LoadUint32(&node.BacklinkCounts[level]))
}

func (h *Index) appendWithSpinlock(node *Node, ptr *uint32, newID uint32, baseM int, level int) bool {
	if node == nil || ptr == nil {
		return false
	}

	// Acquire spinlock to protect against concurrent PruneConnections
	for !h.acquirePruneLock(node) {
		runtime.Gosched()
	}
	defer h.releasePruneLock(node)

	maxCapacity := linkArrayCapacity(baseM, level)
	countPtr := node.linkCountPtr(ptr, level)
	if countPtr == nil {
		return false
	}
	slice := unsafe.Slice(ptr, maxCapacity)
	count := int(atomic.LoadUint32(countPtr))
	if count > maxCapacity {
		count = maxCapacity
	}

	for i := 0; i < count; i++ {
		val := atomic.LoadUint32(&slice[i])
		if val == newID {
			return false // Already exists
		}
	}
	if count >= maxCapacity {
		return false
	}
	atomic.StoreUint32(&slice[count], newID)
	atomic.StoreUint32(countPtr, uint32(count+1))
	if ptr == node.Links[level] {
		atomic.StoreUint32(&node.LinkHeuristic[level], 0)
	}
	return true
}

func (node *Node) linkCountPtr(ptr *uint32, level int) *uint32 {
	if node == nil || level < 0 || level >= MaxLevel {
		return nil
	}
	if ptr == node.Links[level] {
		return &node.LinkCounts[level]
	}
	if ptr == node.Backlinks[level] {
		return &node.BacklinkCounts[level]
	}
	return nil
}

func (h *Index) acquirePruneLock(node *Node) bool {
	return atomic.CompareAndSwapUint32(&node.PruneLock, 0, 1)
}

func (h *Index) releasePruneLock(node *Node) {
	atomic.StoreUint32(&node.PruneLock, 0)
}

// BatchInsert provides optimized batch insertion for better performance with large datasets
func (h *Index) BatchInsert(ctx context.Context, entries []*VectorEntry) error {
	if len(entries) == 0 {
		return nil
	}

	// Use optimized batch processing for all batch sizes
	return h.batchInsertOptimized(ctx, entries)
}

// batchInsertOptimized handles large batch insertions with memory optimization
func (h *Index) batchInsertOptimized(ctx context.Context, entries []*VectorEntry) error {
	workers := min(runtime.GOMAXPROCS(0), len(entries))
	if workers <= 1 || len(entries) < 8 {
		for i, entry := range entries {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
			if err := h.Insert(ctx, entry); err != nil {
				return fmt.Errorf("failed to insert entry at index %d: %w", i, err)
			}
		}
		return nil
	}

	jobs := make(chan int, workers*2)
	errCh := make(chan error, 1)
	var stop atomic.Bool
	var wg sync.WaitGroup

	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				if stop.Load() {
					continue
				}
				select {
				case <-ctx.Done():
					stop.Store(true)
					select {
					case errCh <- ctx.Err():
					default:
					}
					return
				default:
				}
				if err := h.Insert(ctx, entries[idx]); err != nil {
					stop.Store(true)
					select {
					case errCh <- fmt.Errorf("failed to insert entry at index %d: %w", idx, err):
					default:
					}
					return
				}
			}
		}()
	}

sendLoop:
	for i := range entries {
		if stop.Load() {
			break
		}
		select {
		case <-ctx.Done():
			stop.Store(true)
			select {
			case errCh <- ctx.Err():
			default:
			}
			break sendLoop
		case jobs <- i:
		}
	}
	close(jobs)
	wg.Wait()

	select {
	case err := <-errCh:
		return err
	default:
		return nil
	}
}

// Search performs a KNN search using the HNSW algorithm.
func (h *Index) Search(ctx context.Context, query []float32, k int, filter interface {
	Test(idx uint64) bool
}) ([]*SearchResult, error) {

	if k <= 0 {
		return nil, fmt.Errorf("k must be positive, got %d: %w", k, util.ErrInvalidK)
	}
	if k > 4096 {
		return nil, fmt.Errorf("k %d exceeds maximum allowed search result limit of 4096", k)
	}

	if h.size.Load() == 0 {
		return nil, fmt.Errorf("%w", util.ErrEmptyIndex)
	}

	if len(query) != h.config.Dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d",
			len(query), h.config.Dimension)
	}
	qualityFloor := h.config.EfConstruction * 2
	ef := max(h.config.EfSearch, k, qualityFloor)
	if h.quantizer != nil {
		ef = max(ef, min(int(h.size.Load()), h.config.EfConstruction*2))
	}
	scratch := h.acquireSearchScratchWithEF(ef)
	defer h.releaseSearchScratch(scratch)

	size := int(h.size.Load())
	exactCutoff := max(h.config.EfConstruction*2, h.config.EfSearch, k)
	if size <= exactCutoff {
		return h.searchExact(ctx, query, k, filter)
	}

	var queryState any
	if h.quantizer != nil {
		queryState = h.quantizer.PrepareQuery(query)
	}

	// Phase 1: Search from top level to level 1
	ep := h.getEntryPoint()
	for level := h.getMaxLevel(); level > 0; level-- {
		candidate, ok, err := h.greedySearchLevelValue(ctx, query, ep, level, queryState)
		if err != nil {
			return nil, err
		}
		if ok {
			ep = h.nodes.Get(candidate.ID)
		}
	}

	// Phase 2: Search level 0 with ef. Extremely small efSearch values can
	// produce unacceptable tail recall even when the graph topology is sound,
	// so keep a degree/construction-aware floor while still honoring larger
	// caller-specified beams.
	candidates, err := h.searchLevelValuesWithScratch(ctx, query, ep, ef, 0, true, scratch, queryState, filter)
	if err != nil {
		return nil, err
	}
	h.rerankSearchCandidateValues(query, candidates)

	// Convert to results and limit to k
	results := make([]*SearchResult, 0, min(k, len(candidates)))
	for i, candidate := range candidates {
		if i >= k {
			break
		}

		node := h.nodes.Get(candidate.ID)
		if node == nil {
			continue
		}

		// Decompression is off the hot path — the collection layer hydrates
		// vectors from storage. Standalone path reads from off-heap rawVectorStore.
		var resultVector []float32
		if h.provider == nil {
			resultVector, _ = h.getNodeVector(node)
		}

		results = append(results, &SearchResult{
			Ordinal: node.Ordinal,
			ID:      strings.Clone(h.ordinalToID.Get(node.Ordinal)),
			Score:   candidate.Distance,
			Vector:  resultVector,
		})
	}

	// Filter candidates based on the graph filter
	if filter != nil {
		var filtered []*SearchResult
		for _, res := range results {
			if filter.Test(uint64(res.Ordinal)) {
				filtered = append(filtered, res)
			}
		}
		results = filtered
	}

	return results, nil
}

func (h *Index) searchExact(ctx context.Context, query []float32, k int, filter interface {
	Test(idx uint64) bool
}) ([]*SearchResult, error) {
	type exactCandidate struct {
		id       uint32
		distance float32
	}

	candidates := make([]exactCandidate, 0, min(k, int(h.size.Load())))
	for i := 0; i < h.nodes.Len(); i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		node := h.nodes.Get(uint32(i))
		if node == nil {
			continue
		}
		if filter != nil && !filter.Test(uint64(node.Ordinal)) {
			continue
		}

		vec := node.Vector
		if vec == nil {
			if stored, err := h.getNodeVector(node); err == nil {
				vec = stored
			}
		}
		var distance float32
		if vec != nil {
			distance = h.distance(query, vec)
		} else {
			var err error
			distance, err = h.computeDistanceOptimized(query, node, nil)
			if err != nil {
				continue
			}
		}
		candidates = append(candidates, exactCandidate{id: node.Ordinal, distance: distance})
	}

	if len(candidates) == 0 {
		return []*SearchResult{}, nil
	}

	slices.SortFunc(candidates, func(a, b exactCandidate) int {
		if a.distance < b.distance {
			return -1
		}
		if a.distance > b.distance {
			return 1
		}
		if a.id < b.id {
			return -1
		}
		if a.id > b.id {
			return 1
		}
		return 0
	})

	if k > len(candidates) {
		k = len(candidates)
	}
	results := make([]*SearchResult, 0, k)
	for _, candidate := range candidates[:k] {
		node := h.nodes.Get(candidate.id)
		if node == nil {
			continue
		}
		var resultVector []float32
		if h.provider == nil {
			resultVector, _ = h.getNodeVector(node)
		}
		results = append(results, &SearchResult{
			Ordinal: node.Ordinal,
			ID:      strings.Clone(h.ordinalToID.Get(node.Ordinal)),
			Score:   candidate.distance,
			Vector:  resultVector,
		})
	}
	return results, nil
}

func (h *Index) rerankSearchCandidates(query []float32, candidates []*util.Candidate) {
	if h.quantizer == nil || len(candidates) == 0 {
		return
	}
	for _, candidate := range candidates {
		node := h.nodes.Get(candidate.ID)
		if node == nil {
			continue
		}
		vec, err := h.getNodeVector(node)
		if err != nil || vec == nil {
			continue
		}
		candidate.Distance = h.distance(query, vec)
	}
	slices.SortFunc(candidates, compareCandidatePtrs)
}

func (h *Index) rerankSearchCandidateValues(query []float32, candidates []util.Candidate) {
	if h.quantizer == nil || len(candidates) == 0 {
		return
	}
	for i := range candidates {
		node := h.nodes.Get(candidates[i].ID)
		if node == nil {
			continue
		}
		vec, err := h.getNodeVector(node)
		if err != nil || vec == nil {
			continue
		}
		candidates[i].Distance = h.distance(query, vec)
	}
	slices.SortFunc(candidates, compareCandidateValues)
}

// Size returns the number of vectors in the index
func (h *Index) Size() int {
	return int(h.size.Load())
}

// MemoryUsage returns approximate memory usage in bytes
func (h *Index) MemoryUsage() int64 {
	scratch := h.acquireSearchScratch()
	defer h.releaseSearchScratch(scratch)
	return h.calculateMemoryUsage()
}

func (h *Index) RawVectorStoreProfile() map[string]any {

	if h.rawVectorStore == nil {
		return nil
	}
	profile := h.rawVectorStore.Profile()
	return map[string]any{
		"backend":              profile.Backend,
		"vector_count":         profile.VectorCount,
		"dimension":            profile.Dimension,
		"bytes_per_vector":     profile.BytesPerVector,
		"memory_usage":         profile.MemoryUsage,
		"reserved_bytes":       profile.ReservedBytes,
		"reserved_data_bytes":  profile.ReservedDataBytes,
		"reserved_meta_bytes":  profile.ReservedMetaBytes,
		"reserved_guard_bytes": profile.ReservedGuardBytes,
		"live_bytes":           profile.LiveBytes,
		"free_bytes":           profile.FreeBytes,
		"capacity_utilization": profile.CapacityUtilization,
	}
}

// calculateMemoryUsage calculates memory usage without acquiring locks (must be called with lock held)
func (h *Index) calculateMemoryUsage() int64 {
	// If memory mapped, return minimal in-memory usage
	if h.memoryMapped {
		var usage int64

		// Count only essential in-memory structures
		usage += int64(h.nodes.Len() * 8) // Pointer overhead
		// Note: idToIndex map was removed in favor of SegmentedArray

		// Add quantizer memory usage if present
		if h.quantizer != nil {
			usage += h.quantizer.MemoryUsage()
		}

		return usage
	}

	var usage int64
	for i := 0; i < h.nodes.Len(); i++ {
		node := h.nodes.Get(uint32(i))
		if node == nil {
			continue
		}
		if node == nil {
			continue
		}
		// Vector data (original or compressed)
		if node.CompressedVector != nil {
			usage += int64(len(node.CompressedVector))
		}

		// Links
		for i := 0; i < len(node.Links); i++ {
			links := h.getNodeLinks(node, i)
			usage += int64(len(links) * 4) // 4 bytes per uint32
		}

		// Node overhead (approximate)
		usage += 32
	}

	// Add quantizer memory usage if present
	if h.quantizer != nil {
		usage += h.quantizer.MemoryUsage()
	}
	if h.rawVectorStore != nil {
		usage += h.rawVectorStore.MemoryUsage()
	}

	// Add training vectors memory usage if still collecting
	if h.trainingVectors != nil {
		for _, vec := range h.trainingVectors {
			usage += int64(len(vec) * 4)
		}
	}

	return usage
}

// Close shuts down the index
func (h *Index) Close() error {
	h.stopRepairWorker()
	h.setEntryPoint(nil)
	h.searchScratchFree.Store(0)
	if h.reclamation != nil {
		h.reclamation.drain(h)
	}
	for i := range h.searchScratches {
		if arena := h.searchScratches[i].arena; arena != nil {
			_ = arena.Free()
			h.searchScratches[i].arena = nil
		}
	}
	h.searchScratches = nil

	// Free off-heap link storage.
	if h.nodeSFL != nil {
		h.nodeSFL.Free()
		h.nodeSFL = nil
	}
	if h.linkSFL != nil {
		h.linkSFL.Free()
		h.linkSFL = nil
	}
	if h.link0SFL != nil {
		h.link0SFL.Free()
		h.link0SFL = nil
	}

	// Clear all data structures. Registry directories and chunks are off-heap
	// and share registryPool, so detach them before unmapping the pool.
	if h.nodes != nil {
		_ = h.nodes.Close()
		h.nodes = nil
	}
	if h.ordinalToID != nil {
		_ = h.ordinalToID.Close()
		h.ordinalToID = nil
	}
	if h.registryPool != nil {
		h.registryPool.Free()
		h.registryPool = nil
	}
	if h.idToIndex != nil {
		_ = h.idToIndex.Free()
		h.idToIndex = nil
	}

	h.nodes = nil
	h.size.Store(0)
	h.nextOrdinal.Store(0)
	if h.rawVectorStore != nil {
		_ = h.rawVectorStore.Close()
	}
	if h.reclamation != nil {
		h.reclamation.close()
		h.reclamation = nil
	}
	if h.quantizer != nil {
		h.quantizer.Close()
	}

	return nil
}

// generateLevel derives an independent uniform sample from the stable ordinal.
// This avoids a shared PRNG and synchronization on parallel inserts.
func (h *Index) generateLevel(ordinal uint32) int {
	z := uint64(ordinal) + uint64(h.config.RandomSeed) + 0x9e3779b97f4a7c15
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z ^= z >> 31
	u := (float64(z>>11) + 1) / (float64(uint64(1)<<53) + 1)
	level := int(-math.Log(u) * h.config.ML)
	if level >= MaxLevel {
		return MaxLevel - 1
	}
	if level < 0 {
		return 0
	}
	return level
}

// findNodeID returns the stable node index without scanning h.nodes.
func (h *Index) findNodeID(target *Node) uint32 {
	if target == nil {
		return ^uint32(0)
	}
	if int(target.Ordinal) >= h.nodes.Len() || h.nodes.Get(target.Ordinal) != target {
		return ^uint32(0)
	}
	return target.Ordinal
}

// validate checks if the configuration is valid
func (c *Config) validate() error {
	if c.Dimension <= 0 {
		return fmt.Errorf("dimension must be positive")
	}
	if c.M <= 0 {
		return fmt.Errorf("M must be positive")
	}
	if c.EfConstruction <= 0 {
		return fmt.Errorf("EfConstruction must be positive")
	}
	if c.EfSearch <= 0 {
		return fmt.Errorf("EfSearch must be positive")
	}
	if c.ML <= 0 {
		return fmt.Errorf("ML must be positive")
	}
	if c.IDMapCapacity < 0 {
		return fmt.Errorf("IDMapCapacity must be non-negative")
	}
	if c.IDMapCapacity > maxNodeCapacity {
		return fmt.Errorf("IDMapCapacity must not exceed %d", maxNodeCapacity)
	}
	if c.PruneAlpha > 0 && c.PruneAlpha < 1 {
		return fmt.Errorf("PruneAlpha must be >= 1 when set")
	}
	if c.Level0LinkMultiplier > 0 && c.Level0LinkMultiplier < 1 {
		return fmt.Errorf("Level0LinkMultiplier must be >= 1 when set")
	}
	if c.RepairQueueSize < 0 {
		return fmt.Errorf("RepairQueueSize must be non-negative")
	}
	if c.RepairBatchSize < 0 {
		return fmt.Errorf("RepairBatchSize must be non-negative")
	}

	// Validate quantization config if present
	if c.Quantization != nil {
		if err := c.Quantization.Validate(); err != nil {
			return fmt.Errorf("invalid quantization config: %w", err)
		}
	}

	return nil
}

// getTrainingThreshold returns the minimum number of vectors needed for training
func (h *Index) getTrainingThreshold() int {
	if h.config.Quantization == nil {
		return 0
	}

	// Use a reasonable threshold based on quantization type
	switch h.config.Quantization.Type {
	case quant.ProductQuantization:
		// PQ needs more training data for k-means clustering, but keep it reasonable
		return max(100, h.config.Quantization.Codebooks*32)
	case quant.ScalarQuantization:
		// Scalar quantization needs less training data
		return max(50, h.config.Dimension*2)
	default:
		return 100
	}
}

// trainQuantizer trains the quantizer using collected training vectors
func (h *Index) trainQuantizer(ctx context.Context) error {
	if h.quantizer == nil || len(h.trainingVectors) == 0 {
		return fmt.Errorf("no quantizer or training data available")
	}

	// Use a subset of training vectors based on TrainRatio
	trainRatio := h.config.Quantization.TrainRatio
	if trainRatio <= 0 || trainRatio > 1 {
		trainRatio = 0.1 // Default to 10%
	}

	trainCount := int(float64(len(h.trainingVectors)) * trainRatio)
	if trainCount < 1 {
		trainCount = len(h.trainingVectors)
	}

	trainingSet := h.trainingVectors[:trainCount]

	if err := h.quantizer.Train(ctx, trainingSet); err != nil {
		return fmt.Errorf("quantizer training failed: %w", err)
	}

	h.quantizationTrained.Store(true)

	// Clear training vectors to free memory
	h.trainingVectors = nil

	return nil
}

// getNodeVector returns the vector for a node, handling quantization
func (h *Index) getNodeVector(node *Node) ([]float32, error) {
	if node == nil {
		return nil, fmt.Errorf("node is nil")
	}
	if node.Vector != nil {
		return node.Vector, nil
	}
	if h.provider != nil {
		if vec, err := h.provider.GetByOrdinal(node.Ordinal); err == nil && vec != nil {
			return vec, nil
		}
	}
	if ref, ok := h.rawVectorRef(node); ok {
		return h.rawVectorStore.Get(ref)
	}
	if node.CompressedVector != nil && h.quantizer != nil {
		return h.quantizer.Decompress(node.CompressedVector)
	}
	return nil, fmt.Errorf("vector unavailable for ordinal %d (compressed: %v, provider: %v, rawStore: %v)",
		node.Ordinal, node.CompressedVector != nil, h.provider != nil, h.rawVectorStore != nil)
}

// getNodeVectorLocal retrieves a node's vector from local storage only.
// It deliberately skips h.provider to avoid re-entrant lock acquisition
// during serialization (engine holds e.mu.Lock; provider.GetByOrdinal
// would try e.mu.RLock → deadlock).
func (h *Index) getNodeVectorLocal(node *Node) ([]float32, error) {
	if node == nil {
		return nil, fmt.Errorf("node is nil")
	}
	if node.Vector != nil {
		return node.Vector, nil
	}
	if ref, ok := h.rawVectorRef(node); ok {
		vec, err := h.rawVectorStore.Get(ref)
		if err == nil && vec != nil {
			return vec, nil
		}
	}
	if node.CompressedVector != nil && h.quantizer != nil {
		return h.quantizer.Decompress(node.CompressedVector)
	}
	return nil, fmt.Errorf("vector for ordinal %d not in local storage", node.Ordinal)
}

func (h *Index) rawVectorRef(node *Node) (VectorRef, bool) {
	if h.rawVectorStore == nil || node == nil || node.Slot == SentinelNodeID {
		return VectorRef{}, false
	}
	return VectorRef{
		Kind:  VectorEncodingRaw,
		Slot:  node.Slot,
		Bytes: uint32(h.config.Dimension * 4),
		Valid: true,
	}, true
}

func (h *Index) refreshNodeVectorViewsFromRawStore(slotByOrdinal bool) {
	if h.rawVectorStore == nil {
		return
	}
	for i := 0; i < h.nodes.Len(); i++ {
		node := h.nodes.Get(uint32(i))
		if node == nil {
			continue
		}
		if slotByOrdinal {
			node.Slot = node.Ordinal
		}
		node.setVector(nil)
		ref, ok := h.rawVectorRef(node)
		if !ok {
			continue
		}
		if vec, err := h.rawVectorStore.Get(ref); err == nil {
			node.setVector(vec)
		}
	}
}

// SnapshotVectorsFromProvider copies all node vectors from the provider into
// rawVectorStore so that subsequent serialization can proceed without calling
// back into the provider. Must be called before SerializeToBytes when a
// provider is set and the caller cannot re-enter the provider.
func (h *Index) SnapshotVectorsFromProvider(ctx context.Context) error {
	scratch := h.acquireSearchScratch()
	defer h.releaseSearchScratch(scratch)

	if h.provider == nil {
		return nil
	}
	if h.rawVectorStore == nil {
		switch h.config.RawVectorStore {
		case "", RawVectorStoreMemory:
			h.rawVectorStore = NewInMemoryRawVectorStoreWithCapacity(h.config.Dimension, h.config.RawStoreCap)
		case RawVectorStoreSlabby:
			store, err := NewSlabbyRawVectorStore(h.config.Dimension, h.config.RawStoreCap)
			if err != nil {
				return err
			}
			h.rawVectorStore = store
		default:
			return fmt.Errorf("unsupported raw vector store backend: %s", h.config.RawVectorStore)
		}
	}

	for i := 0; i < h.nodes.Len(); i++ {
		node := h.nodes.Get(uint32(i))
		if node == nil {
			continue
		}
		if node == nil || node.CompressedVector != nil {
			continue
		}
		if ref, ok := h.rawVectorRef(node); ok {
			if vec, err := h.rawVectorStore.Get(ref); err == nil && vec != nil {
				node.setVector(vec)
				continue
			}
		}
		vec, err := h.provider.GetByOrdinal(node.Ordinal)
		if err != nil || vec == nil {
			return fmt.Errorf("snapshot vector ordinal %d: %w", node.Ordinal, err)
		}
		ref, err := h.rawVectorStore.Put(vec)
		if err != nil {
			return fmt.Errorf("snapshot store ordinal %d: %w", node.Ordinal, err)
		}
		node.Slot = ref.Slot
		if stored, err := h.rawVectorStore.Get(ref); err == nil {
			node.setVector(stored)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}
	return nil
}

// computeDistance computes distance between vectors, handling quantization
func (h *Index) computeDistance(vec1, vec2 []float32, node1, node2 *Node) (float32, error) {
	// If both nodes are quantized, use quantized distance computation
	if node1 != nil && node2 != nil &&
		node1.CompressedVector != nil && node2.CompressedVector != nil &&
		h.quantizer != nil {
		distance, err := h.quantizer.Distance(node1.CompressedVector, node2.CompressedVector)
		if err != nil {
			return 0, err
		}
		return h.normalizeQuantizedDistance(distance), nil
	}

	// If one is a query vector and the other is quantized
	if node2 != nil && node2.CompressedVector != nil && h.quantizer != nil {
		distance, err := h.quantizer.DistanceToQuery(node2.CompressedVector, vec1, nil)
		if err != nil {
			return 0, err
		}
		return h.normalizeQuantizedDistance(distance), nil
	}

	if vec1 == nil && node1 != nil {
		var err error
		vec1, err = h.getNodeVector(node1)
		if err != nil {
			return 0, err
		}
	}
	if vec2 == nil && node2 != nil {
		var err error
		vec2, err = h.getNodeVector(node2)
		if err != nil {
			return 0, err
		}
	}
	if len(vec1) != h.config.Dimension || len(vec2) != h.config.Dimension {
		return 0, fmt.Errorf("distance vector unavailable or malformed: got %d and %d dimensions, want %d", len(vec1), len(vec2), h.config.Dimension)
	}

	return h.distance(vec1, vec2), nil
}

func (h *Index) Delete(ctx context.Context, id string) error {
	if h.reclamation != nil {
		h.reclamation.tryReclaim(h)
	}
	scratch := h.acquireSearchScratch()
	err := h.deleteNode(ctx, id)
	h.releaseSearchScratch(scratch)
	if h.reclamation != nil {
		h.reclamation.tryReclaim(h)
	}
	return err
}

func (h *Index) DeleteByOrdinal(ctx context.Context, ordinal uint32) error {
	if h.reclamation != nil {
		h.reclamation.tryReclaim(h)
	}
	scratch := h.acquireSearchScratch()
	err := h.deleteNodeByOrdinal(ctx, ordinal)
	h.releaseSearchScratch(scratch)
	if h.reclamation != nil {
		h.reclamation.tryReclaim(h)
	}
	return err
}

// MemoryMappable interface implementation

// CanMemoryMap returns true if the index can be memory mapped
func (h *Index) CanMemoryMap() bool {

	// Can memory map if we have nodes and are not already mapped
	return int(h.size.Load()) > 0 && !h.memoryMapped
}

// EstimateSize returns the estimated size in bytes if memory mapped
func (h *Index) EstimateSize() int64 {
	return h.MemoryUsage()
}

// EnableMemoryMapping enables memory mapping for the index
func (h *Index) EnableMemoryMapping(basePath string) error {
	scratch := h.acquireSearchScratch()
	defer h.releaseSearchScratch(scratch)

	if h.memoryMapped {
		return fmt.Errorf("index is already memory mapped")
	}

	if h.size.Load() == 0 {
		return fmt.Errorf("cannot memory map empty index")
	}

	// Store current memory usage for comparison (calculate inline to avoid deadlock)
	h.originalMemUsage = h.calculateMemoryUsage()

	// Create a temporary file path for the memory-mapped index
	mmapPath := fmt.Sprintf("%s/hnsw_index_%p.mmap", basePath, h)

	// Save the index to the memory-mapped file (use lock-free version since we already hold the lock)
	if err := h.saveToDiskWithoutLock(context.Background(), mmapPath); err != nil {
		return fmt.Errorf("failed to save index for memory mapping: %w", err)
	}

	// Get file size
	stat, err := os.Stat(mmapPath)
	if err != nil {
		return fmt.Errorf("failed to stat memory-mapped file: %w", err)
	}

	h.mmapPath = mmapPath
	h.mmapSize = stat.Size()
	h.memoryMapped = true

	// Clear in-memory data structures to free memory
	if h.rawVectorStore != nil {
		vecMmapPath := fmt.Sprintf("%s/hnsw_vectors_%p.mmap", basePath, h)
		mmapStore, err := h.createMmapRawVectorStore(vecMmapPath)
		if err != nil {
			return fmt.Errorf("failed to create memory-mapped raw vector store: %w", err)
		}
		h.vecMmap = mmapStore.mmap
		oldStore := h.rawVectorStore
		for i := 0; i < h.nodes.Len(); i++ {
			if node := h.nodes.Get(uint32(i)); node != nil {
				node.setVector(nil)
			}
		}
		_ = oldStore.Reset()
		h.rawVectorStore = mmapStore
		h.refreshNodeVectorViewsFromRawStore(true)
	} else if h.quantizer != nil {
		pqMmapPath := fmt.Sprintf("%s/hnsw_pq_%p.mmap", basePath, h)
		pqMmap, err := h.createMmapCompressedVectorStore(pqMmapPath)
		if err != nil {
			return fmt.Errorf("failed to create memory-mapped compressed vector store: %w", err)
		}
		h.pqMmap = pqMmap
	}

	// Clear training vectors
	h.trainingVectors = nil

	return nil
}

// DisableMemoryMapping disables memory mapping and loads data back to RAM
func (h *Index) DisableMemoryMapping() error {

	if !h.memoryMapped {
		return fmt.Errorf("index is not memory mapped")
	}

	// Unmap and clean up vector mmap files first so loadFromDiskImpl can create new stores
	if h.vecMmap != nil {
		path := h.vecMmap.Path()
		h.vecMmap.Close()
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			fmt.Printf("Failed to remove vector memory-mapped file %s: %v\n", path, err)
		}
		h.vecMmap = nil
		if h.quantizer != nil {
			h.quantizer.Close()
			h.quantizer = nil
		}
		h.rawVectorStore = nil
	}
	if h.pqMmap != nil {
		path := h.pqMmap.Path()
		h.pqMmap.Close()
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			fmt.Printf("Failed to remove pq memory-mapped file %s: %v\n", path, err)
		}
		h.pqMmap = nil
	}

	// Reload the index from the memory-mapped file
	if err := h.loadFromDiskImpl(context.Background(), h.mmapPath); err != nil {
		return fmt.Errorf("failed to reload index from memory mapping: %w", err)
	}

	// Clean up the memory-mapped file
	if err := os.Remove(h.mmapPath); err != nil && !os.IsNotExist(err) {
		// Log error but don't fail the operation
		fmt.Printf("Failed to remove memory-mapped file %s: %v\n", h.mmapPath, err)
	}

	h.memoryMapped = false
	h.mmapPath = ""
	h.mmapSize = 0
	h.originalMemUsage = 0

	return nil
}

// IsMemoryMapped returns true if currently using memory mapping
func (h *Index) IsMemoryMapped() bool {
	return h.memoryMapped
}

// MemoryMappedSize returns the size of memory-mapped data
func (h *Index) MemoryMappedSize() int64 {

	if h.memoryMapped {
		return h.mmapSize
	}
	return 0
}

// SaveToDisk persists the HNSW index to disk in binary format
func (h *Index) SaveToDisk(ctx context.Context, path string) error {
	scratch := h.acquireSearchScratch()
	defer h.releaseSearchScratch(scratch)
	return h.saveToDiskImpl(ctx, path)
}

// LoadFromDisk rebuilds HNSW index from disk
func (h *Index) LoadFromDisk(ctx context.Context, path string) error {
	return h.loadFromDiskImpl(ctx, path)
}

// SerializeToBytes serializes the index to an in-memory byte slice using the
// same binary format as SaveToDisk.
func (h *Index) SerializeToBytes() ([]byte, error) {
	scratch := h.acquireSearchScratch()
	defer h.releaseSearchScratch(scratch)

	var buf bytes.Buffer
	writer := bufio.NewWriter(&buf)

	if err := h.writeHeader(writer); err != nil {
		return nil, fmt.Errorf("failed to write header: %w", err)
	}
	if err := h.writeConfig(writer); err != nil {
		return nil, fmt.Errorf("failed to write config: %w", err)
	}
	if err := h.writeNodes(writer); err != nil {
		return nil, fmt.Errorf("failed to write nodes: %w", err)
	}
	if err := h.writeLinks(writer); err != nil {
		return nil, fmt.Errorf("failed to write links: %w", err)
	}
	if err := h.writeMetadata(writer); err != nil {
		return nil, fmt.Errorf("failed to write metadata: %w", err)
	}

	writer.Flush()
	return buf.Bytes(), nil
}

// DeserializeFromBytes restores the index from an in-memory byte slice.
func (h *Index) DeserializeFromBytes(ctx context.Context, data []byte) error {
	reader := bufio.NewReader(bytes.NewReader(data))

	if err := h.readHeader(reader); err != nil {
		return fmt.Errorf("failed to read header: %w", err)
	}
	if err := h.readConfig(reader); err != nil {
		return fmt.Errorf("failed to read config: %w", err)
	}

	// Always restore a local raw vector store during deserialization so the
	// index can answer searches even before external storage is populated.
	if h.rawVectorStore == nil {
		switch h.config.RawVectorStore {
		case "", RawVectorStoreMemory:
			h.rawVectorStore = NewInMemoryRawVectorStoreWithCapacity(h.config.Dimension, h.config.RawStoreCap)
		case RawVectorStoreSlabby:
			store, err := NewSlabbyRawVectorStore(h.config.Dimension, h.config.RawStoreCap)
			if err != nil {
				return fmt.Errorf("failed to create slabby raw vector store: %w", err)
			}
			h.rawVectorStore = store
		default:
			return fmt.Errorf("unsupported raw vector store backend: %s", h.config.RawVectorStore)
		}
	}

	if err := h.readNodes(ctx, reader); err != nil {
		return fmt.Errorf("failed to read nodes: %w", err)
	}
	if err := h.readLinks(ctx, reader); err != nil {
		return fmt.Errorf("failed to read links: %w", err)
	}
	if err := h.readMetadata(reader); err != nil {
		return fmt.Errorf("failed to read metadata: %w", err)
	}

	return h.rebuildIndexState()
}

// GetPersistenceMetadata returns metadata about the current index state
func (h *Index) GetPersistenceMetadata() *HNSWPersistenceMetadata {

	if h.size.Load() == 0 {
		return nil // No metadata for empty index
	}

	return &HNSWPersistenceMetadata{
		Version:       FormatVersion,
		NodeCount:     int(h.size.Load()),
		Dimension:     h.config.Dimension,
		MaxLevel:      h.getMaxLevel(),
		CreatedAt:     time.Now(),
		ChecksumCRC32: h.calculateCRC32(),
		FileSize:      h.estimateFileSize(),
	}
}

// Helper methods for metadata
func (h *Index) estimateFileSize() int64 {
	// Rough estimate of serialized size
	var size int64

	// Header + config
	size += 64

	// Nodes
	for i := 0; i < h.nodes.Len(); i++ {
		node := h.nodes.Get(uint32(i))
		if node == nil {
			continue
		}
		if node != nil {
			vectorBytes := int64(0)
			switch {
			case node.CompressedVector != nil:
				vectorBytes = int64(len(node.CompressedVector))
			case h.provider == nil:
				vectorBytes = int64(h.config.Dimension * 4)
			}
			size += vectorBytes + 24
		}
	}

	// Links
	for i := 0; i < h.nodes.Len(); i++ {
		node := h.nodes.Get(uint32(i))
		if node == nil {
			continue
		}
		if node != nil {
			for i := 0; i < len(node.Links); i++ {
				connections := h.getNodeLinks(node, i)
				if len(connections) > 0 {
					// Memory for the array of links
					size += int64(len(connections) * 4) // 4 bytes per uint32
				}
			}
		}
	}

	return size
}

func (h *Index) freeNodeLinks(node *Node) {
	if node == nil {
		return
	}
	for i, ptr := range node.Links {
		if i != 0 || !isInlineLevel0LinkPtr(node, h.config.M, ptr) {
			h.freeLinkArray(i, ptr)
		}
		node.Links[i] = nil
		atomic.StoreUint32(&node.LinkCounts[i], 0)
	}
	for i, ptr := range node.Backlinks {
		if i != 0 || !isInlineLevel0LinkPtr(node, h.config.M, ptr) {
			h.freeLinkArray(i, ptr)
		}
		node.Backlinks[i] = nil
		atomic.StoreUint32(&node.BacklinkCounts[i], 0)
	}
}

func (h *Index) retireNodeLinksAt(node *Node, epoch uint64) {
	if node == nil {
		return
	}
	for level, ptr := range node.Links {
		if ptr == nil || level == 0 && isInlineLevel0LinkPtr(node, h.config.M, ptr) {
			continue
		}
		kind := retiredUpperLink
		if level == 0 {
			kind = retiredLevel0Link
		}
		base := unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) - SFLMetadataOverhead)
		h.retireAllocationAt(epoch, kind, base)
	}
	for level, ptr := range node.Backlinks {
		if ptr == nil || level == 0 && isInlineLevel0LinkPtr(node, h.config.M, ptr) {
			continue
		}
		kind := retiredUpperLink
		if level == 0 {
			kind = retiredLevel0Link
		}
		base := unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) - SFLMetadataOverhead)
		h.retireAllocationAt(epoch, kind, base)
	}
}

// releaseUnpublishedNode returns allocations that were never made reachable
// through the node registry. No epoch delay is required for these slots.
func (h *Index) releaseUnpublishedNode(node *Node) {
	if node == nil {
		return
	}
	h.releaseUnpublishedVector(node)
	h.freeNodeLinks(node)
	node.CompressedVector = nil
	node.setVector(nil)
	if h.nodeSFL == nil {
		return
	}
	base := unsafe.Pointer(uintptr(unsafe.Pointer(node)) - SFLMetadataOverhead)
	slotSize := int(uint64(SFLMetadataOverhead) + inlineNodeSlotPayloadSize(h.config.M))
	_ = h.nodeSFL.Deallocate(unsafe.Slice((*byte)(base), slotSize))
}

func (h *Index) releaseUnpublishedVector(node *Node) {
	if node == nil || h.provider != nil || h.rawVectorStore == nil || node.Slot == SentinelNodeID {
		return
	}
	ref := VectorRef{
		Kind:  VectorEncodingRaw,
		Slot:  node.Slot,
		Bytes: uint32(h.config.Dimension * 4),
		Valid: true,
	}
	switch store := h.rawVectorStore.(type) {
	case *InMemoryRawVectorStore:
		_ = store.release(ref)
	case *SlabbyRawVectorStore:
		_ = store.release(ref)
	default:
		_ = h.rawVectorStore.Delete(ref)
	}
	node.setVector(nil)
}

func (h *Index) freeLinkArray(level int, ptr *uint32) {
	if ptr == nil {
		return
	}

	capacity := levelMaxLinks(h.config.M, level)
	slack := capacity / 4
	if slack < 4 {
		slack = 4
	}
	expectedCap := capacity + slack

	basePtr := unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) - SFLMetadataOverhead)
	if level == 0 {
		if h.link0SFL != nil {
			slot := unsafe.Slice((*byte)(basePtr), int(SFLMetadataOverhead)+expectedCap*4)
			_ = h.link0SFL.Deallocate(slot)
		}
	} else {
		if h.linkSFL != nil {
			slot := unsafe.Slice((*byte)(basePtr), int(SFLMetadataOverhead)+expectedCap*4)
			_ = h.linkSFL.Deallocate(slot)
		}
	}
}
