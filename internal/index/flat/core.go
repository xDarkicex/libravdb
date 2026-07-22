package flat

import (
	"context"
	"fmt"
	"sync"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/record"
	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/memory"
)

// Core is Flat's map-free, segment-based index. It owns no Go slice or map
// whose size follows record count: records live in immutable off-heap
// generations and visibility is resolved by the generation ordinal radix tree.
type Core struct {
	config  *Config
	mu      sync.RWMutex
	current *record.Generation
}

// PreparedDelta is a fully allocated, unpublished generation. It makes the
// storage-before-publication ordering explicit: transaction code prepares this
// object before its durable commit, then Commit only swaps a generation root.
type PreparedDelta struct {
	core *Core
	base *record.Generation
	next *record.Generation
}

type coreResult struct {
	ref   record.RecordRef
	score float32
}

type coreHeapElement struct {
	ref   record.RecordRef
	score float32
}

// ResultSet owns its result arena and a generation lease. It must be closed.
// Result records borrow immutable generation memory and remain valid until
// Close returns.
type ResultSet struct {
	arena    *memory.Arena
	snapshot *record.Generation
	results  []coreResult
}

func NewCore(config *Config) (*Core, error) {
	if config == nil || config.Dimension <= 0 {
		return nil, fmt.Errorf("flat core requires a positive dimension")
	}
	delta, err := record.NewDelta(record.DeltaConfig{
		ArenaBytes:   4096,
		MaxMutations: 1,
		IDCapacity:   8,
		IDKeyBytes:   4096,
	})
	if err != nil {
		return nil, err
	}
	generation, err := record.NewGeneration(nil, delta)
	if err != nil {
		_ = delta.Close()
		return nil, err
	}
	return &Core{config: config, current: generation}, nil
}

func (idx *Core) NewDelta(arenaBytes uint64, maxMutations uint32, keyBytes uint64) (*record.Delta, error) {
	return record.NewDelta(record.DeltaConfig{
		ArenaBytes:   arenaBytes,
		MaxMutations: maxMutations,
		IDCapacity:   uint64(maxMutations) * 2,
		IDKeyBytes:   keyBytes,
	})
}

func (idx *Core) Snapshot() *record.Generation {
	idx.mu.RLock()
	snapshot := idx.current.Acquire()
	idx.mu.RUnlock()
	return snapshot
}

// CommitDelta publishes an already validated candidate. Storage/WAL callers
// must invoke it only after durable commit; it performs no allocation after
// the new generation has been built.
func (idx *Core) CommitDelta(delta *record.Delta) error {
	prepared, err := idx.PrepareDelta(delta)
	if err != nil {
		return err
	}
	return prepared.Commit()
}

// PrepareDelta allocates and validates the next immutable generation without
// changing the live root. The caller must Commit after durable storage accepts
// the matching mutation, or Abort on any earlier failure.
func (idx *Core) PrepareDelta(delta *record.Delta) (*PreparedDelta, error) {
	idx.mu.Lock()
	base := idx.current
	next, err := record.NewGeneration(base, delta)
	idx.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return &PreparedDelta{core: idx, base: base, next: next}, nil
}

func (p *PreparedDelta) Commit() error {
	if p == nil || p.core == nil || p.next == nil {
		return fmt.Errorf("flat prepared delta is closed")
	}
	p.core.mu.Lock()
	defer p.core.mu.Unlock()
	if p.core.current != p.base {
		return fmt.Errorf("flat generation changed before publication")
	}
	p.core.current = p.next
	p.base.Release()
	p.core = nil
	p.base = nil
	p.next = nil
	return nil
}

func (p *PreparedDelta) Abort() error {
	if p == nil || p.next == nil {
		return nil
	}
	p.next.Release()
	p.core = nil
	p.base = nil
	p.next = nil
	return nil
}

func (idx *Core) CurrentRecord(id record.BytesView) (record.RecordRef, bool) {
	snapshot := idx.Snapshot()
	defer snapshot.Release()
	ref, found := snapshot.Lookup(id)
	return ref, found && !ref.Tombstone()
}

// Size performs a map-free logical count over the immutable generation chain.
// It is intentionally not part of the search hot path; callers use it for
// diagnostics and index metadata only.
func (idx *Core) Size() int {
	snapshot := idx.Snapshot()
	if snapshot == nil {
		return 0
	}
	defer snapshot.Release()
	count := 0
	for generation := snapshot; generation != nil; generation = generation.Parent() {
		for i := 0; i < generation.SegmentLen(); i++ {
			if snapshot.Visible(generation.SegmentAt(i)) {
				count++
			}
		}
	}
	return count
}

// MemoryUsage reports immutable record payload currently retained across the
// generation chain. It intentionally excludes allocator page slack, which is
// not exposed by the Arena API, but never returns the misleading zero value.
func (idx *Core) MemoryUsage() int64 {
	snapshot := idx.Snapshot()
	if snapshot == nil {
		return 0
	}
	defer snapshot.Release()
	var bytes uint64
	for generation := snapshot; generation != nil; generation = generation.Parent() {
		for i := 0; i < generation.SegmentLen(); i++ {
			bytes += generation.SegmentAt(i).Footprint()
		}
	}
	if bytes > uint64(^uint64(0)>>1) {
		return int64(^uint64(0) >> 1)
	}
	return int64(bytes)
}

func (idx *Core) Config() *Config { return idx.config }

// VisitVisible holds one snapshot lease while visiting each logical live
// record exactly once. It is used by persistence and diagnostics, never by
// request ranking.
func (idx *Core) VisitVisible(visit func(record.RecordRef) error) error {
	snapshot := idx.Snapshot()
	if snapshot == nil {
		return nil
	}
	defer snapshot.Release()
	for generation := snapshot; generation != nil; generation = generation.Parent() {
		for i := 0; i < generation.SegmentLen(); i++ {
			ref := generation.SegmentAt(i)
			if !snapshot.Visible(ref) {
				continue
			}
			if err := visit(ref); err != nil {
				return err
			}
		}
	}
	return nil
}

func coreDistance(metric util.DistanceMetric, left, right record.VectorView) (float32, error) {
	if left.Len() != right.Len() {
		return 0, util.ErrDimension
	}
	switch metric {
	case util.CosineDistance:
		return util.CosineDistance_func(left.Float32s(), right.Float32s()), nil
	case util.L2Distance:
		return util.L2Distance_func(left.Float32s(), right.Float32s()), nil
	case util.InnerProduct:
		return util.InnerProduct_func(left.Float32s(), right.Float32s()), nil
	default:
		return 0, fmt.Errorf("unsupported distance metric %d", metric)
	}
}

func coreUpHeap(values []coreHeapElement, i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if values[parent].score >= values[i].score {
			return
		}
		values[parent], values[i] = values[i], values[parent]
		i = parent
	}
}

func coreDownHeap(values []coreHeapElement, i, n int) {
	for {
		largest := i
		left := i*2 + 1
		right := left + 1
		if left < n && values[left].score > values[largest].score {
			largest = left
		}
		if right < n && values[right].score > values[largest].score {
			largest = right
		}
		if largest == i {
			return
		}
		values[i], values[largest] = values[largest], values[i]
		i = largest
	}
}

// SearchBorrowed executes a fully off-heap Flat scan. The only Go objects are
// fixed-size control handles; top-k state and returned rows are arena-backed.
func (idx *Core) SearchBorrowed(ctx context.Context, query record.VectorView, k int, filter interface{ Test(uint64) bool }) (*ResultSet, error) {
	if query.Len() != idx.config.Dimension {
		return nil, util.ErrDimension
	}
	if k <= 0 {
		k = 0
	}
	if k > 4096 {
		return nil, fmt.Errorf("k %d exceeds maximum allowed search result limit of 4096", k)
	}
	snapshot := idx.Snapshot()
	if k == 0 {
		return &ResultSet{snapshot: snapshot}, nil
	}
	arenaBytes := uint64(k) * uint64(unsafe.Sizeof(coreHeapElement{})+unsafe.Sizeof(coreResult{}))
	if arenaBytes < 4096 {
		arenaBytes = 4096
	}
	arena, err := memory.NewArena(arenaBytes, 64)
	if err != nil {
		snapshot.Release()
		return nil, err
	}
	heap, err := memory.ArenaSlice[coreHeapElement](arena, k)
	if err != nil {
		_ = arena.Free()
		snapshot.Release()
		return nil, err
	}
	count := 0
	for generation := snapshot; generation != nil; generation = generation.Parent() {
		for i := 0; i < generation.SegmentLen(); i++ {
			if err := ctx.Err(); err != nil {
				_ = arena.Free()
				snapshot.Release()
				return nil, err
			}
			ref := generation.SegmentAt(i)
			if !snapshot.Visible(ref) || (filter != nil && !filter.Test(uint64(ref.Ordinal()))) {
				continue
			}
			distance, err := coreDistance(idx.config.Metric, query, ref.Vector())
			if err != nil {
				_ = arena.Free()
				snapshot.Release()
				return nil, err
			}
			if count < k {
				heap = heap[:count+1]
				heap[count] = coreHeapElement{ref: ref, score: distance}
				coreUpHeap(heap, count)
				count++
			} else if distance < heap[0].score {
				heap[0] = coreHeapElement{ref: ref, score: distance}
				coreDownHeap(heap, 0, count)
			}
		}
	}
	results, err := memory.ArenaSlice[coreResult](arena, count)
	if err != nil {
		_ = arena.Free()
		snapshot.Release()
		return nil, err
	}
	results = results[:count]
	for i := count - 1; i >= 0; i-- {
		root := heap[0]
		count--
		heap[0] = heap[count]
		coreDownHeap(heap, 0, count)
		results[i] = coreResult{ref: root.ref, score: root.score}
	}
	return &ResultSet{arena: arena, snapshot: snapshot, results: results}, nil
}

func (set *ResultSet) Len() int { return len(set.results) }

func (set *ResultSet) At(i int) (record.RecordRef, float32) {
	if i < 0 || i >= len(set.results) {
		panic("flat: result index out of range")
	}
	return set.results[i].ref, set.results[i].score
}

func (set *ResultSet) Close() error {
	if set == nil {
		return nil
	}
	if set.snapshot != nil {
		set.snapshot.Release()
		set.snapshot = nil
	}
	set.results = nil
	if set.arena != nil {
		err := set.arena.Free()
		set.arena = nil
		return err
	}
	return nil
}

func (idx *Core) Close() error {
	idx.mu.Lock()
	current := idx.current
	idx.current = nil
	idx.mu.Unlock()
	if current != nil {
		current.Release()
	}
	return nil
}
