package record

import (
	"math"
	"unsafe"
)

// DeltaCapacityMutation describes the allocations made for one possible
// first-touch mutation while a delta is staged. The estimator intentionally
// accepts upper-bound plans: callers that cannot know whether a delete will
// find a live record can mark it as an unknown ordinal and reserve its full
// tombstone/path cost.
type DeltaCapacityMutation struct {
	IDBytes       uint64
	VectorBytes   uint64
	MetadataBytes uint64
	Ordinal       uint32
	OrdinalKnown  bool
	ProducesAfter bool
	Tombstone     bool
}

// DeltaCapacityReport is the derived off-heap capacity ledger for one delta
// and its unpublished generation. ArenaBytes is the minimum 64-byte-aligned
// arena capacity for the modeled allocation sequence (or a conservative
// upper bound when the mutation plan contains unknown outcomes). KeyBytes is
// the per-IDMap copied-key arena capacity; the delta and candidate generation
// each receive a separate arena of this size during construction.
type DeltaCapacityReport struct {
	ArenaBytes                 uint64
	KeyBytes                   uint64
	PeakArenaBytes             uint64
	PeakCopiedKeyBytes         uint64
	PeakTrackedOffHeapBytes    uint64
	MutationPointerBytes       uint64
	StagingBytes               uint64
	GenerationRootBytes        uint64
	GenerationSegmentBytes     uint64
	GenerationOrdinalPageBytes uint64
	GenerationOrdinalPages     uint64
}

const (
	deltaArenaAlignment = uint64(64)
	idKeyArenaAlignment = uint64(8)
	ordinalPathLevels   = ordinalLevels - 1
)

func alignCapacity(value, alignment uint64) uint64 {
	if value == 0 {
		return 0
	}
	if value > math.MaxUint64-(alignment-1) {
		return math.MaxUint64
	}
	return (value + alignment - 1) &^ (alignment - 1)
}

func addCapacity(total *uint64, values ...uint64) {
	for _, value := range values {
		if value > math.MaxUint64-*total {
			*total = math.MaxUint64
			return
		}
		*total += value
	}
}

func mulCapacity(left, right uint64) uint64 {
	if left != 0 && right > math.MaxUint64/left {
		return math.MaxUint64
	}
	return left * right
}

// addArenaAllocation applies the same alignment rule as memory.Arena.Alloc:
// the allocation start is aligned, but the allocation size itself is not
// rounded. This makes the final cursor an exact minimum for a fully-known
// sequence, while still allowing callers to add conservative allocations.
func addArenaAllocation(cursor *uint64, size uint64) {
	if *cursor == math.MaxUint64 {
		return
	}
	aligned := alignCapacity(*cursor, deltaArenaAlignment)
	if aligned == math.MaxUint64 || size > math.MaxUint64-aligned {
		*cursor = math.MaxUint64
		return
	}
	*cursor = aligned + size
}

func addKeyAllocation(cursor *uint64, size uint64) {
	if size == 0 || *cursor == math.MaxUint64 {
		return
	}
	aligned := alignCapacity(*cursor, idKeyArenaAlignment)
	if aligned == math.MaxUint64 || size > math.MaxUint64-aligned {
		*cursor = math.MaxUint64
		return
	}
	*cursor = aligned + size
}

func addPrefix(prefixes map[uint64]struct{}, level uint64, ordinal uint32) {
	key := (level << 32) | uint64(ordinal>>uint(level*8))
	prefixes[key] = struct{}{}
}

// EstimateDeltaCapacity derives the arena and copied-key capacities required
// by record.NewDelta followed by record.NewGeneration. The operation order
// mirrors the implementation:
//
//  1. delta mutation-pointer slice;
//  2. staged ID copy, mutationCell, and sealed record/tombstone per plan;
//  3. generation root, segment-pointer slice, and copy-on-write ordinal
//     radix pages.
//
// The caller supplies maxMutations separately because the delta pointer slice
// and generation segment are allocated at the configured maximum while the
// actual mutation log may deduplicate IDs. Known put ordinals are counted by
// distinct radix prefixes. Unknown ordinals reserve all three path levels,
// which is a proved upper bound for the four-level, 8-bit radix tree.
func EstimateDeltaCapacity(maxMutations uint64, mutations []DeltaCapacityMutation) DeltaCapacityReport {
	report := DeltaCapacityReport{}

	mutationPointerBytes := mulCapacity(maxMutations, uint64(unsafe.Sizeof((*mutationCell)(nil))))
	report.MutationPointerBytes = mutationPointerBytes

	var arena uint64
	addArenaAllocation(&arena, mutationPointerBytes)

	var keyBytes uint64
	prefixes := make(map[uint64]struct{}, len(mutations)*ordinalPathLevels)
	unknownPages := uint64(0)
	for _, mutation := range mutations {
		addKeyAllocation(&keyBytes, mutation.IDBytes)
		addArenaAllocation(&arena, mutation.IDBytes)
		addArenaAllocation(&arena, uint64(unsafe.Sizeof(mutationCell{})))

		if mutation.Tombstone {
			addArenaAllocation(&arena, addRecordBytes(mutation.IDBytes, 0, 0))
		} else {
			addArenaAllocation(&arena, addRecordBytes(mutation.IDBytes, mutation.VectorBytes, mutation.MetadataBytes))
		}

		if !mutation.ProducesAfter {
			continue
		}
		if mutation.OrdinalKnown {
			for level := uint64(1); level <= ordinalPathLevels; level++ {
				addPrefix(prefixes, level, mutation.Ordinal)
			}
		} else {
			addCapacity(&unknownPages, ordinalPathLevels)
		}
	}

	report.StagingBytes = arena
	addArenaAllocation(&arena, uint64(unsafe.Sizeof(ordinalNode{})))
	report.GenerationRootBytes = arena - report.StagingBytes
	segmentBytes := mulCapacity(maxMutations, uint64(unsafe.Sizeof(unsafe.Pointer(nil))))
	report.GenerationSegmentBytes = segmentBytes
	addArenaAllocation(&arena, segmentBytes)

	report.GenerationOrdinalPages = uint64(len(prefixes)) + unknownPages
	pageBytes := mulCapacity(report.GenerationOrdinalPages, uint64(unsafe.Sizeof(ordinalNode{})))
	report.GenerationOrdinalPageBytes = pageBytes
	for i := uint64(0); i < report.GenerationOrdinalPages; i++ {
		addArenaAllocation(&arena, uint64(unsafe.Sizeof(ordinalNode{})))
	}

	report.ArenaBytes = arena
	report.PeakArenaBytes = arena
	report.KeyBytes = maxUint64(keyBytes, 1)
	report.PeakCopiedKeyBytes = report.KeyBytes * 2
	if report.PeakCopiedKeyBytes/2 != report.KeyBytes {
		report.PeakCopiedKeyBytes = math.MaxUint64
	}
	addCapacity(&report.PeakTrackedOffHeapBytes, report.PeakArenaBytes, report.PeakCopiedKeyBytes)
	return report
}

func addRecordBytes(idBytes, vectorBytes, metadataBytes uint64) uint64 {
	total := uint64(unsafe.Sizeof(recordHeader{}))
	addCapacity(&total, idBytes, vectorBytes, metadataBytes)
	return total
}

func maxUint64(left, right uint64) uint64 {
	if left > right {
		return left
	}
	return right
}
