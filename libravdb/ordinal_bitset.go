package libravdb

import (
	"context"
	"sync"
)

const ordinalBitsetWordBits = 64

// ordinalBitsetPool reuses []uint64 backing arrays across queries. Each slot is
// sized for the largest collection seen; smaller collections reuse a sub-slice.
var ordinalBitsetPool = sync.Pool{New: func() any { return make([]uint64, 0, 1024) }}
var bitsetMembershipPool = sync.Pool{New: func() any { return &bitsetMembership{} }}

type ordinalBitsetBuffer struct{ words []uint64 }

var ordinalBitsetBufferPool = sync.Pool{New: func() any { return &ordinalBitsetBuffer{words: make([]uint64, 0, 1024)} }}

// acquireOrdinalBitset returns a zeroed []uint64 sized for at least maxOrdinal
// ordinals. The caller owns the slice until ReleaseOrdinalBitset is called.
func acquireOrdinalBitset(maxOrdinal uint32) []uint64 {
	words := int(maxOrdinal)/ordinalBitsetWordBits + 1
	s := ordinalBitsetPool.Get().([]uint64)
	if cap(s) < words {
		s = make([]uint64, words)
	} else {
		s = s[:words]
	}
	for i := range s {
		s[i] = 0
	}
	return s
}

// releaseOrdinalBitset returns a bitset slice to the pool. The caller must
// not read or write the slice after this call.
func releaseOrdinalBitset(s []uint64) {
	ordinalBitsetPool.Put(s[:0])
}

func acquireBitsetMembership(maxOrdinal uint32) *bitsetMembership {
	bm := bitsetMembershipPool.Get().(*bitsetMembership)
	buffer := ordinalBitsetBufferPool.Get().(*ordinalBitsetBuffer)
	words := int(maxOrdinal)/ordinalBitsetWordBits + 1
	if cap(buffer.words) < words {
		buffer.words = make([]uint64, words)
	} else {
		buffer.words = buffer.words[:words]
	}
	for i := range buffer.words {
		buffer.words[i] = 0
	}
	bm.bits = buffer.words
	bm.buffer = buffer
	bm.size = 0
	return bm
}

func releaseBitsetMembership(bm *bitsetMembership) {
	if bm == nil {
		return
	}
	bm.release()
	bitsetMembershipPool.Put(bm)
}

// ordinalMembership abstracts the ordinal allow-list backing store so
// callers can swap between map and dense-bitset without changing the
// GraphFilter interface.
type ordinalMembership interface {
	test(ordinal uint32) bool
	set(ordinal uint32)
	len() int
	clear()
}

// mapMembership is the existing map[uint32]bool path, kept for simple
// hybrid queries where candidate count is unknown.
type mapMembership struct {
	m map[uint32]bool
}

func (mm *mapMembership) test(ordinal uint32) bool { return mm.m[ordinal] }
func (mm *mapMembership) set(ordinal uint32)       { mm.m[ordinal] = true }
func (mm *mapMembership) len() int                 { return len(mm.m) }
func (mm *mapMembership) clear() {
	for k := range mm.m {
		delete(mm.m, k)
	}
}

// bitsetMembership is the zero-alloc dense-bitset path. The backing []uint64
// is pool-allocated and must be released via releaseOrdinalBitset after the
// query completes.
type bitsetMembership struct {
	bits   []uint64
	buffer *ordinalBitsetBuffer
	size   int
}

func (bm *bitsetMembership) test(ordinal uint32) bool {
	word := ordinal / ordinalBitsetWordBits
	bit := ordinal % ordinalBitsetWordBits
	if int(word) >= len(bm.bits) {
		return false
	}
	return bm.bits[word]&(1<<bit) != 0
}

func (bm *bitsetMembership) set(ordinal uint32) {
	word := ordinal / ordinalBitsetWordBits
	bit := ordinal % ordinalBitsetWordBits
	if int(word) >= len(bm.bits) {
		return
	}
	mask := uint64(1) << bit
	if bm.bits[word]&mask == 0 {
		bm.bits[word] |= mask
		bm.size++
	}
}

func (bm *bitsetMembership) len() int { return bm.size }

func (bm *bitsetMembership) clear() {
	for i := range bm.bits {
		bm.bits[i] = 0
	}
	bm.size = 0
}

func (bm *bitsetMembership) release() {
	if bm.bits != nil {
		if bm.buffer != nil {
			bm.buffer.words = bm.bits[:0]
			ordinalBitsetBufferPool.Put(bm.buffer)
			bm.buffer = nil
		} else {
			releaseOrdinalBitset(bm.bits)
		}
		bm.bits = nil
	}
}

// emptyMembership is used as the global membership for a sharded bitmap.
// Shard searches always call ForShard, and a global local-ordinal bitmap
// would be semantically invalid because ordinal values overlap between shards.
type emptyMembership struct{}

func (emptyMembership) test(uint32) bool { return false }
func (emptyMembership) set(uint32)       {}
func (emptyMembership) len() int         { return 0 }
func (emptyMembership) clear()           {}

// buildOrdinalBitmapFromIDs converts a pre-computed set of record IDs to an
// ordinalBitmap without scanning the full collection. Each ID is resolved to
// its ordinal via Collection.Get (O(log N) B-tree lookup per candidate).
// The resulting bitmap uses a pool-allocated dense bitset.
//
// The caller must call bitmap.release() after the query completes to return
// the backing array to the pool.
func (e *Executor) buildOrdinalBitmapFromIDs(ctx context.Context, col *Collection, recordIDs map[string]struct{}) (*ordinalBitmap, error) {
	if len(recordIDs) == 0 {
		return &ordinalBitmap{membership: &mapMembership{m: map[uint32]bool{}}, selectivity: 0}, nil
	}

	matched := 0
	var membership ordinalMembership
	if col.shards != nil {
		byShard := make([]ordinalMembership, len(col.shards))
		byShardBits := make([]*bitsetMembership, len(col.shards))
		for i := range col.shards {
			nextOrdinal, err := col.shards[i].storage.NextOrdinal(ctx)
			if err != nil {
				for _, allocated := range byShardBits {
					if allocated != nil {
						releaseBitsetMembership(allocated)
					}
				}
				return nil, err
			}
			sbm := acquireBitsetMembership(nextOrdinal)
			byShard[i] = sbm
			byShardBits[i] = sbm
		}
		for id := range recordIDs {
			ordinal, err := col.getOrdinal(ctx, id)
			if err != nil {
				continue
			}
			shard := shardForID(id)
			if shard >= 0 && shard < len(byShard) {
				byShard[shard].set(ordinal)
				matched++
			}
		}
		membership = emptyMembership{}
		return acquirePooledOrdinalBitmap(col, membership, byShard, matched, nil, byShardBits), nil
	}

	nextOrdinal, err := col.storage.NextOrdinal(ctx)
	if err != nil {
		return nil, err
	}
	bm := acquireBitsetMembership(nextOrdinal)
	for id := range recordIDs {
		ordinal, err := col.getOrdinal(ctx, id)
		if err != nil {
			continue
		}
		bm.set(ordinal)
		matched++
	}
	membership = bm
	return acquirePooledOrdinalBitmap(col, membership, nil, matched, bm, nil), nil
}

func acquirePooledOrdinalBitmap(col *Collection, membership ordinalMembership, byShard []ordinalMembership, matched int, ownedMembership *bitsetMembership, ownedShards []*bitsetMembership) *ordinalBitmap {
	corpusSize := col.countRecords()
	selectivity := 1.0
	if corpusSize > 0 {
		selectivity = float64(matched) / float64(corpusSize)
	}

	bitmap := ordinalBitmapPool.Get().(*ordinalBitmap)
	bitmap.membership = membership
	bitmap.byMembership = byShard
	bitmap.selectivity = selectivity
	bitmap.ownedMembership = ownedMembership
	bitmap.ownedShards = ownedShards
	bitmap.pooled = true
	return bitmap
}

// decodeBTreeValue unpacks the B-tree encoded value into ordinal, version, and
// graph node ID. Mirrors btree.DecodeValue.
func decodeBTreeValue(val []byte) (ordinal uint32, version uint32, graphNodeID uint64) {
	if len(val) < 4 {
		return 0, 0, 0
	}
	ordinal = uint32(val[0]) | uint32(val[1])<<8 | uint32(val[2])<<16 | uint32(val[3])<<24
	if len(val) >= 8 {
		version = uint32(val[4]) | uint32(val[5])<<8 | uint32(val[6])<<16 | uint32(val[7])<<24
	}
	if len(val) >= 16 {
		graphNodeID = uint64(val[8]) | uint64(val[9])<<8 | uint64(val[10])<<16 | uint64(val[11])<<24 |
			uint64(val[12])<<32 | uint64(val[13])<<40 | uint64(val[14])<<48 | uint64(val[15])<<56
	}
	return
}

// countRecords returns the approximate record count for selectivity estimation.
// Used when the exact count from ListAll is unavailable (bitset path).
func (c *Collection) countRecords() int {
	if c.shards != nil {
		total := 0
		for i := range c.shards {
			n, err := c.shards[i].storage.Count(context.Background())
			if err == nil {
				total += n
			}
		}
		return total
	}
	if c.storage != nil {
		n, _ := c.storage.Count(context.Background())
		return n
	}
	return 0
}
