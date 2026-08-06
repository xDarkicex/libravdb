package libravdb

import (
	"context"
	"fmt"
	"testing"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// TestOrdinalBitset_Basic verifies set / test / clear semantics.
func TestOrdinalBitset_Basic(t *testing.T) {
	bm := &bitsetMembership{bits: make([]uint64, 4)}

	bm.set(0)
	bm.set(63)
	bm.set(64)
	bm.set(200)

	if !bm.test(0) {
		t.Error("bit 0 not set")
	}
	if !bm.test(63) {
		t.Error("bit 63 not set")
	}
	if !bm.test(64) {
		t.Error("bit 64 not set")
	}
	if !bm.test(200) {
		t.Error("bit 200 not set")
	}
	if bm.test(1) {
		t.Error("bit 1 should not be set")
	}
	if bm.len() != 4 {
		t.Errorf("size = %d, want 4", bm.len())
	}

	bm.clear()
	if bm.test(0) {
		t.Error("bit 0 should be clear after reset")
	}
	if bm.len() != 0 {
		t.Errorf("size after clear = %d, want 0", bm.len())
	}
}

// TestOrdinalBitset_Pool verifies acquire / release round-trip.
func TestOrdinalBitset_Pool(t *testing.T) {
	s1 := acquireOrdinalBitset(100)
	s1[0] = 0xDEADBEEF
	releaseOrdinalBitset(s1)

	// Re-acquire: must get a zeroed slice.
	s2 := acquireOrdinalBitset(100)
	for i, w := range s2 {
		if w != 0 {
			t.Fatalf("word %d = %x, want 0 (pool didn't zero)", i, w)
		}
	}
	releaseOrdinalBitset(s2)
}

// TestOrdinalBitset_PoolGrow verifies pool correctly handles growth.
func TestOrdinalBitset_PoolGrow(t *testing.T) {
	small := acquireOrdinalBitset(50)
	releaseOrdinalBitset(small)

	large := acquireOrdinalBitset(100000)
	if cap(large)*ordinalBitsetWordBits < 100000 {
		t.Fatalf("bitset too small: cap=%d words, need >= %d", cap(large), 100000/ordinalBitsetWordBits+1)
	}
	releaseOrdinalBitset(large)
}

// TestOrdinalBitset_Concurrent verifies pool reuse under concurrency.
func TestOrdinalBitset_Concurrent(t *testing.T) {
	done := make(chan struct{})
	for range 10 {
		go func() {
			for i := 0; i < 1000; i++ {
				s := acquireOrdinalBitset(1000)
				s[0] = 1
				s[1] = 1
				releaseOrdinalBitset(s)
			}
			done <- struct{}{}
		}()
	}
	for range 10 {
		<-done
	}
}

// BenchmarkOrdinalBitsetMembershipBuild measures only dense membership writes.
// It deliberately excludes storage ID resolution; BenchmarkOrdinalBitmapFromIDs
// below is the truthful end-to-end query-path measurement.
func BenchmarkOrdinalBitsetMembershipBuild(b *testing.B) {
	const N = 10000
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		bits := acquireOrdinalBitset(uint32(N * 2))
		bm := &bitsetMembership{bits: bits}
		for ord := uint32(0); ord < uint32(N); ord++ {
			bm.set(ord)
		}
		bm.release()
	}
}

func BenchmarkOrdinalBitmapFromIDs(b *testing.B) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:ordinal_bitmap_bench"), WithMetrics(false))
	if err != nil {
		b.Fatal(err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "bench", WithDimension(2), WithFlat())
	if err != nil {
		b.Fatal(err)
	}
	candidates := make(map[string]struct{}, 256)
	for i := 0; i < 256; i++ {
		id := fmt.Sprintf("candidate-%03d", i)
		if err := col.Insert(ctx, id, []float32{float32(i), 1}, nil); err != nil {
			b.Fatal(err)
		}
		candidates[id] = struct{}{}
	}
	exec := newExecutor(db)
	bitmap, err := exec.buildOrdinalBitmapFromIDs(ctx, col, candidates)
	if err != nil {
		b.Fatal(err)
	}
	bitmap.release() // pool warm-up is outside the measured steady state.

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bitmap, err := exec.buildOrdinalBitmapFromIDs(ctx, col, candidates)
		if err != nil {
			b.Fatal(err)
		}
		bitmap.release()
	}
}

// BenchmarkOrdinalBitset_Test measures lookup throughput.
func BenchmarkOrdinalBitset_Test(b *testing.B) {
	const N = 100000
	bits := acquireOrdinalBitset(uint32(N))
	bm := &bitsetMembership{bits: bits}
	for i := uint32(0); i < uint32(N); i += 2 {
		bm.set(i)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_ = bm.test(uint32(b.N % N))
	}

	bm.release()
}

// TestBuildOrdinalBitmapFromIDs_UsesDirectOrdinalLookup verifies candidate IDs
// become ordinals without requiring an authoritative record enumeration.
func TestBuildOrdinalBitmapFromIDs_UsesDirectOrdinalLookup(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:test_ordinal_bitset_no_listall.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "bits", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Insert records so we have real ordinals in the B-tree.
	for i := range 50 {
		id := fmt.Sprintf("rec-%d", i)
		if err := col.Insert(context.Background(), id, []float32{float32(i), 0, 0}, nil); err != nil {
			t.Fatalf("Insert %s: %v", id, err)
		}
	}

	candidates := map[string]struct{}{
		"rec-0":  {},
		"rec-10": {},
		"rec-25": {},
		"rec-49": {},
	}

	exec := newExecutor(db)
	bitmap, err := exec.buildOrdinalBitmapFromIDs(context.Background(), col, candidates)
	if err != nil {
		t.Fatalf("buildOrdinalBitmapFromIDs: %v", err)
	}
	defer bitmap.release()

	// Each candidate must be present in the bitmap.
	if bm, ok := bitmap.membership.(*bitsetMembership); ok {
		if bm.len() != 4 {
			t.Errorf("bitset size = %d, want 4", bm.len())
		}
	} else {
		t.Error("expected bitsetMembership, got map-backed")
	}
}

func TestBuildOrdinalBitmapFromIDs_ShardsUseLocalOrdinals(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:test_ordinal_bitset_sharded"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "bits", WithDimension(2), WithFlat(), WithSharding(true))
	if err != nil {
		t.Fatal(err)
	}

	idsByShard := make([][]string, shardCount)
	for n := 0; n < 256; n++ {
		id := fmt.Sprintf("sharded-%03d", n)
		shard := shardForID(id)
		if len(idsByShard[shard]) >= 2 {
			continue
		}
		if err := col.Insert(ctx, id, []float32{float32(n), 1}, nil); err != nil {
			t.Fatal(err)
		}
		idsByShard[shard] = append(idsByShard[shard], id)
	}
	for shard, ids := range idsByShard {
		if len(ids) != 2 {
			t.Fatalf("shard %d received %d fixtures, want 2", shard, len(ids))
		}
	}

	candidates := make(map[string]struct{})
	for _, ids := range idsByShard {
		for _, id := range ids {
			candidates[id] = struct{}{}
		}
	}
	bitmap, err := newExecutor(db).buildOrdinalBitmapFromIDs(ctx, col, candidates)
	if err != nil {
		t.Fatal(err)
	}
	defer bitmap.release()
	for shard, ids := range idsByShard {
		local, ok := bitmap.ForShard(shard).(*ordinalBitmap)
		if !ok {
			t.Fatalf("shard %d filter type %T", shard, bitmap.ForShard(shard))
		}
		if got := local.membership.len(); got != len(ids) {
			t.Fatalf("shard %d membership=%d, want %d", shard, got, len(ids))
		}
		for _, id := range ids {
			record, err := col.Get(ctx, id)
			if err != nil {
				t.Fatal(err)
			}
			if !local.Test(uint64(record.Ordinal)) {
				t.Fatalf("shard %d omitted %q ordinal %d", shard, id, record.Ordinal)
			}
		}
	}
}

func TestBuildOrdinalBitmapFromIDs_PoolDoesNotLeakCandidates(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:test_ordinal_bitset_reuse"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "bits", WithDimension(2), WithFlat())
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"first", "second"} {
		if err := col.Insert(ctx, id, []float32{1, 0}, nil); err != nil {
			t.Fatal(err)
		}
	}
	exec := newExecutor(db)
	first, err := exec.buildOrdinalBitmapFromIDs(ctx, col, map[string]struct{}{"first": {}})
	if err != nil {
		t.Fatal(err)
	}
	first.release()
	second, err := exec.buildOrdinalBitmapFromIDs(ctx, col, map[string]struct{}{"second": {}})
	if err != nil {
		t.Fatal(err)
	}
	defer second.release()
	firstRecord, _ := col.Get(ctx, "first")
	secondRecord, _ := col.Get(ctx, "second")
	if second.Test(uint64(firstRecord.Ordinal)) {
		t.Fatal("reused bitmap retained a prior candidate")
	}
	if !second.Test(uint64(secondRecord.Ordinal)) {
		t.Fatal("reused bitmap omitted its current candidate")
	}
}

func TestMultiModalExactCandidatesAreShardSafe(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:test_multimodal_exact_sharded"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "docs", WithDimension(2), WithMetric(CosineDistance), WithFlat(), WithSharding(true))
	if err != nil {
		t.Fatal(err)
	}
	ids := []string{"exact-near", "exact-far"}
	for i, id := range ids {
		vector := []float32{0, 1}
		if i == 0 {
			vector = []float32{1, 0}
		}
		if err := col.Insert(ctx, id, vector, nil); err != nil {
			t.Fatal(err)
		}
	}
	result, err := newExecutor(db).executeMultiModalExact(ctx, col, &optimizer.PhysicalPlan{QueryVector: []float32{1, 0}, Limit: 1}, map[string]struct{}{
		"exact-near": {}, "exact-far": {},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 1 || result.Results[0].ID != "exact-near" {
		t.Fatalf("exact shard result=%v, want exact-near", result.Results)
	}
}

// TestOrdinalBitmap_ShardIsolation verifies that ForShard correctly isolates
// ordinal membership between shards.
func TestOrdinalBitmap_ShardIsolation(t *testing.T) {
	// Create per-shard memberships where ordinal 5 is in shard 0 but not shard 1.
	b0 := &bitsetMembership{bits: acquireOrdinalBitset(100)}
	b0.set(5)
	b1 := &bitsetMembership{bits: acquireOrdinalBitset(100)}
	b1.set(10)

	bitmap := &ordinalBitmap{
		membership:   b0,
		byMembership: []ordinalMembership{b0, b1},
		selectivity:  0.5,
	}

	// Shard 0: ordinal 5 passes, ordinal 10 fails.
	s0 := bitmap.ForShard(0)
	if !s0.Test(5) {
		t.Error("shard 0: ordinal 5 should pass")
	}
	if s0.Test(10) {
		t.Error("shard 0: ordinal 10 should NOT pass (belongs to shard 1)")
	}

	// Shard 1: ordinal 10 passes, ordinal 5 fails.
	s1 := bitmap.ForShard(1)
	if !s1.Test(10) {
		t.Error("shard 1: ordinal 10 should pass")
	}
	if s1.Test(5) {
		t.Error("shard 1: ordinal 5 should NOT pass (belongs to shard 0)")
	}

	b0.release()
	b1.release()
}
