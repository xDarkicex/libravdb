package ivfpq

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/xDarkicex/memory"
	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

// trainedPQ builds a small IVF-PQ index with trained ProductQuantizer.
func trainedPQ(t *testing.T, dim, nClusters int) *Index {
	t.Helper()
	cfg := &Config{
		Dimension:     dim,
		NClusters:     nClusters,
		NProbes:       nClusters,
		Metric:        util.L2Distance,
		Quantization:  &quant.QuantizationConfig{Type: quant.ProductQuantization, Codebooks: 2, Bits: 4, TrainRatio: 1},
		MaxIterations: 20,
		Tolerance:     1e-4,
		RandomSeed:    7,
	}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatalf("NewIVFPQ: %v", err)
	}
	train := make([][]float32, 256)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatalf("Train: %v", err)
	}
	return idx
}

// TestSegmentBoundaryAppendAndSearch fills a cluster past the segment
// capacity and verifies later segments are searchable.
func TestSegmentBoundaryAppendAndSearch(t *testing.T) {
	const dim = 8
	idx := trainedPQ(t, dim, 2)
	defer idx.Close()

	const total = 1500
	for i := 0; i < total; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / float32(total+dim)
		}
		entry := &VectorEntry{ID: fmt.Sprintf("sb-%d", i), Ordinal: uint32(i + 1), Vector: v}
		if err := idx.Insert(context.Background(), entry); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	if idx.Size() != total {
		t.Fatalf("Size = %d, want %d", idx.Size(), total)
	}

	sawMulti := false
	for i, c := range idx.gen.clusters {
		c.mutex.RLock()
		n := len(c.storage.segments)
		c.mutex.RUnlock()
		if n > 1 {
			sawMulti = true
			t.Logf("cluster %d: %d segments", i, n)
		}
	}
	if !sawMulti {
		t.Fatal("expected multi-segment cluster")
	}

	q := make([]float32, dim)
	for j := range q {
		q[j] = 0.5
	}
	results, err := idx.Search(context.Background(), q, total, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	found := make(map[uint32]bool, len(results))
	for _, r := range results {
		found[r.Ordinal] = true
	}
	for i := 1; i <= total; i++ {
		if !found[uint32(i)] {
			t.Fatalf("ordinal %d not found", i)
		}
	}
}

// TestCrossSegmentDeleteByOrdinal forces a cross-segment swap-with-last.
func TestCrossSegmentDeleteByOrdinal(t *testing.T) {
	const dim = 8
	idx := trainedPQ(t, dim, 2)
	defer idx.Close()

	const total = 2000
	for i := 0; i < total; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / float32(total+dim)
		}
		entry := &VectorEntry{ID: fmt.Sprintf("xd-%d", i), Ordinal: uint32(i + 1), Vector: v}
		if err := idx.Insert(context.Background(), entry); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}

	var targetOrdinal uint32
	targetClusterID := -1
	for ci, c := range idx.gen.clusters {
		c.mutex.RLock()
		if len(c.storage.segments) > 1 && c.storage.count > 1 {
			seg := c.storage.segments[0]
			if seg.used > 0 {
				targetOrdinal = seg.ordinals[0]
				targetClusterID = ci
			}
		}
		c.mutex.RUnlock()
		if targetOrdinal != 0 {
			break
		}
	}
	if targetOrdinal == 0 {
		t.Skip("no multi-segment cluster found")
	}
	t.Logf("deleting ordinal %d from cluster %d", targetOrdinal, targetClusterID)

	if err := idx.DeleteByOrdinal(context.Background(), targetOrdinal); err != nil {
		t.Fatalf("DeleteByOrdinal: %v", err)
	}
	if idx.Size() != total-1 {
		t.Fatalf("Size = %d, want %d", idx.Size(), total-1)
	}

	q := make([]float32, dim)
	for j := range q {
		q[j] = 0.5
	}
	results, err := idx.Search(context.Background(), q, total-1, nil)
	if err != nil {
		t.Fatalf("Search after delete: %v", err)
	}
	seen := make(map[uint32]bool, len(results))
	for _, r := range results {
		seen[r.Ordinal] = true
	}
	if seen[targetOrdinal] {
		t.Fatalf("deleted ordinal %d still surfaced", targetOrdinal)
	}
	for i := 1; i <= total; i++ {
		if uint32(i) != targetOrdinal && !seen[uint32(i)] {
			t.Fatalf("non-deleted ordinal %d lost", i)
		}
	}
}

// TestClusterStorageRejectsMismatchedCodeWidth verifies fail-closed append.
func TestClusterStorageRejectsMismatchedCodeWidth(t *testing.T) {
	cs := &clusterStorage{segmentCapacity: 4, codeWidth: 8}
	pool, err := newRecordPoolForTest(64 * 1024)
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	defer pool.Free()

	if err := cs.append(42, make([]byte, 4), pool); err == nil {
		t.Fatal("expected error for short code")
	}
	if cs.count != 0 || len(cs.segments) != 0 {
		t.Fatal("state mutated on rejected append")
	}

	cs2 := &clusterStorage{segmentCapacity: 4, codeWidth: 0}
	if err := cs2.append(7, []byte{0x00}, pool); err == nil {
		t.Fatal("expected error for non-empty code on zero-width storage")
	}

	if err := cs.append(99, []byte{1, 2, 3, 4, 5, 6, 7, 8}, pool); err != nil {
		t.Fatalf("correct append: %v", err)
	}
	if cs.count != 1 {
		t.Fatalf("count = %d, want 1", cs.count)
	}
}

func newRecordPoolForTest(bytes uint64) (*memory.Pool, error) {
	slab := uint64(64 * 1024)
	if bytes < slab {
		slab = bytes
	}
	return memory.NewPool(memory.AllocatorConfig{
		PoolSize: bytes, SlabSize: slab, SlabCount: 4,
	}, 64)
}

// TestForcedGCReleasesCallerInput verifies off-heap retention survives GC.
func TestForcedGCReleasesCallerInput(t *testing.T) {
	const dim = 8
	const n = 500
	idx := trainedPQ(t, dim, 2)
	defer idx.Close()

	keep := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / float32(n+dim)
		}
		keep[i] = v
		entry := &VectorEntry{ID: fmt.Sprintf("gc-%d", i), Ordinal: uint32(i + 1), Vector: v}
		if err := idx.Insert(context.Background(), entry); err != nil {
			t.Fatalf("Insert: %v", err)
		}
	}
	for i := range keep {
		keep[i] = nil
	}
	keep = nil
	for i := 0; i < 4; i++ {
		runtime.GC()
		time.Sleep(5 * time.Millisecond)
	}
	if idx.Size() != n {
		t.Fatalf("Size = %d, want %d", idx.Size(), n)
	}
}

// TestRepeatedDeserializeIsIdempotent verifies rehydrate doesn't duplicate.
func TestRepeatedDeserializeIsIdempotent(t *testing.T) {
	const dim = 8
	idx1 := trainedPQ(t, dim, 2)
	defer idx1.Close()

	const n = 60
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / float32(n+dim)
		}
		entry := &VectorEntry{ID: fmt.Sprintf("h-%d", i), Ordinal: uint32(i + 1), Vector: v}
		if err := idx1.Insert(context.Background(), entry); err != nil {
			t.Fatalf("Insert: %v", err)
		}
	}
	data, err := idx1.SerializeToBytes()
	if err != nil {
		t.Fatalf("SerializeToBytes: %v", err)
	}

	cfg := idx1.config
	idx2, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatalf("NewIVFPQ: %v", err)
	}
	defer idx2.Close()
	if err := idx2.DeserializeFromBytes(context.Background(), data); err != nil {
		t.Fatalf("first hydrate: %v", err)
	}
	if idx2.Size() != n {
		t.Fatalf("first hydrate size = %d, want %d", idx2.Size(), n)
	}
	if err := idx2.DeserializeFromBytes(context.Background(), data); err != nil {
		t.Fatalf("second hydrate: %v", err)
	}
	if idx2.Size() != n {
		t.Fatalf("second hydrate size = %d, want %d (duplication)", idx2.Size(), n)
	}
	if err := idx2.DeserializeFromBytes(context.Background(), data); err != nil {
		t.Fatalf("third hydrate: %v", err)
	}
	if idx2.Size() != n {
		t.Fatalf("third hydrate size = %d, want %d", idx2.Size(), n)
	}
}

// TestRecordPoolCeiling verifies the budget bounds the pool.
func TestRecordPoolCeiling(t *testing.T) {
	const budget = 8 * 1024 * 1024
	cfg := &Config{
		Dimension:       8,
		NClusters:       2,
		NProbes:         2,
		Metric:          util.L2Distance,
		Quantization:    &quant.QuantizationConfig{Type: quant.ProductQuantization, Codebooks: 2, Bits: 4, TrainRatio: 1},
		MaxIterations:   20,
		Tolerance:       1e-4,
		RandomSeed:      7,
		RecordPoolBytes: budget,
	}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatalf("NewIVFPQ: %v", err)
	}
	defer idx.Close()
	train := make([][]float32, 128)
	for i := range train {
		v := make([]float32, 8)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatalf("Train: %v", err)
	}
	for i := 0; i < 200; i++ {
		v := make([]float32, 8)
		for j := range v {
			v[j] = float32(i+j) / 200.0
		}
		entry := &VectorEntry{ID: fmt.Sprintf("p-%d", i), Ordinal: uint32(i + 1), Vector: v}
		if err := idx.Insert(context.Background(), entry); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	stats := idx.gen.pool.Stats()
	if stats.Reserved > budget {
		t.Fatalf("reserved %d > budget %d", stats.Reserved, budget)
	}
}

// TestRecordPoolOwnership verifies Close frees the pool.
func TestRecordPoolOwnership(t *testing.T) {
	idx := trainedPQ(t, 8, 2)
	pool := idx.gen.pool
	if pool == nil {
		t.Fatal("no record pool")
	}
	if err := idx.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if idx.gen != nil {
		t.Fatal("generation not nilled after Close")
	}
	if _, err := pool.Allocate(64); err == nil {
		t.Fatal("pool not freed")
	}
}

// TestConcurrentInsertSearch exercises Insert/Search concurrency.
func TestConcurrentInsertSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("concurrent test")
	}
	const dim = 8
	idx := trainedPQ(t, dim, 2)
	defer idx.Close()

	const writers = 4
	const perWriter = 250
	var writerWG sync.WaitGroup
	var searchWG sync.WaitGroup

	stop := make(chan struct{})
	var searchErr error
	var searchMu sync.Mutex

	writerWG.Add(writers)
	searchWG.Add(1)
	for w := 0; w < writers; w++ {
		base := w * perWriter
		go func() {
			defer writerWG.Done()
			for i := 0; i < perWriter; i++ {
				v := make([]float32, dim)
				for j := range v {
					v[j] = float32(base+i+j) / float32(writers*perWriter+dim)
				}
				entry := &VectorEntry{ID: fmt.Sprintf("cw-%d-%d", w, i), Ordinal: uint32(base + i + 1), Vector: v}
				if err := idx.Insert(context.Background(), entry); err != nil {
					t.Errorf("Insert: %v", err)
					return
				}
			}
		}()
	}
	go func() {
		defer searchWG.Done()
		q := make([]float32, dim)
		for {
			select {
			case <-stop:
				return
			default:
			}
			if _, err := idx.Search(context.Background(), q, 10, nil); err != nil {
				searchMu.Lock()
				searchErr = err
				searchMu.Unlock()
				return
			}
		}
	}()

	writerWG.Wait()
	close(stop)
	searchWG.Wait()

	if searchErr != nil {
		t.Fatalf("search error: %v", searchErr)
	}
	if idx.Size() != writers*perWriter {
		t.Fatalf("Size = %d, want %d", idx.Size(), writers*perWriter)
	}
}

// TestPostCloseSafety verifies all public methods handle a closed index safely.
func TestPostCloseSafety(t *testing.T) {
	const dim = 8
	idx := trainedPQ(t, dim, 2)
	if err := idx.Close(); err != nil {
		t.Fatal(err)
	}
	// All public methods must return cleanly without panic.
	if idx.Size() != 0 {
		t.Error("Size after Close should be 0")
	}
	if idx.IsTrained() {
		t.Error("IsTrained after Close should be false")
	}
	cfg := idx.GetConfig()
	if cfg == nil {
		t.Error("GetConfig after Close should return non-nil config")
	}
	info := idx.GetClusterInfo()
	if info != nil {
		t.Error("GetClusterInfo after Close should return nil")
	}
	meta := idx.GetPersistenceMetadata()
	if meta != nil {
		t.Error("GetPersistenceMetadata after Close should return nil")
	}
	if mem := idx.MemoryUsage(); mem != 0 {
		t.Error("MemoryUsage after Close should be 0")
	}
	// Search/Insert/Delete must error without panic.
	v := make([]float32, dim)
	if _, err := idx.Search(context.Background(), v, 5, nil); err == nil {
		t.Error("Search after Close should error")
	}
	if err := idx.Insert(context.Background(), &VectorEntry{ID: "x", Vector: v}); err == nil {
		t.Error("Insert after Close should error")
	}
	if err := idx.DeleteByOrdinal(context.Background(), 1); err == nil {
		t.Error("DeleteByOrdinal after Close should error")
	}
	entries := []*VectorEntry{{ID: "x", Vector: v}}
	if err := idx.BatchInsert(context.Background(), entries); err == nil {
		t.Error("BatchInsert after Close should error")
	}
	data, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	if data != nil {
		t.Error("SerializeToBytes after Close should return nil")
	}
	// Double close must be safe.
	if err := idx.Close(); err != nil {
		t.Error("second Close should not error")
	}
}

// TestConcurrentCloseSearch exercises Search + Close under race.
func TestConcurrentCloseSearch(t *testing.T) {
	const dim = 8
	idx := trainedPQ(t, dim, 2)
	q := make([]float32, dim)
	for i := 0; i < 10; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 10.0
		}
		if err := idx.Insert(context.Background(), &VectorEntry{ID: fmt.Sprintf("cs-%d", i), Ordinal: uint32(i + 1), Vector: v}); err != nil {
			t.Fatal(err)
		}
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			idx.Search(context.Background(), q, 5, nil)
		}
	}()
	go func() {
		defer wg.Done()
		idx.Close()
	}()
	wg.Wait()
}
