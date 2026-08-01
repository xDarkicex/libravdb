package ivfpq

import (
	"context"
	"errors"
	"hash/crc32"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

// TestHydrationRejectsConcurrentInsert proves a successful live write cannot
// disappear behind a replacement staged from an older generation.
func TestHydrationRejectsConcurrentInsert(t *testing.T) {
	const dim = 8
	idx := trainedPQ(t, dim, 2)
	defer idx.Close()

	data, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	before := idx.gen.id
	ctx := &blockCtx{Context: context.Background(), after: 1, blocked: make(chan struct{}), unblock: make(chan struct{})}
	errCh := make(chan error, 1)
	go func() { errCh <- idx.DeserializeFromBytes(ctx, data) }()

	// The hydrate captured the generation and is now staging. Insert succeeds
	// against that live generation before the hydrate is allowed to commit.
	<-ctx.blocked
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(i+1) / 10
	}
	const ordinal uint32 = 900001
	if err := idx.Insert(context.Background(), &VectorEntry{ID: "concurrent", Ordinal: ordinal, Vector: v}); err != nil {
		t.Fatalf("concurrent Insert: %v", err)
	}
	close(ctx.unblock)
	if err := <-errCh; !errors.Is(err, ErrHydrationConflict) {
		t.Fatalf("DeserializeFromBytes error = %v, want ErrHydrationConflict", err)
	}
	if idx.gen.id != before {
		t.Fatalf("stale hydrate replaced live generation: %d -> %d", before, idx.gen.id)
	}
	if idx.Size() != 1 {
		t.Fatalf("Size = %d, want concurrent write retained", idx.Size())
	}
	if err := idx.DeleteByOrdinal(context.Background(), ordinal); err != nil {
		t.Fatalf("concurrent write was lost: %v", err)
	}
}

// TestHydrationExhaustionPreservesLive verifies that staging pool exhaustion
// leaves the live generation unchanged.
func TestHydrationExhaustionPreservesLive(t *testing.T) {
	const dim = 8
	// Small budget: 1 MiB. A large hydration payload will exhaust it.
	const budget uint64 = 1 * 1024 * 1024
	cfg := &Config{
		Dimension:       dim,
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
		t.Fatal(err)
	}
	defer idx.Close()
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 10.0
		}
		if err := idx.Insert(context.Background(), &VectorEntry{ID: "pre", Ordinal: uint32(i + 1), Vector: v}); err != nil {
			t.Fatal(err)
		}
	}
	beforeSize := idx.Size()
	beforeGenID := idx.gen.id
	poolBefore := idx.gen.pool.Stats()

	// Build a v3 payload that demands many more entries than the pool can hold.
	huge := buildHugeV3(t, dim, 2, 200000)
	err = idx.DeserializeFromBytes(context.Background(), huge)
	if err == nil {
		t.Fatal("expected pool exhaustion error")
	}
	t.Logf("exhaustion error: %v", err)

	// Live generation must be unchanged.
	if idx.Size() != beforeSize {
		t.Fatalf("size changed: %d -> %d", beforeSize, idx.Size())
	}
	if idx.gen.id != beforeGenID {
		t.Fatalf("generation pointer changed: %d -> %d", beforeGenID, idx.gen.id)
	}
	poolAfter := idx.gen.pool.Stats()
	if poolAfter.Reserved > budget {
		t.Fatalf("pool reserved %d > budget %d", poolAfter.Reserved, budget)
	}
	if poolAfter.Reserved != poolBefore.Reserved {
		t.Fatal("pool accounting changed during failed hydration")
	}
}

func buildHugeV3(t *testing.T, dim, nClusters, entriesPer int) []byte {
	t.Helper()
	buf := make([]byte, 0, 64*1024)
	w := &sliceWriter{buf: buf}
	w.bytes(ivfpqMagicBytes)
	w.u16(ivfpqFormatVersion)
	w.u8(ivfpqIndexType)
	w.u8(0)
	w.u32(uint32(dim))
	w.u32(uint32(nClusters))
	w.u32(uint32(nClusters))
	w.u8(uint8(util.L2Distance))
	// PQ with Codebooks=2, Bits=4 → CodeSize=1 byte.
	cfg := &quant.QuantizationConfig{Type: quant.ProductQuantization, Codebooks: 2, Bits: 4, TrainRatio: 0.5, CacheSize: 100}
	q := quant.NewProductQuantizer()
	q.Configure(cfg)
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := q.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	state, err := q.SerializeState()
	if err != nil {
		t.Fatal(err)
	}
	q.Close()
	w.u8(qTagPQ)
	w.u32(uint32(len(state)))
	w.raw(state)
	w.u32(20)
	w.f64(1e-4)
	w.i64(0)
	w.u32(uint32(nClusters))
	for i := 0; i < nClusters; i++ {
		w.u32(uint32(i))
		w.u32(uint32(dim))
		for d := 0; d < dim; d++ {
			w.f32(float32(i + d))
		}
	}
	w.u32(uint32(nClusters))
	cs := 1
	for ci := 0; ci < nClusters; ci++ {
		w.u32(uint32(ci))
		w.u32(uint32(entriesPer))
		for e := 0; e < entriesPer; e++ {
			w.u32(uint32(ci*entriesPer + e + 1))
			w.u32(uint32(cs))
			for b := 0; b < cs; b++ {
				w.u8(byte(e & 0x0F))
			}
		}
	}
	ck := crc32.Checksum(w.buf, crc32.MakeTable(crc32.Castagnoli))
	w.u32(ck)
	return w.buf
}

// TestHydrationCancellationDuringStage verifies ctx cancellation mid-stage
// leaves the live generation unchanged. Uses a custom context that cancels
// after a counted number of Err() calls, guaranteeing cancellation triggers
// inside staging (not before parsing).
func TestHydrationCancellationDuringStage(t *testing.T) {
	const dim = 8
	cfg := &Config{
		Dimension:     dim,
		NClusters:     2,
		NProbes:       2,
		Metric:        util.L2Distance,
		Quantization:  &quant.QuantizationConfig{Type: quant.ProductQuantization, Codebooks: 2, Bits: 4, TrainRatio: 1},
		MaxIterations: 20,
		Tolerance:     1e-4,
		RandomSeed:    7,
	}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	beforeSize := idx.Size()
	beforeGenID := idx.gen.id
	beforeGen := idx.gen
	beforeGen.acquire()
	defer beforeGen.release()

	// cancelAfter lets N Err() calls pass, then returns Canceled.
	ctx := &cancelAfterCtx{Context: context.Background(), after: 2}
	huge := buildHugeV3(t, dim, 2, 500)
	err = idx.DeserializeFromBytes(ctx, huge)
	if err == nil {
		t.Fatal("expected cancellation error")
	}
	t.Logf("cancellation error: %v", err)
	if idx.Size() != beforeSize {
		t.Fatalf("size changed: %d -> %d", beforeSize, idx.Size())
	}
	if idx.gen.id != beforeGenID {
		t.Fatalf("generation changed: %d -> %d", beforeGenID, idx.gen.id)
	}
}

// cancelAfterCtx returns context.Canceled after `after` calls to Err().
type cancelAfterCtx struct {
	context.Context
	after  int32
	called atomic.Int32
	done   atomic.Bool
}

func (c *cancelAfterCtx) Err() error {
	n := c.called.Add(1)
	if n > int32(c.after) {
		c.done.Store(true)
		return context.Canceled
	}
	return nil
}
func (c *cancelAfterCtx) Done() <-chan struct{} {
	if c.done.Load() {
		ch := make(chan struct{})
		close(ch)
		return ch
	}
	return nil
}
func (c *cancelAfterCtx) Deadline() (time.Time, bool) { return time.Time{}, false }

// TestRetiredGenerationLifetime verifies old gen not freed while pinned.
func TestRetiredGenerationLifetime(t *testing.T) {
	const dim = 8
	cfg := &Config{
		Dimension:     dim,
		NClusters:     2,
		NProbes:       2,
		Metric:        util.L2Distance,
		Quantization:  &quant.QuantizationConfig{Type: quant.ProductQuantization, Codebooks: 2, Bits: 4, TrainRatio: 1},
		MaxIterations: 20,
		Tolerance:     1e-4,
		RandomSeed:    7,
	}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	old := idx.gen
	old.acquire() // pin like a Search would

	// Hydrate: this swaps to a new generation and retires old.
	data, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.DeserializeFromBytes(context.Background(), data); err != nil {
		t.Fatal(err)
	}
	if idx.gen.id == old.id {
		t.Fatal("generation not swapped")
	}
	if !old.retired.Load() {
		t.Fatal("old generation not retired")
	}
	if old.freed.Load() {
		t.Fatal("old generation freed while pinned")
	}

	// Drop pin — old should be freed now.
	old.release()
	if !old.freed.Load() {
		t.Fatal("old generation not freed after pin release")
	}
}

// TestRepeatedHydrateDoesNotLeak verifies repeated hydrations retire properly.
func TestRepeatedHydrateDoesNotLeak(t *testing.T) {
	const dim = 8
	cfg := &Config{
		Dimension:     dim,
		NClusters:     2,
		NProbes:       2,
		Metric:        util.L2Distance,
		Quantization:  &quant.QuantizationConfig{Type: quant.ProductQuantization, Codebooks: 2, Bits: 4, TrainRatio: 1},
		MaxIterations: 20,
		Tolerance:     1e-4,
		RandomSeed:    7,
	}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}

	var oldGens []*generation
	for i := 0; i < 5; i++ {
		data, err := idx.SerializeToBytes()
		if err != nil {
			t.Fatal(err)
		}
		prev := idx.gen
		prev.acquire()
		if err := idx.DeserializeFromBytes(context.Background(), data); err != nil {
			t.Fatal(err)
		}
		oldGens = append(oldGens, prev)
		if prev.retired.Load() {
			// release pin → should eventually free
			prev.release()
		}
	}
	// All previous generations must be freed.
	for _, og := range oldGens {
		if !og.freed.Load() {
			t.Fatal("retired generation not freed after repeated hydrate")
		}
	}
}

// TestCloseRacingHydration verifies Close during staging doesn't resurrect.
// Uses a blocking custom context to pause hydration after staging has begun
// but before the pointer-swap commit. While paused, Close is called from
// another goroutine. Then hydration resumes, finds idx.gen==nil, and returns
// a closed error without republishing.
func TestCloseRacingHydration(t *testing.T) {
	const dim = 8
	cfg := &Config{
		Dimension:     dim,
		NClusters:     2,
		NProbes:       2,
		Metric:        util.L2Distance,
		Quantization:  &quant.QuantizationConfig{Type: quant.ProductQuantization, Codebooks: 2, Bits: 4, TrainRatio: 1},
		MaxIterations: 20,
		Tolerance:     1e-4,
		RandomSeed:    7,
	}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	huge := buildHugeV3(t, dim, 2, 500)

	// Blocking context: lets N Err() calls pass, then blocks until unblock
	// is closed. This pauses staging deterministically after N calls.
	blockCtx := &blockCtx{Context: context.Background(), after: 1, blocked: make(chan struct{}), unblock: make(chan struct{})}

	var hydErr error
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		hydErr = idx.DeserializeFromBytes(blockCtx, huge)
	}()

	// Wait for hydration to enter staging and block.
	<-blockCtx.blocked

	// Close while hydration is blocked mid-staging.
	idx.Close()

	// Unblock hydration — it will see gen==nil at commit.
	close(blockCtx.unblock)

	wg.Wait()

	if hydErr == nil {
		t.Fatal("hydration must error when racing Close")
	}
	t.Logf("hydration error: %v", hydErr)
	if idx.gen != nil {
		t.Fatal("hydration resurrected closed index")
	}
}

// blockCtx blocks Err() after `after` calls. Used to pause staging deterministically.
type blockCtx struct {
	context.Context
	after   int32
	called  atomic.Int32
	blocked chan struct{}
	unblock chan struct{}
}

func (c *blockCtx) Err() error {
	n := c.called.Add(1)
	if n == int32(c.after)+1 {
		close(c.blocked)
		<-c.unblock
	}
	return nil
}
func (c *blockCtx) Done() <-chan struct{}       { return nil }
func (c *blockCtx) Deadline() (time.Time, bool) { return time.Time{}, false }
