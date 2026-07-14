package hnsw

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/util"
)

func newReclamationTestIndex(t testing.TB, rawStore string) *Index {
	t.Helper()
	index, err := NewHNSW(&Config{
		Dimension:      16,
		M:              8,
		EfConstruction: 32,
		EfSearch:       16,
		ML:             1.0,
		Metric:         util.L2Distance,
		RawVectorStore: rawStore,
		RawStoreCap:    256,
		RandomSeed:     42,
	})
	if err != nil {
		t.Fatal(err)
	}
	return index
}

func TestReclamationWaitsForPreexistingReader(t *testing.T) {
	for _, rawStore := range []string{RawVectorStoreMemory, RawVectorStoreSlabby} {
		t.Run(rawStore, func(t *testing.T) {
			index := newReclamationTestIndex(t, rawStore)
			defer index.Close()

			vector := make([]float32, 16)
			for i := range vector {
				vector[i] = float32(i + 1)
			}
			if err := index.Insert(context.Background(), &VectorEntry{ID: "held", Vector: vector}); err != nil {
				t.Fatal(err)
			}

			reader := index.acquireSearchScratch()
			node := index.nodes.Get(0)
			if node == nil || len(node.Vector) != len(vector) {
				t.Fatal("reader failed to capture published node vector")
			}
			staleVector := node.Vector

			if err := index.Delete(context.Background(), "held"); err != nil {
				index.releaseSearchScratch(reader)
				t.Fatal(err)
			}
			if index.reclamation.enqueuePos.Load() == index.reclamation.dequeuePos.Load() {
				index.releaseSearchScratch(reader)
				t.Fatal("retired storage reclaimed while a preexisting reader was active")
			}
			for i := range vector {
				if staleVector[i] != vector[i] {
					index.releaseSearchScratch(reader)
					t.Fatalf("retired vector changed at dimension %d: got %v want %v", i, staleVector[i], vector[i])
				}
			}

			index.releaseSearchScratch(reader)
			index.reclamation.tryReclaim(index)
			if index.reclamation.enqueuePos.Load() != index.reclamation.dequeuePos.Load() {
				t.Fatal("retired storage did not drain after the preexisting reader left")
			}
		})
	}
}

func TestReclamationReusesPublishedSlotsUnderChurn(t *testing.T) {
	for _, rawStore := range []string{RawVectorStoreMemory, RawVectorStoreSlabby} {
		t.Run(rawStore, func(t *testing.T) {
			index := newReclamationTestIndex(t, rawStore)
			defer index.Close()
			vector := make([]float32, 16)

			for i := 0; i < 2048; i++ {
				vector[0] = float32(i)
				if err := index.Insert(context.Background(), &VectorEntry{ID: "churn", Vector: vector}); err != nil {
					t.Fatalf("insert %d: %v", i, err)
				}
				if err := index.Delete(context.Background(), "churn"); err != nil {
					t.Fatalf("delete %d: %v", i, err)
				}
			}

			if pending := index.reclamation.enqueuePos.Load() - index.reclamation.dequeuePos.Load(); pending != 0 {
				t.Fatalf("retired queue did not drain under churn: %d pending", pending)
			}
			if slabs := index.nodeSFL.Stats().SlabCount; slabs > 1 {
				t.Fatalf("node allocator grew despite reclamation: %d slabs", slabs)
			}
			switch store := index.rawVectorStore.(type) {
			case *InMemoryRawVectorStore:
				if next := store.nextSlot.Load(); next > 1 {
					t.Fatalf("logical vector slots grew despite reclamation: %d", next)
				}
			case *SlabbyRawVectorStore:
				if next := store.nextSlot.Load(); next > 1 {
					t.Fatalf("logical vector slots grew despite reclamation: %d", next)
				}
				if slabs := store.sfl.Stats().SlabCount; slabs > 1 {
					t.Fatalf("raw vector allocator grew despite reclamation: %d slabs", slabs)
				}
			}
		})
	}
}

func TestReclamationConcurrentSearchDeleteReinsert(t *testing.T) {
	index := newReclamationTestIndex(t, RawVectorStoreSlabby)
	defer index.Close()

	vectors := generateTestVectors(128, 16)
	for i, vector := range vectors {
		if err := index.Insert(context.Background(), &VectorEntry{
			ID:     fmt.Sprintf("stable-%d", i),
			Vector: vector,
		}); err != nil {
			t.Fatalf("seed insert %d: %v", i, err)
		}
	}
	if err := index.Insert(context.Background(), &VectorEntry{ID: "moving", Vector: vectors[0]}); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 4)
	var workers sync.WaitGroup
	for worker := 0; worker < 4; worker++ {
		workers.Add(1)
		go func(offset int) {
			defer workers.Done()
			for i := 0; ; i++ {
				select {
				case <-ctx.Done():
					return
				default:
				}
				if _, err := index.Search(ctx, vectors[(i+offset)%len(vectors)], 10, nil); err != nil && ctx.Err() == nil {
					select {
					case errCh <- err:
					default:
					}
					return
				}
			}
		}(worker)
	}

	var mutationErr error
	for i := 0; i < 256; i++ {
		if err := index.Delete(context.Background(), "moving"); err != nil {
			mutationErr = fmt.Errorf("delete %d: %w", i, err)
			break
		}
		if err := index.Insert(context.Background(), &VectorEntry{ID: "moving", Vector: vectors[i%len(vectors)]}); err != nil {
			mutationErr = fmt.Errorf("reinsert %d: %w", i, err)
			break
		}
	}
	cancel()
	workers.Wait()
	if mutationErr != nil {
		t.Fatal(mutationErr)
	}
	select {
	case err := <-errCh:
		t.Fatal(err)
	default:
	}
}

type reclamationTestProvider struct{}

func (reclamationTestProvider) GetByOrdinal(uint32) ([]float32, error) {
	return nil, fmt.Errorf("provider lookup is not expected in this test")
}

func (reclamationTestProvider) Distance([]float32, uint32) (float32, error) {
	return 0, fmt.Errorf("provider distance is not expected in this test")
}

func TestProviderModeReleasesOwnedRawVectorSlots(t *testing.T) {
	for _, rawStore := range []string{RawVectorStoreMemory, RawVectorStoreSlabby} {
		t.Run(rawStore, func(t *testing.T) {
			index, err := NewHNSW(&Config{
				Dimension:      16,
				M:              8,
				EfConstruction: 32,
				EfSearch:       16,
				ML:             1.0,
				Metric:         util.L2Distance,
				Provider:       reclamationTestProvider{},
				RawVectorStore: rawStore,
				RawStoreCap:    16,
				RandomSeed:     42,
			})
			if err != nil {
				t.Fatal(err)
			}
			defer index.Close()

			vector := make([]float32, 16)
			if err := index.Insert(context.Background(), &VectorEntry{ID: "owned", Ordinal: 0, Vector: vector}); err != nil {
				t.Fatal(err)
			}
			if active := index.rawVectorStore.Profile().VectorCount; active != 1 {
				t.Fatalf("active raw vectors after insert = %d, want 1", active)
			}

			if err := index.Insert(context.Background(), &VectorEntry{ID: "owned", Ordinal: 1, Vector: vector}); err == nil {
				t.Fatal("duplicate provider-backed insert succeeded")
			}
			if active := index.rawVectorStore.Profile().VectorCount; active != 1 {
				t.Fatalf("active raw vectors after rollback = %d, want 1", active)
			}

			if err := index.Delete(context.Background(), "owned"); err != nil {
				t.Fatal(err)
			}
			if active := index.rawVectorStore.Profile().VectorCount; active != 0 {
				t.Fatalf("active raw vectors after delete = %d, want 0", active)
			}
		})
	}
}

func TestReclamationMetadataIsCacheLineIsolated(t *testing.T) {
	if size := unsafe.Sizeof(reclamationReaderSlot{}); size != 64 {
		t.Fatalf("reader slot size = %d, want 64", size)
	}
	if size := unsafe.Sizeof(retiredAllocationSlot{}); size != 64 {
		t.Fatalf("retired slot size = %d, want 64", size)
	}
	domain, err := newReclamationDomain(0)
	if err != nil {
		t.Fatal(err)
	}
	defer domain.close()
	if uintptr(unsafe.Pointer(&domain.readers[0]))&63 != 0 {
		t.Fatal("reader epoch table is not 64-byte aligned")
	}
	if uintptr(unsafe.Pointer(&domain.retired[0]))&63 != 0 {
		t.Fatal("retired queue is not 64-byte aligned")
	}
}

type benchmarkEpochSlot struct {
	epoch atomic.Uint64
	_     [56]byte
}

type benchmarkEpochDomain struct {
	global atomic.Uint64
	slots  [64]benchmarkEpochSlot
}

func (d *benchmarkEpochDomain) enter(slot int) {
	for {
		epoch := d.global.Load()
		d.slots[slot].epoch.Store(epoch)
		if d.global.Load() == epoch {
			return
		}
	}
}

func (d *benchmarkEpochDomain) leave(slot int) {
	d.slots[slot].epoch.Store(0)
}

func BenchmarkHNSWSharedReadEpoch(b *testing.B) {
	b.Run("serial", func(b *testing.B) {
		var domain benchmarkEpochDomain
		domain.global.Store(1)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			slot := i & 63
			domain.enter(slot)
			domain.leave(slot)
		}
	})

	b.Run("parallel", func(b *testing.B) {
		var domain benchmarkEpochDomain
		var nextSlot atomic.Uint64
		domain.global.Store(1)

		b.ReportAllocs()
		b.RunParallel(func(pb *testing.PB) {
			slot := int(nextSlot.Add(1)-1) & 63
			for pb.Next() {
				domain.enter(slot)
				domain.leave(slot)
			}
		})
	})
}

func BenchmarkHNSWScratchEpochModes(b *testing.B) {
	for _, enabled := range []bool{false, true} {
		name := "without_epoch"
		if enabled {
			name = "with_epoch"
		}
		b.Run(name, func(b *testing.B) {
			index := newReclamationTestIndex(b, RawVectorStoreMemory)
			domain := index.reclamation
			if !enabled {
				index.reclamation = nil
			}
			defer func() {
				index.reclamation = domain
				_ = index.Close()
			}()

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				scratch := index.acquireSearchScratchWithNodeCountAndEF(5000, 200)
				index.releaseSearchScratch(scratch)
			}
		})
	}
}

func BenchmarkHNSWRawVectorStoreBuild(b *testing.B) {
	const (
		n   = 5000
		dim = 768
	)
	vectors := benchmarkNormalizedVectorsDim(n, dim, 42)
	for _, rawStore := range []string{RawVectorStoreMemory, RawVectorStoreSlabby} {
		b.Run(rawStore, func(b *testing.B) {
			var totalInserts uint64
			var reservedBytes int64
			var liveBytes int64
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				b.StopTimer()
				config := benchmarkNormalizedHNSWConfigDim(dim)
				config.M = 16
				config.EfConstruction = 200
				config.EfSearch = 200
				config.RawVectorStore = rawStore
				config.RawStoreCap = n
				index, err := NewHNSW(&config)
				if err != nil {
					b.Fatal(err)
				}
				b.StartTimer()
				for _, vector := range vectors {
					if err := index.Insert(context.Background(), &VectorEntry{Vector: vector}); err != nil {
						b.Fatal(err)
					}
				}
				totalInserts += n
				b.StopTimer()
				profile := index.rawVectorStore.Profile()
				reservedBytes = profile.ReservedBytes
				liveBytes = profile.LiveBytes
				_ = index.Close()
			}
			if elapsed := b.Elapsed(); elapsed > 0 {
				b.ReportMetric(float64(totalInserts)/elapsed.Seconds(), "ingestion_insert/s")
			}
			b.ReportMetric(n, "nodes/build")
			b.ReportMetric(float64(liveBytes), "raw_live_bytes")
			b.ReportMetric(float64(reservedBytes), "raw_reserved_bytes")
		})
	}
}

func BenchmarkHNSWHyalineReadEpoch(b *testing.B) {
	for _, rawStore := range []string{RawVectorStoreMemory, RawVectorStoreSlabby} {
		b.Run(rawStore, func(b *testing.B) {
			index, err := NewHNSW(&Config{
				Dimension:      768,
				M:              16,
				EfConstruction: 200,
				EfSearch:       200,
				ML:             1.0,
				Metric:         util.L2Distance,
				RawVectorStore: rawStore,
				RawStoreCap:    5000,
			})
			if err != nil {
				b.Fatal(err)
			}
			defer index.Close()

			var vectorStore *SlabbyRawVectorStore
			if store, ok := index.rawVectorStore.(*SlabbyRawVectorStore); ok {
				vectorStore = store
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				slot := i & 63
				index.nodeSFL.HyalineEnter(slot)
				index.linkSFL.HyalineEnter(slot)
				index.link0SFL.HyalineEnter(slot)
				if vectorStore != nil {
					vectorStore.sfl.HyalineEnter(slot)
				}

				if vectorStore != nil {
					vectorStore.sfl.HyalineLeave(slot)
				}
				index.link0SFL.HyalineLeave(slot)
				index.linkSFL.HyalineLeave(slot)
				index.nodeSFL.HyalineLeave(slot)
			}
		})
	}
}
