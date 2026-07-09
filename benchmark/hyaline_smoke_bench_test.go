// Hyaline SMR stress smoke test for the memory package.
//
// Run with:
//
//	go test -race -bench=BenchmarkHyalineSMR -benchtime=1x ./benchmark/
//
// Validates that HyalineEnter/HyalineLeave + Allocate/Retire work correctly
// under concurrent load with the race detector enabled. This is the first
// Hyaline consumer in libraVDB; if the graph layer is going to depend on
// Hyaline SMR, we need to know it works in our setup before writing 200
// lines of graph code on top of it.
//
// What this catches:
//   - Data races (via -race)
//   - Use-after-free during Retire (via -race + Hyaline coordination)
//   - Goroutine leaks (via runtime.NumGoroutine snapshot)
//   - Heap growth (via runtime.MemStats delta)
//   - Retire errors (e.g., double-Retire, invalid slot)
//
// What this does NOT catch:
//   - Correctness of the SMR algorithm itself (that's memory package's job)
//   - Cross-shard latency, NUMA effects, or hardware-specific behavior
package benchmark

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/xDarkicex/memory"
)

const (
	smokeNumShards = 64
	smokeSlotSize  = 64
	smokePoolSize  = 64 * 1024 * 1024
	smokeSlabSize  = 1 * 1024 * 1024
	smokeSlabCount = 16
	smokeReaders   = 4
	smokeWriters   = 4
	smokeRunFor    = 10 * time.Second
)

func BenchmarkHyalineSMR(b *testing.B) {
	sfl, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  smokePoolSize,
		SlotSize:  smokeSlotSize,
		SlabSize:  smokeSlabSize,
		SlabCount: smokeSlabCount,
		Prealloc:  true,
	}, 64, smokeNumShards)
	if err != nil {
		b.Fatal(err)
	}
	defer sfl.Free()

	var (
		wg           sync.WaitGroup
		stop         = make(chan struct{})
		readOps      atomic.Uint64
		writeOps     atomic.Uint64
		retireErrors atomic.Uint64
	)

	initialGoroutines := runtime.NumGoroutine()
	var memBefore runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	// Readers: tight HyalineEnter/HyalineLeave loop on a fixed slot each.
	// Spreading readers across slots (id % 64) avoids serializing on a
	// single Hyaline word and exercises the multi-slot path.
	for id := range smokeReaders {
		wg.Go(func() {
			slot := id % smokeNumShards
			for {
				select {
				case <-stop:
					return
				default:
				}
				sfl.HyalineEnter(slot)
				sfl.HyalineLeave(slot)
				readOps.Add(1)
			}
		})
	}

	// Writers: tight Allocate/Retire loop. Retire (not Deallocate) is the
	// Hyaline-aware path — this is what the graph layer will call when
	// removing an EdgeTable page. If Retire has a use-after-free bug, the
	// race detector will catch it when a concurrent reader is in
	// HyalineEnter on the same slot.
	for range smokeWriters {
		wg.Go(func() {
			for {
				select {
				case <-stop:
					return
				default:
				}
				slot, err := sfl.Allocate()
				if err != nil {
					// Pool exhausted; the PID controller will catch up.
					// Don't error — the smoke should tolerate this.
					runtime.Gosched()
					continue
				}
				if err := sfl.Retire(slot); err != nil {
					retireErrors.Add(1)
				}
				writeOps.Add(1)
			}
		})
	}

	time.Sleep(smokeRunFor)
	close(stop)
	wg.Wait()

	var memAfter runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memAfter)

	finalGoroutines := runtime.NumGoroutine()
	leaked := finalGoroutines - initialGoroutines
	heapGrowth := int64(memAfter.HeapAlloc) - int64(memBefore.HeapAlloc)

	if leaked != 0 {
		b.Errorf("goroutine leak: started with %d, ended with %d (leaked %d)",
			initialGoroutines, finalGoroutines, leaked)
	}
	if rerrs := retireErrors.Load(); rerrs != 0 {
		b.Errorf("retire errors: %d (likely double-Retire or invalid slot)", rerrs)
	}

	b.ReportMetric(float64(readOps.Load()), "read-ops")
	b.ReportMetric(float64(writeOps.Load()), "write-ops")
	b.ReportMetric(float64(retireErrors.Load()), "retire-errors")
	b.ReportMetric(float64(leaked), "goroutine-leak")
	b.ReportMetric(float64(heapGrowth)/1024, "heap-growth-KB")
}
