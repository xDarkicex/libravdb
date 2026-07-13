package hnsw

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"math/bits"
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/libravdb/internal/util/simd"
)

const (
	semanticFixtureHeaderBytes = 64
	semanticFixtureDim         = 768
	semanticFixtureK           = 10
)

var semanticFixtureMagic = [8]byte{'L', 'V', 'S', 'E', 'M', '0', '0', '1'}

type semanticFixture struct {
	data     []byte
	vectors  [][]float32
	queries  [][]float32
	dim      int
	modelTag uint64
}

func loadSemanticFixture(tb testing.TB, path string) semanticFixture {
	tb.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		tb.Fatalf("read semantic fixture: %v", err)
	}
	if len(data) < semanticFixtureHeaderBytes {
		tb.Fatalf("semantic fixture is %d bytes; header requires %d", len(data), semanticFixtureHeaderBytes)
	}
	if string(data[:len(semanticFixtureMagic)]) != string(semanticFixtureMagic[:]) {
		tb.Fatalf("semantic fixture has invalid magic %q", data[:len(semanticFixtureMagic)])
	}

	dim := int(binary.LittleEndian.Uint32(data[8:12]))
	vectorCount := int(binary.LittleEndian.Uint32(data[12:16]))
	queryCount := int(binary.LittleEndian.Uint32(data[16:20]))
	modelTag := binary.LittleEndian.Uint64(data[24:32])
	if dim <= 0 || vectorCount <= 0 || queryCount <= 0 {
		tb.Fatalf("semantic fixture has invalid shape vectors=%d queries=%d dim=%d", vectorCount, queryCount, dim)
	}

	floatCount := (vectorCount + queryCount) * dim
	wantBytes := semanticFixtureHeaderBytes + floatCount*4
	if floatCount < 0 || wantBytes != len(data) {
		tb.Fatalf("semantic fixture size=%d, want=%d for vectors=%d queries=%d dim=%d", len(data), wantBytes, vectorCount, queryCount, dim)
	}

	flat := unsafe.Slice((*float32)(unsafe.Pointer(&data[semanticFixtureHeaderBytes])), floatCount)
	vectors := make([][]float32, vectorCount)
	for i := range vectors {
		start := i * dim
		vectors[i] = flat[start : start+dim : start+dim]
	}
	queries := make([][]float32, queryCount)
	queryBase := vectorCount * dim
	for i := range queries {
		start := queryBase + i*dim
		queries[i] = flat[start : start+dim : start+dim]
	}

	return semanticFixture{
		data:     data,
		vectors:  vectors,
		queries:  queries,
		dim:      dim,
		modelTag: modelTag,
	}
}

func semanticExactTruth(vectors, queries [][]float32, k int) [][]int {
	truth := make([][]int, len(queries))
	for qi, query := range queries {
		bestIDs := make([]int, k)
		bestDistances := make([]float32, k)
		for i := range bestIDs {
			bestIDs[i] = -1
			bestDistances[i] = float32(math.Inf(1))
		}

		admit := func(id int, distance float32) {
			if !(distance < bestDistances[k-1]) {
				return
			}
			position := k - 1
			for position > 0 && distance < bestDistances[position-1] {
				bestDistances[position] = bestDistances[position-1]
				bestIDs[position] = bestIDs[position-1]
				position--
			}
			bestDistances[position] = distance
			bestIDs[position] = id
		}

		i := 0
		if simd.HasL2Batch8Ptr() {
			for ; i+8 <= len(vectors); i += 8 {
				d0, d1, d2, d3, d4, d5, d6, d7 := simd.L2Distance8Ptr(
					query,
					unsafe.Pointer(unsafe.SliceData(vectors[i])),
					unsafe.Pointer(unsafe.SliceData(vectors[i+1])),
					unsafe.Pointer(unsafe.SliceData(vectors[i+2])),
					unsafe.Pointer(unsafe.SliceData(vectors[i+3])),
					unsafe.Pointer(unsafe.SliceData(vectors[i+4])),
					unsafe.Pointer(unsafe.SliceData(vectors[i+5])),
					unsafe.Pointer(unsafe.SliceData(vectors[i+6])),
					unsafe.Pointer(unsafe.SliceData(vectors[i+7])),
				)
				admit(i, d0)
				admit(i+1, d1)
				admit(i+2, d2)
				admit(i+3, d3)
				admit(i+4, d4)
				admit(i+5, d5)
				admit(i+6, d6)
				admit(i+7, d7)
			}
		}
		for ; i < len(vectors); i++ {
			admit(i, util.L2Distance_func(query, vectors[i]))
		}
		truth[qi] = bestIDs
	}
	return truth
}

// BenchmarkHNSWSemanticScale validates construction and search against a real
// embedding fixture. It is opt-in because a 50k x 768 fixture is about 154 MB.
//
// Run:
//
//	LIBRAVDB_SEMANTIC_FIXTURE=/path/to/nomic-longmemeval-50k.semantic.f32 \
//	  go test ./internal/index/hnsw -run '^$' -bench BenchmarkHNSWSemanticScale \
//	  -benchtime=1x -count=1
func BenchmarkHNSWSemanticScale(b *testing.B) {
	fixturePath := os.Getenv("LIBRAVDB_SEMANTIC_FIXTURE")
	if fixturePath == "" {
		b.Skip("set LIBRAVDB_SEMANTIC_FIXTURE to a generated semantic fixture")
	}

	fixture := loadSemanticFixture(b, fixturePath)
	if fixture.dim != semanticFixtureDim {
		b.Fatalf("fixture dim=%d, want=%d", fixture.dim, semanticFixtureDim)
	}
	ids := benchmarkIDs(len(fixture.vectors))
	entries := make([]VectorEntry, len(fixture.vectors))
	for i := range entries {
		entries[i] = VectorEntry{ID: ids[i], Vector: fixture.vectors[i]}
	}

	b.Logf("computing exact truth: vectors=%d queries=%d dim=%d model_tag=%016x", len(fixture.vectors), len(fixture.queries), fixture.dim, fixture.modelTag)
	truthStarted := time.Now()
	truth := semanticExactTruth(fixture.vectors, fixture.queries, semanticFixtureK)
	b.Logf("exact truth completed in %s", time.Since(truthStarted))
	repairFlush := os.Getenv("LIBRAVDB_SEMANTIC_REPAIR") == "1"
	graphM := 36
	if value := os.Getenv("LIBRAVDB_SEMANTIC_M"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed <= 0 {
			b.Fatalf("invalid LIBRAVDB_SEMANTIC_M=%q", value)
		}
		graphM = parsed
	}
	serialPrefix := 0
	if value := os.Getenv("LIBRAVDB_SEMANTIC_SERIAL_PREFIX"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed < 0 || parsed > len(entries) {
			b.Fatalf("invalid LIBRAVDB_SEMANTIC_SERIAL_PREFIX=%q", value)
		}
		serialPrefix = parsed
	}

	workerValues := []int{1, 4}
	if value := os.Getenv("LIBRAVDB_SEMANTIC_WORKERS"); value != "" {
		workers, err := strconv.Atoi(value)
		if err != nil || workers <= 0 {
			b.Fatalf("invalid LIBRAVDB_SEMANTIC_WORKERS=%q", value)
		}
		workerValues = []int{workers}
	}
	for _, workers := range workerValues {
		workers := workers
		b.Run("workers_"+strconv.Itoa(workers), func(b *testing.B) {
			var totalInserts int
			var totalRepairs int
			var recallEF200 float64
			var recallEF208 float64
			var recallEF216 float64
			var recallEF224 float64
			var recallEF300 float64
			var belowExactEF200 int
			var belowExactEF208 int
			var belowExactEF216 int
			var belowExactEF224 int
			var belowExactEF300 int
			var searchLatencyEF200 []int64
			var searchLatencyEF300 []int64
			ctx := context.Background()

			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				b.StopTimer()
				config := benchmarkNormalizedHNSWConfigDim(fixture.dim)
				config.M = graphM
				config.EfConstruction = 200
				config.EfSearch = 200
				config.RawStoreCap = len(fixture.vectors)
				config.IDMapCapacity = 1 << bits.Len(uint(len(fixture.vectors)*2-1))
				if repairFlush {
					config.RepairQueueSize = len(fixture.vectors) * 2
					config.RepairBatchSize = 256
				}
				index, err := NewHNSW(&config)
				if err != nil {
					b.Fatal(err)
				}
				preloadBenchmarkRawVectors(b, index, fixture.vectors)

				var next atomic.Uint64
				var wg sync.WaitGroup
				errCh := make(chan error, workers)
				start := make(chan struct{})
				for worker := 0; worker < workers; worker++ {
					wg.Add(1)
					go func() {
						defer wg.Done()
						<-start
						for {
							i := int(next.Add(1) - 1)
							if i >= len(entries) {
								return
							}
							if err := index.Insert(ctx, &entries[i]); err != nil {
								select {
								case errCh <- fmt.Errorf("insert %d: %w", i, err):
								default:
								}
								return
							}
						}
					}()
				}

				previousProcs := runtime.GOMAXPROCS(workers)
				b.StartTimer()
				for i := 0; i < serialPrefix; i++ {
					if err := index.Insert(ctx, &entries[i]); err != nil {
						b.StopTimer()
						runtime.GOMAXPROCS(previousProcs)
						index.Close()
						b.Fatalf("serial prefix insert %d: %v", i, err)
					}
				}
				next.Store(uint64(serialPrefix))
				close(start)
				wg.Wait()
				if repairFlush {
					totalRepairs += index.FlushRepairs(0)
				}
				b.StopTimer()
				runtime.GOMAXPROCS(previousProcs)
				select {
				case err := <-errCh:
					index.Close()
					b.Fatal(err)
				default:
				}
				if got := int(index.size.Load()); got != len(entries) {
					index.Close()
					b.Fatalf("published %d nodes, want %d", got, len(entries))
				}
				totalInserts += len(entries)

				truthSets := benchmarkTruthSetsForExternalIDs(b, index, truth, ids)
				ordinalBuf200 := make([]uint32, 0, semanticFixtureK)
				ordinalBuf208 := make([]uint32, 0, semanticFixtureK)
				ordinalBuf216 := make([]uint32, 0, semanticFixtureK)
				ordinalBuf224 := make([]uint32, 0, semanticFixtureK)
				ordinalBuf300 := make([]uint32, 0, semanticFixtureK)
				warmupQueries := min(8, len(fixture.queries))
				for qi := 0; qi < warmupQueries; qi++ {
					if _, _, err := searchExplicitEFOrdinals(ctx, index, fixture.queries[qi], semanticFixtureK, 200, ordinalBuf200); err != nil {
						index.Close()
						b.Fatalf("ef=200 warmup query %d: %v", qi, err)
					}
					if _, _, err := searchExplicitEFOrdinals(ctx, index, fixture.queries[qi], semanticFixtureK, 300, ordinalBuf300); err != nil {
						index.Close()
						b.Fatalf("ef=300 warmup query %d: %v", qi, err)
					}
				}

				searchOne := func(qi, ef int, ordinalBuf []uint32) []uint32 {
					started := time.Now()
					ordinals, _, err := searchExplicitEFOrdinals(ctx, index, fixture.queries[qi], semanticFixtureK, ef, ordinalBuf)
					latency := time.Since(started).Nanoseconds()
					if err != nil {
						index.Close()
						b.Fatalf("ef=%d query %d: %v", ef, qi, err)
					}
					recall := recallOrdinalsAtK(ordinals, truthSets[qi], semanticFixtureK)
					if ef == 200 {
						searchLatencyEF200 = append(searchLatencyEF200, latency)
						recallEF200 += recall
						if recall < 1 {
							belowExactEF200++
						}
					} else {
						searchLatencyEF300 = append(searchLatencyEF300, latency)
						recallEF300 += recall
						if recall < 1 {
							belowExactEF300++
						}
					}
					return ordinals
				}
				for qi := range fixture.queries {
					if qi&1 == 0 {
						ordinalBuf200 = searchOne(qi, 200, ordinalBuf200)
						ordinalBuf300 = searchOne(qi, 300, ordinalBuf300)
					} else {
						ordinalBuf300 = searchOne(qi, 300, ordinalBuf300)
						ordinalBuf200 = searchOne(qi, 200, ordinalBuf200)
					}

					for _, probe := range []struct {
						ef         int
						buffer     *[]uint32
						total      *float64
						belowExact *int
					}{
						{ef: 208, buffer: &ordinalBuf208, total: &recallEF208, belowExact: &belowExactEF208},
						{ef: 216, buffer: &ordinalBuf216, total: &recallEF216, belowExact: &belowExactEF216},
						{ef: 224, buffer: &ordinalBuf224, total: &recallEF224, belowExact: &belowExactEF224},
					} {
						ordinals, _, err := searchExplicitEFOrdinals(ctx, index, fixture.queries[qi], semanticFixtureK, probe.ef, *probe.buffer)
						if err != nil {
							index.Close()
							b.Fatalf("ef=%d query %d: %v", probe.ef, qi, err)
						}
						recall := recallOrdinalsAtK(ordinals, truthSets[qi], semanticFixtureK)
						*probe.total += recall
						if recall < 1 {
							*probe.belowExact++
						}
						*probe.buffer = ordinals
					}
				}
				index.Close()
			}

			searches := b.N * len(fixture.queries)
			if elapsed := b.Elapsed(); elapsed > 0 {
				b.ReportMetric(float64(totalInserts)/elapsed.Seconds(), "graph_ready_insert/s")
			}
			b.ReportMetric(recallEF200/float64(searches), "recall_ef200@10")
			b.ReportMetric(recallEF208/float64(searches), "recall_ef208@10")
			b.ReportMetric(recallEF216/float64(searches), "recall_ef216@10")
			b.ReportMetric(recallEF224/float64(searches), "recall_ef224@10")
			b.ReportMetric(recallEF300/float64(searches), "recall_ef300@10")
			b.ReportMetric(float64(belowExactEF200)/float64(b.N), "queries_below_1_ef200/build")
			b.ReportMetric(float64(belowExactEF208)/float64(b.N), "queries_below_1_ef208/build")
			b.ReportMetric(float64(belowExactEF216)/float64(b.N), "queries_below_1_ef216/build")
			b.ReportMetric(float64(belowExactEF224)/float64(b.N), "queries_below_1_ef224/build")
			b.ReportMetric(float64(belowExactEF300)/float64(b.N), "queries_below_1_ef300/build")
			b.ReportMetric(float64(percentileDuration(searchLatencyEF200, 0.50).Nanoseconds()), "search_ef200_p50-ns")
			b.ReportMetric(float64(percentileDuration(searchLatencyEF200, 0.99).Nanoseconds()), "search_ef200_p99-ns")
			b.ReportMetric(float64(percentileDuration(searchLatencyEF300, 0.50).Nanoseconds()), "search_ef300_p50-ns")
			b.ReportMetric(float64(percentileDuration(searchLatencyEF300, 0.99).Nanoseconds()), "search_ef300_p99-ns")
			b.ReportMetric(float64(len(fixture.vectors)), "nodes/build")
			b.ReportMetric(float64(graphM), "M")
			b.ReportMetric(float64(serialPrefix), "serial_prefix")
			b.ReportMetric(float64(totalRepairs)/float64(b.N), "repairs/build")
			b.ReportMetric(float64(workers), "workers")
		})
	}
	runtime.KeepAlive(fixture.data)
}
