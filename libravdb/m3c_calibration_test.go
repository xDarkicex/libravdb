package libravdb

// M3c calibration harness: sweep N x dimension x selectivity, time the three
// hybrid operators end-to-end, find the exact-vs-ANN crossover candidate
// count, fit exactCandidateFraction per dimension, and report recall of the
// approximate paths against the exact result (the honest reference).
//
// Run: go test -v -run TestM3c -timeout 60m ./libravdb/

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

type m3cCell struct {
	N   int
	d   int
	sel float64
	k   int
	exN int // actual matching candidate count (int(sel*N) clamped >=1)

	exactNs    int64
	filteredNs int64
	iterNs     int64
	exactRec   float64 // always 1.0 by construction (reference)
	filtRec    float64 // vs exact
	iterRec    float64 // vs exact
}

func m3cBuild(tb testing.TB, n, dim int, sel float64) (*Collection, []float32, func()) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(fmt.Sprintf(":memory:m3c_%d_%d_%d", n, dim, int(sel*1000))), WithMetrics(false))
	if err != nil {
		tb.Fatal(err)
	}
	var closeOnce sync.Once
	cleanup := func() { closeOnce.Do(func() { _ = db.Close() }) }
	tb.Cleanup(cleanup)

	col, err := db.CreateCollection(ctx, "c",
		WithDimension(dim),
		WithProductionHNSW(),
		WithMetadataSchema(map[string]FieldType{"cat": IntField}),
	)
	if err != nil {
		tb.Fatal(err)
	}

	rng := rand.New(rand.NewSource(int64(n)*100000 + int64(dim)*100 + int64(sel*1000)))
	matchCount := int(float64(n) * sel)
	if matchCount < 1 {
		matchCount = 1
	}
	entries := make([]VectorEntry, n)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = rng.Float32()*2 - 1
		}
		cat := int64(0)
		if i < matchCount {
			cat = 1
		}
		entries[i] = VectorEntry{ID: fmt.Sprintf("id-%06d", i), Vector: v, Metadata: map[string]interface{}{"cat": cat}}
	}
	if err := col.InsertBatch(ctx, entries); err != nil {
		tb.Fatal(err)
	}

	q := make([]float32, dim)
	for j := range q {
		q[j] = rng.Float32()*2 - 1
	}
	return col, q, cleanup
}

func m3cPred() []optimizer.RelationalPredicate {
	return []optimizer.RelationalPredicate{{Column: "cat", Operator: 12, Value: []byte("1")}}
}

// m3cExact: production-equivalent candidate enumeration plus SIMD exact
// scoring. Its timer deliberately includes enumeration, matching dispatcher
// accounting rather than timing only the final score loop.
func m3cExact(col *Collection, q []float32, k int) (*SearchResults, int64) {
	start := time.Now()
	records, _ := col.ListAll(context.Background())
	var cand []Record
	for _, r := range records {
		if recordMatchesPredicates(r, m3cPred()) {
			cand = append(cand, r)
		}
	}
	res := scoreAndSelectTopK(col, cand, q, k)
	return res, time.Since(start).Nanoseconds()
}

func m3cExactAll(col *Collection, q []float32, k int) *SearchResults {
	records, _ := col.ListAll(context.Background())
	return scoreAndSelectTopK(col, records, q, k)
}

// m3cFiltered executes the production filtered operator end to end,
// including bitmap materialization and its exact fallback policy.
func m3cFiltered(col *Collection, q []float32, k int) (*SearchResults, int64) {
	start := time.Now()
	plan := &optimizer.PhysicalPlan{
		CollectionName:     col.name,
		Kind:               optimizer.QueryKindKNN,
		HasVectorSearch:    true,
		HasRelationalQuery: true,
		Predicates:         m3cPred(),
		QueryVector:        q,
		Limit:              k,
		RecallContract:     optimizer.RecallBounded,
	}
	metrics := &QueryMetrics{PlanChosen: DispatchFilteredANN}
	res, err := newExecutor(col.db).executeFilteredANN(context.Background(), plan, metrics, &hybridConstraints{})
	if err != nil {
		return &SearchResults{}, time.Since(start).Nanoseconds()
	}
	return res, time.Since(start).Nanoseconds()
}

// m3cIter executes the production one-traversal iterative operator, not the
// historical repeated unfiltered-query stand-in.
func m3cIter(col *Collection, q []float32, k int) (*SearchResults, int64) {
	start := time.Now()
	plan := &optimizer.PhysicalPlan{
		CollectionName:     col.name,
		Kind:               optimizer.QueryKindKNN,
		HasVectorSearch:    true,
		HasRelationalQuery: true,
		Predicates:         m3cPred(),
		QueryVector:        q,
		Limit:              k,
		RecallContract:     optimizer.RecallBounded,
	}
	metrics := &QueryMetrics{PlanChosen: DispatchIterativeANNThenFilter}
	results, err := newExecutor(col.db).executeIterativeANNThenFilter(context.Background(), plan, metrics, &hybridConstraints{})
	if err != nil {
		return &SearchResults{}, time.Since(start).Nanoseconds()
	}
	return results, time.Since(start).Nanoseconds()
}

// m3cPure is the required same-fixture baseline. It distinguishes a weak
// base graph from a filtered-operator defect.
func m3cPure(col *Collection, q []float32, k, ef int) (*SearchResults, int64) {
	start := time.Now()
	results, err := col.Query(context.Background()).WithVector(q).WithEfSearch(ef).Limit(k).Execute()
	if err != nil {
		return &SearchResults{}, time.Since(start).Nanoseconds()
	}
	return results, time.Since(start).Nanoseconds()
}

func m3cRecall(got *SearchResults, want map[string]bool) float64 {
	if len(want) == 0 || got == nil {
		return 0
	}
	hits := 0
	for _, r := range got.Results {
		if want[r.ID] {
			hits++
		}
	}
	return float64(hits) / float64(len(want))
}

// m3cBest runs fn `runs` times, returns the best (min) time and the recall of
// that fastest run. Best-of is the standard microbenchmark latency estimator.
func m3cBest(fn func() (*SearchResults, int64), want map[string]bool, runs int) (int64, float64) {
	bestNs := int64(1 << 62)
	var bestRecall float64
	for i := 0; i < runs; i++ {
		res, ns := fn()
		if ns < bestNs {
			bestNs = ns
			bestRecall = m3cRecall(res, want)
		}
	}
	return bestNs, bestRecall
}

func TestM3cFilteredSanity(t *testing.T) {
	const k = 10
	col, q, cleanup := m3cBuild(t, 5_000, 32, 0.10)
	defer cleanup()

	exact, _ := m3cExact(col, q, k)
	want := make(map[string]bool, len(exact.Results))
	for _, result := range exact.Results {
		want[result.ID] = true
	}
	filtered, _ := m3cFiltered(col, q, k)
	if len(filtered.Results) != k {
		t.Fatalf("filtered result count = %d, want %d", len(filtered.Results), k)
	}
	if recall := m3cRecall(filtered, want); recall < 0.8 {
		t.Fatalf("filtered recall = %.3f, want at least 0.8", recall)
	}
}

// TestProductionHNSWRecallFloor512 is a deterministic quality gate for the
// default production profile. It compares pure HNSW against brute-force truth
// on several unseen 512d queries; a fast profile is intentionally not covered
// by this contract.
func TestProductionHNSWRecallFloor512(t *testing.T) {
	const (
		n       = 5_000
		dim     = 512
		k       = 10
		queries = 10
	)
	col, _, cleanup := m3cBuild(t, n, dim, 0.10)
	defer cleanup()
	rng := rand.New(rand.NewSource(0x51512))
	var total float64
	for i := 0; i < queries; i++ {
		query := make([]float32, dim)
		for j := range query {
			query[j] = rng.Float32()*2 - 1
		}
		exact := m3cExactAll(col, query, k)
		want := make(map[string]bool, len(exact.Results))
		for _, result := range exact.Results {
			want[result.ID] = true
		}
		approx, _ := m3cPure(col, query, k, ProductionHNSWEfSearch)
		total += m3cRecall(approx, want)
	}
	if average := total / queries; average < 0.97 {
		t.Fatalf("production HNSW 512d recall@%d = %.3f, want >= 0.970", k, average)
	}
}

// TestM3cCalibration runs the matrix and prints per-cell results plus the
// fitted exactCandidateFraction per dimension.
func TestM3cCalibration(t *testing.T) {
	const k = 10
	ns := []int{1_000, 5_000, 20_000, 50_000}
	dims := []int{32, 128, 512}
	sels := []float64{0.02, 0.10, 0.50}
	const runs = 3

	// dim -> list of (N, crossoverCandidates) observations
	type obs struct {
		n         int
		cross     float64 // candidate count where exact == filtered
		crossFrac float64 // cross / N
	}
	fits := map[int][]obs{}

	fmt.Printf("%-7s %-5s %-6s %-8s %12s %12s %12s %7s %7s %7s\n",
		"N", "dim", "sel", "cand", "exact_us", "filt_us", "iter_us", "pRec", "fRec", "iRec")

	for _, dim := range dims {
		for _, n := range ns {
			for _, sel := range sels {
				col, q, cleanup := m3cBuild(t, n, dim, sel)
				exN := int(float64(n) * sel)
				if exN < 1 {
					exN = 1
				}

				// Reference = exact (filters first, then top-k).
				exactRef, _ := m3cExact(col, q, k)
				want := make(map[string]bool, len(exactRef.Results))
				for _, r := range exactRef.Results {
					want[r.ID] = true
				}
				pureRef := m3cExactAll(col, q, k)
				pureWant := make(map[string]bool, len(pureRef.Results))
				for _, r := range pureRef.Results {
					pureWant[r.ID] = true
				}
				exactNs, _ := m3cBest(func() (*SearchResults, int64) {
					return m3cExact(col, q, k)
				}, nil, runs)

				filtNs, filtRec := m3cBest(func() (*SearchResults, int64) {
					return m3cFiltered(col, q, k)
				}, want, runs)
				iterNs, iterRec := m3cBest(func() (*SearchResults, int64) {
					return m3cIter(col, q, k)
				}, want, runs)
				_, pureRec := m3cBest(func() (*SearchResults, int64) {
					return m3cPure(col, q, k, ProductionHNSWEfSearch)
				}, pureWant, runs)

				fmt.Printf("%-7d %-5d %-6.2f %-8d %12.1f %12.1f %12.1f %7.3f %7.3f %7.3f\n",
					n, dim, sel, exN, float64(exactNs)/1e3, float64(filtNs)/1e3,
					float64(iterNs)/1e3, pureRec, filtRec, iterRec)

				_ = exN
				cleanup()
			}

			// Per (N,dim): estimate the crossover candidate count from the
			// ratio of exact-cost-per-candidate to filtered cost. exact cost
			// ~= alpha * candidates; filtered ~= beta. alpha from the largest
			// sel cell (0.50), beta from the same cell.
			// crossover = beta / alpha.
			col, q, cleanup := m3cBuild(t, n, dim, 0.50)
			exactNs, _ := m3cBest(func() (*SearchResults, int64) {
				return m3cExact(col, q, k)
			}, nil, runs)
			filtNs, _ := m3cBest(func() (*SearchResults, int64) {
				return m3cFiltered(col, q, k)
			}, nil, runs)
			cand50 := int(float64(n) * 0.50)
			if cand50 < 1 {
				cand50 = 1
			}
			if exactNs > 0 && filtNs > 0 {
				alpha := float64(exactNs) / float64(cand50) // ns per candidate
				beta := float64(filtNs)                     // flat filtered cost
				cross := beta / alpha
				fits[dim] = append(fits[dim], obs{n: n, cross: cross, crossFrac: cross / float64(n)})
			}
			cleanup()
		}
	}

	fmt.Println("\n=== Fitted exactCandidateFraction per dimension ===")
	for _, dim := range dims {
		os := fits[dim]
		if len(os) == 0 {
			continue
		}
		sumFrac := 0.0
		for _, o := range os {
			sumFrac += o.crossFrac
		}
		meanFrac := sumFrac / float64(len(os))
		fmt.Printf("dim=%4d: exactCandidateFraction = %.4f  (per-N crossover fracs: %v)\n",
			dim, meanFrac, func() []float64 {
				var f []float64
				for _, o := range os {
					f = append(f, o.crossFrac)
				}
				return f
			}())
	}
}
