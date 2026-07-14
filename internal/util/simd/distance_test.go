package simd

import (
	"math"
	"runtime"
	"strconv"
	"testing"
	"time"
	"unsafe"

	"golang.org/x/sys/cpu"
)

var distanceBenchmarkSink float32
var distancePredicateBenchmarkSink uint32

func BenchmarkL2Distance8PtrNEON(b *testing.B) {
	if runtime.GOARCH != "arm64" {
		b.Skipf("NEON pointer batch implementation only enabled for arm64")
		return
	}
	for _, dimension := range []int{64, 256, 768} {
		dimension := dimension
		b.Run(strconv.Itoa(dimension), func(b *testing.B) {
			vectors := make([][]float32, 9)
			for i := range vectors {
				vectors[i], _ = deterministicVectors(dimension + i)
				vectors[i] = vectors[i][:dimension]
			}
			b.SetBytes(int64(dimension * 4 * len(vectors)))
			b.ReportAllocs()
			b.ResetTimer()
			var sum float32
			//nolint:staticcheck // The arm64 NEON call is an intentional panic stub when analyzed on amd64.
			for i := 0; i < b.N; i++ {
				d0, d1, d2, d3, d4, d5, d6, d7 := L2Distance8PtrNEON(
					vectors[0],
					unsafe.Pointer(&vectors[1][0]),
					unsafe.Pointer(&vectors[2][0]),
					unsafe.Pointer(&vectors[3][0]),
					unsafe.Pointer(&vectors[4][0]),
					unsafe.Pointer(&vectors[5][0]),
					unsafe.Pointer(&vectors[6][0]),
					unsafe.Pointer(&vectors[7][0]),
					unsafe.Pointer(&vectors[8][0]),
				)
				sum += d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7
			}
			distanceBenchmarkSink = sum
		})
	}
}

func BenchmarkL2Distance8VsTwo4NEONInterleaved(b *testing.B) {
	if runtime.GOARCH != "arm64" {
		b.Skipf("NEON pointer batch implementation only enabled for arm64")
		return
	}
	const dimension = 768
	vectors := make([][]float32, 9)
	vectors[0], _ = deterministicVectors(dimension)
	for i := 1; i < len(vectors); i++ {
		_, vectors[i] = deterministicVectors(dimension + i)
		vectors[i] = vectors[i][:dimension]
	}
	ptrs := [8]unsafe.Pointer{}
	for i := range ptrs {
		ptrs[i] = unsafe.Pointer(&vectors[i+1][0])
	}

	var elapsed [2]time.Duration
	var sum float32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		//nolint:staticcheck // The arm64 NEON call is an intentional panic stub when analyzed on amd64.
		for offset := 0; offset < 2; offset++ {
			mode := (i + offset) & 1
			//nolint:staticcheck // The value is consumed on arm64; amd64 analysis sees the following NEON panic stub.
			start := time.Now()
			if mode == 0 {
				d0, d1, d2, d3, d4, d5, d6, d7 := L2Distance8PtrNEON(
					vectors[0], ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
				)
				sum += d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7
			} else {
				d0, d1, d2, d3 := L2Distance4PtrNEON(vectors[0], ptrs[0], ptrs[1], ptrs[2], ptrs[3])
				d4, d5, d6, d7 := L2Distance4PtrNEON(vectors[0], ptrs[4], ptrs[5], ptrs[6], ptrs[7])
				sum += d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7
			}
			elapsed[mode] += time.Since(start)
		}
	}
	b.StopTimer()
	distanceBenchmarkSink = sum
	if b.N > 0 {
		eightMean := float64(elapsed[0].Nanoseconds()) / float64(b.N)
		twoFourMean := float64(elapsed[1].Nanoseconds()) / float64(b.N)
		b.ReportMetric(eightMean, "eight_mean-ns")
		b.ReportMetric(twoFourMean, "two_x4_mean-ns")
		b.ReportMetric(twoFourMean/eightMean, "eight_speedup")
	}
}

func BenchmarkL2Distance8AlignedNEONInterleaved(b *testing.B) {
	if runtime.GOARCH != "arm64" {
		b.Skipf("NEON pointer batch implementation only enabled for arm64")
		return
	}
	const dimension = 768
	vectors := make([][]float32, 9)
	for i := range vectors {
		vectors[i], _ = deterministicVectors(dimension + i)
		vectors[i] = vectors[i][:dimension]
	}
	ptrs := [8]unsafe.Pointer{}
	for i := range ptrs {
		ptrs[i] = unsafe.Pointer(&vectors[i+1][0])
	}

	var elapsed [2]time.Duration
	var sum float32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		//nolint:staticcheck // The arm64 NEON call is an intentional panic stub when analyzed on amd64.
		for offset := 0; offset < 2; offset++ {
			mode := (i + offset) & 1
			//nolint:staticcheck // The value is consumed on arm64; amd64 analysis sees the following NEON panic stub.
			start := time.Now()
			if mode == 0 {
				d0, d1, d2, d3, d4, d5, d6, d7 := L2Distance8PtrNEON(
					vectors[0], ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
				)
				sum += d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7
			} else {
				d0, d1, d2, d3, d4, d5, d6, d7 := L2Distance8AlignedPtrNEON(
					vectors[0], ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
				)
				sum += d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7
			}
			elapsed[mode] += time.Since(start)
		}
	}
	b.StopTimer()
	distanceBenchmarkSink = sum
	if b.N > 0 {
		baselineMean := float64(elapsed[0].Nanoseconds()) / float64(b.N)
		alignedMean := float64(elapsed[1].Nanoseconds()) / float64(b.N)
		b.ReportMetric(baselineMean, "baseline_mean-ns")
		b.ReportMetric(alignedMean, "one_pass_mean-ns")
		b.ReportMetric(baselineMean/alignedMean, "one_pass_speedup")
	}
}

func BenchmarkL2AnyLessThan8AlignedNEONInterleaved(b *testing.B) {
	if runtime.GOARCH != "arm64" {
		b.Skipf("NEON pointer batch implementation only enabled for arm64")
		return
	}
	const dimension = 768
	vectors := make([][]float32, 9)
	for i := range vectors {
		vectors[i], _ = deterministicVectors(dimension + i)
		vectors[i] = vectors[i][:dimension]
	}
	ptrs := [8]unsafe.Pointer{}
	for i := range ptrs {
		ptrs[i] = unsafe.Pointer(&vectors[i+1][0])
	}
	d0, d1, d2, d3, d4, d5, d6, d7 := L2Distance8AlignedPtrNEON(
		vectors[0], ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
	)
	distances := [...]float32{d0, d1, d2, d3, d4, d5, d6, d7}
	minimum := distances[0]
	maximum := distances[0]
	for _, distance := range distances[1:] {
		minimum = min(minimum, distance)
		maximum = max(maximum, distance)
	}

	for _, tc := range []struct {
		name   string
		cutoff float32
	}{
		{name: "no_rejection", cutoff: minimum},
		{name: "rejection", cutoff: maximum},
	} {
		tc := tc
		b.Run(tc.name, func(b *testing.B) {
			var elapsed [2]time.Duration
			var sink uint32
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				//nolint:staticcheck // The arm64 NEON call is an intentional panic stub when analyzed on amd64.
				for offset := 0; offset < 2; offset++ {
					mode := (i + offset) & 1
					//nolint:staticcheck // The value is consumed on arm64; amd64 analysis sees the following NEON panic stub.
					start := time.Now()
					if mode == 0 {
						d0, d1, d2, d3, d4, d5, d6, d7 := L2Distance8AlignedPtrNEON(
							vectors[0], ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
						)
						if d0 < tc.cutoff || d1 < tc.cutoff || d2 < tc.cutoff || d3 < tc.cutoff ||
							d4 < tc.cutoff || d5 < tc.cutoff || d6 < tc.cutoff || d7 < tc.cutoff {
							sink++
						}
					} else {
						sink += L2AnyLessThan8AlignedPtrNEON(
							vectors[0], ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], tc.cutoff,
						)
					}
					elapsed[mode] += time.Since(start)
				}
			}
			b.StopTimer()
			distancePredicateBenchmarkSink = sink
			if b.N > 0 {
				baselineMean := float64(elapsed[0].Nanoseconds()) / float64(b.N)
				predicateMean := float64(elapsed[1].Nanoseconds()) / float64(b.N)
				b.ReportMetric(baselineMean, "baseline_mean-ns")
				b.ReportMetric(predicateMean, "predicate_mean-ns")
				b.ReportMetric(baselineMean/predicateMean, "predicate_speedup")
			}
		})
	}
}

func BenchmarkL2Distance8PtrAVX2VsTwo4Interleaved(b *testing.B) {
	if runtime.GOARCH != "amd64" || !HasL2Batch8Ptr() {
		b.Skip("AVX2 pointer batch implementation is unavailable")
	}
	const dimension = 768
	vectors := make([][]float32, 9)
	vectors[0], _ = deterministicVectors(dimension)
	for i := 1; i < len(vectors); i++ {
		_, vectors[i] = deterministicVectors(dimension + i)
		vectors[i] = vectors[i][:dimension]
	}
	ptrs := [8]unsafe.Pointer{}
	for i := range ptrs {
		ptrs[i] = unsafe.Pointer(&vectors[i+1][0])
	}

	var elapsed [2]time.Duration
	var sum float32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for offset := 0; offset < 2; offset++ {
			mode := (i + offset) & 1
			start := time.Now()
			if mode == 0 {
				d0, d1, d2, d3 := L2Distance4AVX2(vectors[0], vectors[1], vectors[2], vectors[3], vectors[4])
				d4, d5, d6, d7 := L2Distance4AVX2(vectors[0], vectors[5], vectors[6], vectors[7], vectors[8])
				sum += d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7
			} else {
				d0, d1, d2, d3, d4, d5, d6, d7 := L2Distance8Ptr(
					vectors[0], ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
				)
				sum += d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7
			}
			elapsed[mode] += time.Since(start)
		}
	}
	b.StopTimer()
	distanceBenchmarkSink = sum
	if b.N > 0 {
		twoFourMean := float64(elapsed[0].Nanoseconds()) / float64(b.N)
		eightMean := float64(elapsed[1].Nanoseconds()) / float64(b.N)
		b.ReportMetric(twoFourMean, "two_x4_mean-ns")
		b.ReportMetric(eightMean, "x8_ptr_mean-ns")
		b.ReportMetric(twoFourMean/eightMean, "x8_ptr_speedup")
	}
}

func scalarDot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func scalarL2(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

func deterministicVectors(n int) ([]float32, []float32) {
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(math.Sin(float64(i*17+3))) * 3
		b[i] = float32(math.Cos(float64(i*11+5))) * 2
	}
	return a, b
}

func TestL2DistanceSIMDMatchesScalar(t *testing.T) {
	l2 := l2DistanceSIMDForTest(t)
	for _, n := range []int{1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 127, 128, 129} {
		a, b := deterministicVectors(n)
		want := scalarL2(a, b)
		got := l2(a, b)
		if diff := math.Abs(float64(got - want)); diff > 1e-3 {
			t.Fatalf("L2DistanceSIMD len=%d got=%v want=%v diff=%v", n, got, want, diff)
		}
	}
}

func TestL2Distance4NEONMatchesScalar(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skipf("NEON batch implementation only enabled for arm64")
	}
	testL2Distance4(t, L2Distance4NEON)
}

func TestL2Distance4PtrNEONMatchesScalar(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skipf("NEON pointer batch implementation only enabled for arm64")
	}
	for _, n := range []int{1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 127, 128, 129} {
		q, b0 := deterministicVectors(n)
		_, b1 := deterministicVectors(n + 1)
		_, b2 := deterministicVectors(n + 2)
		_, b3 := deterministicVectors(n + 3)
		b1 = b1[:n]
		b2 = b2[:n]
		b3 = b3[:n]

		got0, got1, got2, got3 := L2Distance4PtrNEON(
			q,
			unsafe.Pointer(&b0[0]),
			unsafe.Pointer(&b1[0]),
			unsafe.Pointer(&b2[0]),
			unsafe.Pointer(&b3[0]),
		)
		want := [4]float32{
			scalarL2(q, b0),
			scalarL2(q, b1),
			scalarL2(q, b2),
			scalarL2(q, b3),
		}
		got := [4]float32{got0, got1, got2, got3}
		for i := range got {
			if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-3 {
				t.Fatalf("L2Distance4PtrNEON len=%d lane=%d got=%v want=%v diff=%v", n, i, got[i], want[i], diff)
			}
		}
	}
}

func TestL2Distance8PtrNEONMatchesScalar(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skipf("NEON pointer batch implementation only enabled for arm64")
	}
	for _, n := range []int{1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 127, 128, 129} {
		q, b0 := deterministicVectors(n)
		_, b1 := deterministicVectors(n + 1)
		_, b2 := deterministicVectors(n + 2)
		_, b3 := deterministicVectors(n + 3)
		_, b4 := deterministicVectors(n + 4)
		_, b5 := deterministicVectors(n + 5)
		_, b6 := deterministicVectors(n + 6)
		_, b7 := deterministicVectors(n + 7)
		b1 = b1[:n]
		b2 = b2[:n]
		b3 = b3[:n]
		b4 = b4[:n]
		b5 = b5[:n]
		b6 = b6[:n]
		b7 = b7[:n]

		got0, got1, got2, got3, got4, got5, got6, got7 := L2Distance8PtrNEON(
			q,
			unsafe.Pointer(&b0[0]),
			unsafe.Pointer(&b1[0]),
			unsafe.Pointer(&b2[0]),
			unsafe.Pointer(&b3[0]),
			unsafe.Pointer(&b4[0]),
			unsafe.Pointer(&b5[0]),
			unsafe.Pointer(&b6[0]),
			unsafe.Pointer(&b7[0]),
		)
		want := [8]float32{
			scalarL2(q, b0),
			scalarL2(q, b1),
			scalarL2(q, b2),
			scalarL2(q, b3),
			scalarL2(q, b4),
			scalarL2(q, b5),
			scalarL2(q, b6),
			scalarL2(q, b7),
		}
		got := [8]float32{got0, got1, got2, got3, got4, got5, got6, got7}
		for i := range got {
			if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-3 {
				t.Fatalf("L2Distance8PtrNEON len=%d lane=%d got=%v want=%v diff=%v", n, i, got[i], want[i], diff)
			}
		}
	}
}

func TestL2Distance8AlignedPtrNEONMatchesScalar(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skipf("NEON pointer batch implementation only enabled for arm64")
	}
	for _, n := range []int{16, 32, 64, 256, 768} {
		q, b0 := deterministicVectors(n)
		_, b1 := deterministicVectors(n + 1)
		_, b2 := deterministicVectors(n + 2)
		_, b3 := deterministicVectors(n + 3)
		_, b4 := deterministicVectors(n + 4)
		_, b5 := deterministicVectors(n + 5)
		_, b6 := deterministicVectors(n + 6)
		_, b7 := deterministicVectors(n + 7)
		b1 = b1[:n]
		b2 = b2[:n]
		b3 = b3[:n]
		b4 = b4[:n]
		b5 = b5[:n]
		b6 = b6[:n]
		b7 = b7[:n]

		got0, got1, got2, got3, got4, got5, got6, got7 := L2Distance8AlignedPtrNEON(
			q,
			unsafe.Pointer(&b0[0]),
			unsafe.Pointer(&b1[0]),
			unsafe.Pointer(&b2[0]),
			unsafe.Pointer(&b3[0]),
			unsafe.Pointer(&b4[0]),
			unsafe.Pointer(&b5[0]),
			unsafe.Pointer(&b6[0]),
			unsafe.Pointer(&b7[0]),
		)
		baseline0, baseline1, baseline2, baseline3, baseline4, baseline5, baseline6, baseline7 := L2Distance8PtrNEON(
			q,
			unsafe.Pointer(&b0[0]),
			unsafe.Pointer(&b1[0]),
			unsafe.Pointer(&b2[0]),
			unsafe.Pointer(&b3[0]),
			unsafe.Pointer(&b4[0]),
			unsafe.Pointer(&b5[0]),
			unsafe.Pointer(&b6[0]),
			unsafe.Pointer(&b7[0]),
		)
		want := [8]float32{
			scalarL2(q, b0),
			scalarL2(q, b1),
			scalarL2(q, b2),
			scalarL2(q, b3),
			scalarL2(q, b4),
			scalarL2(q, b5),
			scalarL2(q, b6),
			scalarL2(q, b7),
		}
		got := [8]float32{got0, got1, got2, got3, got4, got5, got6, got7}
		baseline := [8]float32{baseline0, baseline1, baseline2, baseline3, baseline4, baseline5, baseline6, baseline7}
		for i := range got {
			if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-2 {
				t.Fatalf("L2Distance8AlignedPtrNEON len=%d lane=%d got=%v want=%v diff=%v", n, i, got[i], want[i], diff)
			}
			if diff := math.Abs(float64(got[i] - baseline[i])); diff > 1e-2 {
				t.Fatalf("L2Distance8AlignedPtrNEON len=%d lane=%d got=%v baseline=%v diff=%v", n, i, got[i], baseline[i], diff)
			}
		}
	}
}

func TestL2AnyLessThan8AlignedPtrNEONMatchesDistances(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skipf("NEON pointer batch implementation only enabled for arm64")
	}
	for _, n := range []int{16, 32, 64, 256, 768} {
		vectors := make([][]float32, 9)
		vectors[0], _ = deterministicVectors(n)
		for i := 1; i < len(vectors); i++ {
			_, vectors[i] = deterministicVectors(n + i)
			vectors[i] = vectors[i][:n]
		}
		ptrs := [8]unsafe.Pointer{}
		for i := range ptrs {
			ptrs[i] = unsafe.Pointer(&vectors[i+1][0])
		}
		d0, d1, d2, d3, d4, d5, d6, d7 := L2Distance8AlignedPtrNEON(
			vectors[0], ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
		)
		distances := [...]float32{d0, d1, d2, d3, d4, d5, d6, d7}
		cutoffs := []float32{-1, 0, d0, d3, d7, math.MaxFloat32}
		for _, cutoff := range cutoffs {
			var want uint32
			for _, distance := range distances {
				if distance < cutoff {
					want = 1
					break
				}
			}
			got := L2AnyLessThan8AlignedPtrNEON(
				vectors[0], ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], cutoff,
			)
			if got != want {
				t.Fatalf("len=%d cutoff=%v got=%d want=%d distances=%v", n, cutoff, got, want, distances)
			}
		}
	}

	vectors := make([][]float32, 9)
	vectors[0], _ = deterministicVectors(16)
	for i := 1; i < len(vectors); i++ {
		_, vectors[i] = deterministicVectors(16 + i)
		vectors[i] = vectors[i][:16]
	}
	vectors[0][0] = float32(math.NaN())
	got := L2AnyLessThan8AlignedPtrNEON(
		vectors[0],
		unsafe.Pointer(&vectors[1][0]), unsafe.Pointer(&vectors[2][0]),
		unsafe.Pointer(&vectors[3][0]), unsafe.Pointer(&vectors[4][0]),
		unsafe.Pointer(&vectors[5][0]), unsafe.Pointer(&vectors[6][0]),
		unsafe.Pointer(&vectors[7][0]), unsafe.Pointer(&vectors[8][0]),
		math.MaxFloat32,
	)
	if got != 0 {
		t.Fatalf("NaN distances must not compare less than cutoff: got=%d", got)
	}
}

func TestL2Batch8PtrArchitectureAPI(t *testing.T) {
	if !HasL2Batch8Ptr() {
		t.Skip("x8 pointer SIMD is unavailable on this CPU")
	}
	for _, n := range []int{16, 17, 64, 255, 256, 768} {
		vectors := make([][]float32, 9)
		vectors[0], _ = deterministicVectors(n)
		for i := 1; i < len(vectors); i++ {
			_, vectors[i] = deterministicVectors(n + i)
			vectors[i] = vectors[i][:n]
		}
		ptrs := [8]unsafe.Pointer{}
		for i := range ptrs {
			ptrs[i] = unsafe.Pointer(&vectors[i+1][0])
		}
		d0, d1, d2, d3, d4, d5, d6, d7 := L2Distance8Ptr(
			vectors[0], ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
		)
		distances := [...]float32{d0, d1, d2, d3, d4, d5, d6, d7}
		if runtime.GOARCH == "amd64" {
			e0, e1, e2, e3 := L2Distance4AVX2(vectors[0], vectors[1], vectors[2], vectors[3], vectors[4])
			e4, e5, e6, e7 := L2Distance4AVX2(vectors[0], vectors[5], vectors[6], vectors[7], vectors[8])
			existing := [...]float32{e0, e1, e2, e3, e4, e5, e6, e7}
			for lane := range distances {
				if math.Float32bits(distances[lane]) != math.Float32bits(existing[lane]) {
					t.Fatalf("amd64 x8 changed distance bits: len=%d lane=%d x8=%v x4=%v", n, lane, distances[lane], existing[lane])
				}
			}
		}
		p0, p1, p2, p3 := L2Distance4Ptr(vectors[0], ptrs[0], ptrs[1], ptrs[2], ptrs[3])
		pointerFour := [...]float32{p0, p1, p2, p3}
		for lane, got := range pointerFour {
			if diff := math.Abs(float64(got - distances[lane])); diff > 1e-2 {
				t.Fatalf("x4 pointer len=%d lane=%d got=%v x8=%v diff=%v", n, lane, got, distances[lane], diff)
			}
		}
		for lane, got := range distances {
			want := scalarL2(vectors[0], vectors[lane+1])
			if diff := math.Abs(float64(got - want)); diff > 1e-2 {
				t.Fatalf("len=%d lane=%d got=%v want=%v diff=%v", n, lane, got, want, diff)
			}
		}

		for _, cutoff := range []float32{-1, d0, d3, d7, math.MaxFloat32} {
			var want uint32
			for _, distance := range distances {
				if distance < cutoff {
					want = 1
					break
				}
			}
			got := L2AnyLessThan8Ptr(
				vectors[0], ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], cutoff,
			)
			if got != want {
				t.Fatalf("len=%d cutoff=%v predicate=%d want=%d distances=%v", n, cutoff, got, want, distances)
			}
		}
	}
}

func TestL2Distance4AVX2MatchesScalar(t *testing.T) {
	if runtime.GOARCH != "amd64" || !cpu.X86.HasAVX2 || !cpu.X86.HasFMA {
		t.Skipf("AVX2 batch implementation not enabled on this machine")
	}
	testL2Distance4(t, L2Distance4AVX2)
}

func testL2Distance4(t *testing.T, l2x4 func([]float32, []float32, []float32, []float32, []float32) (float32, float32, float32, float32)) {
	t.Helper()
	for _, n := range []int{1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 127, 128, 129} {
		q, b0 := deterministicVectors(n)
		_, b1 := deterministicVectors(n + 1)
		_, b2 := deterministicVectors(n + 2)
		_, b3 := deterministicVectors(n + 3)
		b1 = b1[:n]
		b2 = b2[:n]
		b3 = b3[:n]

		got0, got1, got2, got3 := l2x4(q, b0, b1, b2, b3)
		want := [4]float32{
			scalarL2(q, b0),
			scalarL2(q, b1),
			scalarL2(q, b2),
			scalarL2(q, b3),
		}
		got := [4]float32{got0, got1, got2, got3}
		for i := range got {
			if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-3 {
				t.Fatalf("L2Distance4NEON len=%d lane=%d got=%v want=%v diff=%v", n, i, got[i], want[i], diff)
			}
		}
	}
}

func TestDotProductSIMDMatchesScalar(t *testing.T) {
	dot := dotProductSIMDForTest(t)
	for _, n := range []int{1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 127, 128, 129} {
		a, b := deterministicVectors(n)
		want := scalarDot(a, b)
		got := dot(a, b)
		if diff := math.Abs(float64(got - want)); diff > 1e-3 {
			t.Fatalf("DotProductSIMD len=%d got=%v want=%v diff=%v", n, got, want, diff)
		}
	}
}

func l2DistanceSIMDForTest(t *testing.T) func([]float32, []float32) float32 {
	t.Helper()
	switch runtime.GOARCH {
	case "arm64":
		return L2DistanceNEON
	case "amd64":
		if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
			return L2DistanceAVX2
		}
	}
	t.Skipf("no SIMD L2 implementation enabled for %s", runtime.GOARCH)
	return nil
}

func dotProductSIMDForTest(t *testing.T) func([]float32, []float32) float32 {
	t.Helper()
	switch runtime.GOARCH {
	case "arm64":
		return DotProductNEON
	case "amd64":
		if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
			return DotProductAVX2
		}
	}
	t.Skipf("no SIMD dot implementation enabled for %s", runtime.GOARCH)
	return nil
}
