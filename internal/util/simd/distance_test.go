package simd

import (
	"math"
	"runtime"
	"testing"
	"unsafe"

	"golang.org/x/sys/cpu"
)

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
