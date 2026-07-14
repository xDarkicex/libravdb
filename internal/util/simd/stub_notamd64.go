//go:build !amd64

package simd

// DotProductAVX2 is a stub for non-amd64 architectures.
// It will never be called because distance.go checks cpu.X86.HasAVX2.
func DotProductAVX2(a []float32, b []float32) float32 {
	panic("AVX2 not supported on this architecture")
}

// L2DistanceAVX2 is a stub for non-amd64 architectures.
func L2DistanceAVX2(a []float32, b []float32) float32 {
	panic("AVX2 not supported on this architecture")
}

func L2Distance4AVX2(q []float32, b0 []float32, b1 []float32, b2 []float32, b3 []float32) (d0 float32, d1 float32, d2 float32, d3 float32) {
	panic("AVX2 not supported on this architecture")
}
