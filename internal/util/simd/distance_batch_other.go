//go:build !arm64 && !amd64

package simd

import "unsafe"

func HasL2Batch8Ptr() bool { return false }

func L2Distance4Ptr(
	q []float32,
	b0, b1, b2, b3 unsafe.Pointer,
) (d0, d1, d2, d3 float32) {
	panic("SIMD x4 pointer L2 is not supported on this architecture")
}

func L2Distance8Ptr(
	q []float32,
	b0, b1, b2, b3, b4, b5, b6, b7 unsafe.Pointer,
) (d0, d1, d2, d3, d4, d5, d6, d7 float32) {
	panic("SIMD x8 pointer L2 is not supported on this architecture")
}

func L2AnyLessThan8Ptr(
	q []float32,
	b0, b1, b2, b3, b4, b5, b6, b7 unsafe.Pointer,
	cutoff float32,
) uint32 {
	panic("SIMD x8 pointer L2 is not supported on this architecture")
}
