//go:build arm64

package simd

import "unsafe"

func HasL2Batch8Ptr() bool { return true }

func L2Distance4Ptr(
	q []float32,
	b0, b1, b2, b3 unsafe.Pointer,
) (d0, d1, d2, d3 float32) {
	return L2Distance4PtrNEON(q, b0, b1, b2, b3)
}

func L2Distance8Ptr(
	q []float32,
	b0, b1, b2, b3, b4, b5, b6, b7 unsafe.Pointer,
) (d0, d1, d2, d3, d4, d5, d6, d7 float32) {
	if len(q)&15 == 0 {
		return L2Distance8AlignedPtrNEON(q, b0, b1, b2, b3, b4, b5, b6, b7)
	}
	return L2Distance8PtrNEON(q, b0, b1, b2, b3, b4, b5, b6, b7)
}

func L2AnyLessThan8Ptr(
	q []float32,
	b0, b1, b2, b3, b4, b5, b6, b7 unsafe.Pointer,
	cutoff float32,
) uint32 {
	if len(q)&15 == 0 {
		return L2AnyLessThan8AlignedPtrNEON(q, b0, b1, b2, b3, b4, b5, b6, b7, cutoff)
	}
	d0, d1, d2, d3, d4, d5, d6, d7 := L2Distance8PtrNEON(q, b0, b1, b2, b3, b4, b5, b6, b7)
	if d0 < cutoff || d1 < cutoff || d2 < cutoff || d3 < cutoff ||
		d4 < cutoff || d5 < cutoff || d6 < cutoff || d7 < cutoff {
		return 1
	}
	return 0
}
