//go:build arm64

package simd

import "unsafe"

func DotProductNEON(a, b []float32) float32
func L2DistanceNEON(a, b []float32) float32
func L2Distance4NEON(q, b0, b1, b2, b3 []float32) (d0, d1, d2, d3 float32)
func L2Distance4PtrNEON(q []float32, b0, b1, b2, b3 unsafe.Pointer) (d0, d1, d2, d3 float32)
func L2Distance8PtrNEON(q []float32, b0, b1, b2, b3, b4, b5, b6, b7 unsafe.Pointer) (d0, d1, d2, d3, d4, d5, d6, d7 float32)

// L2Distance8AlignedPtrNEON computes eight distances in one query pass. The
// query dimension must be a multiple of 16.
func L2Distance8AlignedPtrNEON(q []float32, b0, b1, b2, b3, b4, b5, b6, b7 unsafe.Pointer) (d0, d1, d2, d3, d4, d5, d6, d7 float32)

// L2AnyLessThan8AlignedPtrNEON reports whether any squared L2 distance is
// strictly below cutoff. The query dimension must be a multiple of 16.
func L2AnyLessThan8AlignedPtrNEON(q []float32, b0, b1, b2, b3, b4, b5, b6, b7 unsafe.Pointer, cutoff float32) uint32

//go:noescape
func PrefetchL1(ptr unsafe.Pointer)

//go:noescape
func Prefetch8L1(ptrs *[8]unsafe.Pointer)
