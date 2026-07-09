//go:build arm64

package simd

import "unsafe"

func DotProductNEON(a, b []float32) float32
func L2DistanceNEON(a, b []float32) float32
func L2Distance4NEON(q, b0, b1, b2, b3 []float32) (d0, d1, d2, d3 float32)
func L2Distance4PtrNEON(q []float32, b0, b1, b2, b3 unsafe.Pointer) (d0, d1, d2, d3 float32)

//go:noescape
func PrefetchL1(ptr unsafe.Pointer)
