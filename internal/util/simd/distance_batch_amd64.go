//go:build amd64

package simd

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

func HasL2Batch8Ptr() bool { return cpu.X86.HasAVX2 && cpu.X86.HasFMA }

func L2Distance4Ptr(
	q []float32,
	b0, b1, b2, b3 unsafe.Pointer,
) (d0, d1, d2, d3 float32) {
	return l2Distance4PtrAVX2(q, (*byte)(b0), (*byte)(b1), (*byte)(b2), (*byte)(b3))
}

func L2Distance8Ptr(
	q []float32,
	b0, b1, b2, b3, b4, b5, b6, b7 unsafe.Pointer,
) (d0, d1, d2, d3, d4, d5, d6, d7 float32) {
	return l2Distance8PtrAVX2(
		q,
		(*byte)(b0), (*byte)(b1), (*byte)(b2), (*byte)(b3),
		(*byte)(b4), (*byte)(b5), (*byte)(b6), (*byte)(b7),
	)
}

func L2AnyLessThan8Ptr(
	q []float32,
	b0, b1, b2, b3, b4, b5, b6, b7 unsafe.Pointer,
	cutoff float32,
) uint32 {
	return l2AnyLessThan8PtrAVX2(
		q,
		(*byte)(b0), (*byte)(b1), (*byte)(b2), (*byte)(b3),
		(*byte)(b4), (*byte)(b5), (*byte)(b6), (*byte)(b7),
		cutoff,
	)
}
