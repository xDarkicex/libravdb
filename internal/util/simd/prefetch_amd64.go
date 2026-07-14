//go:build amd64

package simd

import "unsafe"

func Prefetch8L1(ptrs *[8]unsafe.Pointer) {
	prefetch8L1AMD64((*[8]*byte)(unsafe.Pointer(ptrs)))
}
