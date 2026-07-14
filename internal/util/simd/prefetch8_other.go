//go:build !arm64 && !amd64

package simd

import "unsafe"

func Prefetch8L1(_ *[8]unsafe.Pointer) {}
