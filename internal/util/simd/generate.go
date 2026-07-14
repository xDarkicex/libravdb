//go:build ignore

package main

import (
	"strconv"

	"github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/operand"
	"github.com/mmcloughlin/avo/reg"
)

const unroll = 4

func main() {
	genDotProduct()
	genL2()
	genL2x4()
	genL2x4Ptr()
	genL2x8Ptr()
	genL2AnyLessThan8Ptr()
	genPrefetch()
	build.Generate()
}

func genPrefetch() {
	build.TEXT("prefetch8L1AMD64", build.NOSPLIT, "func(ptrs *[8]*byte)")
	build.Pragma("noescape")
	ptrs := build.Load(build.Param("ptrs"), build.GP64())
	for i := 0; i < 8; i++ {
		ptr := build.GP64()
		build.MOVQ(operand.Mem{Base: ptrs, Disp: i * 8}, ptr)
		build.PREFETCHT0(operand.Mem{Base: ptr})
	}
	build.RET()
}

func genDotProduct() {
	build.TEXT("DotProductAVX2", build.NOSPLIT, "func(a, b []float32) float32")
	build.Pragma("noescape")
	build.Doc("DotProductAVX2 computes the dot product of two float32 slices using AVX2, unrolled.")

	aPtr := build.Load(build.Param("a").Base(), build.GP64())
	bPtr := build.Load(build.Param("b").Base(), build.GP64())
	n := build.Load(build.Param("a").Len(), build.GP64())

	accs := make([]reg.VecVirtual, unroll)
	for i := 0; i < unroll; i++ {
		accs[i] = build.YMM()
		build.VXORPS(accs[i], accs[i], accs[i])
	}

	build.Label("dot_loop")
	build.CMPQ(n, operand.Imm(unroll*8))
	build.JL(operand.LabelRef("dot_tail"))

	for i := 0; i < unroll; i++ {
		va := build.YMM()
		vb := build.YMM()
		build.VMOVUPS(operand.Mem{Base: aPtr, Disp: i * 32}, va)
		build.VMOVUPS(operand.Mem{Base: bPtr, Disp: i * 32}, vb)
		build.VFMADD231PS(va, vb, accs[i])
	}

	build.ADDQ(operand.Imm(unroll*32), aPtr)
	build.ADDQ(operand.Imm(unroll*32), bPtr)
	build.SUBQ(operand.Imm(unroll*8), n)
	build.JMP(operand.LabelRef("dot_loop"))

	build.Label("dot_tail")
	for i := 1; i < unroll; i++ {
		build.VADDPS(accs[i], accs[0], accs[0])
	}

	build.Label("dot_tail_8")
	build.CMPQ(n, operand.Imm(8))
	build.JL(operand.LabelRef("dot_tail_1"))

	va := build.YMM()
	vb := build.YMM()
	build.VMOVUPS(operand.Mem{Base: aPtr}, va)
	build.VMOVUPS(operand.Mem{Base: bPtr}, vb)
	build.VFMADD231PS(va, vb, accs[0])

	build.ADDQ(operand.Imm(32), aPtr)
	build.ADDQ(operand.Imm(32), bPtr)
	build.SUBQ(operand.Imm(8), n)
	build.JMP(operand.LabelRef("dot_tail_8"))

	build.Label("dot_tail_1")
	xmmSum := accs[0].AsX()
	temp := build.XMM()
	build.VEXTRACTF128(operand.Imm(1), accs[0], temp)
	build.VADDPS(temp, xmmSum, xmmSum)
	build.VHADDPS(xmmSum, xmmSum, xmmSum)
	build.VHADDPS(xmmSum, xmmSum, xmmSum)

	build.Label("dot_scalar_loop")
	build.CMPQ(n, operand.Imm(0))
	build.JE(operand.LabelRef("dot_done"))

	sA := build.XMM()
	sB := build.XMM()
	build.VMOVSS(operand.Mem{Base: aPtr}, sA)
	build.VMOVSS(operand.Mem{Base: bPtr}, sB)
	build.VMULSS(sA, sB, sA)
	build.VADDSS(sA, xmmSum, xmmSum)

	build.ADDQ(operand.Imm(4), aPtr)
	build.ADDQ(operand.Imm(4), bPtr)
	build.SUBQ(operand.Imm(1), n)
	build.JMP(operand.LabelRef("dot_scalar_loop"))

	build.Label("dot_done")
	build.Store(xmmSum, build.ReturnIndex(0))
	build.RET()
}

func genL2() {
	build.TEXT("L2DistanceAVX2", build.NOSPLIT, "func(a, b []float32) float32")
	build.Pragma("noescape")
	build.Doc("L2DistanceAVX2 computes the L2 distance of two float32 slices using AVX2, unrolled.")

	aPtr := build.Load(build.Param("a").Base(), build.GP64())
	bPtr := build.Load(build.Param("b").Base(), build.GP64())
	n := build.Load(build.Param("a").Len(), build.GP64())

	accs := make([]reg.VecVirtual, unroll)
	for i := 0; i < unroll; i++ {
		accs[i] = build.YMM()
		build.VXORPS(accs[i], accs[i], accs[i])
	}

	build.Label("l2_loop")
	build.CMPQ(n, operand.Imm(unroll*8))
	build.JL(operand.LabelRef("l2_tail"))

	for i := 0; i < unroll; i++ {
		va := build.YMM()
		vb := build.YMM()
		build.VMOVUPS(operand.Mem{Base: aPtr, Disp: i * 32}, va)
		build.VMOVUPS(operand.Mem{Base: bPtr, Disp: i * 32}, vb)
		build.VSUBPS(vb, va, va)
		build.VFMADD231PS(va, va, accs[i])
	}

	build.ADDQ(operand.Imm(unroll*32), aPtr)
	build.ADDQ(operand.Imm(unroll*32), bPtr)
	build.SUBQ(operand.Imm(unroll*8), n)
	build.JMP(operand.LabelRef("l2_loop"))

	build.Label("l2_tail")
	for i := 1; i < unroll; i++ {
		build.VADDPS(accs[i], accs[0], accs[0])
	}

	build.Label("l2_tail_8")
	build.CMPQ(n, operand.Imm(8))
	build.JL(operand.LabelRef("l2_tail_1"))

	va := build.YMM()
	vb := build.YMM()
	build.VMOVUPS(operand.Mem{Base: aPtr}, va)
	build.VMOVUPS(operand.Mem{Base: bPtr}, vb)
	build.VSUBPS(vb, va, va)
	build.VFMADD231PS(va, va, accs[0])

	build.ADDQ(operand.Imm(32), aPtr)
	build.ADDQ(operand.Imm(32), bPtr)
	build.SUBQ(operand.Imm(8), n)
	build.JMP(operand.LabelRef("l2_tail_8"))

	build.Label("l2_tail_1")
	xmmSum := accs[0].AsX()
	temp := build.XMM()
	build.VEXTRACTF128(operand.Imm(1), accs[0], temp)
	build.VADDPS(temp, xmmSum, xmmSum)
	build.VHADDPS(xmmSum, xmmSum, xmmSum)
	build.VHADDPS(xmmSum, xmmSum, xmmSum)

	build.Label("l2_scalar_loop")
	build.CMPQ(n, operand.Imm(0))
	build.JE(operand.LabelRef("l2_done"))

	sA := build.XMM()
	sB := build.XMM()
	build.VMOVSS(operand.Mem{Base: aPtr}, sA)
	build.VMOVSS(operand.Mem{Base: bPtr}, sB)
	build.VSUBSS(sB, sA, sA)
	build.VMULSS(sA, sA, sA)
	build.VADDSS(sA, xmmSum, xmmSum)

	build.ADDQ(operand.Imm(4), aPtr)
	build.ADDQ(operand.Imm(4), bPtr)
	build.SUBQ(operand.Imm(1), n)
	build.JMP(operand.LabelRef("l2_scalar_loop"))

	build.Label("l2_done")
	build.Store(xmmSum, build.ReturnIndex(0))
	build.RET()
}

func genL2x4() {
	build.TEXT("L2Distance4AVX2", build.NOSPLIT, "func(q, b0, b1, b2, b3 []float32) (d0, d1, d2, d3 float32)")
	build.Pragma("noescape")
	build.Doc("L2Distance4AVX2 computes four L2 distances against one query using AVX2.")

	qPtr := build.Load(build.Param("q").Base(), build.GP64())
	b0Ptr := build.Load(build.Param("b0").Base(), build.GP64())
	b1Ptr := build.Load(build.Param("b1").Base(), build.GP64())
	b2Ptr := build.Load(build.Param("b2").Base(), build.GP64())
	b3Ptr := build.Load(build.Param("b3").Base(), build.GP64())
	n := build.Load(build.Param("q").Len(), build.GP64())

	acc0 := build.YMM()
	acc1 := build.YMM()
	acc2 := build.YMM()
	acc3 := build.YMM()
	build.VXORPS(acc0, acc0, acc0)
	build.VXORPS(acc1, acc1, acc1)
	build.VXORPS(acc2, acc2, acc2)
	build.VXORPS(acc3, acc3, acc3)

	build.Label("l2x4_loop")
	build.CMPQ(n, operand.Imm(8))
	build.JL(operand.LabelRef("l2x4_tail"))

	qv := build.YMM()
	tmp := build.YMM()
	build.VMOVUPS(operand.Mem{Base: qPtr}, qv)

	build.VMOVUPS(operand.Mem{Base: b0Ptr}, tmp)
	build.VSUBPS(qv, tmp, tmp)
	build.VFMADD231PS(tmp, tmp, acc0)

	build.VMOVUPS(operand.Mem{Base: b1Ptr}, tmp)
	build.VSUBPS(qv, tmp, tmp)
	build.VFMADD231PS(tmp, tmp, acc1)

	build.VMOVUPS(operand.Mem{Base: b2Ptr}, tmp)
	build.VSUBPS(qv, tmp, tmp)
	build.VFMADD231PS(tmp, tmp, acc2)

	build.VMOVUPS(operand.Mem{Base: b3Ptr}, tmp)
	build.VSUBPS(qv, tmp, tmp)
	build.VFMADD231PS(tmp, tmp, acc3)

	build.ADDQ(operand.Imm(32), qPtr)
	build.ADDQ(operand.Imm(32), b0Ptr)
	build.ADDQ(operand.Imm(32), b1Ptr)
	build.ADDQ(operand.Imm(32), b2Ptr)
	build.ADDQ(operand.Imm(32), b3Ptr)
	build.SUBQ(operand.Imm(8), n)
	build.JMP(operand.LabelRef("l2x4_loop"))

	build.Label("l2x4_tail")
	x0 := acc0.AsX()
	x1 := acc1.AsX()
	x2 := acc2.AsX()
	x3 := acc3.AsX()
	reduceYMM(acc0, x0)
	reduceYMM(acc1, x1)
	reduceYMM(acc2, x2)
	reduceYMM(acc3, x3)

	build.Label("l2x4_scalar_loop")
	build.CMPQ(n, operand.Imm(0))
	build.JE(operand.LabelRef("l2x4_done"))

	qs := build.XMM()
	bs := build.XMM()
	build.VMOVSS(operand.Mem{Base: qPtr}, qs)

	build.VMOVSS(operand.Mem{Base: b0Ptr}, bs)
	build.VSUBSS(qs, bs, bs)
	build.VMULSS(bs, bs, bs)
	build.VADDSS(bs, x0, x0)

	build.VMOVSS(operand.Mem{Base: b1Ptr}, bs)
	build.VSUBSS(qs, bs, bs)
	build.VMULSS(bs, bs, bs)
	build.VADDSS(bs, x1, x1)

	build.VMOVSS(operand.Mem{Base: b2Ptr}, bs)
	build.VSUBSS(qs, bs, bs)
	build.VMULSS(bs, bs, bs)
	build.VADDSS(bs, x2, x2)

	build.VMOVSS(operand.Mem{Base: b3Ptr}, bs)
	build.VSUBSS(qs, bs, bs)
	build.VMULSS(bs, bs, bs)
	build.VADDSS(bs, x3, x3)

	build.ADDQ(operand.Imm(4), qPtr)
	build.ADDQ(operand.Imm(4), b0Ptr)
	build.ADDQ(operand.Imm(4), b1Ptr)
	build.ADDQ(operand.Imm(4), b2Ptr)
	build.ADDQ(operand.Imm(4), b3Ptr)
	build.SUBQ(operand.Imm(1), n)
	build.JMP(operand.LabelRef("l2x4_scalar_loop"))

	build.Label("l2x4_done")
	build.Store(x0, build.ReturnIndex(0))
	build.Store(x1, build.ReturnIndex(1))
	build.Store(x2, build.ReturnIndex(2))
	build.Store(x3, build.ReturnIndex(3))
	build.RET()
}

func loadL2PtrParameters(count int) (reg.Register, []reg.Register, reg.Register) {
	qPtr := build.Load(build.Param("q").Base(), build.GP64())
	n := build.Load(build.Param("q").Len(), build.GP64())
	ptrs := make([]reg.Register, count)
	for i := range ptrs {
		ptrs[i] = build.Load(build.Param("b"+strconv.Itoa(i)), build.GP64())
	}
	return qPtr, ptrs, n
}

func emitL2PtrBody(prefix string, qPtr reg.Register, ptrs []reg.Register, n reg.Register) []reg.VecVirtual {
	accs := make([]reg.VecVirtual, len(ptrs))
	for i := range accs {
		accs[i] = build.YMM()
		build.VXORPS(accs[i], accs[i], accs[i])
	}

	build.Label(prefix + "_loop")
	build.CMPQ(n, operand.Imm(8))
	build.JL(operand.LabelRef(prefix + "_tail"))

	qv := build.YMM()
	tmp := build.YMM()
	build.VMOVUPS(operand.Mem{Base: qPtr}, qv)
	for i := range ptrs {
		build.VMOVUPS(operand.Mem{Base: ptrs[i]}, tmp)
		build.VSUBPS(qv, tmp, tmp)
		build.VFMADD231PS(tmp, tmp, accs[i])
		build.ADDQ(operand.Imm(32), ptrs[i])
	}
	build.ADDQ(operand.Imm(32), qPtr)
	build.SUBQ(operand.Imm(8), n)
	build.JMP(operand.LabelRef(prefix + "_loop"))

	build.Label(prefix + "_tail")
	return accs
}

func emitL2x8ScalarTail(prefix string, qPtr reg.Register, ptrs []reg.Register, n reg.Register, sums []reg.Register) {
	build.Label(prefix + "_scalar_loop")
	build.CMPQ(n, operand.Imm(0))
	build.JE(operand.LabelRef(prefix + "_done"))

	qs := build.XMM()
	bs := build.XMM()
	build.VMOVSS(operand.Mem{Base: qPtr}, qs)
	for i := range ptrs {
		build.VMOVSS(operand.Mem{Base: ptrs[i]}, bs)
		build.VSUBSS(qs, bs, bs)
		build.VMULSS(bs, bs, bs)
		build.VADDSS(bs, sums[i], sums[i])
		build.ADDQ(operand.Imm(4), ptrs[i])
	}
	build.ADDQ(operand.Imm(4), qPtr)
	build.SUBQ(operand.Imm(1), n)
	build.JMP(operand.LabelRef(prefix + "_scalar_loop"))

	build.Label(prefix + "_done")
}

func genL2x8Ptr() {
	build.TEXT("l2Distance8PtrAVX2", build.NOSPLIT, "func(q []float32, b0, b1, b2, b3, b4, b5, b6, b7 *byte) (d0, d1, d2, d3, d4, d5, d6, d7 float32)")
	build.Pragma("noescape")
	build.Doc("l2Distance8PtrAVX2 computes eight squared L2 distances in one query pass using AVX2.")

	qPtr, ptrs, n := loadL2PtrParameters(8)
	accs := emitL2PtrBody("l2x8ptr", qPtr, ptrs, n)
	sums := make([]reg.Register, len(accs))
	for i := range accs {
		sums[i] = accs[i].AsX()
		reduceYMM(accs[i], sums[i])
	}
	emitL2x8ScalarTail("l2x8ptr", qPtr, ptrs, n, sums)
	for i := range sums {
		build.Store(sums[i], build.ReturnIndex(i))
	}
	build.RET()
}

func genL2AnyLessThan8Ptr() {
	build.TEXT("l2AnyLessThan8PtrAVX2", build.NOSPLIT, "func(q []float32, b0, b1, b2, b3, b4, b5, b6, b7 *byte, cutoff float32) uint32")
	build.Pragma("noescape")
	build.Doc("l2AnyLessThan8PtrAVX2 reports whether any squared L2 distance is below cutoff using AVX2.")

	qPtr, ptrs, n := loadL2PtrParameters(8)
	accs := emitL2PtrBody("l2x8any", qPtr, ptrs, n)
	sums := make([]reg.Register, len(accs))
	for i := range accs {
		sums[i] = accs[i].AsX()
		reduceYMM(accs[i], sums[i])
	}
	emitL2x8ScalarTail("l2x8any", qPtr, ptrs, n, sums)

	cutoff := build.Load(build.Param("cutoff"), build.XMM())
	for i := range sums {
		next := "l2x8any_compare_next_" + strconv.Itoa(i)
		build.VUCOMISS(cutoff, sums[i])
		build.JP(operand.LabelRef(next))
		build.JAE(operand.LabelRef(next))
		result := build.GP32()
		build.MOVL(operand.U32(1), result)
		build.Store(result, build.ReturnIndex(0))
		build.RET()
		build.Label(next)
	}
	result := build.GP32()
	build.MOVL(operand.U32(0), result)
	build.Store(result, build.ReturnIndex(0))
	build.RET()
}

func genL2x4Ptr() {
	build.TEXT("l2Distance4PtrAVX2", build.NOSPLIT, "func(q []float32, b0, b1, b2, b3 *byte) (d0, d1, d2, d3 float32)")
	build.Pragma("noescape")
	build.Doc("l2Distance4PtrAVX2 computes four squared L2 distances in one query pass using AVX2.")

	qPtr, ptrs, n := loadL2PtrParameters(4)
	accs := emitL2PtrBody("l2x4ptrraw", qPtr, ptrs, n)
	sums := make([]reg.Register, len(accs))
	for i := range accs {
		sums[i] = accs[i].AsX()
		reduceYMM(accs[i], sums[i])
	}
	emitL2x8ScalarTail("l2x4ptrraw", qPtr, ptrs, n, sums)
	for i := range sums {
		build.Store(sums[i], build.ReturnIndex(i))
	}
	build.RET()
}

func reduceYMM(acc reg.VecVirtual, dst reg.Register) {
	temp := build.XMM()
	build.VEXTRACTF128(operand.Imm(1), acc, temp)
	build.VADDPS(temp, dst, dst)
	build.VHADDPS(dst, dst, dst)
	build.VHADDPS(dst, dst, dst)
}
