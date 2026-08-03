package maxsim

import (
	"testing"
	"unsafe"
)

func TestTokenLayout(t *testing.T) {
	var token Token
	
	if size := unsafe.Sizeof(token); size != 32 {
		t.Fatalf("Token expected to be 32 bytes, got %d", size)
	}
	
	if off := unsafe.Offsetof(token.CentroidID); off != 0 {
		t.Fatalf("CentroidID expected at offset 0, got %d", off)
	}
	if off := unsafe.Offsetof(token.Residual); off != 4 {
		t.Fatalf("Residual expected at offset 4, got %d", off)
	}
}

func TestApproxMaxSimPruningCorrectness(t *testing.T) {
	// A simple 1D similarity metric for testing
	simFunc := func(c1, c2 uint32) float32 {
		diff := float32(c1) - float32(c2)
		if diff < 0 {
			diff = -diff
		}
		return 100.0 - diff
	}
	
	// Pre-define alternating residuals for the 12 doc tokens
	residuals := []float32{3.0, -2.5, 4.1, -1.0, 5.0, -5.0, 0.0, 2.2, -3.1, 4.5, -4.9, 1.1}
	
	exactSimFunc := func(q []float32, token Token) float32 {
		baseSim := simFunc(uint32(q[0]), token.CentroidID)
		// Extract the signed residual for this token based on its residual array index
		// (Storing the index in the first byte of the Residual array for testing)
		resIdx := int(token.Residual[0])
		return baseSim + residuals[resIdx]
	}
	
	// 4 Query tokens
	queryCentroids := []uint32{10, 25, 40, 55}
	queryExact := [][]float32{{10.0}, {25.0}, {40.0}, {55.0}}
	
	// 12 Document tokens
	docCentroids := []uint32{5, 12, 15, 20, 22, 30, 35, 42, 48, 50, 58, 60}
	docTokens := make([]Token, 12)
	for i := 0; i < 12; i++ {
		docTokens[i] = Token{CentroidID: docCentroids[i]}
		docTokens[i].Residual[0] = byte(i) // Store index to fetch signed residual later
	}
	
	// maxResidual is precisely the maximum positive magnitude in our residual array (5.0)
	maxResidual := float32(5.0)
	
	trueSim := TrueMaxSim(queryExact, docTokens, exactSimFunc)
	approxSim := ApproxMaxSim(queryCentroids, docCentroids, simFunc, maxResidual)
	
	if approxSim < trueSim {
		t.Fatalf("Pruning contract violated: approx upper bound %f is less than true similarity %f", approxSim, trueSim)
	}
}

func BenchmarkApproxMaxSim(b *testing.B) {
	qCentroids := []uint32{1, 2, 3, 4}
	dCentroids := []uint32{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	
	simFunc := func(c1, c2 uint32) float32 {
		return 0.5 // Stub 
	}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		_ = ApproxMaxSim(qCentroids, dCentroids, simFunc, 0.1)
	}
}
