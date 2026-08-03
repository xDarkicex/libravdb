package maxsim

import (
	"math"
)

// Token represents a single factorized 32-byte representation of a multi-vector embedding.
// Two tokens fit perfectly into a single 64-byte cache line for AVX-512 VPGATHERDD unpacking.
type Token struct {
	CentroidID uint32
	Residual   [28]byte // 4-bit or 8-bit scalar quantized residuals
}

// ApproxMaxSim computes the coarse upper-bound approximation for ColBERT-style late interaction.
// It executes mathematically bounded algebraic pruning using only the 4-byte Centroid IDs,
// eliminating the need to read the 28-byte residuals for irrelevant documents.
//
// The formula ensures the returned score is strictly >= the true MaxSim score (for similarity, 
// or <= for distance metrics) by applying a residual-sign correction.
func ApproxMaxSim(queryCentroids []uint32, docCentroids []uint32, centroidSimilarity func(c1, c2 uint32) float32, maxResidualBound float32) float32 {
	var totalScore float32 = 0
	
	// MaxSim Algebra: Outer sum monoid over query tokens
	for _, qc := range queryCentroids {
		var maxSimForToken float32 = -math.MaxFloat32
		
		// Inner max monoid over document tokens
		for _, dc := range docCentroids {
			// Base similarity using the global codebook
			sim := centroidSimilarity(qc, dc)
			
			// Residual-sign correction:
			// Adding the maximum possible residual contribution ensures the bound
			// is strictly conservative (false positives are allowed, false negatives are not).
			upperBoundSim := sim + maxResidualBound
			
			if upperBoundSim > maxSimForToken {
				maxSimForToken = upperBoundSim
			}
		}
		
		totalScore += maxSimForToken
	}
	
	return totalScore
}

// TrueMaxSim evaluates the exact multi-vector similarity, expanding the residuals.
// In the full engine, this is dispatched to highly optimized SIMD assembly.
func TrueMaxSim(queryTokens [][]float32, docTokens []Token, exactSimilarity func(q []float32, t Token) float32) float32 {
	var totalScore float32 = 0
	
	// MaxSim Algebra: Outer sum monoid over query tokens
	for _, qToken := range queryTokens {
		var maxSimForToken float32 = -math.MaxFloat32
		
		// Inner max monoid over document tokens
		for _, dt := range docTokens {
			sim := exactSimilarity(qToken, dt)
			if sim > maxSimForToken {
				maxSimForToken = sim
			}
		}
		
		totalScore += maxSimForToken
	}
	
	return totalScore
}
