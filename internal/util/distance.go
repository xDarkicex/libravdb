package util

import (
	"fmt"

	"github.com/xDarkicex/libravdb/internal/util/simd"
	"golang.org/x/sys/cpu"
)

// DistanceMetric defines supported distance metrics
type DistanceMetric int

const (
	L2Distance DistanceMetric = iota
	InnerProduct
	CosineDistance
)

// DistanceFunc represents a distance function
type DistanceFunc func(a, b []float32) float32

var (
	hasAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasFMA
	hasNEON = cpu.ARM64.HasASIMD
)

// GetDistanceFunc returns the appropriate distance function
func GetDistanceFunc(metric DistanceMetric) (DistanceFunc, error) {
	switch metric {
	case L2Distance:
		if hasAVX2 {
			return l2DistanceAVX2Checked, nil
		}
		if hasNEON {
			return l2DistanceNEONChecked, nil
		}
		return L2Distance_func, nil
	case InnerProduct:
		if hasAVX2 {
			return func(a, b []float32) float32 {
				checkVectorDimensions(a, b)
				return -simd.DotProductAVX2(a, b) // Negative for max-heap behavior
			}, nil
		}
		if hasNEON {
			return func(a, b []float32) float32 {
				checkVectorDimensions(a, b)
				return -simd.DotProductNEON(a, b)
			}, nil
		}
		return InnerProduct_func, nil
	case CosineDistance:
		if hasAVX2 {
			return func(a, b []float32) float32 {
				checkVectorDimensions(a, b)
				dist := 1.0 - simd.DotProductAVX2(a, b)
				if dist < 0 {
					return 0
				}
				return dist
			}, nil
		}
		if hasNEON {
			return func(a, b []float32) float32 {
				checkVectorDimensions(a, b)
				dist := 1.0 - simd.DotProductNEON(a, b)
				if dist < 0 {
					return 0
				}
				return dist
			}, nil
		}
		return CosineDistance_func, nil
	default:
		return nil, fmt.Errorf("unsupported distance metric: %v", metric)
	}
}

func checkVectorDimensions(a, b []float32) {
	if len(a) != len(b) {
		panic("vector dimensions must match")
	}
}

func l2DistanceAVX2Checked(a, b []float32) float32 {
	checkVectorDimensions(a, b)
	return simd.L2DistanceAVX2(a, b)
}

func l2DistanceNEONChecked(a, b []float32) float32 {
	checkVectorDimensions(a, b)
	return simd.L2DistanceNEON(a, b)
}

// L2Distance_func calculates Euclidean distance
func L2Distance_func(a, b []float32) float32 {
	checkVectorDimensions(a, b)

	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum // Return squared L2 distance as per Component 5 optimization
}

// InnerProduct_func calculates inner product (dot product)
func InnerProduct_func(a, b []float32) float32 {
	checkVectorDimensions(a, b)

	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return -sum // Negative for max-heap behavior in nearest neighbor search
}

// CosineDistance_func calculates cosine distance
func CosineDistance_func(a, b []float32) float32 {
	checkVectorDimensions(a, b)

	var dotProduct float32

	for i := range a {
		dotProduct += a[i] * b[i]
	}

	// Assuming vectors are pre-normalized to unit length (Component 5 optimization)
	dist := 1.0 - dotProduct // Convert similarity to distance
	if dist < 0 {
		return 0
	}
	return dist
}
