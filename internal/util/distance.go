package util

import (
	"fmt"
	"math"
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

// GetDistanceFunc returns the appropriate distance function
func GetDistanceFunc(metric DistanceMetric) (DistanceFunc, error) {
	switch metric {
	case L2Distance:
		return L2Distance_func, nil
	case InnerProduct:
		return InnerProduct_func, nil
	case CosineDistance:
		return CosineDistance_func, nil
	default:
		return nil, fmt.Errorf("unsupported distance metric: %v", metric)
	}
}

// L2Distance_func calculates Euclidean distance
func L2Distance_func(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vector dimensions must match")
	}

	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// InnerProduct_func calculates inner product (dot product)
func InnerProduct_func(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vector dimensions must match")
	}

	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return -sum // Negative for max-heap behavior in nearest neighbor search
}

// CosineDistance_func calculates cosine distance
func CosineDistance_func(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vector dimensions must match")
	}

	var dotProduct, normA, normB float32

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	normA = float32(math.Sqrt(float64(normA)))
	normB = float32(math.Sqrt(float64(normB)))

	if normA == 0 || normB == 0 {
		// Returning NaN surfaces zero-vector bugs in caller code rather than
		// silently reporting a meaningless 1.0 "max distance." NaN propagates
		// through any distance comparison and is the standard math convention
		// (matches numpy/scipy behavior on cosine divide-by-zero).
		return float32(math.NaN())
	}

	cosine := dotProduct / (normA * normB)
	return 1.0 - cosine // Convert similarity to distance
}
