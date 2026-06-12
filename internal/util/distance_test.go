package util

import (
	"math"
	"testing"
)

func TestCosineDistance_Identical(t *testing.T) {
	v := []float32{1, 2, 3}
	d := CosineDistance_func(v, v)
	if math.Abs(float64(d)) > 1e-6 {
		t.Errorf("identical vectors: want distance ~0, got %v", d)
	}
}

func TestCosineDistance_Opposite(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{-1, 0, 0}
	d := CosineDistance_func(a, b)
	if math.Abs(float64(d-2.0)) > 1e-6 {
		t.Errorf("opposite vectors: want distance 2.0, got %v", d)
	}
}

func TestCosineDistance_Orthogonal(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{0, 1, 0}
	d := CosineDistance_func(a, b)
	if math.Abs(float64(d-1.0)) > 1e-6 {
		t.Errorf("orthogonal vectors: want distance 1.0, got %v", d)
	}
}

func TestCosineDistance_ZeroVectorReturnsNaN(t *testing.T) {
	a := []float32{1, 2, 3}
	zero := []float32{0, 0, 0}
	for _, pair := range [][2][]float32{{a, zero}, {zero, a}, {zero, zero}} {
		d := CosineDistance_func(pair[0], pair[1])
		if !math.IsNaN(float64(d)) {
			t.Errorf("zero vector pair %v: want NaN, got %v", pair, d)
		}
	}
}
