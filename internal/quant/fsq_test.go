package quant

import (
	"context"
	"math"
	"testing"
)

func TestFSQQuantizerCompressDecompress(t *testing.T) {
	vectors := [][]float32{
		{-1, 0, 1, 2},
		{-0.5, 0.25, 1.5, 2.5},
		{0, 0.5, 2, 3},
		{0.5, 0.75, 2.5, 3.5},
	}

	fq := NewFSQQuantizer()
	config := &QuantizationConfig{
		Type:       FiniteScalarQuantization,
		Bits:       4,
		TrainRatio: 1,
		Levels:     []int{16, 16, 8, 8},
	}
	if err := fq.Configure(config); err != nil {
		t.Fatalf("Configure: %v", err)
	}
	if err := fq.Train(context.Background(), vectors); err != nil {
		t.Fatalf("Train: %v", err)
	}

	compressed, err := fq.Compress(vectors[1])
	if err != nil {
		t.Fatalf("Compress: %v", err)
	}
	decompressed, err := fq.Decompress(compressed)
	if err != nil {
		t.Fatalf("Decompress: %v", err)
	}
	if len(decompressed) != len(vectors[1]) {
		t.Fatalf("dimension got %d want %d", len(decompressed), len(vectors[1]))
	}

	for i := range decompressed {
		if math.IsNaN(float64(decompressed[i])) || math.IsInf(float64(decompressed[i]), 0) {
			t.Fatalf("decompressed[%d] is not finite: %f", i, decompressed[i])
		}
		if decompressed[i] < fq.minValues[i]-1e-4 || decompressed[i] > fq.maxValues[i]+1e-4 {
			t.Fatalf("decompressed[%d]=%f outside trained range [%f,%f]", i, decompressed[i], fq.minValues[i], fq.maxValues[i])
		}
	}
}

func TestFSQQuantizerDistanceToQueryOrdersNearerVector(t *testing.T) {
	vectors := [][]float32{
		{0, 0, 0, 0},
		{1, 1, 1, 1},
		{4, 4, 4, 4},
		{8, 8, 8, 8},
	}
	fq := NewFSQQuantizer()
	if err := fq.Configure(&QuantizationConfig{Type: FiniteScalarQuantization, Bits: 6, TrainRatio: 1}); err != nil {
		t.Fatalf("Configure: %v", err)
	}
	if err := fq.Train(context.Background(), vectors); err != nil {
		t.Fatalf("Train: %v", err)
	}

	near, err := fq.Compress(vectors[1])
	if err != nil {
		t.Fatalf("Compress near: %v", err)
	}
	far, err := fq.Compress(vectors[3])
	if err != nil {
		t.Fatalf("Compress far: %v", err)
	}

	query := []float32{1.1, 1.1, 1.1, 1.1}
	nearDist, err := fq.DistanceToQuery(near, query, nil)
	if err != nil {
		t.Fatalf("near distance: %v", err)
	}
	farDist, err := fq.DistanceToQuery(far, query, nil)
	if err != nil {
		t.Fatalf("far distance: %v", err)
	}
	if nearDist >= farDist {
		t.Fatalf("near distance %f should be less than far distance %f", nearDist, farDist)
	}
}

func TestFSQQuantizerRegistered(t *testing.T) {
	factory := NewFSQQuantizerFactory()
	if !factory.Supports(FiniteScalarQuantization) {
		t.Fatal("FSQ factory should support FSQ")
	}
	quantizer, err := factory.Create(&QuantizationConfig{
		Type:       FiniteScalarQuantization,
		Bits:       4,
		TrainRatio: 1,
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, ok := quantizer.(*FSQQuantizer); !ok {
		t.Fatalf("Create returned %T, want *FSQQuantizer", quantizer)
	}
}
