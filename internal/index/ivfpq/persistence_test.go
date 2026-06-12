package ivfpq

import (
	"context"
	"fmt"
	"testing"
)

func TestSerializeDeserializeRoundTrip(t *testing.T) {
	dim := 32
	nClusters := 8
	nVectors := 200

	config := &Config{
		Dimension:     dim,
		NClusters:     nClusters,
		NProbes:       4,
		Metric:        0, // L2
		Quantization:  nil,
		MaxIterations: 20,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatal(err)
	}
	// No PQ for basic round-trip test — validates centroids + inverted lists.
	trainVecs := make([][]float32, nVectors)
	for i := range trainVecs {
		trainVecs[i] = make([]float32, dim)
		for j := range trainVecs[i] {
			trainVecs[i][j] = float32(i+j) / float32(nVectors+dim)
		}
	}
	if err := idx.Train(context.Background(), trainVecs); err != nil {
		t.Fatal(err)
	}
	// Insert training vectors.
	for i, vec := range trainVecs {
		entry := &VectorEntry{
			ID:     fmt.Sprintf("v%d", i),
			Vector: vec,
		}
		if err := idx.Insert(context.Background(), entry); err != nil {
			t.Fatal(err)
		}
	}

	data, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	if data == nil {
		t.Fatal("SerializeToBytes returned nil for trained index")
	}
	t.Logf("serialized size: %d bytes", len(data))

	// Deserialize into a fresh index with same config.
	idx2, err := NewIVFPQ(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := idx2.DeserializeFromBytes(context.Background(), data); err != nil {
		t.Fatal(err)
	}
	if !idx2.IsTrained() {
		t.Fatal("deserialized index not trained")
	}

	// Verify centroids match.
	for i := range idx.clusters {
		c1 := idx.clusters[i].Centroid
		c2 := idx2.clusters[i].Centroid
		if len(c1) != len(c2) {
			t.Fatalf("centroid %d length mismatch: %d vs %d", i, len(c1), len(c2))
		}
		for d := range c1 {
			if c1[d] != c2[d] {
				t.Fatalf("centroid %d[%d] mismatch: %f vs %f", i, d, c1[d], c2[d])
			}
		}
	}
	t.Log("centroids match")

	idx.Close()
	idx2.Close()
}

func TestSerializeUntrainedReturnsNil(t *testing.T) {
	config := &Config{Dimension: 16, NClusters: 4, NProbes: 2, MaxIterations: 10, Tolerance: 1e-4}
	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatal(err)
	}
	data, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	if data != nil {
		t.Fatal("expected nil for untrained index")
	}
	idx.Close()
}

func TestDeserializeRejectsCorruptData(t *testing.T) {
	config := &Config{Dimension: 16, NClusters: 4, NProbes: 2, MaxIterations: 10, Tolerance: 1e-4}
	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	// Bogus data.
	err = idx.DeserializeFromBytes(context.Background(), []byte{0x00, 0x01, 0x02, 0x03})
	if err == nil {
		t.Fatal("expected error for corrupt data")
	}
	t.Logf("corrupt data error: %v", err)
}

func TestDeserializeRejectsConfigMismatch(t *testing.T) {
	dim := 16
	config1 := &Config{Dimension: dim, NClusters: 4, NProbes: 2, MaxIterations: 10, Tolerance: 1e-4}
	idx1, err := NewIVFPQ(config1)
	if err != nil {
		t.Fatal(err)
	}
	trainVecs := make([][]float32, 100)
	for i := range trainVecs {
		trainVecs[i] = make([]float32, dim)
		trainVecs[i][0] = float32(i)
	}
	if err := idx1.Train(context.Background(), trainVecs); err != nil {
		t.Fatal(err)
	}
	data, err := idx1.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	idx1.Close()

	// Different config (dimension mismatch).
	config2 := &Config{Dimension: 32, NClusters: 4, NProbes: 2, MaxIterations: 10, Tolerance: 1e-4}
	idx2, err := NewIVFPQ(config2)
	if err != nil {
		t.Fatal(err)
	}
	defer idx2.Close()
	err = idx2.DeserializeFromBytes(context.Background(), data)
	if err == nil {
		t.Fatal("expected config mismatch error")
	}
	t.Logf("config mismatch error: %v", err)
}
