package ivfpq

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"math"
	"testing"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

// --- v2 PQ round-trip with nil-config bridge target ---

func TestV2PQRoundTripNilConfig(t *testing.T) {
	const dim = 8
	const clusters = 2
	const codebooks = 2
	const bits = 4
	subDim := dim / codebooks

	// Build v2 payload by hand.
	pqCentroids := make([][][]float32, codebooks)
	for s := 0; s < codebooks; s++ {
		pqCentroids[s] = make([][]float32, 4)
		for c := 0; c < 4; c++ {
			pqCentroids[s][c] = make([]float32, subDim)
			for d := 0; d < subDim; d++ {
				pqCentroids[s][c][d] = float32(s*100 + c*10 + d)
			}
		}
	}
	centroids := make([][]float32, clusters)
	for i := 0; i < clusters; i++ {
		centroids[i] = make([]float32, dim)
		for d := 0; d < dim; d++ {
			centroids[i][d] = float32(i*10 + d)
		}
	}
	const n = 20
	codeSize := 1
	records := make([][]pendingRecord, clusters)
	for i := 0; i < n; i++ {
		ci := i % clusters
		records[ci] = append(records[ci], pendingRecord{
			ordinal: uint32(i + 1),
			code:    []byte{byte(i % 4)},
		})
	}

	data := buildV2Bytes(t, dim, clusters, codebooks, bits, subDim, centroids, pqCentroids, records)
	t.Logf("v2 bytes: %d", len(data))

	// Target with Quantization:nil (bridge pattern).
	cfg := &Config{Dimension: dim, NClusters: clusters, NProbes: clusters, Metric: util.L2Distance, MaxIterations: 20, Tolerance: 1e-4, RandomSeed: 7}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	if err := idx.DeserializeFromBytes(context.Background(), data); err != nil {
		t.Fatalf("v2 PQ nil-config: %v", err)
	}
	if idx.Size() != n {
		t.Fatalf("size=%d want=%d", idx.Size(), n)
	}
	if idx.gen.quantizer == nil || !idx.gen.quantizer.IsTrained() {
		t.Fatal("quantizer not trained")
	}
	if got := idx.codeSize(); got != codeSize {
		t.Fatalf("codeSize=%d want=%d", got, codeSize)
	}
}

// --- v2 no-quantizer (covered by TestSerializeDeserializeRoundTrip) ---

func TestV2NoQuantizerRoundTrip(t *testing.T) {
	// The existing TestSerializeDeserializeRoundTrip in persistence_test.go
	// validates nil-quantizer round trips through the v3 format writer.
	// This test validates that v3 nil-quant payloads round-trip correctly
	// when deserialized into a fresh target with Quantization:nil.
	const dim = 8
	cfg := &Config{Dimension: dim, NClusters: 2, NProbes: 2, Metric: util.L2Distance, MaxIterations: 20, Tolerance: 1e-4, RandomSeed: 7}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 8.0
		}
		if err := idx.Insert(context.Background(), &VectorEntry{ID: "nq", Ordinal: uint32(i + 1), Vector: v}); err != nil {
			t.Fatal(err)
		}
	}
	data, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	// Fresh target with matching config but nil Quantization.
	tgtCfg := &Config{Dimension: dim, NClusters: 2, NProbes: 2, Metric: util.L2Distance, MaxIterations: 20, Tolerance: 1e-4, RandomSeed: 7}
	tgt, err := NewIVFPQ(tgtCfg)
	if err != nil {
		t.Fatal(err)
	}
	defer tgt.Close()
	if err := tgt.DeserializeFromBytes(context.Background(), data); err != nil {
		t.Fatalf("v3 nil-quant: %v", err)
	}
	if tgt.Size() != 8 {
		t.Fatalf("size=%d", tgt.Size())
	}
	if tgt.gen.quantizer != nil {
		t.Fatal("unexpected quantizer on nil-quant target")
	}
}

// --- v3 PQ nil-config ---

func TestV3PQNilConfig(t *testing.T) {
	const dim = 8
	cfg := &Config{Dimension: dim, NClusters: 2, NProbes: 2, Metric: util.L2Distance,
		Quantization: &quant.QuantizationConfig{Type: quant.ProductQuantization, Codebooks: 2, Bits: 4, TrainRatio: 1},
		MaxIterations: 20, Tolerance: 1e-4, RandomSeed: 7,
	}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	train := make([][]float32, 128)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 30; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 30.0
		}
		if err := idx.Insert(context.Background(), &VectorEntry{ID: "x", Ordinal: uint32(i + 1), Vector: v}); err != nil {
			t.Fatal(err)
		}
	}
	data, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	// Nil-config target.
	tgtCfg := &Config{Dimension: dim, NClusters: 2, NProbes: 2, Metric: util.L2Distance, MaxIterations: 20, Tolerance: 1e-4, RandomSeed: 7}
	tgt, err := NewIVFPQ(tgtCfg)
	if err != nil {
		t.Fatal(err)
	}
	defer tgt.Close()
	if err := tgt.DeserializeFromBytes(context.Background(), data); err != nil {
		t.Fatalf("v3 PQ nil-config: %v", err)
	}
	if tgt.Size() != 30 {
		t.Fatalf("size=%d", tgt.Size())
	}
	if tgt.config.Quantization == nil {
		t.Fatal("Quantization not populated")
	}
}

// --- Scalar round-trip ---

func TestScalarRoundTrip(t *testing.T) {
	const dim = 8
	cfg := &Config{Dimension: dim, NClusters: 2, NProbes: 2, Metric: util.L2Distance,
		Quantization: &quant.QuantizationConfig{Type: quant.ScalarQuantization, Bits: 8, TrainRatio: 1},
		MaxIterations: 20, Tolerance: 1e-4, RandomSeed: 7,
	}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	train := make([][]float32, 128)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 40; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 40.0
		}
		if err := idx.Insert(context.Background(), &VectorEntry{ID: "s", Ordinal: uint32(i + 1), Vector: v}); err != nil {
			t.Fatal(err)
		}
	}
	data, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("scalar bytes: %d", len(data))

	tgt, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer tgt.Close()
	if err := tgt.DeserializeFromBytes(context.Background(), data); err != nil {
		t.Fatalf("scalar round-trip: %v", err)
	}
	if tgt.Size() != 40 {
		t.Fatalf("size=%d", tgt.Size())
	}
	if _, ok := tgt.gen.quantizer.(*quant.ScalarQuantizer); !ok {
		t.Fatalf("quantizer is %T", tgt.gen.quantizer)
	}
}

// --- FSQ round-trip ---

func TestFSQRoundTrip(t *testing.T) {
	const dim = 8
	cfg := &Config{Dimension: dim, NClusters: 2, NProbes: 2, Metric: util.L2Distance,
		Quantization: &quant.QuantizationConfig{Type: quant.FiniteScalarQuantization, Bits: 8, TrainRatio: 1},
		MaxIterations: 20, Tolerance: 1e-4, RandomSeed: 7,
	}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	train := make([][]float32, 128)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 30; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 30.0
		}
		if err := idx.Insert(context.Background(), &VectorEntry{ID: "f", Ordinal: uint32(i + 1), Vector: v}); err != nil {
			t.Fatal(err)
		}
	}
	data, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	tgt, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer tgt.Close()
	if err := tgt.DeserializeFromBytes(context.Background(), data); err != nil {
		t.Fatalf("FSQ round-trip: %v", err)
	}
	if tgt.Size() != 30 {
		t.Fatalf("size=%d", tgt.Size())
	}
	if _, ok := tgt.gen.quantizer.(*quant.FSQQuantizer); !ok {
		t.Fatalf("quantizer is %T", tgt.gen.quantizer)
	}
}

// --- Malformed/truncated rejection ---

func TestPersistenceRejectsMalformed(t *testing.T) {
	cfg := &Config{Dimension: 8, NClusters: 2, NProbes: 2, MaxIterations: 10, Tolerance: 1e-4}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	// short
	if err := idx.DeserializeFromBytes(context.Background(), []byte{1, 2, 3, 4}); err == nil {
		t.Fatal("expected error")
	}
	// truncated header
	bad := append([]byte(nil), ivfpqMagicBytes...)
	bad = append(bad, 0x00)
	if err := idx.DeserializeFromBytes(context.Background(), bad); err == nil {
		t.Fatal("expected error")
	}
}

func TestPersistenceRejectsV1(t *testing.T) {
	cfg := &Config{Dimension: 16, NClusters: 4, NProbes: 2, MaxIterations: 10, Tolerance: 1e-4}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	buf := make([]byte, 0, 1024)
	w := &sliceWriter{buf: buf}
	w.bytes(ivfpqMagicBytes)
	w.u16(1)
	w.u8(ivfpqIndexType)
	w.u8(0)
	w.u32(16)
	w.u32(4)
	w.u32(2)
	w.u8(0)
	w.u8(0)
	w.u32(0)
	w.u8(0)
	w.u32(10)
	w.f64(1e-4)
	w.i64(0)
	w.u32(4)
	for i := 0; i < 4; i++ {
		w.u32(uint32(i))
		w.u32(16)
		for d := 0; d < 16; d++ {
			w.f32(0)
		}
	}
	w.u32(0)
	w.u32(4)
	for i := 0; i < 4; i++ {
		w.u32(uint32(i))
		w.u32(0)
	}
	cs := crc32.Checksum(w.buf, crc32.MakeTable(crc32.Castagnoli))
	w.u32(cs)
	if err := idx.DeserializeFromBytes(context.Background(), w.buf); err == nil {
		t.Fatal("expected v1 rejection")
	}
}

func TestPersistenceRejectsTrailingBytes(t *testing.T) {
	const dim = 8
	cfg := &Config{Dimension: dim, NClusters: 2, NProbes: 2, Metric: util.L2Distance, Quantization: &quant.QuantizationConfig{Type: quant.ProductQuantization, Codebooks: 2, Bits: 4, TrainRatio: 1}, MaxIterations: 20, Tolerance: 1e-4, RandomSeed: 7}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	data, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	// Append trailing bytes with valid CRC.
	bad := append(data, 0x00, 0x01, 0x02, 0x03)
	if err := idx.DeserializeFromBytes(context.Background(), bad); err == nil {
		t.Fatal("expected trailing byte rejection")
	}
}

// --- v2 builder ---

func buildV2Bytes(t *testing.T, dim, clusters, codebooks, bits, subDim int, centroids [][]float32, pqCentroids [][][]float32, records [][]pendingRecord) []byte {
	t.Helper()
	buf := make([]byte, 0, 8*1024)
	w := &sliceWriter{buf: buf}
	w.bytes(ivfpqMagicBytes)
	w.u16(ivfpqFormatVersionLegacy)
	w.u8(ivfpqIndexType)
	w.u8(0)
	w.u32(uint32(dim))
	w.u32(uint32(clusters))
	w.u32(uint32(clusters))
	w.u8(uint8(util.L2Distance))
	if codebooks > 0 {
		w.u8(1)
		w.u32(uint32(codebooks))
		w.u8(uint8(bits))
	} else {
		w.u8(0)
		w.u32(0)
		w.u8(0)
	}
	w.u32(20)
	w.f64(1e-4)
	w.i64(0)
	// centroids
	w.u32(uint32(clusters))
	for i, c := range centroids {
		w.u32(uint32(i))
		w.u32(uint32(len(c)))
		for _, v := range c {
			w.f32(v)
		}
	}
	// PQ codebooks
	if pqCentroids != nil {
		w.u32(uint32(len(pqCentroids)))
		cps := 0
		if len(pqCentroids) > 0 {
			cps = len(pqCentroids[0])
		}
		w.u32(uint32(cps))
		w.u32(uint32(subDim))
		for _, ss := range pqCentroids {
			for _, c := range ss {
				for _, v := range c {
					w.f32(v)
				}
			}
		}
	} else {
		w.u32(0)
	}
	// inverted lists
	w.u32(uint32(clusters))
	for ci, recs := range records {
		w.u32(uint32(ci))
		w.u32(uint32(len(recs)))
		for _, rec := range recs {
			w.u32(rec.ordinal)
			w.u32(uint32(len(rec.code)))
			if len(rec.code) > 0 {
				w.raw(rec.code)
			}
		}
	}
	cs := crc32.Checksum(w.buf, crc32.MakeTable(crc32.Castagnoli))
	w.u32(cs)
	return w.buf
}


// --- hardening: truncation at multiple structural boundaries ---

func TestPersistenceV3TruncationBoundaries(t *testing.T) {
	const dim = 8
	cfg := &Config{Dimension: dim, NClusters: 2, NProbes: 2, Metric: util.L2Distance, Quantization: &quant.QuantizationConfig{Type: quant.ProductQuantization, Codebooks: 2, Bits: 4, TrainRatio: 1}, MaxIterations: 20, Tolerance: 1e-4, RandomSeed: 7}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	good, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	truncations := []struct {
		name   string
		offset int
	}{
		{"header", 8},
		{"version", 10},
		{"index_type", 12},
		{"flags", 13},
		{"dim", 14},
		{"clusters", 18},
		{"probes", 22},
		{"metric", 23},
		{"quant_tag", 24},
		{"quant_len", 25},
		{"maxIter", len(good) - 74},
		{"centroid_count", len(good) - 70},
		{"invlist_count", len(good) - 40},
		{"mid_codebook", len(good) - 50},
	}
	for _, tc := range truncations {
		t.Run(tc.name, func(t *testing.T) {
			trunc := make([]byte, tc.offset)
			copy(trunc, good)
			err := idx.DeserializeFromBytes(context.Background(), trunc)
			if err == nil {
				t.Fatalf("expected error for truncation at %d (%s)", tc.offset, tc.name)
			}
		})
	}
}

func TestPersistenceV2TruncationBoundaries(t *testing.T) {
	const dim = 8
	centroids := make([][]float32, 2)
	for i := range centroids {
		centroids[i] = make([]float32, dim)
		for d := range centroids[i] {
			centroids[i][d] = float32(i*10 + d)
		}
	}
	cfg := &Config{Dimension: dim, NClusters: 2, NProbes: 2, Metric: util.L2Distance, MaxIterations: 20, Tolerance: 1e-4, RandomSeed: 7}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	good := buildV2TruncTest(t, dim, 2, centroids, nil)
	for _, off := range []int{8, 10, 12, 14, 18, 22, 23, 24, 28, 29, 33, 100, 120} {
		if off > len(good) {
			continue
		}
		trunc := make([]byte, off)
		copy(trunc, good)
		if err := idx.DeserializeFromBytes(context.Background(), trunc); err == nil {
			t.Fatalf("expected truncation error at offset %d", off)
		}
	}
}

func buildV2TruncTest(t *testing.T, dim, clusters int, centroids [][]float32, records [][]pendingRecord) []byte {
	t.Helper()
	buf := make([]byte, 0, 1024)
	w := &sliceWriter{buf: buf}
	w.bytes(ivfpqMagicBytes)
	w.u16(ivfpqFormatVersionLegacy)
	w.u8(ivfpqIndexType)
	w.u8(0)
	w.u32(uint32(dim))
	w.u32(uint32(clusters))
	w.u32(uint32(clusters))
	w.u8(0)
	w.u8(0)
	w.u32(0)
	w.u8(0)
	w.u32(20)
	w.f64(1e-4)
	w.i64(0)
	w.u32(uint32(clusters))
	for i, c := range centroids {
		w.u32(uint32(i))
		w.u32(uint32(len(c)))
		for _, v := range c {
			w.f32(v)
		}
	}
	w.u32(0)
	w.u32(uint32(clusters))
	for ci := 0; ci < clusters; ci++ {
		w.u32(uint32(ci))
		w.u32(0)
	}
	cs := crc32.Checksum(w.buf, crc32.MakeTable(crc32.Castagnoli))
	w.u32(cs)
	return w.buf
}

// --- hostile values ---

func TestPersistenceRejectsHostileValues(t *testing.T) {
	for _, tc := range []struct {
		name string
		data []byte
	}{
		{"huge_dim_v3", hostileV3Field(0, 1<<20, 0, 0)},
		{"huge_clusters_v3", hostileV3Field(1, 0, 1<<20, 0)},
		{"huge_entries_v3", hostileV3Entry(2, 1<<30)},
		{"huge_qstate_len", hostileV3QStateLen(1 << 30)},
		{"bad_v2_codebooks", hostileV2BadCodebooks()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &Config{Dimension: 8, NClusters: 2, NProbes: 2, MaxIterations: 10, Tolerance: 1e-4}
			idx, err := NewIVFPQ(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer idx.Close()
			if err := idx.DeserializeFromBytes(context.Background(), tc.data); err == nil {
				t.Fatalf("expected rejection")
			}
		})
	}
}

func hostileV3Field(mode int, dim, clusters, probes uint32) []byte {
	// mode: 0=patch dim, 1=patch clusters, 2=patch probes
	buf := make([]byte, 0, 1024)
	w := &sliceWriter{buf: buf}
	w.bytes(ivfpqMagicBytes)
	w.u16(ivfpqFormatVersion)
	w.u8(ivfpqIndexType)
	w.u8(0)
	if mode == 0 {
		w.u32(dim)
	} else {
		w.u32(8)
	}
	if mode == 1 {
		w.u32(clusters)
	} else {
		w.u32(2)
	}
	if mode == 2 {
		w.u32(probes)
	} else {
		w.u32(2)
	}
	w.u8(0)
	w.u8(qTagNone)
	w.u32(20)
	w.f64(1e-4)
	w.i64(0)
	w.u32(2)
	for i := 0; i < 2; i++ {
		w.u32(uint32(i))
		w.u32(8)
		for d := 0; d < 8; d++ {
			w.f32(0)
		}
	}
	w.u32(2)
	for i := 0; i < 2; i++ {
		w.u32(uint32(i))
		w.u32(0)
	}
	cs := crc32.Checksum(w.buf, crc32.MakeTable(crc32.Castagnoli))
	w.u32(cs)
	return w.buf
}

func hostileV3Entry(clusters int, entries uint32) []byte {
	buf := make([]byte, 0, 1024)
	w := &sliceWriter{buf: buf}
	w.bytes(ivfpqMagicBytes)
	w.u16(ivfpqFormatVersion)
	w.u8(ivfpqIndexType)
	w.u8(0)
	w.u32(8)
	w.u32(uint32(clusters))
	w.u32(uint32(clusters))
	w.u8(0)
	w.u8(qTagNone)
	w.u32(20)
	w.f64(1e-4)
	w.i64(0)
	w.u32(uint32(clusters))
	for i := 0; i < clusters; i++ {
		w.u32(uint32(i))
		w.u32(8)
		for d := 0; d < 8; d++ {
			w.f32(0)
		}
	}
	w.u32(uint32(clusters))
	for i := 0; i < clusters; i++ {
		w.u32(uint32(i))
		w.u32(entries)
	}
	cs := crc32.Checksum(w.buf, crc32.MakeTable(crc32.Castagnoli))
	w.u32(cs)
	return w.buf
}

func hostileV3QStateLen(huge uint32) []byte {
	buf := make([]byte, 0, 1024)
	w := &sliceWriter{buf: buf}
	w.bytes(ivfpqMagicBytes)
	w.u16(ivfpqFormatVersion)
	w.u8(ivfpqIndexType)
	w.u8(0)
	w.u32(8)
	w.u32(2)
	w.u32(2)
	w.u8(0)
	w.u8(qTagPQ)
	w.u32(huge)
	w.u32(20)
	w.f64(1e-4)
	w.i64(0)
	w.u32(2)
	for i := 0; i < 2; i++ {
		w.u32(uint32(i))
		w.u32(8)
		for d := 0; d < 8; d++ {
			w.f32(0)
		}
	}
	w.u32(2)
	for i := 0; i < 2; i++ {
		w.u32(uint32(i))
		w.u32(0)
	}
	cs := crc32.Checksum(w.buf, crc32.MakeTable(crc32.Castagnoli))
	w.u32(cs)
	return w.buf
}

func hostileV2BadCodebooks() []byte {
	buf := make([]byte, 0, 1024)
	w := &sliceWriter{buf: buf}
	w.bytes(ivfpqMagicBytes)
	w.u16(ivfpqFormatVersionLegacy)
	w.u8(ivfpqIndexType)
	w.u8(0)
	w.u32(8)
	w.u32(2)
	w.u32(2)
	w.u8(0)
	w.u8(1)
	w.u32(3) // codebooks=3, but dim=8 → not divisible
	w.u8(4)
	w.u32(20)
	w.f64(1e-4)
	w.i64(0)
	w.u32(2)
	for i := 0; i < 2; i++ {
		w.u32(uint32(i))
		w.u32(8)
		for d := 0; d < 8; d++ {
			w.f32(0)
		}
	}
	w.u32(3)
	w.u32(4)
	w.u32(4)
	for s := 0; s < 3; s++ {
		for c := 0; c < 4; c++ {
			for d := 0; d < 4; d++ {
				w.f32(0)
			}
		}
	}
	w.u32(2)
	for i := 0; i < 2; i++ {
		w.u32(uint32(i))
		w.u32(0)
	}
	cs := crc32.Checksum(w.buf, crc32.MakeTable(crc32.Castagnoli))
	w.u32(cs)
	return w.buf
}

// --- quantizer state hardening ---

func TestPQDeserializeStateHardening(t *testing.T) {
	// Valid 2-subspace PQ state.
	cfg := &quant.QuantizationConfig{Type: quant.ProductQuantization, Codebooks: 2, Bits: 4, TrainRatio: 0.5, CacheSize: 100}
	q := quant.NewProductQuantizer()
	if err := q.Configure(cfg); err != nil {
		t.Fatal(err)
	}
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, 8)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := q.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	good, err := q.SerializeState()
	if err != nil {
		t.Fatal(err)
	}
	// Round trip OK.
	q2 := quant.NewProductQuantizer()
	if err := q2.DeserializeState(good); err != nil {
		t.Fatalf("valid round trip: %v", err)
	}
	q2.Close()

	// Truncated at various offsets.
	for _, off := range []int{4, 8, 16, 24, 30, len(good) - 8, len(good) - 1} {
		q3 := quant.NewProductQuantizer()
		err := q3.DeserializeState(good[:off])
		if err == nil {
			t.Fatalf("expected error for truncated PQ state at %d", off)
		}
		q3.Close()
	}
	// Trailing bytes.
	q4 := quant.NewProductQuantizer()
	excess := append([]byte(nil), good...)
	excess = append(excess, 0, 0, 0, 0)
	if err := q4.DeserializeState(excess); err == nil {
		t.Fatal("expected error for trailing bytes")
	}
	q4.Close()
}

func TestSQDeserializeStateHardening(t *testing.T) {
	cfg := &quant.QuantizationConfig{Type: quant.ScalarQuantization, Bits: 8, TrainRatio: 0.5}
	q := quant.NewScalarQuantizer()
	if err := q.Configure(cfg); err != nil {
		t.Fatal(err)
	}
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, 8)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := q.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	good, err := q.SerializeState()
	if err != nil {
		t.Fatal(err)
	}
	q2 := quant.NewScalarQuantizer()
	if err := q2.DeserializeState(good); err != nil {
		t.Fatalf("valid round trip: %v", err)
	}
	q2.Close()
	for _, off := range []int{4, 8, 12, len(good) - 16, len(good) - 1} {
		q3 := quant.NewScalarQuantizer()
		if err := q3.DeserializeState(good[:off]); err == nil {
			t.Fatalf("expected error at offset %d", off)
		}
		q3.Close()
	}
	q4 := quant.NewScalarQuantizer()
	excess := append([]byte(nil), good...)
	excess = append(excess, 1, 2, 3, 4)
	if err := q4.DeserializeState(excess); err == nil {
		t.Fatal("expected trailing byte error")
	}
	q4.Close()
}

func TestFSQDeserializeStateHardening(t *testing.T) {
	cfg := &quant.QuantizationConfig{Type: quant.FiniteScalarQuantization, Bits: 8, TrainRatio: 0.5}
	q := quant.NewFSQQuantizer()
	if err := q.Configure(cfg); err != nil {
		t.Fatal(err)
	}
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, 8)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := q.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	good, err := q.SerializeState()
	if err != nil {
		t.Fatal(err)
	}
	q2 := quant.NewFSQQuantizer()
	if err := q2.DeserializeState(good); err != nil {
		t.Fatalf("valid round trip: %v", err)
	}
	q2.Close()
	for _, off := range []int{4, 8, 12, len(good) - 36, len(good) - 1} {
		q3 := quant.NewFSQQuantizer()
		if err := q3.DeserializeState(good[:off]); err == nil {
			t.Fatalf("expected error at offset %d", off)
		}
		q3.Close()
	}
	q4 := quant.NewFSQQuantizer()
	excess := append([]byte(nil), good...)
	excess = append(excess, 0xff, 0xff)
	if err := q4.DeserializeState(excess); err == nil {
		t.Fatal("expected trailing byte error")
	}
	q4.Close()
}

// --- failed hydration leaves target unchanged ---

func TestFailedHydrationPreservesLiveState(t *testing.T) {
	const dim = 8
	cfg := &Config{Dimension: dim, NClusters: 2, NProbes: 2, Metric: util.L2Distance, Quantization: &quant.QuantizationConfig{Type: quant.ProductQuantization, Codebooks: 2, Bits: 4, TrainRatio: 1}, MaxIterations: 20, Tolerance: 1e-4, RandomSeed: 7}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 10.0
		}
		if err := idx.Insert(context.Background(), &VectorEntry{ID: fmt.Sprintf("pre-%d", i), Ordinal: uint32(i + 1), Vector: v}); err != nil {
			t.Fatal(err)
		}
	}
	sizeBefore := idx.Size()
	trainedBefore := idx.IsTrained()
	centBefore := make([]float32, dim)
	copy(centBefore, idx.gen.clusters[0].Centroid)
	qtBefore := false
	if idx.gen.quantizer != nil {
		qtBefore = idx.gen.quantizer.IsTrained()
	}

	// Attempt to hydrate with a truncated v3 payload.
	bad := make([]byte, 16)
	copy(bad, []byte("LIBRAIVF"))
	binary.LittleEndian.PutUint16(bad[8:], ivfpqFormatVersion)
	bad[10] = ivfpqIndexType
	if err := idx.DeserializeFromBytes(context.Background(), bad); err == nil {
		t.Fatal("expected error from truncated payload")
	}

	if idx.Size() != sizeBefore {
		t.Fatalf("size mutated: %d -> %d", sizeBefore, idx.Size())
	}
	if idx.IsTrained() != trainedBefore {
		t.Fatal("trained flag mutated")
	}
	for d, v := range centBefore {
		if idx.gen.clusters[0].Centroid[d] != v {
			t.Fatal("centroid mutated")
		}
	}
	qtAfter := false
	if idx.gen.quantizer != nil {
		qtAfter = idx.gen.quantizer.IsTrained()
	}
	if qtAfter != qtBefore {
		t.Fatal("quantizer trained state mutated")
	}
}

// suppress unused import warnings
var _ = binary.PutUvarint
var _ = math.MaxFloat32

// --- adversarial PQ shape regression ---

func TestPQDeserializeStateAdversarialShape(t *testing.T) {
	q := quant.NewProductQuantizer()
	defer q.Close()
	// dim=65536, subsp=1, subDim=65536, cps=65536
	// Implied centroid state = 32 + 1*65536*65536*4 = ~17 GB, far
	// exceeding the 64 MiB quantizer state ceiling. Rejected by the
	// byte cap, not by uint32 overflow (the multiplication is int64).
	buf := make([]byte, 0, 32)
	w := &sliceWriter{buf: buf}
	w.u32(65536) // dim
	w.u32(1)     // subsp
	w.u32(65536) // cps
	w.u32(65536) // subDim
	w.u32(4)     // bits
	w.f64(0.5)   // trainRatio
	w.u32(1000)  // cacheSize
	if err := q.DeserializeState(w.buf); err == nil {
		t.Fatal("adversarial PQ shape should be rejected by byte ceiling")
	}
}

// --- small payload, large entry count ---

func TestPersistenceRejectsLargeCountSmallPayload(t *testing.T) {
	cfg := &Config{Dimension: 8, NClusters: 2, NProbes: 2, MaxIterations: 10, Tolerance: 1e-4}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	// v3 payload with 2 clusters, first cluster claims 1000000 entries
	// but only has 20 bytes left.
	buf := make([]byte, 0, 1024)
	w := &sliceWriter{buf: buf}
	w.bytes(ivfpqMagicBytes)
	w.u16(ivfpqFormatVersion)
	w.u8(ivfpqIndexType)
	w.u8(0)
	w.u32(8)
	w.u32(2)
	w.u32(2)
	w.u8(0)
	w.u8(qTagNone)
	w.u32(20)
	w.f64(1e-4)
	w.i64(0)
	w.u32(2)
	for i := 0; i < 2; i++ {
		w.u32(uint32(i))
		w.u32(8)
		for d := 0; d < 8; d++ {
			w.f32(0)
		}
	}
	w.u32(2)
	w.u32(0)
	w.u32(1000000) // huge entry count
	cs := crc32.Checksum(w.buf, crc32.MakeTable(crc32.Castagnoli))
	w.u32(cs)
	err = idx.DeserializeFromBytes(context.Background(), w.buf)
	if err == nil {
		t.Fatal("expected rejection for huge entry count with no payload")
	}
}

// --- v2 PQ codebook byte ceiling ---

func TestV2PQCodebookExceedsCeiling(t *testing.T) {
	cfg := &Config{Dimension: 8, NClusters: 2, NProbes: 2, MaxIterations: 10, Tolerance: 1e-4}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	// v2 PQ with subspaces=1024, centroidsPerSS=65536, subDim=4
	// alloc = 1024*65536*4 = 268M floats → well over ceiling.
	buf := make([]byte, 0, 1024)
	w := &sliceWriter{buf: buf}
	w.bytes(ivfpqMagicBytes)
	w.u16(ivfpqFormatVersionLegacy)
	w.u8(ivfpqIndexType)
	w.u8(0)
	w.u32(4096) // dim = subspaces*subDim = 1024*4
	w.u32(2)
	w.u32(2)
	w.u8(0)
	w.u8(1)
	w.u32(1024)  // codebooks
	w.u8(4)      // bits
	w.u32(20)
	w.f64(1e-4)
	w.i64(0)
	w.u32(2)
	for i := 0; i < 2; i++ {
		w.u32(uint32(i))
		w.u32(4096)
		for d := 0; d < 4096; d++ {
			w.f32(0)
		}
	}
	w.u32(1024)    // subspaces
	w.u32(65536)   // centroidsPerSS
	w.u32(4)       // subDim
	cs := crc32.Checksum(w.buf, crc32.MakeTable(crc32.Castagnoli))
	w.u32(cs)
	if err := idx.DeserializeFromBytes(context.Background(), w.buf); err == nil {
		t.Fatal("expected rejection: v2 codebook ceiling exceeded")
	}
}

// --- FSQ invalid levels/bitWidths ---

func TestFSQDeserializeStateInvalidInvariants(t *testing.T) {
	q := quant.NewFSQQuantizer()
	defer q.Close()
	// Write valid header, then invalid levels/bitWidths.
	buf := make([]byte, 0, 16+8*36)
	w := &sliceWriter{buf: buf}
	w.u32(8)     // dim
	w.u32(8)     // bits
	w.f64(0.5)   // trainRatio
	// levels: write 5 (bitWidth should be 2), but write bitWidth=1
	for i := 0; i < 8; i++ {
		w.u32(5)   // level=5, bitsForLevel(5)==3
	}
	for i := 0; i < 8; i++ {
		w.u32(1)   // wrong: expected 3, got 1
	}
	// Fill remaining f32 arrays with zeros.
	for i := 0; i < 8*7; i++ {
		w.f32(0)
	}
	if err := q.DeserializeState(w.buf); err == nil {
		t.Fatal("expected FSQ bitWidth invariant rejection")
	}
}

// --- large-code round trips (regression for 256-byte cap) ---

func TestScalarLargeCodeRoundTrip(t *testing.T) {
	const dim = 384
	const bits = 8
	codeSz := (dim*bits + 7) / 8 // 384
	if codeSz <= 256 {
		t.Fatalf("test precondition: code size %d must be >256", codeSz)
	}
	cfg := &Config{Dimension: dim, NClusters: 2, NProbes: 2, Metric: util.L2Distance,
		Quantization: &quant.QuantizationConfig{Type: quant.ScalarQuantization, Bits: bits, TrainRatio: 0.5},
		MaxIterations: 10, Tolerance: 1e-4, RandomSeed: 7,
	}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 10.0
		}
		if err := idx.Insert(context.Background(), &VectorEntry{ID: "L", Ordinal: uint32(i + 1), Vector: v}); err != nil {
			t.Fatal(err)
		}
	}
	data, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	tgt, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer tgt.Close()
	if err := tgt.DeserializeFromBytes(context.Background(), data); err != nil {
		t.Fatalf("large scalar round-trip: %v", err)
	}
	if tgt.Size() != 10 {
		t.Fatalf("size=%d", tgt.Size())
	}
	if tgt.codeSize() != codeSz {
		t.Fatalf("codeSize=%d want=%d", tgt.codeSize(), codeSz)
	}
}

func TestFSQLargeCodeRoundTrip(t *testing.T) {
	const dim = 512
	const bits = 8
	// FSQ: bits=8 → levels=256 per dim, bitWidths=8 each. CodeSize = ceil(512*8/8) = 512.
	codeSz := 512
	if codeSz <= 256 {
		t.Fatalf("test precondition: code size %d must be >256", codeSz)
	}
	cfg := &Config{Dimension: dim, NClusters: 2, NProbes: 2, Metric: util.L2Distance,
		Quantization: &quant.QuantizationConfig{Type: quant.FiniteScalarQuantization, Bits: bits, TrainRatio: 0.5},
		MaxIterations: 10, Tolerance: 1e-4, RandomSeed: 7,
	}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 8.0
		}
		if err := idx.Insert(context.Background(), &VectorEntry{ID: "LF", Ordinal: uint32(i + 1), Vector: v}); err != nil {
			t.Fatal(err)
		}
	}
	data, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	tgt, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer tgt.Close()
	if err := tgt.DeserializeFromBytes(context.Background(), data); err != nil {
		t.Fatalf("large FSQ round-trip: %v", err)
	}
	if tgt.Size() != 8 {
		t.Fatalf("size=%d", tgt.Size())
	}
	if tgt.codeSize() != codeSz {
		t.Fatalf("codeSize=%d want=%d", tgt.codeSize(), codeSz)
	}
}

// --- code length exceeds format ceiling ---

func TestPersistenceRejectsCodeLenAboveFormatMax(t *testing.T) {
	cfg := &Config{Dimension: 8, NClusters: 2, NProbes: 2, MaxIterations: 10, Tolerance: 1e-4}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	// v3 payload with a code whose length is 262145 (above format max 262144).
	const aboveMax = 262145
	buf := make([]byte, 0, 1024)
	w := &sliceWriter{buf: buf}
	w.bytes(ivfpqMagicBytes)
	w.u16(ivfpqFormatVersion)
	w.u8(ivfpqIndexType)
	w.u8(0)
	w.u32(8)
	w.u32(2)
	w.u32(2)
	w.u8(0)
	w.u8(qTagNone)
	w.u32(20)
	w.f64(1e-4)
	w.i64(0)
	w.u32(2)
	for i := 0; i < 2; i++ {
		w.u32(uint32(i))
		w.u32(8)
		for d := 0; d < 8; d++ {
			w.f32(0)
		}
	}
	w.u32(2)
	w.u32(0)
	w.u32(1)             // 1 entry
	w.u32(1)             // ordinal
	w.u32(uint32(aboveMax)) // code length above ceiling
	cs := crc32.Checksum(w.buf, crc32.MakeTable(crc32.Castagnoli))
	w.u32(cs)
	if err := idx.DeserializeFromBytes(context.Background(), w.buf); err == nil {
		t.Fatal("expected rejection for code length above format max")
	}
}

// --- FSQ custom-level round trip (>16 bits) ---

func TestFSQCustomLevelRoundTrip(t *testing.T) {
	const dim = 512
	// Alternating levels: odd dims use 1<<17 (17 bits), even dims use 8 bits.
	const highLevel = 1 << 17
	const lowLevel = 256
	levels := make([]int, dim)
	expanded := make([]int, dim)
	for i := range levels {
		if i%2 == 0 {
			levels[i] = highLevel
		} else {
			levels[i] = lowLevel
		}
		// The quantizer expands Levels via d % len(Levels); since we provide
		// a full dim-length slice, each entry maps 1:1.
		expanded[i] = levels[i]
	}
	// bitsForLevel(highLevel) = 17, bitsForLevel(lowLevel) = 8.
	// total bits = 256*17 + 256*8 = 4352 + 2048 = 6400; 6400/8 = 800.
	totalBits := 256*17 + 256*8
	expectedCS := (totalBits + 7) / 8
	t.Logf("codeSize=%d totalBits=%d", expectedCS, totalBits)

	cfg := &Config{Dimension: dim, NClusters: 2, NProbes: 2, Metric: util.L2Distance,
		Quantization: &quant.QuantizationConfig{Type: quant.FiniteScalarQuantization, Bits: 8, TrainRatio: 0.5, Levels: levels},
		MaxIterations: 10, Tolerance: 1e-4, RandomSeed: 7,
	}
	idx, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	train := make([][]float32, 64)
	for i := range train {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / 32.0
		}
		train[i] = v
	}
	if err := idx.Train(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	const n = 6
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i+j) / float32(n)
		}
		if err := idx.Insert(context.Background(), &VectorEntry{ID: fmt.Sprintf("CL%d", i), Ordinal: uint32(i + 1), Vector: v}); err != nil {
			t.Fatal(err)
		}
	}
	data, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	tgt, err := NewIVFPQ(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer tgt.Close()
	if err := tgt.DeserializeFromBytes(context.Background(), data); err != nil {
		t.Fatalf("custom-level FSQ round-trip: %v", err)
	}
	if tgt.Size() != n {
		t.Fatalf("size=%d", tgt.Size())
	}
	if tgt.codeSize() != expectedCS {
		t.Fatalf("codeSize=%d want=%d", tgt.codeSize(), expectedCS)
	}

	// Verify config.Levels restored.
	fq, ok := tgt.gen.quantizer.(*quant.FSQQuantizer)
	if !ok {
		t.Fatal("quantizer not FSQ")
	}
	qCfg := fq.Config()
	if len(qCfg.Levels) != dim {
		t.Fatalf("restored Levels len=%d want=%d", len(qCfg.Levels), dim)
	}
	for i := 0; i < dim; i++ {
		if qCfg.Levels[i] != expanded[i] {
			t.Fatalf("restored Levels[%d]=%d want=%d", i, qCfg.Levels[i], expanded[i])
		}
	}

	// Search parity: source and restored must return the same
	// ordinals in the same order.
	q := make([]float32, dim)
	for j := range q {
		q[j] = 0.5
	}
	srcResult, err := idx.Search(context.Background(), q, n, nil)
	if err != nil {
		t.Fatalf("source Search: %v", err)
	}
	dstResult, err := tgt.Search(context.Background(), q, n, nil)
	if err != nil {
		t.Fatalf("target Search: %v", err)
	}
	if len(srcResult) != len(dstResult) {
		t.Fatalf("result count: src=%d dst=%d", len(srcResult), len(dstResult))
	}
	for i := range srcResult {
		if srcResult[i].Ordinal != dstResult[i].Ordinal {
			t.Fatalf("result[%d] ordinal: src=%d dst=%d", i, srcResult[i].Ordinal, dstResult[i].Ordinal)
		}
	}
}

// --- FSQ level exceeds uint32 max rejected ---

func TestFSQExceedsUint32MaxRejected(t *testing.T) {
	bigLevel := int64(math.MaxUint32) + 1
	// Test directly against the FSQ quantizer Configure so the asserted
	// error source is unambiguous (not generic bits validation).
	cfg := &quant.QuantizationConfig{Type: quant.FiniteScalarQuantization, Bits: 8, TrainRatio: 0.5,
		Levels: []int{int(bigLevel)},
	}
	q := quant.NewFSQQuantizer()
	err := q.Configure(cfg)
	q.Close()
	if err == nil {
		t.Fatal("expected FSQ rejection of level exceeding uint32 max")
	}
	t.Logf("FSQ Configure rejected: %v", err)
}

// Required for test to compile.
func bitsForLevel(level int) int {
	bits := 0
	value := level - 1
	for value > 0 {
		bits++
		value >>= 1
	}
	if bits == 0 {
		return 1
	}
	return bits
}
