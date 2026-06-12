package libravdb

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"math/rand"
	"os"
	"testing"
)

// restartBenchConfig holds the dataset parameters for restart benchmarks.
type restartBenchConfig struct {
	dim       int
	count     int
	indexType IndexType
	// IVF-PQ specific
	nClusters  int
	nProbes    int
	nSubspaces int // PQ codebooks
	bits       int // PQ bits
}

// BenchmarkRestartPersisted measures New() time when a valid persisted index chunk
// exists — the fast path (deserialize only). Reopens the same file multiple times.
func BenchmarkRestartPersisted(b *testing.B) {
	cfg := restartBenchConfig{dim: 128, count: 1000, indexType: HNSW}
	benchmarkRestartPersisted(b, cfg)
}

// BenchmarkRestartRebuild measures New() time when the index chunk is corrupt
// and the engine must rebuild indexes from Records — the fallback path.
// Runs a single iteration because the rebuild mutates the file.
func BenchmarkRestartRebuild(b *testing.B) {
	cfg := restartBenchConfig{dim: 128, count: 1000, indexType: HNSW}
	benchmarkRestartRebuild(b, cfg)
}

// BenchmarkRestartFlatPersisted measures Flat index restart with valid persistence.
func BenchmarkRestartFlatPersisted(b *testing.B) {
	cfg := restartBenchConfig{dim: 128, count: 1000, indexType: Flat}
	benchmarkRestartPersisted(b, cfg)
}

// BenchmarkRestartFlatRebuild measures Flat index restart with forced rebuild.
func BenchmarkRestartFlatRebuild(b *testing.B) {
	cfg := restartBenchConfig{dim: 128, count: 1000, indexType: Flat}
	benchmarkRestartRebuild(b, cfg)
}

// BenchmarkRestartIVFPQPersisted measures IVF-PQ index restart with valid
// persisted index chunk — centroids, codebooks, and inverted lists are
// deserialized without retraining.
func BenchmarkRestartIVFPQPersisted(b *testing.B) {
	cfg := restartBenchConfig{
		dim: 128, count: 1000, indexType: IVFPQ,
		nClusters: 16, nProbes: 4, nSubspaces: 16, bits: 8,
	}
	benchmarkRestartPersisted(b, cfg)
}

// BenchmarkRestartIVFPQRebuild measures IVF-PQ index restart when the index
// chunk is corrupt — the engine falls back to retraining k-means and PQ from
// storage records.
func BenchmarkRestartIVFPQRebuild(b *testing.B) {
	cfg := restartBenchConfig{
		dim: 128, count: 1000, indexType: IVFPQ,
		nClusters: 16, nProbes: 4, nSubspaces: 16, bits: 8,
	}
	benchmarkRestartRebuild(b, cfg)
}

// BenchmarkRestartIVFPQ10KPersisted measures IVF-PQ persisted reopen at 10K vectors.
func BenchmarkRestartIVFPQ10KPersisted(b *testing.B) {
	cfg := restartBenchConfig{
		dim: 128, count: 10000, indexType: IVFPQ,
		nClusters: 64, nProbes: 16, nSubspaces: 16, bits: 8,
	}
	benchmarkRestartPersisted(b, cfg)
}

// BenchmarkRestartIVFPQ10KRebuild measures IVF-PQ rebuild at 10K vectors.
func BenchmarkRestartIVFPQ10KRebuild(b *testing.B) {
	cfg := restartBenchConfig{
		dim: 128, count: 10000, indexType: IVFPQ,
		nClusters: 64, nProbes: 16, nSubspaces: 16, bits: 8,
	}
	benchmarkRestartRebuild(b, cfg)
}

// BenchmarkRestartIVFPQ50KPersisted measures IVF-PQ persisted reopen at 50K vectors.
func BenchmarkRestartIVFPQ50KPersisted(b *testing.B) {
	cfg := restartBenchConfig{
		dim: 128, count: 50000, indexType: IVFPQ,
		nClusters: 128, nProbes: 32, nSubspaces: 16, bits: 8,
	}
	benchmarkRestartPersisted(b, cfg)
}

// BenchmarkRestartIVFPQ50KRebuild measures IVF-PQ rebuild at 50K vectors.
func BenchmarkRestartIVFPQ50KRebuild(b *testing.B) {
	cfg := restartBenchConfig{
		dim: 128, count: 50000, indexType: IVFPQ,
		nClusters: 128, nProbes: 32, nSubspaces: 16, bits: 8,
	}
	benchmarkRestartRebuild(b, cfg)
}

func benchmarkRestartPersisted(b *testing.B, cfg restartBenchConfig) {
	b.Helper()
	ctx := context.Background()

	path := testDBPathBench(b)
	_ = createAndCompact(b, ctx, path, cfg)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		db, err := Open(WithStoragePath(path))
		if err != nil {
			b.Fatalf("reopen: %v", err)
		}
		db.Close()
	}

	b.ReportMetric(float64(cfg.count), "vectors")
	if cfg.indexType == IVFPQ {
		b.ReportMetric(float64(cfg.nClusters), "nClusters")
		b.ReportMetric(float64(cfg.nSubspaces), "subspaces")
		b.ReportMetric(float64(cfg.bits), "bits")
	}
}

func benchmarkRestartRebuild(b *testing.B, cfg restartBenchConfig) {
	b.Helper()
	ctx := context.Background()

	path := testDBPathBench(b)
	_ = createAndCompact(b, ctx, path, cfg)
	corruptIndexChunk(b, path)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		copyPath := path + ".copy"
		copyFile(b, path, copyPath)

		b.StartTimer()
		db, err := Open(WithStoragePath(copyPath))
		if err != nil {
			b.Fatalf("reopen: %v", err)
		}
		b.StopTimer()

		db.Close()
		os.Remove(copyPath)
	}

	b.ReportMetric(float64(cfg.count), "vectors")
	if cfg.indexType == IVFPQ {
		b.ReportMetric(float64(cfg.nClusters), "nClusters")
		b.ReportMetric(float64(cfg.nSubspaces), "subspaces")
		b.ReportMetric(float64(cfg.bits), "bits")
	}
}

func createAndCompact(tb testing.TB, ctx context.Context, path string, cfg restartBenchConfig) *Database {
	tb.Helper()

	db, err := Open(WithStoragePath(path))
	if err != nil {
		tb.Fatalf("new db: %v", err)
	}

	var opts []CollectionOption
	opts = append(opts, WithDimension(cfg.dim), WithMetric(CosineDistance))
	switch cfg.indexType {
	case HNSW:
		opts = append(opts, WithHNSW(16, 100, 50))
	case Flat:
		opts = append(opts, WithFlat())
	case IVFPQ:
		nClusters := cfg.nClusters
		if nClusters <= 0 {
			nClusters = 16
		}
		nProbes := cfg.nProbes
		if nProbes <= 0 {
			nProbes = 4
		}
		opts = append(opts, WithIVFPQ(nClusters, nProbes))
		if cfg.nSubspaces > 0 && cfg.bits > 0 {
			opts = append(opts, WithProductQuantization(cfg.nSubspaces, cfg.bits, 1.0))
		}
	default:
		tb.Fatalf("unsupported index type: %v", cfg.indexType)
	}

	col, err := db.CreateCollection(ctx, "restart_bench", opts...)
	if err != nil {
		db.Close()
		tb.Fatalf("create collection: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	batch := make([]VectorEntry, 0, 500)
	for i := 0; i < cfg.count; i++ {
		vec := make([]float32, cfg.dim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		batch = append(batch, VectorEntry{
			ID:     fmt.Sprintf("vec_%d", i),
			Vector: vec,
		})
		if len(batch) == 500 || i == cfg.count-1 {
			if err := col.InsertBatch(ctx, batch); err != nil {
				db.Close()
				tb.Fatalf("insert batch: %v", err)
			}
			batch = batch[:0]
		}
	}

	if err := db.storage.(interface{ Compact() error }).Compact(); err != nil {
		db.Close()
		tb.Fatalf("compact: %v", err)
	}
	db.Close()
	return db
}

// corruptIndexChunk flips a byte in the index chunk data on disk, simulating
// data corruption (e.g. bit rot). The metapage CRC remains valid, but the
// index block checksum no longer matches, so the engine falls back to
// rebuildIndexesFromRecords on restart.
func corruptIndexChunk(tb testing.TB, path string) {
	tb.Helper()
	f, err := os.OpenFile(path, os.O_RDWR, 0644)
	if err != nil {
		tb.Fatalf("open for corruption: %v", err)
	}
	defer f.Close()

	// Page size is always 4096.
	const pageSize = 4096

	// Read header (page 0) to find active metapage.
	headerBuf := make([]byte, pageSize)
	if _, err := f.ReadAt(headerBuf, 0); err != nil {
		tb.Fatalf("read header: %v", err)
	}
	formatVersion := binary.LittleEndian.Uint16(headerBuf[8:10])
	var activeMetaPage uint64
	if formatVersion >= 2 {
		activeMetaPage = binary.LittleEndian.Uint64(headerBuf[64:72])
	} else {
		activeMetaPage = binary.LittleEndian.Uint64(headerBuf[40:48])
	}

	// Read the active metapage.
	metaBuf := make([]byte, pageSize)
	if _, err := f.ReadAt(metaBuf, int64(activeMetaPage)*pageSize); err != nil {
		tb.Fatalf("read active metapage: %v", err)
	}

	// Verify metapage CRC to ensure we're reading correctly.
	if crc32.Checksum(metaBuf[:88], crc32.MakeTable(crc32.Castagnoli)) != binary.LittleEndian.Uint32(metaBuf[88:92]) {
		// Active metapage is corrupt — try the other one.
		otherPage := uint64(1)
		if activeMetaPage == 1 {
			otherPage = 2
		}
		if _, err := f.ReadAt(metaBuf, int64(otherPage)*pageSize); err != nil {
			tb.Fatalf("read fallback metapage: %v", err)
		}
		if crc32.Checksum(metaBuf[:88], crc32.MakeTable(crc32.Castagnoli)) != binary.LittleEndian.Uint32(metaBuf[88:92]) {
			tb.Fatalf("no valid metapage found")
		}
	}

	indexOffset := binary.LittleEndian.Uint64(metaBuf[68:76])
	indexLength := binary.LittleEndian.Uint64(metaBuf[76:84])
	if indexOffset == 0 || indexLength == 0 {
		tb.Fatalf("no index chunk to corrupt")
	}

	// Corrupt one byte in the index chunk payload (past the chunk header).
	chunkHeaderSize := int64(16) // chunkMagic(4)+kind(2)+version(2)+payloadLen(4)+checksum(4)
	corruptPos := int64(indexOffset) + chunkHeaderSize
	if _, err := f.WriteAt([]byte{0xFF}, corruptPos); err != nil {
		tb.Fatalf("corrupt index chunk at %d: %v", corruptPos, err)
	}

	if err := f.Sync(); err != nil {
		tb.Fatalf("sync after corruption: %v", err)
	}
}

func copyFile(tb testing.TB, src, dst string) {
	tb.Helper()
	data, err := os.ReadFile(src)
	if err != nil {
		tb.Fatalf("read src: %v", err)
	}
	if err := os.WriteFile(dst, data, 0644); err != nil {
		tb.Fatalf("write dst: %v", err)
	}
}
