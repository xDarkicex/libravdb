package ivfpq

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"math"
	"os"

	"github.com/xDarkicex/libravdb/internal/quant"
)

// ---------------------------------------------------------------------------
const (
	ivfpqFormatVersion uint16 = 2
	ivfpqIndexType     uint8  = 2 // matches index.IndexType(IVFPQ)
)

var ivfpqMagicBytes = []byte("LIBRAIVF")

// ivfpqClusterMeta stores per-cluster entry metadata deserialized from the
// inverted lists section. Each entry records its ordinal and compressed PQ codes.
type ivfpqClusterMeta struct {
	entries []ivfpqEntryMeta
}

// ivfpqEntryMeta pairs an ordinal with its compressed PQ representation.
type ivfpqEntryMeta struct {
	compressed []byte
	ordinal    uint32
}

// deserializedMeta stores per-cluster entry metadata between DeserializeFromBytes
// and PopulateEntriesFromStorage. Cluster membership is authoritative — it was
// captured during serialize, so PopulateEntriesFromStorage can place entries
// directly by ordinal without recomputing centroid distances.
type deserializedMeta struct {
	clusters []ivfpqClusterMeta
}

// SerializeToBytes serializes all trained IVF-PQ artifacts to a byte slice.
// Returns (nil, nil) if the index is not trained.
func (idx *Index) SerializeToBytes() ([]byte, error) {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()

	if !idx.trained {
		return nil, nil
	}

	buf := make([]byte, 0, 64*1024)
	w := &sliceWriter{buf: buf}

	// ---- Header ----
	w.bytes(ivfpqMagicBytes)
	w.u16(ivfpqFormatVersion)
	w.u8(ivfpqIndexType)
	w.u8(0) // flags (reserved)

	// ---- Config ----
	w.u32(uint32(idx.config.Dimension))
	w.u32(uint32(idx.config.NClusters))
	w.u32(uint32(idx.config.NProbes))
	w.u8(uint8(idx.config.Metric))
	if idx.quantizer != nil && idx.config.Quantization != nil {
		w.u8(1) // quantType = PQ
		w.u32(uint32(idx.config.Quantization.Codebooks))
		w.u8(uint8(idx.config.Quantization.Bits))
	} else {
		w.u8(0)
		w.u32(0)
		w.u8(0)
	}
	w.u32(uint32(idx.config.MaxIterations))
	w.f64(idx.config.Tolerance)
	w.i64(idx.config.RandomSeed)

	// ---- Coarse centroids ----
	w.u32(uint32(len(idx.clusters)))
	for _, cluster := range idx.clusters {
		w.u32(uint32(cluster.ID))
		w.u32(uint32(len(cluster.Centroid)))
		for _, v := range cluster.Centroid {
			w.f32(v)
		}
	}

	// ---- PQ codebooks ----
	if idx.quantizer != nil && idx.config.Quantization != nil {
		codebooks := extractCodebooks(idx.quantizer)
		subspaces := len(codebooks)
		subDim := idx.config.Dimension / idx.config.Quantization.Codebooks
		w.u32(uint32(subspaces))
		if subspaces > 0 && len(codebooks[0]) > 0 {
			centroidsPerSS := uint32(len(codebooks[0]))
			w.u32(centroidsPerSS)
			w.u32(uint32(subDim))
			for _, subspace := range codebooks {
				for _, centroid := range subspace {
					for _, v := range centroid {
						w.f32(v)
					}
				}
			}
		} else {
			w.u32(0)
			w.u32(0)
		}
	} else {
		w.u32(0)
	}

	// ---- Inverted lists: (ordinal, compressed) per cluster ----
	w.u32(uint32(len(idx.clusters)))
	for _, cluster := range idx.clusters {
		cluster.mutex.RLock()
		w.u32(uint32(cluster.ID))
		w.u32(uint32(len(cluster.Entries)))
		for _, entry := range cluster.Entries {
			compressed := cluster.CompressedVectors[entry.ID]
			w.u32(entry.Ordinal)
			w.u32(uint32(len(compressed)))
			if len(compressed) > 0 {
				w.raw(compressed)
			}
		}
		cluster.mutex.RUnlock()
	}

	// ---- Footer: CRC32 Castagnoli over all preceding bytes ----
	checksum := crc32.Checksum(w.buf, crc32.MakeTable(crc32.Castagnoli))
	w.u32(checksum)

	return w.buf, nil
}

// DeserializeFromBytes restores trained IVF-PQ centroids and codebooks from
// serialized bytes without retraining. Cluster entries and compressed vectors
// are stored in idx.deserMeta for later population via PopulateEntriesFromStorage.
func (idx *Index) DeserializeFromBytes(ctx context.Context, data []byte) error {
	idx.mutex.Lock()
	defer idx.mutex.Unlock()

	// Clear previous idToCluster map — rebuilt by PopulateEntriesFromStorage.
	idx.idToCluster.Range(func(key, _ any) bool {
		idx.idToCluster.Delete(key)
		return true
	})

	if len(data) < 20 {
		return fmt.Errorf("IVF-PQ data too short: %d bytes", len(data))
	}

	// Validate checksum before consuming any data to avoid panics on corrupt input.
	payloadEnd := len(data) - 4
	expected := crc32.Checksum(data[:payloadEnd], crc32.MakeTable(crc32.Castagnoli))
	got := binary.LittleEndian.Uint32(data[payloadEnd:])
	if got != expected {
		return fmt.Errorf("IVF-PQ checksum mismatch: got %08X, expected %08X", got, expected)
	}

	r := &sliceReader{buf: data}

	// ---- Header ----
	magic, err := r.bytes(8)
	if err != nil {
		return fmt.Errorf("failed reading magic: %w", err)
	}
	if string(magic) != string(ivfpqMagicBytes) {
		return fmt.Errorf("invalid IVF-PQ magic bytes: got %q, expected %q", magic, ivfpqMagicBytes)
	}
	version, err := r.u16()
	if err != nil {
		return err
	}
	if version == 1 {
		return fmt.Errorf("IVF-PQ format v1 is obsolete and contained a positional compressed-code matching bug; rebuild the index from records")
	}
	if version != ivfpqFormatVersion {
		return fmt.Errorf("unsupported IVF-PQ format version: %d, expected %d", version, ivfpqFormatVersion)
	}
	indexType, err := r.u8()
	if err != nil {
		return err
	}
	if indexType != ivfpqIndexType {
		return fmt.Errorf("index type mismatch: stored %d, expected %d", indexType, ivfpqIndexType)
	}
	if _, err := r.u8(); err != nil { // flags
		return err
	}

	// ---- Config (validate compatibility) ----
	dimV, err := r.u32()
	if err != nil {
		return err
	}
	dim := int(dimV)
	nClustersV, err := r.u32()
	if err != nil {
		return err
	}
	nClusters := int(nClustersV)
	nProbesV, err := r.u32()
	if err != nil {
		return err
	}
	nProbes := int(nProbesV)
	metricV, err := r.u8()
	if err != nil {
		return err
	}
	metric := metricV
	quantTypeV, err := r.u8()
	if err != nil {
		return err
	}
	quantType := quantTypeV
	storedCodebooksV, err := r.u32()
	if err != nil {
		return err
	}
	storedCodebooks := int(storedCodebooksV)
	storedBitsV, err := r.u8()
	if err != nil {
		return err
	}
	storedBits := int(storedBitsV)
	storedMaxIterV, err := r.u32()
	if err != nil {
		return err
	}
	storedMaxIter := int(storedMaxIterV)
	if _, err := r.f64(); err != nil { // tolerance (restore, not validate)
		return err
	}
	if _, err := r.i64(); err != nil { // randomSeed (restore, not validate)
		return err
	}

	if dim != idx.config.Dimension {
		return fmt.Errorf("config mismatch: dimension stored=%d current=%d", dim, idx.config.Dimension)
	}
	if nClusters != idx.config.NClusters {
		return fmt.Errorf("config mismatch: nClusters stored=%d current=%d", nClusters, idx.config.NClusters)
	}
	if nProbes != idx.config.NProbes {
		return fmt.Errorf("config mismatch: nProbes stored=%d current=%d", nProbes, idx.config.NProbes)
	}
	if metric != uint8(idx.config.Metric) {
		return fmt.Errorf("config mismatch: metric stored=%d current=%d", metric, idx.config.Metric)
	}
	if idx.config.Quantization != nil {
		if quantType != 1 {
			return fmt.Errorf("config mismatch: quant stored=0, current has PQ")
		}
		if storedCodebooks != idx.config.Quantization.Codebooks {
			return fmt.Errorf("config mismatch: codebooks stored=%d current=%d", storedCodebooks, idx.config.Quantization.Codebooks)
		}
		if storedBits != idx.config.Quantization.Bits {
			return fmt.Errorf("config mismatch: PQ bits stored=%d current=%d", storedBits, idx.config.Quantization.Bits)
		}
	} else if quantType != 0 {
		// Reconstitute quantizer from stored payload. The bridge creates an
		// empty index without quantization config during deserialization;
		// the stored binary payload is authoritative.
		idx.config.Quantization = &quant.QuantizationConfig{
			Type:       quant.ProductQuantization,
			Codebooks:  storedCodebooks,
			Bits:       storedBits,
			TrainRatio: 0.1,
			CacheSize:  1000,
		}
		q, err := quant.Create(idx.config.Quantization)
		if err != nil {
			return fmt.Errorf("reconstitute quantizer from stored payload: %w", err)
		}
		idx.quantizer = q
	}
	idx.config.MaxIterations = storedMaxIter

	// ---- Coarse centroids ----
	clusterCountV, err := r.u32()
	if err != nil {
		return err
	}
	clusterCount := int(clusterCountV)
	if clusterCount != nClusters {
		return fmt.Errorf("cluster count mismatch: stored %d, config %d", clusterCount, nClusters)
	}
	for i := 0; i < nClusters; i++ {
		cidV, err := r.u32()
		if err != nil {
			return err
		}
		cid := int(cidV)
		if cid != i {
			return fmt.Errorf("cluster ID sequence broken at %d: got %d", i, cid)
		}
		centDimsV, err := r.u32()
		if err != nil {
			return err
		}
		centDims := int(centDimsV)
		if centDims != dim {
			return fmt.Errorf("centroid dimension mismatch at cluster %d: %d vs %d", i, centDims, dim)
		}
		idx.clusters[i].Centroid = make([]float32, dim)
		var norm2 float32
		for d := 0; d < dim; d++ {
			c, err := r.f32()
			if err != nil {
				return err
			}
			idx.clusters[i].Centroid[d] = c
			norm2 += c * c
		}
		idx.clusters[i].centroidNorm2 = norm2
	}

	// ---- PQ codebooks ----
	subspaceCountV, err := r.u32()
	if err != nil {
		return err
	}
	subspaceCount := int(subspaceCountV)
	if subspaceCount > 0 {
		centroidsPerSSV, err := r.u32()
		if err != nil {
			return err
		}
		centroidsPerSS := int(centroidsPerSSV)
		subDimV, err := r.u32()
		if err != nil {
			return err
		}
		subDim := int(subDimV)
		codebookData := make([][][]float32, subspaceCount)
		for s := 0; s < subspaceCount; s++ {
			codebookData[s] = make([][]float32, centroidsPerSS)
			for c := 0; c < centroidsPerSS; c++ {
				codebookData[s][c] = make([]float32, subDim)
				for d := 0; d < subDim; d++ {
					cb, err := r.f32()
					if err != nil {
						return err
					}
					codebookData[s][c][d] = cb
				}
			}
		}
		if idx.quantizer != nil {
			injectCodebooks(idx.quantizer, codebookData, dim, subspaceCount, subDim)
		}
	}

	// ---- Inverted lists: per-cluster (ordinal, compressed) entries ----
	nClusters2V, err := r.u32()
	if err != nil {
		return err
	}
	nClusters2 := int(nClusters2V)
	if nClusters2 != nClusters {
		return fmt.Errorf("inverted list cluster count mismatch: %d vs %d", nClusters2, nClusters)
	}
	meta := &deserializedMeta{
		clusters: make([]ivfpqClusterMeta, nClusters),
	}
	for i := 0; i < nClusters; i++ {
		clusterIDV, err := r.u32()
		if err != nil {
			return err
		}
		_ = int(clusterIDV) // cluster ID (validated above)
		entryCountV, err := r.u32()
		if err != nil {
			return err
		}
		entryCount := int(entryCountV)
		meta.clusters[i].entries = make([]ivfpqEntryMeta, 0, entryCount)
		for e := 0; e < entryCount; e++ {
			if e%1024 == 0 {
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
				}
			}

			ordinalV, err := r.u32()
			if err != nil {
				return err
			}
			ordinal := ordinalV
			compressedLenV, err := r.u32()
			if err != nil {
				return err
			}
			compressedLen := int(compressedLenV)
			var compressed []byte
			if compressedLen > 0 {
				compressed = make([]byte, compressedLen)
				raw, err := r.raw(compressedLen)
				if err != nil {
					return err
				}
				copy(compressed, raw)
			}
			meta.clusters[i].entries = append(meta.clusters[i].entries, ivfpqEntryMeta{
				ordinal:    ordinal,
				compressed: compressed,
			})
		}
	}
	idx.deserMeta = meta

	// ---- Footer (CRC32 already validated at entry) ----
	if _, err := r.u32(); err != nil { // checksum already validated
		return err
	}

	idx.trained = true
	return nil
}

// HasDeserializedMeta reports whether deserialized entry metadata is pending
// population. Used by the collection layer to decide whether to call
// PopulateEntriesFromStorage or rebuild from scratch.
func (idx *Index) HasDeserializedMeta() bool {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	return idx.deserMeta != nil
}

// PopulateEntriesFromStorage wires storage records into the already-deserialized
// index using authoritative cluster membership from the persisted inverted lists.
// No centroid-distance recomputation is performed — each entry is placed directly
// into its stored cluster by ordinal.
func (idx *Index) PopulateEntriesFromStorage(provider EntryProvider) error {
	idx.mutex.Lock()
	defer idx.mutex.Unlock()

	if !idx.trained {
		return fmt.Errorf("index not trained; call DeserializeFromBytes first")
	}
	if idx.deserMeta == nil {
		// Index was fully rebuilt from records (not deserialized from a
		// checkpoint). Entries are already present; nothing to populate.
		return nil
	}

	// Compute total entry count from persisted metadata for pre-sizing.
	totalEntries := 0
	for cid := range idx.deserMeta.clusters {
		if cid >= len(idx.clusters) {
			break
		}
		totalEntries += len(idx.deserMeta.clusters[cid].entries)
	}

	// Build ordinal → entry map from storage in a single pass.
	// Map is pre-sized to avoid rehash cascades during insertion.
	ordinalToEntry := make(map[uint32]*VectorEntry, totalEntries)
	err := provider.IterateEntries(func(id string, ordinal uint32, vector []float32, metadata map[string]interface{}) error {
		ordinalToEntry[ordinal] = &VectorEntry{
			ID:       id,
			Ordinal:  ordinal,
			Vector:   vector,
			Metadata: metadata,
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("iterate storage: %w", err)
	}

	// Place entries directly by ordinal using stored cluster membership.
	// Slices and maps are pre-sized to avoid incremental growth allocations.
	for cid := range idx.deserMeta.clusters {
		if cid >= len(idx.clusters) {
			break
		}
		cluster := idx.clusters[cid]
		clusterMeta := idx.deserMeta.clusters[cid]
		n := len(clusterMeta.entries)
		if n == 0 {
			continue
		}
		cluster.mutex.Lock()
		if cap(cluster.Entries) < n {
			cluster.Entries = make([]*VectorEntry, 0, n)
		}
		if cluster.CompressedVectors == nil {
			cluster.CompressedVectors = make(map[string][]byte, n)
		}
		for _, em := range clusterMeta.entries {
			entry, ok := ordinalToEntry[em.ordinal]
			if !ok {
				continue
			}
			cluster.Entries = append(cluster.Entries, entry)
			idx.idToCluster.Store(entry.ID, cid)
			if len(em.compressed) > 0 {
				cluster.CompressedVectors[entry.ID] = em.compressed
				entry.Compressed = em.compressed
			}
			idx.size++
		}
		cluster.mutex.Unlock()
	}

	idx.deserMeta = nil // consumed
	return nil
}

// EntryProvider abstracts storage iteration for entry reconstruction.
type EntryProvider interface {
	IterateEntries(fn func(id string, ordinal uint32, vector []float32, metadata map[string]interface{}) error) error
}

// ---------------------------------------------------------------------------
// Codebook extraction/injection
// ---------------------------------------------------------------------------

func extractCodebooks(q quant.Quantizer) [][][]float32 {
	if qcw, ok := q.(interface{ GetCodebooks() [][][]float32 }); ok {
		return qcw.GetCodebooks()
	}
	return nil
}

func injectCodebooks(q quant.Quantizer, codebooks [][][]float32, dimension, subspaces, subDim int) {
	if qcw, ok := q.(interface {
		SetCodebooks(codebooks [][][]float32, dimension, subspaces, subDim int)
	}); ok {
		qcw.SetCodebooks(codebooks, dimension, subspaces, subDim)
	}
}

// ---------------------------------------------------------------------------
// Binary I/O helpers (allocation-free, slice-backed)
// ---------------------------------------------------------------------------

type sliceWriter struct{ buf []byte }

func (w *sliceWriter) u8(v uint8)     { w.buf = append(w.buf, v) }
func (w *sliceWriter) u16(v uint16)   { w.buf = binary.LittleEndian.AppendUint16(w.buf, v) }
func (w *sliceWriter) u32(v uint32)   { w.buf = binary.LittleEndian.AppendUint32(w.buf, v) }
func (w *sliceWriter) i64(v int64)    { w.buf = binary.LittleEndian.AppendUint64(w.buf, uint64(v)) }
func (w *sliceWriter) bytes(v []byte) { w.buf = append(w.buf, v...) }
func (w *sliceWriter) f32(v float32) {
	w.buf = binary.LittleEndian.AppendUint32(w.buf, math.Float32bits(v))
}
func (w *sliceWriter) f64(v float64) {
	w.buf = binary.LittleEndian.AppendUint64(w.buf, math.Float64bits(v))
}
func (w *sliceWriter) raw(v []byte) { w.buf = append(w.buf, v...) }

type sliceReader struct {
	buf []byte
	pos int
}

func (r *sliceReader) remaining() int { return len(r.buf) - r.pos }

func (r *sliceReader) u8() (uint8, error) {
	if r.remaining() < 1 {
		return 0, fmt.Errorf("sliceReader: truncated input, need 1 byte, have %d", r.remaining())
	}
	v := r.buf[r.pos]
	r.pos++
	return v, nil
}
func (r *sliceReader) bytes(n int) ([]byte, error) {
	if r.remaining() < n {
		return nil, fmt.Errorf("sliceReader: truncated input, need %d bytes, have %d", n, r.remaining())
	}
	v := r.buf[r.pos : r.pos+n]
	r.pos += n
	return v, nil
}
func (r *sliceReader) u16() (uint16, error) {
	if r.remaining() < 2 {
		return 0, fmt.Errorf("sliceReader: truncated input, need 2 bytes, have %d", r.remaining())
	}
	v := binary.LittleEndian.Uint16(r.buf[r.pos:])
	r.pos += 2
	return v, nil
}
func (r *sliceReader) u32() (uint32, error) {
	if r.remaining() < 4 {
		return 0, fmt.Errorf("sliceReader: truncated input, need 4 bytes, have %d", r.remaining())
	}
	v := binary.LittleEndian.Uint32(r.buf[r.pos:])
	r.pos += 4
	return v, nil
}
func (r *sliceReader) i64() (int64, error) {
	if r.remaining() < 8 {
		return 0, fmt.Errorf("sliceReader: truncated input, need 8 bytes, have %d", r.remaining())
	}
	v := binary.LittleEndian.Uint64(r.buf[r.pos:])
	r.pos += 8
	return int64(v), nil
}
func (r *sliceReader) f32() (float32, error) {
	if r.remaining() < 4 {
		return 0, fmt.Errorf("sliceReader: truncated input, need 4 bytes, have %d", r.remaining())
	}
	v := binary.LittleEndian.Uint32(r.buf[r.pos:])
	r.pos += 4
	return math.Float32frombits(v), nil
}
func (r *sliceReader) f64() (float64, error) {
	if r.remaining() < 8 {
		return 0, fmt.Errorf("sliceReader: truncated input, need 8 bytes, have %d", r.remaining())
	}
	v := binary.LittleEndian.Uint64(r.buf[r.pos:])
	r.pos += 8
	return math.Float64frombits(v), nil
}
func (r *sliceReader) raw(n int) ([]byte, error) {
	if r.remaining() < n {
		return nil, fmt.Errorf("sliceReader: truncated input, need %d bytes, have %d", n, r.remaining())
	}
	v := r.buf[r.pos : r.pos+n]
	r.pos += n
	return v, nil
}

// ---------------------------------------------------------------------------
// SaveToDisk / LoadFromDisk — file-based wrappers
// ---------------------------------------------------------------------------

func (idx *Index) SaveToDisk(_ context.Context, path string) error {
	data, err := idx.SerializeToBytes()
	if err != nil {
		return err
	}
	if data == nil {
		return fmt.Errorf("IVF-PQ index not trained; nothing to save")
	}
	return os.WriteFile(path, data, 0644)
}

func (idx *Index) LoadFromDisk(ctx context.Context, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return idx.DeserializeFromBytes(ctx, data)
}
