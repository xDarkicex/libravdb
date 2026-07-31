package ivfpq

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"math"
	"os"

	"github.com/xDarkicex/memory"
	"github.com/xDarkicex/libravdb/internal/quant"
)

const (
	ivfpqFormatVersion      uint16 = 3
	ivfpqFormatVersionLegacy uint16 = 2
	ivfpqIndexType          uint8  = 2
)

var ivfpqMagicBytes = []byte("LIBRAIVF")

const (
	qTagNone uint8 = 0
	qTagPQ   uint8 = 1
	qTagSQ   uint8 = 2
	qTagFSQ  uint8 = 3
)

const (
	maxHydrateDim        = 65536
	maxHydrateClusters   = 16384
	maxHydrateProbes     = 16384
	maxHydrateRecsList   = 1 << 24
	maxHydrateQState     = 1 << 26
	maxHydrateSubsp      = 1024
	maxHydrateCPerSS     = 65536
	maxHydrateSubDim     = 65536
	maxHydrateTotalRecs  = 1 << 28
	maxHydrateCodeBytes  = 262144
)

func qTag(q quant.Quantizer) uint8 {
	if q == nil {
		return qTagNone
	}
	switch q.(type) {
	case *quant.ScalarQuantizer:
		return qTagSQ
	case *quant.ProductQuantizer:
		return qTagPQ
	case *quant.FSQQuantizer:
		return qTagFSQ
	}
	return math.MaxUint8
}

// --- SerializeToBytes v3 ---

func (idx *Index) SerializeToBytes() ([]byte, error) {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	if idx.gen == nil || !idx.gen.trained {
		return nil, nil
	}
	buf := make([]byte, 0, 64*1024)
	w := &sliceWriter{buf: buf}
	w.bytes(ivfpqMagicBytes)
	w.u16(ivfpqFormatVersion)
	w.u8(ivfpqIndexType)
	w.u8(0)
	g := idx.gen
	w.u32(uint32(g.config.Dimension))
	w.u32(uint32(g.config.NClusters))
	w.u32(uint32(g.config.NProbes))
	w.u8(uint8(g.config.Metric))
	tag := qTag(g.quantizer)
	if tag == math.MaxUint8 {
		return nil, fmt.Errorf("persist: unknown quantizer %T", g.quantizer)
	}
	w.u8(tag)
	if tag != qTagNone {
		s, err := g.quantizer.SerializeState()
		if err != nil {
			return nil, fmt.Errorf("persist: %w", err)
		}
		w.u32(uint32(len(s)))
		w.raw(s)
	}
	w.u32(uint32(g.config.MaxIterations))
	w.f64(g.config.Tolerance)
	w.i64(g.config.RandomSeed)
	w.u32(uint32(len(g.clusters)))
	for _, c := range g.clusters {
		w.u32(uint32(c.ID))
		w.u32(uint32(len(c.Centroid)))
		for _, v := range c.Centroid {
			w.f32(v)
		}
	}
	w.u32(uint32(len(g.clusters)))
	cs := 0
	if g.quantizer != nil && g.quantizer.IsTrained() {
		cs = g.quantizer.CodeSize()
	}
	for _, c := range g.clusters {
		c.mutex.RLock()
		w.u32(uint32(c.ID))
		w.u32(uint32(c.storage.count))
		for _, seg := range c.storage.segments {
			for j := uint32(0); j < seg.used; j++ {
				w.u32(seg.ordinals[j])
				var cd []byte
				if cs > 0 {
					cd = seg.codes[int(j)*cs : int(j+1)*cs]
				}
				w.u32(uint32(len(cd)))
				if len(cd) > 0 {
					w.raw(cd)
				}
			}
		}
		c.mutex.RUnlock()
	}
	ck := crc32.Checksum(w.buf, crc32.MakeTable(crc32.Castagnoli))
	w.u32(ck)
	return w.buf, nil
}

// --- pending state ---

type pendingIndex struct {
	dim             int
	nClusters       int
	nProbes         int
	metric          uint8
	maxIter         int
	tolerance       float64
	randomSeed      int64
	quantTag        uint8
	pqConfig        *quant.QuantizationConfig
	quantState      []byte
	legacyCodebooks [][][]float32
	legacySubDim    int
	centroids       [][]float32
	records         [][]pendingRecord
}

type pendingRecord struct {
	ordinal uint32
	code    []byte
}

// --- validation helpers ---

func checkDim(d int) error {
	if d <= 0 || d > maxHydrateDim {
		return fmt.Errorf("dimension %d out of range", d)
	}
	return nil
}
func checkClust(n int) error {
	if n <= 0 || n > maxHydrateClusters {
		return fmt.Errorf("clusters %d out of range", n)
	}
	return nil
}
func checkEntryCount(v uint32) (int, error) {
	if int64(v) > maxHydrateRecsList {
		return 0, fmt.Errorf("entry count %d exceeds max", v)
	}
	return int(v), nil
}
func checkQStateLen(v uint32) (int, error) {
	if int64(v) > maxHydrateQState {
		return 0, fmt.Errorf("quant state length %d exceeds max", v)
	}
	return int(v), nil
}
func safeMul(a, b int) (int, error) {
	if a < 0 || b < 0 {
		return 0, fmt.Errorf("negative product")
	}
	if b != 0 && a > math.MaxInt/b {
		return 0, fmt.Errorf("overflow: %d * %d", a, b)
	}
	return a * b, nil
}

// --- DeserializeFromBytes ---

func (idx *Index) DeserializeFromBytes(ctx context.Context, data []byte) error {
	// Capture the live generation as a staging base. We hold only RLock
	// during capture; parsing and staging happen independently.
	idx.mutex.RLock()
	if idx.gen == nil {
		idx.mutex.RUnlock()
		return fmt.Errorf("index closed")
	}
	base := idx.gen
	base.acquire()
	idx.mutex.RUnlock()
	defer base.release()

	// Parse and validate — go heap only, no live state mutation.
	if len(data) < 20 {
		return fmt.Errorf("IVF-PQ data too short")
	}
	pe := len(data) - 4
	if pe <= 0 {
		return fmt.Errorf("IVF-PQ data too short")
	}
	if binary.LittleEndian.Uint32(data[pe:]) != crc32.Checksum(data[:pe], crc32.MakeTable(crc32.Castagnoli)) {
		return fmt.Errorf("checksum mismatch")
	}
	r := &sliceReader{buf: data}
	m, err := r.bytes(8)
	if err != nil {
		return fmt.Errorf("magic: %w", err)
	}
	if string(m) != string(ivfpqMagicBytes) {
		return fmt.Errorf("bad magic")
	}
	ver, err := r.u16()
	if err != nil {
		return fmt.Errorf("version: %w", err)
	}
	switch ver {
	case 1:
		return fmt.Errorf("v1 obsolete")
	case ivfpqFormatVersionLegacy, ivfpqFormatVersion:
	default:
		return fmt.Errorf("unsupported version %d", ver)
	}
	it, err := r.u8()
	if err != nil {
		return fmt.Errorf("index type: %w", err)
	}
	if it != ivfpqIndexType {
		return fmt.Errorf("index type mismatch")
	}
	if _, err := r.u8(); err != nil {
		return fmt.Errorf("flags: %w", err)
	}

	var p *pendingIndex
	switch ver {
	case ivfpqFormatVersionLegacy:
		p, err = parsePendingV2(r)
	case ivfpqFormatVersion:
		p, err = parsePendingV3(r)
	}
	if err != nil {
		return fmt.Errorf("parse: %w", err)
	}
	if r.pos != len(data) {
		return fmt.Errorf("trailing bytes: pos=%d len=%d", r.pos, len(data))
	}
	// Validate against base config, not idx.config.
	if p.dim != base.config.Dimension {
		return fmt.Errorf("dim mismatch %d vs %d", p.dim, base.config.Dimension)
	}
	if p.nClusters != base.config.NClusters {
		return fmt.Errorf("clusters mismatch %d vs %d", p.nClusters, base.config.NClusters)
	}
	if p.nProbes != base.config.NProbes {
		return fmt.Errorf("probes mismatch %d vs %d", p.nProbes, base.config.NProbes)
	}
	if p.metric != uint8(base.config.Metric) {
		return fmt.Errorf("metric mismatch")
	}

	var q quant.Quantizer
	if p.quantTag != qTagNone {
		q, err = buildQuantizerFromPending(p)
		if err != nil {
			return fmt.Errorf("quantizer: %w", err)
		}
	}
	cs := 0
	if q != nil {
		cs = q.CodeSize()
	}
	for ci, recs := range p.records {
		for ri, rec := range recs {
			if cs == 0 && len(rec.code) != 0 {
				return fmt.Errorf("c%d e%d: code without quantizer", ci, ri)
			}
			if cs > 0 && len(rec.code) != cs {
				return fmt.Errorf("c%d e%d: code len %d != %d", ci, ri, len(rec.code), cs)
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	// STAGE: build a complete replacement generation in a fresh pool.
	// No live state is mutated during staging. A failure at any point
	// frees only the replacement; the live generation remains untouched.
	replPool, err := memory.NewPool(base.poolCfg, 64)
	if err != nil {
		return fmt.Errorf("create replacement pool: %w", err)
	}
	var committed bool
	defer func() {
		if !committed {
			replPool.Free()
			if q != nil {
				q.Close()
			}
		}
	}()

	replClusters := make([]*Cluster, p.nClusters)
	var totalRecords int64
	for ci, recs := range p.records {
		if err := ctx.Err(); err != nil {
			return err
		}
		entries := len(recs)
		totalRecords += int64(entries)
		stg := &clusterStorage{segmentCapacity: 1024, codeWidth: uint32(cs)}
		cl := &Cluster{ID: ci, Centroid: make([]float32, p.dim), storage: stg}
		for ri, rec := range recs {
			if ri%1024 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			if err := stg.append(rec.ordinal, rec.code, replPool); err != nil {
				return fmt.Errorf("stage cluster %d: %w", ci, err)
			}
		}
		replClusters[ci] = cl
	}
	// Centroids.
	for i, cent := range p.centroids {
		replClusters[i].Centroid = cent
		var n2 float32
		for _, v := range cent {
			n2 += v * v
		}
		replClusters[i].centroidNorm2 = n2
	}

	// Build replacement config (deep-clone quantization if needed).
	replConfig := cloneConfig(base.config)
	replConfig.MaxIterations = p.maxIter
	replConfig.Tolerance = p.tolerance
	replConfig.RandomSeed = p.randomSeed
	if q != nil {
		if cfg := q.Config(); cfg != nil {
			replConfig.Quantization = cfg
		}
	} else {
		replConfig.Quantization = nil
	}

	replGen := newGeneration(replPool, base.poolCfg, replClusters, q, replConfig, int(totalRecords), true)
	committed = true

	// COMMIT: pointer swap under exclusive lock.
	idx.mutex.Lock()
	if idx.gen == nil {
		idx.mutex.Unlock()
		replGen.retired.Store(true)
		replGen.release()
		return fmt.Errorf("index closed during hydration commit")
	}
	previous := idx.gen
	idx.gen = replGen
	idx.config = replConfig
	idx.mutex.Unlock()

	// RETIRE the old generation outside the lock.
	previous.retired.Store(true)
	previous.release()
	return nil
}

// cloneConfig returns a shallow copy of c with its own Quantization.
func cloneConfig(c *Config) *Config {
	clone := *c
	if c.Quantization != nil {
		qc := *c.Quantization
		if c.Quantization.Levels != nil {
			qc.Levels = append([]int(nil), c.Quantization.Levels...)
		}
		clone.Quantization = &qc
	}
	return &clone
}

// --- parsePendingV3 ---

func parsePendingV3(r *sliceReader) (*pendingIndex, error) {
	p := &pendingIndex{}
	dv, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("dim: %w", err)
	}
	p.dim = int(dv)
	if err := checkDim(p.dim); err != nil {
		return nil, err
	}
	nv, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("clusters: %w", err)
	}
	p.nClusters = int(nv)
	if err := checkClust(p.nClusters); err != nil {
		return nil, err
	}
	pv, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("probes: %w", err)
	}
	p.nProbes = int(pv)
	if p.nProbes <= 0 || p.nProbes > maxHydrateProbes {
		return nil, fmt.Errorf("probes %d out of range", p.nProbes)
	}
	mv, err := r.u8()
	if err != nil {
		return nil, fmt.Errorf("metric: %w", err)
	}
	p.metric = mv
	tv, err := r.u8()
	if err != nil {
		return nil, fmt.Errorf("quantTag: %w", err)
	}
	p.quantTag = tv
	if p.quantTag != qTagNone {
		ql, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("qstate len: %w", err)
		}
		qlen, err := checkQStateLen(ql)
		if err != nil {
			return nil, err
		}
		p.quantState, err = r.raw(qlen)
		if err != nil {
			return nil, fmt.Errorf("qstate: %w", err)
		}
	}
	miv, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("maxIter: %w", err)
	}
	p.maxIter = int(miv)
	p.tolerance, err = r.f64()
	if err != nil {
		return nil, fmt.Errorf("tolerance: %w", err)
	}
	p.randomSeed, err = r.i64()
	if err != nil {
		return nil, fmt.Errorf("seed: %w", err)
	}
	// centroids
	cv, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("centroid count: %w", err)
	}
	if int(cv) != p.nClusters {
		return nil, fmt.Errorf("centroid count mismatch: %d vs %d", cv, p.nClusters)
	}
	if _, err := safeMul(p.nClusters, p.dim); err != nil {
		return nil, fmt.Errorf("centroid alloc: %w", err)
	}
	p.centroids = make([][]float32, p.nClusters)
	for i := 0; i < p.nClusters; i++ {
		if _, err := r.u32(); err != nil {
			return nil, fmt.Errorf("centroid[%d] id: %w", i, err)
		}
		cd, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("centroid[%d] dim: %w", i, err)
		}
		if int(cd) != p.dim {
			return nil, fmt.Errorf("centroid[%d] dim %d != %d", i, cd, p.dim)
		}
		c := make([]float32, p.dim)
		for d := 0; d < p.dim; d++ {
			c[d], err = r.f32()
			if err != nil {
				return nil, fmt.Errorf("centroid[%d][%d]: %w", i, d, err)
			}
		}
		p.centroids[i] = c
	}
	// inverted lists
	cv2, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("invlist count: %w", err)
	}
	if int(cv2) != p.nClusters {
		return nil, fmt.Errorf("invlist count mismatch: %d vs %d", cv2, p.nClusters)
	}
	p.records = make([][]pendingRecord, p.nClusters)
	var totalRecs int64
	for ci := 0; ci < p.nClusters; ci++ {
		if _, err := r.u32(); err != nil {
			return nil, fmt.Errorf("invlist[%d] id: %w", ci, err)
		}
		ec, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("invlist[%d] count: %w", ci, err)
		}
		ecnt, err := checkEntryCount(ec)
		if err != nil {
			return nil, fmt.Errorf("invlist[%d]: %w", ci, err)
		}
		totalRecs += int64(ecnt)
		if totalRecs > maxHydrateTotalRecs {
			return nil, fmt.Errorf("total records %d exceeds ceiling", totalRecs)
		}
		remain := len(r.buf) - r.pos
		maxCap := ecnt
		if remain > 0 && remain/8 < maxCap {
			maxCap = remain / 8
		}
		recs := make([]pendingRecord, 0, maxCap)
		for e := 0; e < ecnt; e++ {
			ov, err := r.u32()
			if err != nil {
				return nil, fmt.Errorf("invlist[%d][%d] ord: %w", ci, e, err)
			}
			cl, err := r.u32()
			if err != nil {
				return nil, fmt.Errorf("invlist[%d][%d] clen: %w", ci, e, err)
			}
			if cl > maxHydrateCodeBytes {
				return nil, fmt.Errorf("invlist[%d][%d] code len %d exceeds %d", ci, e, cl, maxHydrateCodeBytes)
			}
			var code []byte
			if cl > 0 {
				code, err = r.raw(int(cl))
				if err != nil {
					return nil, fmt.Errorf("invlist[%d][%d] code: %w", ci, e, err)
				}
			}
			recs = append(recs, pendingRecord{ordinal: ov, code: code})
		}
		p.records[ci] = recs
	}
	if _, err := r.u32(); err != nil {
		return nil, fmt.Errorf("footer: %w", err)
	}
	return p, nil
}

// --- parsePendingV2 ---

func parsePendingV2(r *sliceReader) (*pendingIndex, error) {
	p := &pendingIndex{}
	dv, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("dim: %w", err)
	}
	p.dim = int(dv)
	if err := checkDim(p.dim); err != nil {
		return nil, err
	}
	nv, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("clusters: %w", err)
	}
	p.nClusters = int(nv)
	if err := checkClust(p.nClusters); err != nil {
		return nil, err
	}
	pv, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("probes: %w", err)
	}
	p.nProbes = int(pv)
	if p.nProbes <= 0 || p.nProbes > maxHydrateProbes {
		return nil, fmt.Errorf("probes %d out of range", p.nProbes)
	}
	mv, err := r.u8()
	if err != nil {
		return nil, fmt.Errorf("metric: %w", err)
	}
	p.metric = mv
	tv, err := r.u8()
	if err != nil {
		return nil, fmt.Errorf("quantTag: %w", err)
	}
	p.quantTag = tv
	scb, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("v2 codebooks: %w", err)
	}
	sb, err := r.u8()
	if err != nil {
		return nil, fmt.Errorf("v2 bits: %w", err)
	}
	miv, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("maxIter: %w", err)
	}
	p.maxIter = int(miv)
	p.tolerance, err = r.f64()
	if err != nil {
		return nil, fmt.Errorf("tolerance: %w", err)
	}
	p.randomSeed, err = r.i64()
	if err != nil {
		return nil, fmt.Errorf("seed: %w", err)
	}
	switch p.quantTag {
	case 0:
	case 1:
		cb := int(scb)
		bits := int(sb)
		if cb <= 0 || cb > maxHydrateSubsp {
			return nil, fmt.Errorf("v2 codebooks %d out of range", cb)
		}
		if bits < 1 || bits > 16 {
			return nil, fmt.Errorf("v2 bits %d out of range", bits)
		}
		p.pqConfig = &quant.QuantizationConfig{
			Type: quant.ProductQuantization, Codebooks: cb, Bits: bits,
			TrainRatio: 0.1, CacheSize: 1000,
		}
	default:
		return nil, fmt.Errorf("v2 unsupported quant tag %d", p.quantTag)
	}
	// centroids
	cv, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("centroid count: %w", err)
	}
	if int(cv) != p.nClusters {
		return nil, fmt.Errorf("centroid count mismatch: %d vs %d", cv, p.nClusters)
	}
	p.centroids = make([][]float32, p.nClusters)
	for i := 0; i < p.nClusters; i++ {
		if _, err := r.u32(); err != nil {
			return nil, fmt.Errorf("centroid[%d] id: %w", i, err)
		}
		cd, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("centroid[%d] dim: %w", i, err)
		}
		if int(cd) != p.dim {
			return nil, fmt.Errorf("centroid[%d] dim %d != %d", i, cd, p.dim)
		}
		c := make([]float32, p.dim)
		for d := 0; d < p.dim; d++ {
			c[d], err = r.f32()
			if err != nil {
				return nil, fmt.Errorf("centroid[%d][%d]: %w", i, d, err)
			}
		}
		p.centroids[i] = c
	}
	// legacy PQ codebooks
	if p.quantTag == 1 {
		sc, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("v2 subspaces: %w", err)
		}
		scnt := int(sc)
		if scnt <= 0 || scnt > maxHydrateSubsp {
			return nil, fmt.Errorf("v2 subspaces %d out of range", scnt)
		}
		if scnt != p.pqConfig.Codebooks {
			return nil, fmt.Errorf("v2 subspaces %d != codebooks %d", scnt, p.pqConfig.Codebooks)
		}
		cps, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("v2 cps: %w", err)
		}
		cp := int(cps)
		if cp <= 0 || cp > maxHydrateCPerSS {
			return nil, fmt.Errorf("v2 cps %d out of range", cp)
		}
		sd, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("v2 subdim: %w", err)
		}
		p.legacySubDim = int(sd)
		if p.legacySubDim <= 0 || p.legacySubDim > maxHydrateSubDim {
			return nil, fmt.Errorf("v2 subdim %d out of range", p.legacySubDim)
		}
		prod, err := safeMul(scnt, p.legacySubDim)
		if err != nil {
			return nil, fmt.Errorf("v2 PQ shape overflow")
		}
		if prod != p.dim {
			return nil, fmt.Errorf("v2 PQ shape mismatch: %d*%d != %d", scnt, p.legacySubDim, p.dim)
		}
		alloc, err := safeMul(scnt, cp)
		if err != nil {
			return nil, fmt.Errorf("v2 PQ alloc overflow")
		}
		alloc, err = safeMul(alloc, p.legacySubDim)
		if err != nil {
			return nil, fmt.Errorf("v2 PQ alloc overflow")
		}
		if int64(alloc)*4 > maxHydrateQState {
			return nil, fmt.Errorf("v2 PQ alloc %d floats exceeds quant state ceiling", alloc)
		}
		cb := make([][][]float32, scnt)
		for s := 0; s < scnt; s++ {
			cb[s] = make([][]float32, cp)
			for c := 0; c < cp; c++ {
				cb[s][c] = make([]float32, p.legacySubDim)
				for d := 0; d < p.legacySubDim; d++ {
					cb[s][c][d], err = r.f32()
					if err != nil {
						return nil, fmt.Errorf("v2 cb[%d][%d][%d]: %w", s, c, d, err)
					}
				}
			}
		}
		p.legacyCodebooks = cb
	}
	// inverted lists
	cv2, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("v2 invlist count: %w", err)
	}
	if int(cv2) != p.nClusters {
		return nil, fmt.Errorf("v2 invlist count mismatch: %d vs %d", cv2, p.nClusters)
	}
	p.records = make([][]pendingRecord, p.nClusters)
	var v2TotalRecs int64
	for ci := 0; ci < p.nClusters; ci++ {
		if _, err := r.u32(); err != nil {
			return nil, fmt.Errorf("v2 invlist[%d] id: %w", ci, err)
		}
		ec, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("v2 invlist[%d] count: %w", ci, err)
		}
		ecnt, err := checkEntryCount(ec)
		if err != nil {
			return nil, fmt.Errorf("v2 invlist[%d]: %w", ci, err)
		}
		v2TotalRecs += int64(ecnt)
		if v2TotalRecs > maxHydrateTotalRecs {
			return nil, fmt.Errorf("v2 total records %d exceeds ceiling", v2TotalRecs)
		}
		remain := len(r.buf) - r.pos
		maxCap := ecnt
		if remain > 0 && remain/8 < maxCap {
			maxCap = remain / 8
		}
		recs := make([]pendingRecord, 0, maxCap)
		for e := 0; e < ecnt; e++ {
			ov, err := r.u32()
			if err != nil {
				return nil, fmt.Errorf("v2 invlist[%d][%d] ord: %w", ci, e, err)
			}
			cl, err := r.u32()
			if err != nil {
				return nil, fmt.Errorf("v2 invlist[%d][%d] clen: %w", ci, e, err)
			}
			if cl > maxHydrateCodeBytes {
				return nil, fmt.Errorf("v2 invlist[%d][%d] code len %d exceeds %d", ci, e, cl, maxHydrateCodeBytes)
			}
			var code []byte
			if cl > 0 {
				code, err = r.raw(int(cl))
				if err != nil {
					return nil, fmt.Errorf("v2 invlist[%d][%d] code: %w", ci, e, err)
				}
			}
			recs = append(recs, pendingRecord{ordinal: ov, code: code})
		}
		p.records[ci] = recs
	}
	if _, err := r.u32(); err != nil {
		return nil, fmt.Errorf("v2 footer: %w", err)
	}
	return p, nil
}

// --- buildQuantizerFromPending ---

func buildQuantizerFromPending(p *pendingIndex) (quant.Quantizer, error) {
	switch p.quantTag {
	case qTagPQ:
		if p.quantState != nil {
			pq := quant.NewProductQuantizer()
			if err := pq.DeserializeState(p.quantState); err != nil {
				return nil, fmt.Errorf("PQ: %w", err)
			}
			return pq, nil
		}
		if p.pqConfig == nil {
			return nil, fmt.Errorf("v2 PQ missing config")
		}
		if p.dim%p.pqConfig.Codebooks != 0 {
			return nil, fmt.Errorf("v2 PQ: dim %d %% codebooks %d != 0", p.dim, p.pqConfig.Codebooks)
		}
		pq := quant.NewProductQuantizer()
		if err := pq.Configure(p.pqConfig); err != nil {
			return nil, err
		}
		pq.SetCodebooks(p.legacyCodebooks, p.dim, p.pqConfig.Codebooks, p.legacySubDim)
		return pq, nil
	case qTagSQ:
		if p.quantState == nil {
			return nil, fmt.Errorf("scalar missing state")
		}
		sq := quant.NewScalarQuantizer()
		if err := sq.DeserializeState(p.quantState); err != nil {
			return nil, fmt.Errorf("SQ: %w", err)
		}
		return sq, nil
	case qTagFSQ:
		if p.quantState == nil {
			return nil, fmt.Errorf("FSQ missing state")
		}
		fq := quant.NewFSQQuantizer()
		if err := fq.DeserializeState(p.quantState); err != nil {
			return nil, fmt.Errorf("FSQ: %w", err)
		}
		return fq, nil
	}
	return nil, fmt.Errorf("unknown quant tag %d", p.quantTag)
}

// --- I/O helpers ---

type sliceWriter struct{ buf []byte }

func (w *sliceWriter) u8(v uint8)     { w.buf = append(w.buf, v) }
func (w *sliceWriter) u16(v uint16)   { w.buf = binary.LittleEndian.AppendUint16(w.buf, v) }
func (w *sliceWriter) u32(v uint32)   { w.buf = binary.LittleEndian.AppendUint32(w.buf, v) }
func (w *sliceWriter) i64(v int64)    { w.buf = binary.LittleEndian.AppendUint64(w.buf, uint64(v)) }
func (w *sliceWriter) bytes(v []byte) { w.buf = append(w.buf, v...) }
func (w *sliceWriter) f32(v float32)  { w.buf = binary.LittleEndian.AppendUint32(w.buf, math.Float32bits(v)) }
func (w *sliceWriter) f64(v float64)  { w.buf = binary.LittleEndian.AppendUint64(w.buf, math.Float64bits(v)) }
func (w *sliceWriter) raw(v []byte)   { w.buf = append(w.buf, v...) }

type sliceReader struct {
	buf []byte
	pos int
}

func (r *sliceReader) u8() (uint8, error) {
	if r.pos >= len(r.buf) {
		return 0, fmt.Errorf("truncated at pos %d", r.pos)
	}
	v := r.buf[r.pos]
	r.pos++
	return v, nil
}
func (r *sliceReader) bytes(n int) ([]byte, error) {
	if r.pos+n > len(r.buf) {
		return nil, fmt.Errorf("truncated at pos %d, need %d bytes", r.pos, n)
	}
	v := r.buf[r.pos : r.pos+n]
	r.pos += n
	return v, nil
}
func (r *sliceReader) u16() (uint16, error) {
	if r.pos+2 > len(r.buf) {
		return 0, fmt.Errorf("truncated at pos %d, need u16", r.pos)
	}
	v := binary.LittleEndian.Uint16(r.buf[r.pos:])
	r.pos += 2
	return v, nil
}
func (r *sliceReader) u32() (uint32, error) {
	if r.pos+4 > len(r.buf) {
		return 0, fmt.Errorf("truncated at pos %d, need u32", r.pos)
	}
	v := binary.LittleEndian.Uint32(r.buf[r.pos:])
	r.pos += 4
	return v, nil
}
func (r *sliceReader) i64() (int64, error) {
	if r.pos+8 > len(r.buf) {
		return 0, fmt.Errorf("truncated at pos %d, need i64", r.pos)
	}
	v := binary.LittleEndian.Uint64(r.buf[r.pos:])
	r.pos += 8
	return int64(v), nil
}
func (r *sliceReader) f32() (float32, error) {
	if r.pos+4 > len(r.buf) {
		return 0, fmt.Errorf("truncated at pos %d, need f32", r.pos)
	}
	v := binary.LittleEndian.Uint32(r.buf[r.pos:])
	r.pos += 4
	return math.Float32frombits(v), nil
}
func (r *sliceReader) f64() (float64, error) {
	if r.pos+8 > len(r.buf) {
		return 0, fmt.Errorf("truncated at pos %d, need f64", r.pos)
	}
	v := binary.LittleEndian.Uint64(r.buf[r.pos:])
	r.pos += 8
	return math.Float64frombits(v), nil
}
func (r *sliceReader) raw(n int) ([]byte, error) {
	if r.pos+n > len(r.buf) {
		return nil, fmt.Errorf("truncated at pos %d, need %d raw bytes", r.pos, n)
	}
	v := r.buf[r.pos : r.pos+n]
	r.pos += n
	return v, nil
}

func (idx *Index) SaveToDisk(_ context.Context, path string) error {
	d, err := idx.SerializeToBytes()
	if err != nil {
		return err
	}
	if d == nil {
		return fmt.Errorf("not trained")
	}
	return os.WriteFile(path, d, 0644)
}
func (idx *Index) LoadFromDisk(ctx context.Context, path string) error {
	d, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return idx.DeserializeFromBytes(ctx, d)
}
