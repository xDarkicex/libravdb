package libravdb

import (
	"container/list"
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/internal/util"
)

// TemporalANNConfig controls the optional snapshot-index cache.
type TemporalANNConfig struct {
	// MaxBytes is the approximate memory limit for all cached snapshot indexes.
	// Zero means no caching (exact-only default).
	MaxBytes int64
	// MaxEntries caps the number of cached snapshot indexes regardless of size.
	MaxEntries int
	// MinCandidates is the minimum graph/scalar candidate count before temporal
	// ANN is considered. Below this threshold, exact scoring is faster.
	MinCandidates int
	// EfSearch overrides the default HNSW search breadth for cached indexes.
	EfSearch int
}

// temporalIndexKey identifies a cache entry.
type temporalIndexKey struct {
	collection string
	lsn        uint64
	dimension  int
	metric     int
	m          int
	efConst    int
}

// temporalIndexEntry is one cached immutable snapshot HNSW index.
type temporalIndexEntry struct {
	key         temporalIndexKey
	index       index.Index
	ordinalToID []string // local-ordinal → record ID
	snapshot    *TemporalSnapshot
	bytes       int64
	builtAt     time.Time
	element     *list.Element // LRU list position
	// leaseCount tracks in-flight queries using this entry.
	leaseCount atomic.Int32
}

// temporalIndexCache is a bounded LRU cache of snapshot HNSW indexes.
type temporalIndexCache struct {
	mu         sync.Mutex
	entries    map[temporalIndexKey]*temporalIndexEntry
	lru        *list.List
	maxBytes   int64
	maxEntries int
	totalBytes int64
	// singleflight coordination
	building map[temporalIndexKey]*sync.Once
	db       *Database
}

func newTemporalIndexCache(db *Database, maxBytes int64, maxEntries int) *temporalIndexCache {
	return &temporalIndexCache{
		entries:    make(map[temporalIndexKey]*temporalIndexEntry),
		lru:        list.New(),
		maxBytes:   maxBytes,
		maxEntries: maxEntries,
		building:   make(map[temporalIndexKey]*sync.Once),
		db:         db,
	}
}

func (c *temporalIndexCache) getOrBuild(ctx context.Context, key temporalIndexKey, col *Collection) (*temporalIndexEntry, error) {
	c.mu.Lock()
	if entry, ok := c.entries[key]; ok {
		c.lru.MoveToFront(entry.element)
		entry.leaseCount.Add(1)
		c.mu.Unlock()
		return entry, nil
	}

	// Singleflight: only one goroutine builds per key.
	once, exists := c.building[key]
	if !exists {
		once = &sync.Once{}
		c.building[key] = once
	}
	c.mu.Unlock()

	var entry *temporalIndexEntry
	var buildErr error
	once.Do(func() {
		entry, buildErr = c.buildEntry(ctx, key, col)
		c.mu.Lock()
		delete(c.building, key)
		if buildErr == nil && entry != nil {
			c.entries[key] = entry
			entry.element = c.lru.PushFront(entry)
			c.totalBytes += entry.bytes
			c.evictLocked()
			entry.leaseCount.Add(1) // lease for this request
		}
		c.mu.Unlock()
	})

	if buildErr != nil {
		return nil, buildErr
	}

	// If another goroutine built it while we waited, grab the lease.
	c.mu.Lock()
	if entry == nil {
		entry = c.entries[key]
	}
	if entry != nil && entry.leaseCount.Load() == 0 {
		entry.leaseCount.Add(1)
		c.lru.MoveToFront(entry.element)
	}
	c.mu.Unlock()

	if entry == nil {
		return nil, fmt.Errorf("failed to build temporal index for %s@LSN%d", key.collection, key.lsn)
	}
	return entry, nil
}

func (c *temporalIndexCache) buildEntry(ctx context.Context, key temporalIndexKey, col *Collection) (*temporalIndexEntry, error) {
	snap, err := c.db.SnapshotAtLSN(ctx, key.lsn)
	if err != nil {
		return nil, fmt.Errorf("temporal ANN snapshot: %w", err)
	}

	// Collect all visible vectors at this snapshot.
	var vectors [][]float32
	var ids []string
	err = col.ListVisibleAtLSN(ctx, key.lsn, func(r *Record) bool {
		if len(r.Vector) == 0 {
			return true
		}
		vectors = append(vectors, r.Vector)
		ids = append(ids, r.ID)
		return true
	})
	if err != nil {
		snap.Close()
		return nil, err
	}
	if len(vectors) == 0 {
		snap.Close()
		return nil, fmt.Errorf("no vectors visible at LSN %d", key.lsn)
	}

	// Build immutable HNSW index with a local-ordinal provider.
	localOrdinalToID := make([]string, len(ids))
	copy(localOrdinalToID, ids)

	distFn, _ := util.GetDistanceFunc(util.DistanceMetric(col.config.Metric))
	provider := &temporalVectorProvider{
		vectors:    vectors,
		dimension:  col.Dimension(),
		metric:     util.DistanceMetric(col.config.Metric),
		distanceFn: distFn,
	}

	hnswCfg := &index.HNSWConfig{
		Dimension:      key.dimension,
		M:              key.m,
		EfConstruction: key.efConst,
		EfSearch:       key.efConst * 2,
		ML:             1.0 / float64(key.m),
		Metric:         util.DistanceMetric(key.metric),
		Provider:       provider,
		IDMapCapacity:  len(vectors),
	}

	idx, err := index.NewHNSW(hnswCfg)
	if err != nil {
		snap.Close()
		return nil, fmt.Errorf("create temporal HNSW: %w", err)
	}

	// Batch insert all vectors with local ordinals.
	entries := make([]*index.VectorEntry, len(vectors))
	for i, v := range vectors {
		entries[i] = &index.VectorEntry{
			ID:      ids[i],
			Vector:  cloneFloat32Slice(v),
			Ordinal: uint32(i),
			Version: 1,
		}
	}
	if err := idx.BatchInsert(ctx, entries); err != nil {
		idx.Close()
		snap.Close()
		return nil, fmt.Errorf("batch insert temporal HNSW: %w", err)
	}

	estBytes := int64(len(vectors) * (key.dimension*4 + 64)) // vector + HNSW per-node overhead
	return &temporalIndexEntry{
		key:         key,
		index:       idx,
		ordinalToID: localOrdinalToID,
		snapshot:    snap,
		bytes:       estBytes,
		builtAt:     time.Now(),
	}, nil
}

func (c *temporalIndexCache) release(entry *temporalIndexEntry) {
	entry.leaseCount.Add(-1)
}

func (c *temporalIndexCache) evictLocked() {
	for (c.maxEntries > 0 && len(c.entries) > c.maxEntries) ||
		(c.maxBytes > 0 && c.totalBytes > c.maxBytes) {
		elem := c.lru.Back()
		if elem == nil {
			break
		}
		entry := elem.Value.(*temporalIndexEntry)
		if entry.leaseCount.Load() > 0 {
			break // can't evict in-use entry
		}
		c.lru.Remove(elem)
		delete(c.entries, entry.key)
		c.totalBytes -= entry.bytes
		entry.index.Close()
		entry.snapshot.Close()
	}
}

func (c *temporalIndexCache) close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for key, entry := range c.entries {
		entry.index.Close()
		entry.snapshot.Close()
		delete(c.entries, key)
	}
	c.lru.Init()
	c.totalBytes = 0
}

// temporalVectorProvider implements hnsw.VectorProvider for temporal indexes.
type temporalVectorProvider struct {
	vectors    [][]float32
	dimension  int
	metric     util.DistanceMetric
	distanceFn func([]float32, []float32) float32
}

func (p *temporalVectorProvider) GetByOrdinal(ordinal uint32) ([]float32, error) {
	if int(ordinal) >= len(p.vectors) {
		return nil, fmt.Errorf("ordinal %d out of range", ordinal)
	}
	return p.vectors[ordinal], nil
}

func (p *temporalVectorProvider) Distance(a []float32, ordinal uint32) (float32, error) {
	b, err := p.GetByOrdinal(ordinal)
	if err != nil {
		return 0, err
	}
	return p.distanceFn(a, b), nil
}

func (p *temporalVectorProvider) Dim() int { return p.dimension }

func cloneFloat32Slice(src []float32) []float32 {
	if len(src) == 0 {
		return nil
	}
	dst := make([]float32, len(src))
	copy(dst, src)
	return dst
}

// --- Dispatch integration ---

// temporalANNEnabled returns true if temporal ANN caching is configured.
func (db *Database) temporalANNEnabled() bool {
	return db.config.TemporalANN.MaxBytes > 0 || db.config.TemporalANN.MaxEntries > 0
}

// executeMultiModalWithANN attempts temporal ANN for the candidate set.
// Falls back to exact scoring if ANN is disabled, the candidate set is too
// small, or no cache entry is available.
func (e *Executor) executeMultiModalWithANN(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan, candidates map[string]struct{}) (*SearchResults, error) {
	if !e.db.temporalANNEnabled() {
		return e.scoreCandidatesAtLSN(ctx, col, plan, candidates, plan.SnapshotLSN)
	}

	snapshotLSN := plan.SnapshotLSN
	minCandidates := e.db.config.TemporalANN.MinCandidates
	if minCandidates <= 0 {
		minCandidates = 100
	}
	if len(candidates) < minCandidates {
		return e.scoreCandidatesAtLSN(ctx, col, plan, candidates, snapshotLSN)
	}

	// Build cache key.
	cfg := col.Config()
	key := temporalIndexKey{
		collection: col.name,
		lsn:        snapshotLSN,
		dimension:  cfg.Dimension,
		metric:     int(cfg.Metric),
		m:          cfg.M,
		efConst:    cfg.EfConstruction,
	}

	entry, err := e.db.temporalCache.getOrBuild(ctx, key, col)
	if err != nil {
		// Cache miss/build failure → fall back to exact.
		return e.scoreCandidatesAtLSN(ctx, col, plan, candidates, snapshotLSN)
	}
	defer e.db.temporalCache.release(entry)

	// Map graph candidate IDs to local ordinals in the temporal index.
	localOrdinals := make(map[uint32]bool, len(candidates))
	for id := range candidates {
		for ord, oid := range entry.ordinalToID {
			if oid == id {
				localOrdinals[uint32(ord)] = true
				break
			}
		}
	}
	if len(localOrdinals) == 0 {
		return e.scoreCandidatesAtLSN(ctx, col, plan, candidates, snapshotLSN)
	}

	// Build a local-ordinal bitmap filter for the temporal HNSW.
	filter := &temporalCandidateFilter{allowed: localOrdinals}
	efSearch := e.db.config.TemporalANN.EfSearch
	if efSearch <= 0 {
		efSearch = entry.key.efConst * 4
	}

	// Search the temporal HNSW with the candidate filter.
	k := plan.Limit
	if k <= 0 {
		k = 10
	}
	results, err := entry.index.Search(ctx, plan.QueryVector, k*10, filter)
	if err != nil {
		return e.scoreCandidatesAtLSN(ctx, col, plan, candidates, snapshotLSN)
	}

	// Exact rerank: map local ordinals back to record IDs, get historical
	// vectors, and compute exact scores.
	type scored struct {
		id    string
		score float32
	}
	var reranked []scored
	for _, r := range results {
		if int(r.Ordinal) >= len(entry.ordinalToID) {
			continue
		}
		recID := entry.ordinalToID[r.Ordinal]
		if _, ok := candidates[recID]; !ok {
			continue // not in graph/scalar candidate set
		}
		rec, err := col.GetAtLSN(ctx, recID, snapshotLSN)
		if err != nil || rec == nil || len(rec.Vector) == 0 {
			continue
		}
		score := computeVectorScore(col, optimizer.VectorFuncProjection{
			IsDistance:  true,
			QueryVector: plan.QueryVector,
		}, rec.Vector)
		reranked = append(reranked, scored{id: recID, score: score})
	}

	// Sort by score ascending (distance).
	for i := 1; i < len(reranked); i++ {
		for j := i; j > 0 && reranked[j].score < reranked[j-1].score; j-- {
			reranked[j], reranked[j-1] = reranked[j-1], reranked[j]
		}
	}
	if len(reranked) > k {
		reranked = reranked[:k]
	}

	out := &SearchResults{Results: make([]*SearchResult, len(reranked)), Total: len(reranked)}
	for i, s := range reranked {
		out.Results[i] = &SearchResult{ID: s.id, Score: s.score}
	}
	return out, nil
}

// temporalCandidateFilter implements index.GraphFilter for temporal ANN.
type temporalCandidateFilter struct {
	allowed map[uint32]bool
}

func (f *temporalCandidateFilter) Test(idx uint64) bool {
	return f.allowed[uint32(idx)]
}
