package libravdb

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/bits"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/internal/storage"
)

const (
	costModelStatisticsVersion = uint32(2)
	costModelMaxTopValues      = 256
	costModelDistinctBits      = 2048
	costModelNumericSampleSize = 256
	costModelHistogramBuckets  = 8
)

// collectionCostModelState is deliberately collection-local: queries only
// read immutable snapshots and successful mutations merely invalidate them.
// This keeps the estimator off the write hot path.
type collectionCostModelState struct {
	mu         sync.RWMutex
	statistics *CostModelStatistics
	dirty      bool
	pending    uint32
}

func newCollectionCostModelState(payload []byte, currentDataLSN uint64) *collectionCostModelState {
	state := &collectionCostModelState{dirty: true}
	if len(payload) == 0 {
		return state
	}
	var stats CostModelStatistics
	if err := json.Unmarshal(payload, &stats); err != nil || stats.Version != costModelStatisticsVersion {
		return state
	}
	state.statistics = cloneCostModelStatistics(&stats)
	state.dirty = stats.DataLSN != currentDataLSN
	return state
}

func (s *collectionCostModelState) markDirty() {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.dirty = true
	s.mu.Unlock()
}

func (s *collectionCostModelState) snapshot() (*CostModelStatistics, bool) {
	if s == nil {
		return nil, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.dirty || s.statistics == nil || s.statistics.Version != costModelStatisticsVersion {
		return nil, false
	}
	return cloneCostModelStatistics(s.statistics), true
}

func (s *collectionCostModelState) replace(stats *CostModelStatistics) {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.statistics = cloneCostModelStatistics(stats)
	s.dirty = false
	s.pending = 0
	s.mu.Unlock()
}

func (s *collectionCostModelState) mergeFeedback(plan *optimizer.PhysicalPlan, metrics *QueryMetrics) (*CostModelStatistics, bool) {
	if s == nil || plan == nil || metrics == nil {
		return nil, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.dirty || s.statistics == nil || s.statistics.Version != costModelStatisticsVersion {
		return nil, false
	}
	mergeCostModelFeedback(s.statistics, plan, metrics)
	s.pending++
	// Persist at a bounded cadence. Statistics refreshes and mutations already
	// reset this state, so no stale feedback can be published.
	if s.pending < 16 {
		return nil, false
	}
	s.pending = 0
	return cloneCostModelStatistics(s.statistics), true
}

func (db *Database) recordCostModelFeedback(plan *optimizer.PhysicalPlan, metrics *QueryMetrics) {
	if db == nil || plan == nil || metrics == nil {
		return
	}
	collection, err := db.GetCollection(plan.CollectionName)
	if err != nil {
		return
	}
	updated, shouldPersist := collection.costModel.mergeFeedback(plan, metrics)
	if !shouldPersist {
		return
	}
	payload, err := json.Marshal(updated)
	if err != nil {
		return
	}
	store, ok := db.storage.(storage.CostModelStatisticsStore)
	if !ok {
		return
	}
	published, err := store.SetCollectionCostModelStatsIfDataLSN(context.Background(), plan.CollectionName, updated.DataLSN, payload)
	if err != nil || !published {
		collection.costModel.markDirty()
	}
}

func mergeCostModelFeedback(stats *CostModelStatistics, plan *optimizer.PhysicalPlan, metrics *QueryMetrics) {
	if stats == nil {
		return
	}
	now := time.Now().UTC()
	if plan.HasGraphTraversal && metrics.ActGraphCandidates >= 0 {
		if stats.Graph.PatternSamples == nil {
			stats.Graph.PatternSamples = make(map[string]CostModelGraphPatternSample)
		}
		key := costModelGraphPatternKey(plan)
		sample := stats.Graph.PatternSamples[key]
		sample.Observations++
		sample.Seeds += uint64(max(0, metrics.GraphSeeds))
		sample.Vertices += uint64(max(0, metrics.GraphVertices))
		sample.Candidates += uint64(max(0, metrics.ActGraphCandidates))
		sample.UpdatedAt = now
		stats.Graph.PatternSamples[key] = sample
	}
	if metrics.ANNEf > 0 && (metrics.PlanChosen == DispatchFilteredANN || metrics.PlanChosen == DispatchIterativeANNThenFilter) {
		mergeRankBucket(&stats.RankBucketYields, metrics.ANNEf, metrics.FilterValidHits, metrics.ExecutionNanos)
	}
	if stats.Calibrations == nil {
		stats.Calibrations = make(map[string]CostModelCalibrationProfile)
	}
	key := costModelCalibrationKey(stats)
	profile := stats.Calibrations[key]
	profile.LastCalibrated = now
	candidates := uint64(max(1, metrics.ActConjunctionCandidates))
	switch metrics.PlanChosen {
	case DispatchExactCandidateScan:
		profile.ExactSamples++
		profile.ExactCandidates += candidates
		profile.ExactNanos += metrics.ExecutionNanos
	case DispatchFilteredANN:
		profile.FilteredSamples++
		profile.FilteredCandidates += uint64(max(1, metrics.ANNEf))
		profile.FilteredNanos += metrics.ExecutionNanos
	case DispatchIterativeANNThenFilter:
		profile.IterativeSamples++
		profile.IterativeCandidates += uint64(max(1, metrics.ANNEf))
		profile.IterativeNanos += metrics.ExecutionNanos
	}
	stats.Calibrations[key] = profile
}

func mergeRankBucket(buckets *[]CostModelRankBucket, upperRank, validHits int, nanos uint64) {
	if upperRank <= 0 {
		return
	}
	const lowerRank = uint32(1)
	upper := uint32(upperRank)
	for i := range *buckets {
		bucket := &(*buckets)[i]
		if bucket.LowerRank == lowerRank && bucket.UpperRank == upper {
			bucket.Candidates += uint64(upperRank)
			bucket.Valid += uint64(max(0, validHits))
			bucket.DistanceNanos += nanos
			return
		}
	}
	*buckets = append(*buckets, CostModelRankBucket{
		LowerRank: lowerRank, UpperRank: upper, Candidates: uint64(upperRank),
		Valid: uint64(max(0, validHits)), DistanceNanos: nanos,
	})
	// Bound the persisted envelope count even when callers use many ef values.
	if len(*buckets) > 32 {
		sort.Slice(*buckets, func(i, j int) bool { return (*buckets)[i].Candidates > (*buckets)[j].Candidates })
		*buckets = (*buckets)[:32]
	}
}

func costModelGraphPatternKey(plan *optimizer.PhysicalPlan) string {
	key := fmt.Sprintf("seed=%t,label=%s,anchor=%t", plan.HasExplicitSeed, plan.SeedLabel, plan.HasVectorAnchor)
	for _, edge := range plan.GraphEdges {
		key += fmt.Sprintf("|k=%d,d=%d,min=%d,max=%d", edge.EdgeKind, edge.Direction, edge.QuantMin, edge.QuantMax)
	}
	return key
}

func costModelCalibrationKey(stats *CostModelStatistics) string {
	return fmt.Sprintf("%s/d=%d/m=%d", stats.HardwareProfile, stats.Dimension, stats.Metric)
}

// AnalyzeCollection performs a bounded-memory, exact metadata scan and
// persists the resulting versioned statistics. It is explicit by design:
// edge deployments should choose when to spend an O(N) analysis pass rather
// than have an ordinary query unexpectedly do it.
func (db *Database) AnalyzeCollection(ctx context.Context, name string) (CostModelStatistics, error) {
	if db == nil {
		return CostModelStatistics{}, fmt.Errorf("database is nil")
	}
	col, err := db.GetCollection(name)
	if err != nil {
		return CostModelStatistics{}, err
	}
	store, ok := db.storage.(storage.CostModelStatisticsStore)
	if !ok {
		return CostModelStatistics{}, fmt.Errorf("storage engine does not support cost-model statistics persistence")
	}
	beforeLSN, err := store.CollectionDataLSN(name)
	if err != nil {
		return CostModelStatistics{}, fmt.Errorf("read collection statistics watermark: %w", err)
	}
	stats, err := col.buildCostModelStatistics(ctx)
	if err != nil {
		return CostModelStatistics{}, err
	}
	stats.DataLSN = beforeLSN
	payload, err := json.Marshal(stats)
	if err != nil {
		return CostModelStatistics{}, fmt.Errorf("encode cost-model statistics: %w", err)
	}
	published, err := store.SetCollectionCostModelStatsIfDataLSN(ctx, name, beforeLSN, payload)
	if err != nil {
		return CostModelStatistics{}, fmt.Errorf("persist cost-model statistics: %w", err)
	}
	if !published {
		return CostModelStatistics{}, fmt.Errorf("collection changed during analysis; retry AnalyzeCollection")
	}
	col.costModel.replace(&stats)
	return *cloneCostModelStatistics(&stats), nil
}

func (c *Collection) buildCostModelStatistics(ctx context.Context) (CostModelStatistics, error) {
	stats := CostModelStatistics{
		Version:         costModelStatisticsVersion,
		CollectionName:  c.name,
		Fields:          make(map[string]CostModelFieldStats),
		Dimension:       c.config.Dimension,
		Metric:          c.config.Metric,
		HardwareProfile: runtime.GOOS + "/" + runtime.GOARCH,
		RefreshedAt:     time.Now().UTC(),
	}
	fields := make(map[string]*costModelFieldAccumulator)
	if err := c.Iterate(ctx, func(record Record) error {
		stats.RowCount++
		for field, value := range record.Metadata {
			acc := fields[field]
			if acc == nil {
				acc = newCostModelFieldAccumulator()
				fields[field] = acc
			}
			acc.observe(value)
		}
		return nil
	}); err != nil {
		return CostModelStatistics{}, err
	}
	for field, acc := range fields {
		stats.Fields[field] = acc.statistics(stats.RefreshedAt)
	}
	if c.graph != nil {
		graphStats := c.graph.Stats()
		stats.Graph.EdgeCount = graphStats.EdgesAdded - graphStats.EdgesRemoved
		stats.Graph.VertexCount = stats.RowCount
		if stats.Graph.VertexCount > 0 {
			stats.Graph.AverageBranching = float64(stats.Graph.EdgeCount) / float64(stats.Graph.VertexCount)
		}
		stats.Graph.PatternSamples = make(map[string]CostModelGraphPatternSample)
	}
	stats.Calibrations = make(map[string]CostModelCalibrationProfile)
	return stats, nil
}

func costModelMetadataKey(value interface{}) string {
	return "text:" + recordMetaToString(value)
}

type costModelFieldAccumulator struct {
	count         uint64
	nullCount     uint64
	distinctBits  [costModelDistinctBits / 64]uint64
	heavyHitters  map[string]uint64
	numericSample []float64
	numericSeen   uint64
}

func newCostModelFieldAccumulator() *costModelFieldAccumulator {
	return &costModelFieldAccumulator{heavyHitters: make(map[string]uint64, costModelMaxTopValues)}
}

func (a *costModelFieldAccumulator) observe(value interface{}) {
	a.count++
	if value == nil {
		a.nullCount++
		return
	}
	key := costModelMetadataKey(value)
	hash := costModelHash(key)
	a.distinctBits[(hash%costModelDistinctBits)/64] |= uint64(1) << (hash % 64)
	if count, ok := a.heavyHitters[key]; ok {
		a.heavyHitters[key] = count + 1
	} else if len(a.heavyHitters) < costModelMaxTopValues {
		a.heavyHitters[key] = 1
	} else {
		for candidate, count := range a.heavyHitters {
			if count <= 1 {
				delete(a.heavyHitters, candidate)
			} else {
				a.heavyHitters[candidate] = count - 1
			}
		}
	}
	if numeric, ok := metadataNumericValue(value); ok {
		a.numericSeen++
		if len(a.numericSample) < costModelNumericSampleSize {
			a.numericSample = append(a.numericSample, numeric)
			return
		}
		// Deterministic reservoir replacement avoids a per-field unbounded
		// allocation while keeping the sample representative across values.
		position := costModelHash(key+fmt.Sprintf("/%d", a.numericSeen)) % a.numericSeen
		if position < costModelNumericSampleSize {
			a.numericSample[position] = numeric
		}
	}
}

func (a *costModelFieldAccumulator) statistics(refreshedAt time.Time) CostModelFieldStats {
	stats := CostModelFieldStats{
		Count:         a.count,
		NullCount:     a.nullCount,
		Distinct:      costModelDistinctEstimate(a.distinctBits[:]),
		TopValues:     cloneUint64Map(a.heavyHitters),
		Histogram:     costModelHistogram(a.numericSample, a.numericSeen),
		LastRefreshed: refreshedAt,
	}
	return stats
}

func costModelHash(value string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(value))
	return h.Sum64()
}

func costModelDistinctEstimate(words []uint64) uint64 {
	setBits := 0
	for _, word := range words {
		setBits += bits.OnesCount64(word)
	}
	if setBits == 0 {
		return 0
	}
	if setBits >= costModelDistinctBits {
		return costModelDistinctBits
	}
	estimate := -float64(costModelDistinctBits) * math.Log(1-float64(setBits)/costModelDistinctBits)
	return uint64(math.Ceil(estimate))
}

func costModelHistogram(sample []float64, observed uint64) []CostModelHistogramBucket {
	if len(sample) == 0 || observed == 0 {
		return nil
	}
	sorted := append([]float64(nil), sample...)
	sort.Float64s(sorted)
	bucketCount := min(costModelHistogramBuckets, len(sorted))
	buckets := make([]CostModelHistogramBucket, 0, bucketCount)
	for bucket := 0; bucket < bucketCount; bucket++ {
		start := bucket * len(sorted) / bucketCount
		end := (bucket+1)*len(sorted)/bucketCount - 1
		count := uint64((bucket+1)*int(observed)/bucketCount - bucket*int(observed)/bucketCount)
		buckets = append(buckets, CostModelHistogramBucket{Lower: sorted[start], Upper: sorted[end], Count: count})
	}
	return buckets
}

func cloneCostModelStatistics(stats *CostModelStatistics) *CostModelStatistics {
	if stats == nil {
		return nil
	}
	copyStats := *stats
	copyStats.Fields = make(map[string]CostModelFieldStats, len(stats.Fields))
	for field, value := range stats.Fields {
		value.TopValues = cloneUint64Map(value.TopValues)
		value.Histogram = append([]CostModelHistogramBucket(nil), value.Histogram...)
		copyStats.Fields[field] = value
	}
	copyStats.Graph.LabelCounts = cloneUint64Map(stats.Graph.LabelCounts)
	copyStats.Graph.EdgeKindCounts = make(map[uint16]uint64, len(stats.Graph.EdgeKindCounts))
	for kind, count := range stats.Graph.EdgeKindCounts {
		copyStats.Graph.EdgeKindCounts[kind] = count
	}
	copyStats.Graph.PatternSamples = make(map[string]CostModelGraphPatternSample, len(stats.Graph.PatternSamples))
	for pattern, sample := range stats.Graph.PatternSamples {
		copyStats.Graph.PatternSamples[pattern] = sample
	}
	copyStats.RankBucketYields = append([]CostModelRankBucket(nil), stats.RankBucketYields...)
	copyStats.Calibrations = make(map[string]CostModelCalibrationProfile, len(stats.Calibrations))
	for key, profile := range stats.Calibrations {
		copyStats.Calibrations[key] = profile
	}
	return &copyStats
}

func cloneUint64Map(source map[string]uint64) map[string]uint64 {
	if source == nil {
		return nil
	}
	result := make(map[string]uint64, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}
