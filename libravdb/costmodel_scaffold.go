package libravdb

import (
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// CostModelStatistics is the versioned collection-scoped payload persisted by
// the storage engine. A payload is invalidated by every base-row mutation, so
// the planner never consumes statistics known to predate the live records.
type CostModelStatistics struct {
	Version          uint32
	CollectionName   string
	DataLSN          uint64
	RefreshedAt      time.Time
	RowCount         uint64
	Fields           map[string]CostModelFieldStats
	Graph            CostModelGraphStats
	RankBucketYields []CostModelRankBucket
	Calibrations     map[string]CostModelCalibrationProfile
	Dimension        int
	Metric           DistanceMetric
	HardwareProfile  string
}

// CostModelFieldStats is a bounded metadata synopsis. Distinct is a
// linear-counting estimate, TopValues is a bounded heavy-hitter summary, and
// Histogram is a reservoir-derived numeric distribution.
type CostModelFieldStats struct {
	Count         uint64
	NullCount     uint64
	Distinct      uint64
	TopValues     map[string]uint64
	Histogram     []CostModelHistogramBucket
	LastRefreshed time.Time
}

type CostModelHistogramBucket struct {
	Lower float64
	Upper float64
	Count uint64
}

// CostModelGraphStats contains graph-wide counters plus bounded observed
// cardinalities keyed by normalized MATCH pattern shape.
type CostModelGraphStats struct {
	VertexCount      uint64
	EdgeCount        uint64
	LabelCounts      map[string]uint64
	EdgeKindCounts   map[uint16]uint64
	AverageBranching float64
	PatternSamples   map[string]CostModelGraphPatternSample
}

type CostModelGraphPatternSample struct {
	Observations uint64
	Seeds        uint64
	Vertices     uint64
	Candidates   uint64
	UpdatedAt    time.Time
}

// CostModelRankBucket records observed predicate yield by ANN rank range.
type CostModelRankBucket struct {
	LowerRank     uint32
	UpperRank     uint32
	Candidates    uint64
	Valid         uint64
	DistanceNanos uint64
}

// CostModelCalibrationProfile is a bounded online calibration aggregate for
// one hardware/dimension/metric tuple. It stores totals rather than raw query
// history so it remains edge-friendly and serializable.
type CostModelCalibrationProfile struct {
	ExactSamples        uint64
	ExactCandidates     uint64
	ExactNanos          uint64
	FilteredSamples     uint64
	FilteredCandidates  uint64
	FilteredNanos       uint64
	IterativeSamples    uint64
	IterativeCandidates uint64
	IterativeNanos      uint64
	LastCalibrated      time.Time
}

// CostModelObservation is one retained estimate/actual execution sample.
//
// This inspection ring is separate from the bounded aggregates persisted in
// CostModelStatistics. It is diagnostic only and never influences dispatch.
type CostModelObservation struct {
	CollectionName string
	RecordedAt     time.Time
	Metrics        QueryMetrics
}

// costModelStats keeps a bounded feedback buffer so estimates can eventually
// be compared with observed cardinalities and latency without changing query
// correctness. Its size is fixed so it cannot grow with query volume.
type costModelStats struct {
	mu           sync.RWMutex
	maxSamples   int
	observations []CostModelObservation
}

func newCostModelStats(maxSamples int) *costModelStats {
	if maxSamples <= 0 {
		maxSamples = 2048
	}
	return &costModelStats{maxSamples: maxSamples}
}

func (s *costModelStats) record(collection string, metrics *QueryMetrics) {
	if s == nil || metrics == nil {
		return
	}
	copyMetrics := *metrics
	copyMetrics.Transitions = append([]string(nil), metrics.Transitions...)
	copyMetrics.FilterCandidatesPerHit = append([]int(nil), metrics.FilterCandidatesPerHit...)

	s.mu.Lock()
	defer s.mu.Unlock()
	s.observations = append(s.observations, CostModelObservation{
		CollectionName: collection,
		RecordedAt:     time.Now(),
		Metrics:        copyMetrics,
	})
	if excess := len(s.observations) - s.maxSamples; excess > 0 {
		copy(s.observations, s.observations[excess:])
		s.observations = s.observations[:s.maxSamples]
	}
}

func (s *costModelStats) snapshot() []CostModelObservation {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make([]CostModelObservation, len(s.observations))
	for i := range s.observations {
		result[i] = s.observations[i]
		result[i].Metrics.Transitions = append([]string(nil), s.observations[i].Metrics.Transitions...)
		result[i].Metrics.FilterCandidatesPerHit = append([]int(nil), s.observations[i].Metrics.FilterCandidatesPerHit...)
	}
	return result
}

func (s *costModelStats) reset() {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.observations = nil
	s.mu.Unlock()
}

// CostModelObservations returns bounded cost-model feedback samples.
//
// This is an inspection hook; dispatch reads persisted aggregates instead.
func (db *Database) CostModelObservations() []CostModelObservation {
	if db == nil {
		return nil
	}
	return db.costModelStats.snapshot()
}

// ResetCostModelObservations clears the in-memory calibration buffer.
//
// This only clears diagnostic history; persisted aggregates are unaffected.
func (db *Database) ResetCostModelObservations() {
	if db != nil {
		db.costModelStats.reset()
	}
}

// hybridCardinalityEstimate consumes a validated collection statistics snapshot
// when one exists and otherwise preserves the legacy conservative heuristic.
// Source and confidence remain visible so explain output can distinguish the
// two cases.
func (e *Executor) hybridCardinalityEstimate(plan *optimizer.PhysicalPlan, corpusSize, k int) hybridCardinalityEstimate {
	if corpusSize < 1 {
		corpusSize = 1
	}
	if k <= 0 {
		k = 10
	}

	estimate := hybridCardinalityEstimate{
		source:      "provisional_heuristic",
		confidence:  0.05,
		assumptions: []string{"equality_selectivity=0.10", "other_selectivity=0.50", "graph_branching_factor=5", "predicate_independence=true"},
	}
	var statistics *CostModelStatistics
	if e != nil && e.db != nil {
		if collection, err := e.db.GetCollection(plan.CollectionName); err == nil {
			if snapshot, ok := collection.costModel.snapshot(); ok && snapshot.RowCount == uint64(corpusSize) {
				statistics = snapshot
			}
		}
	}

	scalarSel := 1.0
	if plan.HasRelationalQuery && len(plan.Predicates) > 0 {
		for _, predicate := range plan.Predicates {
			if predicate.Operator == 12 && statistics != nil { // KindEquals
				fieldStats, knownField := statistics.Fields[predicate.Column]
				if knownField && statistics.RowCount > 0 {
					if count, heavyHitter := fieldStats.TopValues["text:"+string(predicate.Value)]; heavyHitter {
						scalarSel *= float64(count) / float64(statistics.RowCount)
						estimate.source = "analyzed_collection_statistics"
						estimate.confidence = 0.90
						estimate.assumptions = []string{"equality_top_value_frequency", "statistics_version=1"}
						continue
					}
					if fieldStats.Distinct > 0 {
						scalarSel *= 1.0 / float64(fieldStats.Distinct)
						estimate.source = "analyzed_collection_statistics"
						estimate.confidence = 0.60
						estimate.assumptions = []string{"equality_distinct_uniform_fallback", "statistics_version=1"}
						continue
					}
				}
				// A statistics snapshot may not cover every field. Preserve the
				// legacy equality fallback for those fields.
				scalarSel *= 0.10
			} else {
				if predicate.Operator == 12 { // KindEquals without statistics
					scalarSel *= 0.10
					continue
				}
				scalarSel *= 0.50
			}
		}
	}
	estimate.scalarCandidates = int(float64(corpusSize) * scalarSel)

	if plan.HasGraphTraversal && len(plan.GraphEdges) > 0 {
		seeds := 1
		if plan.HasVectorAnchor {
			seeds = k * 2
		} else if plan.SeedLabel != "" {
			seeds = 10
		}
		value := float64(seeds)
		if statistics != nil {
			if sample, ok := statistics.Graph.PatternSamples[costModelGraphPatternKey(plan)]; ok && sample.Observations >= 3 && sample.Seeds > 0 {
				value = float64(seeds) * float64(sample.Candidates) / float64(sample.Seeds)
				estimate.source = "analyzed_collection_statistics"
				estimate.confidence = max(estimate.confidence, 0.75)
				estimate.assumptions = append(estimate.assumptions, "graph_pattern_sample")
			}
		}
		for _, edge := range plan.GraphEdges {
			if statistics != nil {
				if _, ok := statistics.Graph.PatternSamples[costModelGraphPatternKey(plan)]; ok {
					break
				}
			}
			hops := int(edge.QuantMax)
			if hops == 0 {
				hops = 1
			}
			if hops > 10 {
				hops = 10
			}
			for hop := 0; hop < hops; hop++ {
				value *= 5.0
			}
		}
		if value > float64(corpusSize) {
			value = float64(corpusSize)
		}
		estimate.graphCandidates = int(value)
		estimate.graphSeeds = seeds
	}

	switch {
	case plan.HasRelationalQuery && plan.HasGraphTraversal:
		scalarFraction := float64(estimate.scalarCandidates) / float64(corpusSize)
		graphFraction := float64(estimate.graphCandidates) / float64(corpusSize)
		estimate.conjunctionCandidates = int(float64(corpusSize) * scalarFraction * graphFraction)
		if estimate.conjunctionCandidates < 1 {
			estimate.conjunctionCandidates = 1
		}
	case plan.HasRelationalQuery:
		estimate.conjunctionCandidates = estimate.scalarCandidates
	case plan.HasGraphTraversal:
		estimate.conjunctionCandidates = estimate.graphCandidates
	default:
		estimate.conjunctionCandidates = corpusSize
	}
	return estimate
}

type hybridCardinalityEstimate struct {
	scalarCandidates      int
	graphCandidates       int
	conjunctionCandidates int
	graphSeeds            int
	source                string
	confidence            float64
	assumptions           []string
}

func calibratedExactCandidateCap(stats *CostModelStatistics) int {
	if stats == nil {
		return 0
	}
	profile, ok := stats.Calibrations[costModelCalibrationKey(stats)]
	if !ok || profile.ExactSamples < 4 || profile.FilteredSamples < 4 || profile.ExactCandidates == 0 || profile.FilteredNanos == 0 {
		return 0
	}
	exactPerCandidate := float64(profile.ExactNanos) / float64(profile.ExactCandidates)
	filteredPerQuery := float64(profile.FilteredNanos) / float64(profile.FilteredSamples)
	if exactPerCandidate <= 0 || filteredPerQuery <= 0 {
		return 0
	}
	cap := int(filteredPerQuery / exactPerCandidate)
	if cap < 100 {
		cap = 100
	}
	if cap > exactCandidateCap {
		cap = exactCandidateCap
	}
	return cap
}
