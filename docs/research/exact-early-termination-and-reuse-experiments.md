# Exact Early-Termination and Distance-Reuse Experiments

Date: 2026-07-12

Target configuration: 5,000 normalized random vectors, 768 dimensions,
`M=36`, `efConstruction=200`, `efSearch=200`, Apple M2.

## Tail-Norm Lower Bound

For a checkpoint `t`, squared L2 has the exact lower bound

```text
D(c, s) >= P_t + (||c_tail|| - ||s_tail||)^2
```

where `P_t` is the squared L2 accumulated through dimension `t`.

A deterministic 1-in-64 sample of the real x8 diversity-pruning traffic
observed 1,266,992 candidate/selected pairs at 128-dimensional checkpoints.

| Metric | Result |
|---|---:|
| Plain partial-L2 termination rate | 0.2262% |
| Tail-bound termination rate | 0.2294% |
| Plain dimensions saved per pair | 0.2895 |
| Tail-bound dimensions saved per pair | 0.2936 |
| Earliest observed tail-bound hit | 640d |

The tail norm adds only 0.0041 saved dimensions per pair. It does not justify
8-20 bytes of metadata per vector, checkpoint branches, square roots, or
8-to-4-to-2 lane compaction. The experimental threshold kernel and its unused
ARM64 assembly were removed.

For a future exact implementation, independently rounding both stored norms
down is not sufficient to make the absolute norm difference conservative.
Safe metadata needs intervals. Given tail norm intervals `[aLo, aHi]` and
`[bLo, bHi]`, the conservative separation is:

```text
max(0, aLo-bHi, bLo-aHi)
```

## Diversity Rejection Position

The x8 kernel's first rejecting lane was measured across 10,135,942 batches.

| Metric | Result |
|---|---:|
| Batches containing a rejection | 26.74% |
| Batches rejecting in lanes 0-3 | 15.93% |
| Share of rejections in lanes 0-3 | 59.55% |
| First batch rejection rate | 32.88% |
| First batch lanes 0-3 rejection rate | 20.73% |

A universal 4+4 split is not attractive because the production one-pass x8
kernel is already 13-15% faster than two x4 calls. Reordering selected vectors
with a move-to-front occluder cache was then tested because the diversity
predicate is order-independent.

| Mode | Graph inserts/s, three runs | Recall@10 | Level-0 links |
|---|---|---:|---:|
| Baseline | 860.2, 861.4, 861.0 | 1.000 | 840,707 |
| Move to front | 857.8, 855.5, 858.1 | 1.000 | 840,707 |

The exact graph was preserved, but construction regressed by 0.4-0.7%. The
experiment was removed.

## Flash-Style Pair-Distance Reuse

Candidate/selected ID pairs were tracked within each insertion to measure the
maximum opportunity for an exact fixed-capacity distance cache.

| Metric | Result |
|---|---:|
| Observed x8 pairs | 81,087,536 |
| Repeated pairs | 2.095% |
| Batches with any repeated pair | 7.787% |
| Fully reusable x8 batches | 0.08133% |

The hit rate is too low for hashing, probing, and partial-lane repacking to pay
for themselves. Flash's compact construction mode remains a separate optional
experiment; exact pair-distance reuse does not transfer profitably to the main
HNSW mode on this workload.

## Scatter Entry Diagnostic

Four independent high-level entry seeds with `ef=50` each were merged under a
total nominal beam budget of 200. Recall fell to 0.838-0.842. Partitioning the
beam destroys too much local frontier state and does not repair the rare
concurrent-build miss. The diagnostic was removed.

## Invalid Proposed Stopping Rules

Two research-agent suggestions must not be implemented:

1. In a single sorted SoA pool, stopping when
   `beam[cursor].distance >= beam[k-1].distance` is effectively true once the
   cursor passes `k`; it would terminate after roughly `k` expansions and is
   not equivalent to HNSW's separate frontier/result stopping rule.
2. A global minimum edge length cannot prove that an expanded node's neighbors
   are outside the result radius. The reverse triangle inequality requires an
   upper bound on the traversed edge length for that proof, not a lower bound.

## Clean Profile After Rejections

Workers=4 concurrent graph-ready construction reached 3,225 inserts/s in the
profiled run, with recall@10=0.999 at ef=200 and 1.000 at ef=300.

| Symbol | Flat CPU | Cumulative CPU |
|---|---:|---:|
| `L2Distance8AlignedPtrNEON` | 49.47% | 49.47% |
| `rejectBySelectedHeuristic` | 1.76% | 44.89% |
| `searchAndSelectForConstructionWithScratch` | 0.53% | 32.92% |
| `searchLevelScratchValues` | 9.68% | 27.11% |
| `soaCandidateQueue.Insert` | 1.06% | 5.63% |

The remaining wall is exact vector arithmetic and the number of diversity
comparisons. Queue layout, early termination, probe reordering, and exact
distance reuse are no longer primary targets for this fixture.

## Final-Reduction Predicate

The production x8 distance kernel was specialized for diversity pruning. It
performs the same FMLA sequence as `L2Distance8AlignedPtrNEON`, but returns a
single predicate and stops final horizontal reduction at the first distance
strictly below the cutoff. It has no midpoint checks and does not alter graph
mathematics.

Five paired kernel runs improved both rejecting and non-rejecting batches by
`1.007x-1.010x`. Three thermally balanced full-build rounds used the order
baseline, predicate, predicate, baseline:

| Round | Baseline inserts/s | Predicate inserts/s | Speedup |
|---:|---:|---:|---:|
| 1 | 861.7 | 862.8 | 1.001x |
| 2 | 862.6 | 868.7 | 1.007x |
| 3 | 857.0 | 857.4 | 1.001x |

Every build produced exactly 840,707 level-0 links and recall@10=1.000. The
predicate is retained as the ARM64 diversity-pruning default. The ordinary x8
distance function remains in use wherever callers need the eight scores.
