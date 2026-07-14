# Semantic HNSW Scale Validation

## Purpose

The existing 5k/768d benchmark uses normalized random vectors. It is useful as
an isotropic stress test, but it does not represent the clustered geometry of
production text embeddings. `BenchmarkHNSWSemanticScale` adds an opt-in scale
check over real Nomic embeddings without adding a 154 MB fixture to Git or CI.

The initial fixture uses:

- 50,000 deduplicated LongMemEval conversation messages.
- 100 held-out LongMemEval questions.
- Nomic Embed Text v1.5 Q8 GGUF, 768 dimensions, normalized float32 output.
- Document text truncated to 512 Unicode characters, matching a normal memory
  chunk rather than exercising the model's long-context path.
- Exact brute-force top-10 truth computed with LibraVDB's SIMD L2 kernels.

The question labels are not used as ground truth. Recall measures whether HNSW
returns the exact nearest vectors under squared L2 for the generated embedding
space.

## Generate

The generator imports the embedding engine from the sibling `libravdbd`
module, so run it with that module as the working directory. Metal access is
required on macOS.

```sh
cd ../libravdbd

LIBRAVDB_LLAMA_LIB=/opt/homebrew/opt/libravdbd/models/llama/llama-darwin-arm64/lib/libllama.dylib \
go run ../libraVDB/scripts/semanticfixture/generate.go \
  -backend gguf \
  -corpus ./data/longmemeval_s_cleaned.json \
  -model /opt/homebrew/opt/libravdbd/models/nomic-embed-text-v1.5/nomic-embed-text-v1.5.Q8_0.gguf \
  -llama-lib /opt/homebrew/opt/libravdbd/models/llama/llama-darwin-arm64/lib/libllama.dylib \
  -output /tmp/nomic-longmemeval-50k-gguf-q8.semantic.f32 \
  -vectors 50000 \
  -queries 100 \
  -max-chars 512
```

The file starts with a 64-byte header and stores document vectors followed by
query vectors as contiguous little-endian float32 values. The header contains
the dimension, vector/query counts, and the first 64 bits of the model SHA-256.

## Benchmark

```sh
LIBRAVDB_SEMANTIC_FIXTURE=/tmp/nomic-longmemeval-50k-gguf-q8.semantic.f32 \
go test ./internal/index/hnsw \
  -run '^$' \
  -bench '^BenchmarkHNSWSemanticScale$' \
  -benchtime=1x \
  -count=1 \
  -v
```

The benchmark builds the same `M=36`, `efConstruction=200` graph with one and
four construction workers. Vector ingestion is preloaded outside the timed
region, so `graph_ready_insert/s` measures topology construction rather than
the unavoidable owned-vector copy. It reports recall@10 and p50/p99 search
latency at `efSearch=200` and `300`.

`RawStoreCap` is set to the fixture count and `IDMapCapacity` is rounded up with
headroom before construction. Leaving `RawStoreCap=0` selects the 64 MB minimum
raw-vector pool, which is intentionally insufficient for 50k vectors at 768d.

Search measurements restore the host `GOMAXPROCS`, warm both ef values, and
alternate their execution order by query to avoid a systematic cache advantage.
Set `LIBRAVDB_SEMANTIC_WORKERS` to a positive integer to run only one
construction concurrency level during a diagnostic repeat.
Set `LIBRAVDB_SEMANTIC_M` to override the default `M=36` for a topology sweep.
Set `LIBRAVDB_SEMANTIC_SERIAL_PREFIX` to build a stable prefix before releasing
the configured construction workers.
Set `LIBRAVDB_SEMANTIC_REPAIR=1` to configure the deferred repair queue and
include a synchronous `FlushRepairs` in graph-ready construction time.

## Initial 50k Result

Fixture SHA-256:
`a176d920c50b0d8e635c522e1452b4f282c0c99b9b223089c66a9ae86bf4a243`.

The serial `M=36`, `efConstruction=200` build was deterministic in the first
run:

- Graph-ready construction: 628 inserts/s.
- Recall@10: 1.000 at ef=200 and ef=300.
- ef=200 search: p50 0.634 ms, p99 1.240 ms.
- ef=300 search: p50 1.045 ms, p99 1.417 ms.

Four-worker construction reached 1.7k-1.9k inserts/s, but topology varied by
schedule. Across four builds, ef=200 recall ranged from 0.996 to 1.000. Some
builds became exact at ef=208, while persistent misses remained through ef=300
in other builds. This proves that the remaining concurrent misses include both
beam-depth and construction-topology failures; raising ef alone is not a full
correction.

A synchronous deferred-repair flush restored 1.000 recall at every measured ef,
but repaired 47,445 of 50,000 nodes and reduced graph-ready throughput to 1,246
inserts/s. The current dirty trigger is therefore too broad to serve as the
default correction; repair must become selective before it can be production
viable.

### Rejected: lock-contention-scoped repair

The adjacency mutation path was verified before testing this hypothesis.
`connectLinkWithHeuristic` acquires `PruneLock` before reading the target list
and holds it through pruning and publication, so contending writers do not
prune stale snapshots or lose one another's updates.

An experimental 32-bit per-node mask recorded failed first lock acquisitions at
M=16. One 50k build marked 14,082 endpoint nodes: 7,539 adjacency targets and
9,021 insertion sources. Repairing target nodes only remained within the time
budget at 3,093 graph-ready inserts/s, but degraded recall to 0.995 at ef=200
and only 0.998 at ef=300. Lock contention is therefore not a useful proxy for
topology damage. The instrumentation and repair path were removed.

### Rejected: committed-only construction candidates

Another experiment removed the explicit snapshot of in-flight nodes from
construction neighbor selection. At M=16 on the semantic fixture it produced
3,181-4,086 inserts/s with 0.996-0.999 recall, which did not improve the quality
envelope. On the 5k isotropic fixture it reduced throughput from approximately
2.85-3.05k to 2.67-2.83k inserts/s, did not improve ef=200 recall, and made the
ef=300 result less reliable. The existing in-flight candidate behavior was
restored.

### Rejected: padded x4 SIMD traversal tails

The raw-pointer traversal path normally scores neighbors in x8 and x4 SIMD
batches, then uses scalar distance calls for the final one to three neighbors.
An exact experiment replaced tails of two or three vectors with one x4 call,
duplicating the last valid pointer into unused lanes and discarding those
results. This preserves every admitted distance and changes no graph math.

Three M=16/four-worker semantic builds averaged 3,469 graph-ready inserts/s
with the padded x4 tail. The immediately following scalar-tail control averaged
3,553 inserts/s under the same machine state. Recall remained inside the normal
schedule-dependent envelope in both groups. The extra duplicate-lane work and
batch admission overhead produced a 2.4% construction regression, so the scalar
tail was retained.

### Rejected: batched in-flight snapshot distances

The scaled profile charged 7.19 seconds of cumulative CPU to scalar distance
evaluation of the in-flight construction snapshot. An exact experiment gathered
the same eligible node IDs and scored full groups through the existing x8/x4
pointer kernels, retaining scalar tails and the existing `distance > 0` rule.

The apparent profile opportunity did not survive end-to-end measurement. At
four workers, batched snapshots averaged 3,540 graph-ready inserts/s versus
3,553 for the immediately preceding scalar control (-0.4%). At eight workers,
where snapshots can fill more SIMD lanes, batching averaged 3,165 inserts/s
versus 3,283 for scalar (-3.6%). Snapshot widths and eligibility are too small
and irregular to amortize gathering and batch setup. The batched path was
removed completely.

### Current scaled profile

An M=16/four-worker 50k semantic CPU profile attributes approximately 85% of
sampled construction CPU to exact SIMD distance work:

- `L2Distance8AlignedPtrNEON`: 33.27% flat.
- `L2DistanceNEON`: 26.24% flat.
- `L2Distance4PtrNEON`: 7.61% flat.
- x8 diversity predicate: 4.63% flat.
- `searchLevelScratchValues`: 13.48% flat, 61.12% cumulative.

By comparison, `memmove` was 1.17%, candidate ordering was 0.91%, hash metadata
was below 1%, and SoA admission was 3.08% cumulative. This profile does not
support another queue or collision-bookkeeping rewrite. Remaining exact-path
experiments must reduce completed vector work or combine predicate evaluation
with distance calculation. Larger throughput gains now require changing the
construction algorithm/representation or addressing the separate durability
pipeline.

The lower-M four-worker sweep produced:

| M | Graph-ready inserts/s | ef=200 recall | Higher-ef outcome |
|---:|---:|---:|---|
| 32 | 2,087 | 0.998 | 0.999 through ef=300 |
| 24 | 2,602 | 0.998 | 0.999 from ef=216 through 300 |
| 20 | 2,871-2,965 | 0.994-0.999 | 0.997-0.999 at ef=300 |
| 16 | 2,946-3,408 | 0.996-0.998 | 0.998-0.999 at ef=300 in repeated builds |

One M=16 concurrent build reached 1.000 at ef=224, demonstrating the same
schedule variance seen at M=36. A serial M=16 control produced 0.999 at ef=200
and 1.000 at ef=300. Therefore:

- Lower M genuinely restores construction and search speed on semantic data.
- M=16's serial topology needs a wider beam for exact recall.
- Concurrent mutation introduces additional non-monotonic topology variance;
  tuning M or ef alone cannot eliminate it.
- Lock-contention-scoped correction and blanket repair are not viable. Any
  further topology correction needs a quality signal rather than a concurrency
  signal.

## Interpretation

- Random 5k/768d remains the adversarial topology benchmark.
- Semantic 50k/768d is the production-geometry benchmark.
- Neither substitutes for a million-vector run; the same format and benchmark
  can consume a larger fixture without code changes.
- A result below 1.0 recall is recorded rather than hidden. Both average recall
  and the number of non-exact queries are reported so rare misses remain visible.
