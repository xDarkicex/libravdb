# Task: stop allocating a fresh buffer for every JSON value libraVDB encodes

You are working in the libraVDB repository (`github.com/xDarkicex/libravdb`, the `libravdb/`
and `internal/` packages). The engine is a single-file embedded database with a Cypher-style
SQL dialect, a graph layer, and a pgwire server. Its JSON codec is
[`apexJSON/v2`](https://github.com/xDarkicex/apexJSON) (`github.com/xDarkicex/apexJSON/v2`).

## The goal

Several of libraVDB's hot paths encode a Go value to JSON **once per row, per column, or per
value**. Each of those calls goes through `apexjson.Marshal`, which allocates a fresh result
buffer every time, and the caller usually converts the result to a `string` — so each encoded
value costs two allocations where one would do.

Replace those with an encode into a **reusable, pooled scratch buffer**, so the per-value
allocation disappears and only the copy that the API actually requires remains.

This is a performance change with a hard correctness requirement: **the bytes produced must
not change.** Not their order, not their formatting, not their escaping, not their
determinism.

## What is actually there

`apexjson.Marshal` is called from 24 places. They are not equivalent, and the difference
matters:

**Category A — the result becomes a `string`.** One allocation is saved per call, because the
intermediate `[]byte` goes away and the `string` copy remains.

- `libravdb/json_sql.go:539` — `encodeJSONValue(value) (string, error)`
- `libravdb/json_sql.go:942` — `encodeJSONValueToBytes` (returns `[]byte`; see below)
- `libravdb/json_sql.go:177` — the `default:` branch of `decodeJSONValue`, a decode fallback
- `libravdb/executor.go:3528` — `conflictColumnValue`, a conflict-detection path
- `libravdb/cte_execute.go:2934` and `:2941` — CTE JSON values
- `libravdb/sql_merge.go:470` — `MERGE` relationship properties
- `libravdb/costmodel_statistics.go:115`, `:237` and `libravdb/costmodel_scaffold.go:80` —
  statistics payloads, not per-row
- `internal/optimizer/scalar.go:323` — scalar function arguments
- `internal/pgwire/results.go:753`, `:758`, `:763`, `:768`, `:773`, `:778`, `:783` — per-column
  result encoding on the wire path. **This is the hottest of the group: it runs once per
  column per row.**

**Category B — the result is returned as owned `[]byte` or copied into a WAL frame.** A pooled
buffer saves nothing here, because the bytes must be owned by the caller afterwards: encoding
into a pooled buffer and copying out costs the same allocation as allocating directly. **Leave
these on `apexjson.Marshal` unless the whole call path can consume the buffer in place.**

- `internal/graph/properties.go:93` and `:108` — the edge property envelope, which is copied
  into the graph WAL frame
- `libravdb/json_sql.go:942` — returns `[]byte` to its caller

**Do not assume the categories are right. Measure.** The point of this task is a measured
result, and the categories are a hypothesis to check, not a decision.

## The approach

There is already a pooled-buffer idiom in this repository, and the new code should look like
it rather than invent a second one:

```go
// internal/util/encoding.go
func AcquireBinaryEncoder(estimated int) *BinaryEncoder   // takes from BinaryEncoderPool,
                                                          // resets Buf to [:0], grows capacity
func ReleaseBinaryEncoder(enc *BinaryEncoder)             // returns it; drops the buffer if it
                                                          // grew past maxPooledEncoderCapacity
```

Mirror that shape for a JSON encoder: a `sync.Pool` of `*struct{ Buf []byte }`, an
`Acquire`/`Release` pair, a capacity hint, and a cap on how large a pooled buffer may grow
before it is dropped rather than returned to the pool.

Then at the Category A call sites:

```go
enc := AcquireJSONEncoder(estimated)
defer ReleaseJSONEncoder(enc)
enc.Buf, err = apexjson.AppendMarshal(enc.Buf, value)
if err != nil { /* same error handling as before */ }
return string(enc.Buf), nil
```

`apexjson.AppendMarshal(dst []byte, v any) ([]byte, error)` already exists and is exported
(`encode.go:209` in the apexJSON repo). It appends the encoding to `dst` and returns the
result. If `dst` has insufficient capacity, `append` grows it onto the heap — so the pooled
buffer must start with a sensible capacity, and its grown capacity is what the next user
inherits. That is the whole trick: the buffer grows once and is reused after that.

Two constraints on the buffer's lifetime:

- The buffer must not be returned to the pool while anything still references it. A
  `string(enc.Buf)` **copies**, so the string is safe after `Release`; a `[]byte` handed to a
  caller is **not**.
- `Release` must be safe on the error path, and must not put a nil or a huge buffer back.

`apexjson.Marshal` sizes its own buffer with a capacity hint it computes from the value;
`AppendMarshal` does not, so the hint is now the caller's problem. Use the same reasoning
apexJSON uses (`map`/`slice` length times a per-element estimate, `string` length times two,
a floor of 64 bytes) — or start the pooled buffer at a fixed size and let it grow.

## The environment

- The sibling layout is **load-bearing**. `go.mod` carries
  `replace github.com/xDarkicex/lexer => ../lexer` and
  `replace github.com/xDarkicex/apexJSON/v2 => ../apexJSON/v2`. Do not remove them, and do
  not vendor. `../apexJSON` is the working copy of the codec.
- `go.mod` requires `github.com/xDarkicex/apexJSON/v2 v2.0.4`. **Keep that version.** v2.0.3
  cannot encode a map under Go 1.27: it reads the direct-interface flag from the runtime
  type's kind byte, and Go 1.27 moved that flag to the TFlag byte, so the encoder walked a
  map header as though it were the address of a slot holding it and killed the process. The
  fix is in v2.0.4 and in the sibling tree.
- The module proxy resolves `github.com/xDarkicex/apexJSON/v2` from the **root** tag
  (`v2.0.4`), taking the module from the `v2/` subdirectory. If you need to publish, both
  `v2.0.4` and `v2/v2.0.4` exist for each release.
- This module declares `go 1.25.7`, so `go test` runs on the local 1.25.7 toolchain by
  default. To exercise the Go 1.27 path, use `GOTOOLCHAIN=go1.27.1 go test ./...`. Both
  toolchains must pass.
- Push to `origin` (`xDarkicex/libravdb`). There is also a `jason` remote that is a fork —
  leave it alone.
- Commits use conventional commit prefixes: `fix:`, `perf:`, `test:`, `feat:`.
- Releases are tagged at the patch level: the current tag is `v1.6.15`, so the next one is
  `v1.6.16`.

## Acceptance criteria

1. **`go test ./...` passes on Go 1.25.7 and on Go 1.27.1** (`GOTOOLCHAIN=go1.27.1`). No
   test is skipped or weakened to make it pass.
2. **The bytes do not change.** For every call site you change, the encoded output must be
   byte-identical to what it was. Write a test that asserts this — for example, encode the
   same value through both paths and compare. The pgwire result encoder has seven separate
   call sites with slightly different types; each needs to be covered, not just one.
3. **A measured allocation win, reported.** There are existing benchmarks that exercise JSON
   columns: `BenchmarkSQLJSONExpansion` and `BenchmarkSQLJSONMembership` in
   `libravdb/sql_aggregate_benchmark_test.go`, and `BenchmarkJSONBSetNative` /
   `BenchmarkJSONBInsertNative` in `libravdb/json_sql_mutation_benchmark_test.go`. Add a
   benchmark that exercises a JSON-valued column across many rows, and report
   **allocs/op and B/op before and after**.
4. **Measure the way this repository's numbers are meant to be measured.** The machine is
   shared and its load varies by several-fold within a run, so a single before/after pair is
   not evidence. Run the before and after builds **interleaved**, take the **minimum** of
   at least ten runs of each, and report the deltas. If a number is not stable across
   independent runs, say so instead of quoting it.
5. **Nothing outside the scope changes.** No public API changes, no behaviour changes, no
   dependency version changes other than as noted above.

## What not to do

- Do not change what the engine produces. A JSON encoding that differs by one byte is a
  regression, even if it is faster.
- Do not convert the Category B sites. Copying out of a pooled buffer costs the same
  allocation as allocating directly, and it makes the ownership harder to reason about.
- Do not add a second pooling idiom. Follow `internal/util/encoding.go`.
- Do not widen the scope to the SQL planner, the graph layer, or the storage engine.
- Do not "optimise" by caching encoded values across calls. The values are arbitrary and
  mutable; a cache would be a correctness bug waiting to happen.

## Report back

- Which call sites you changed, and which you deliberately left alone, with the measurement
  that decided it.
- The before/after allocation numbers, with the method.
- The test that proves the bytes did not change.
- Anything you found that contradicts the categories above.
