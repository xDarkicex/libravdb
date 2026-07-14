# Cross-Platform SIMD Parity

Date: 2026-07-12

## Architecture

HNSW now uses an architecture-neutral raw-pointer distance API:

- `HasL2Batch8Ptr`
- `L2Distance4Ptr`
- `L2Distance8Ptr`
- `L2AnyLessThan8Ptr`

ARM64 dispatches to the existing NEON kernels. AMD64 dispatches to generated
AVX2/FMA kernels when the CPU feature flags are present. Other architectures
retain the existing scalar/slice fallback path.

The shared API is used by:

- search traversal neighbor scoring;
- construction candidate expansion;
- diversity-pruning rejection.

## Generation

Run:

```sh
go generate ./internal/util/simd
```

`generate.go` uses Avo to generate both `distance_amd64.s` and
`stub_amd64.go`. Generated functions are marked `//go:noescape` so pointer and
slice arguments do not create GC-visible hot-path allocations.

The generated amd64 kernels include:

- four-pointer squared L2;
- eight-pointer squared L2 in one query pass;
- eight-pointer any-distance-below-cutoff predicate;
- existing scalar, x4 slice, and prefetch kernels.

Avo targets x86. ARM64 NEON remains checked-in Go assembly and is exposed
through the same build-tagged API.

## Validation

- ARM64 HNSW and SIMD suites pass.
- ARM64 5k/768d construction remains recall@10=1.000, 840,707 level-0 links,
  zero allocations, and 873.5 graph inserts/s in the validation run.
- Generated amd64 output is reproducible across consecutive `go generate`
  runs.
- Linux amd64, Linux arm64, Windows amd64, and generic Linux ppc64le targets
  cross-compile.
- Rosetta does not expose AVX2 through `x/sys/cpu`; therefore local amd64
  emulation validates fallback behavior only. AVX2 execution and performance
  must be measured on a real amd64 CI host.

CI now rejects stale generated amd64 output and runs graph, HNSW, and SIMD
tests in the configured amd64/arm64 architecture matrix.

## Future Quantized Path

Int8, binary/Hamming, scalar quantization, and product quantization are a
separate index-representation project. They require explicit recall, storage,
reranking, persistence, and migration semantics. They should not be mixed into
the exact float32 SIMD path as an incremental optimization.
