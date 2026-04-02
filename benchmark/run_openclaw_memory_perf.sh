#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

: "${GOCACHE:=/tmp/libraVDB-gocache}"
: "${OPENCLAW_BENCHTIME:=1x}"

echo "== OpenClaw memory profile =="
LIBRAVDB_RUN_OPENCLAW_PERF=1 GOCACHE="$GOCACHE" go test -run '^TestOpenClawMemoryProfile$' -v ./benchmark

echo
echo "== OpenClaw memory benchmarks =="
GOCACHE="$GOCACHE" go test -run '^$' -bench '^BenchmarkOpenClawMemory' -benchmem -benchtime="$OPENCLAW_BENCHTIME" ./benchmark
