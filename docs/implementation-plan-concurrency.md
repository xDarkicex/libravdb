# Implementation Plan: Robust Batch and Concurrent Connection Handling for `libravdb`

## Goal

Strengthen `libravdb` so it remains correct, performant, and predictable under high concurrent write and read load without relying on application-side wrappers as the primary safety mechanism.

The design should:
- preserve correctness
- avoid uncontrolled CPU and memory spikes
- maintain strong throughput for bulk ingestion
- keep search latency stable under write pressure
- work well in sharded/load-balanced production deployments

## Core Principles

- Correctness comes first: scheduling must not change write semantics.
- Concurrency must be bounded, not caller-amplified.
- Backpressure is better than uncontrolled saturation.
- Defaults should be safe for production, with opt-in aggressive modes for bulk loading.
- Search and ingestion should not interfere more than necessary.
- The library should expose operational signals for tuning and autoscaling.

## Non-Goals

- Do not serialize all writes behind a global lock.
- Do not permanently reduce all batch operations to single-threaded execution.
- Do not require users to build external admission-control wrappers just to use the library safely.
- Do not change public CRUD semantics or HNSW correctness guarantees.

## Current Problem Summary

Today, batch execution can multiply concurrency in two ways:
- multiple callers may issue batch writes concurrently
- each batch may also use internal worker parallelism

This can lead to:
- excessive CPU saturation
- memory spikes
- noisy-neighbor behavior between reads and writes
- unstable desktop or production node behavior under pressure

The system currently optimizes for throughput, but not yet for global fairness, bounded resource usage, or adaptive backpressure.

## Target Architecture

Introduce an internal write scheduling and resource-control layer inside `libravdb`.

### Key components

1. Write admission controller
2. Batch scheduler
3. Adaptive concurrency policy
4. Memory budget manager for batch execution
5. Reader/writer isolation controls
6. Operational metrics and pressure signals
7. Explicit execution modes

## Phase 1: Add Internal Write Admission Control

### Objective

Ensure the number of concurrent mutation-heavy operations is bounded at the collection or database level.

### Implementation

Add an internal scheduler that all mutation paths pass through:
- `Insert`
- `InsertBatch`
- `Update`
- `Delete`
- batch update/delete variants
- streaming ingestion paths

### Behavior

Each write operation must acquire execution permission before entering heavy work.

Introduce internal limits such as:
- max concurrent write jobs
- max concurrent batch jobs
- max concurrent index mutation jobs

### Notes

- Keep this internal initially.
- Default limits should be conservative and production-safe.
- This scheduler should govern all write paths consistently.

## Phase 2: Separate External and Internal Concurrency

### Objective

Prevent concurrency multiplication from callers plus batch worker pools.

### Implementation

Define two distinct control layers:

- External concurrency:
  - how many write jobs can execute simultaneously
- Internal batch concurrency:
  - how many workers a single batch may use

### Behavior

A batch job should not be allowed to use full internal concurrency when the system is already under write load.

For example:
- if only one batch is running, it may use more workers
- if multiple batches are active, each batch should automatically scale down worker count

### Result

This preserves throughput while preventing oversubscription.

## Phase 3: Adaptive Batch Concurrency

### Objective

Make batch worker counts dynamic instead of fixed.

### Implementation

Replace static “use caller-requested max concurrency directly” behavior with adaptive selection based on:
- current active write jobs
- batch size
- chunk size
- estimated memory cost
- system mode
- collection/index type
- optional runtime pressure indicators

### Policy ideas

- small batches:
  - run serially
  - or with low concurrency
- medium batches:
  - 2 workers
- large offline ingestion:
  - allow higher concurrency
- if write queue is saturated:
  - reduce worker count automatically
- if memory estimate is high:
  - reduce worker count and/or chunk size

### API behavior

Keep `BatchOptions.MaxConcurrency`, but treat it as an upper bound, not a guaranteed execution level.

## Phase 4: Introduce Memory-Aware Batch Execution

### Objective

Prevent batch execution from causing uncontrolled memory spikes.

### Implementation

Extend batch planning to estimate:
- vector payload memory
- temporary batch buffers
- index mutation working memory
- storage write-path overhead
- quantization overhead if applicable

### Behavior

Before launching a batch:
- compute an estimated cost
- compare it against a configurable memory budget
- if above budget:
  - reduce chunk size
  - reduce worker count
  - queue the job until budget is available

### Additional behavior

During execution:
- track in-flight reserved memory
- release reservations on completion
- allow waiting or rejection when budget is exhausted

## Phase 5: Reader/Writer Isolation

### Objective

Protect search latency from ingestion spikes.

### Implementation Options

Implement one or more of the following:
- lower write concurrency when search pressure is high
- reserve CPU budget for reads
- limit concurrent HNSW mutation work per collection
- isolate read and write execution lanes
- prefer reads when system is in latency-sensitive mode

### Minimal initial target

Add a mode where write throughput is intentionally limited to preserve search responsiveness.

## Phase 6: Execution Modes

### Objective

Provide safe but flexible operating profiles.

### Modes

#### `latency`

- prioritize search responsiveness
- low write concurrency
- smaller chunks
- conservative memory usage

#### `balanced`

- default mode
- moderate write concurrency
- adaptive chunking
- bounded resource usage

#### `bulk_ingest`

- higher write parallelism
- larger chunk sizes
- optimized for offline loading
- acceptable to trade some search latency for throughput

### Implementation

Add a configuration option at DB or collection level:
- `ExecutionModeLatency`
- `ExecutionModeBalanced`
- `ExecutionModeBulkIngest`

Use mode as a policy input for:
- worker count
- queue depth
- memory budget
- read/write fairness

## Phase 7: Queueing and Backpressure

### Objective

Replace uncontrolled saturation with explicit pressure handling.

### Implementation

When write capacity is exhausted:
- queue operations up to a bounded depth
- or reject immediately with a retryable error
- or block with context cancellation support

### Required behavior

Support:
- bounded queue size
- context-aware waiting
- retryable “write pressure” errors
- visibility into queue depth and wait times

### Error model

Introduce retryable operational errors such as:
- write queue full
- memory budget exceeded
- ingestion throttled
- concurrency limit reached

## Phase 8: Operational Metrics and Introspection

### Objective

Make production tuning and autoscaling possible.

### Metrics to expose

- active write jobs
- active batch workers
- queued write jobs
- average queue wait time
- rejected/throttled writes
- estimated reserved write memory
- adaptive chosen concurrency per batch
- chunk size actually used
- read latency during write load
- write throughput by mode

### Debug/introspection endpoints

Expose internal state such as:
- scheduler status
- current mode
- active budget usage
- per-collection pressure state

## Phase 9: Preserve Correctness Semantics

### Objective

Ensure scheduling changes do not alter behavior.

### Guarantees to preserve

- ordinal assignment semantics
- storage-first persistence behavior
- HNSW/provider consistency
- delete/update semantics
- rollback behavior on partial failures
- public API semantics for insert/update/delete/search

### Validation

Add tests for:
- queued batch execution
- cancellation while queued
- adaptive worker reduction
- memory-budget throttling
- concurrent writers plus readers
- no lost writes
- no duplicate ordinals
- stable reopen/rebuild behavior under throttled ingestion

## Phase 10: Streaming Integration

### Objective

Ensure streaming ingestion uses the same safety model.

### Implementation

Route `StreamingBatchInsert` through the same scheduler and memory budget system.

### Behavior

Streaming workers should:
- respect write permits
- respect memory reservations
- adapt flush behavior under pressure
- downgrade concurrency instead of amplifying pressure

## Phase 11: Public API Additions

### Additive Configuration

Add optional configuration structures such as:

```go
type SchedulerConfig struct {
    MaxConcurrentWrites    int
    MaxConcurrentBatches   int
    MaxWriteQueueDepth     int
    MaxWriteMemoryBytes    int64
    DefaultExecutionMode   ExecutionMode
    EnableAdaptiveBatching bool
    EnableReadPriority     bool
}
```

```go
type ExecutionMode string

const (
    ExecutionModeLatency    ExecutionMode = "latency"
    ExecutionModeBalanced   ExecutionMode = "balanced"
    ExecutionModeBulkIngest ExecutionMode = "bulk_ingest"
)
```

### BatchOptions behavior changes

Retain:
- `ChunkSize`
- `MaxConcurrency`

But redefine behavior:
- `MaxConcurrency` is an upper bound
- actual concurrency is chosen by the scheduler
- `ChunkSize` may be reduced internally when needed for memory safety

## Phase 12: Rollout Strategy

### Step 1

Add scheduler infrastructure behind current APIs.

### Step 2

Route batch insert through scheduler with conservative defaults.

### Step 3

Add adaptive concurrency and memory budgeting.

### Step 4

Integrate streaming and other write paths.

### Step 5

Add read-priority mode.

### Step 6

Expose metrics and operational introspection.

### Step 7

Tune defaults based on benchmark and production-style load tests.

## Testing Plan

### Functional correctness

- concurrent inserts preserve correctness
- queued writes remain ordered and durable where required
- updates and deletes behave correctly under scheduler pressure
- reopen/rebuild remains correct after throttled ingestion

### Performance/regression

- single-writer throughput remains strong
- balanced mode prevents uncontrolled CPU saturation
- latency mode protects read performance under write load
- bulk_ingest mode still scales well for offline load jobs

### Stress tests

- many concurrent batch writers
- mixed read/write workloads
- forced memory budget exhaustion
- queue overflow and cancellation
- shard-style parallel deployments

## Success Criteria

The implementation is successful when:
- `libravdb` remains correct under concurrent write pressure
- CPU and memory usage are bounded and predictable
- search latency is protected in latency-sensitive mode
- bulk ingestion still performs well in high-throughput mode
- multiple callers cannot accidentally multiply concurrency into instability
- production operators can observe and tune the system without wrapping the library externally

## Recommended Initial Default

For first rollout, use:
- execution mode: `balanced`
- low-to-moderate write concurrency
- adaptive batch concurrency enabled
- bounded write queue
- memory-budget enforcement enabled
- `BatchOptions.MaxConcurrency` treated as advisory upper bound

This gives a safer core immediately while preserving room for higher-throughput bulk modes when explicitly requested.
