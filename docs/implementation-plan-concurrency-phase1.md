# Phase 1 Implementation Plan: Concurrent and Batch Write Admission Control

## Goal

Add a narrow first-phase concurrency layer to `libravdb` that makes concurrent and batch writes safer immediately without changing write semantics or attempting the full adaptive scheduler design yet.

This phase is intended to harden the core for plugin use, especially when one agent or multiple subagents may issue many writes against the same collection.

## Scope

Phase 1 includes:
- bounded write admission control
- bounded queued writers
- conservative write concurrency defaults
- batch worker concurrency clamped to collection write parallelism
- streaming insert worker concurrency clamped to collection write parallelism
- focused tests for queueing and cancellation

Phase 1 does not include:
- adaptive concurrency policies
- memory-budget-aware scheduling
- read/write prioritization
- execution modes
- metrics/introspection surface for the scheduler

## Problem Statement

Today, write pressure can build up in two layers:
- multiple callers can issue writes concurrently
- batch and streaming paths can each create internal worker pools

Even though collection mutation paths are still serialized internally, callers can still pile up blocked goroutines and oversized worker pools. That is enough to cause avoidable CPU pressure, desktop disruption, and poor behavior under subagent-style usage.

## Phase 1 Design

### 1. Per-Collection Write Controller

Each collection gets an internal write controller that:
- limits how many write operations may actively execute
- bounds how many additional writers may wait
- respects context cancellation while waiting

The controller is intentionally simple:
- active writes are governed by permits
- waiting writers are counted explicitly
- if the waiting queue is full, new writes fail fast

### 2. Admission Applied to Mutation Paths

All collection mutation entry points should acquire a write permit before entering the heavy write path:
- `Insert`
- `insertBatch`
- `Update`
- `Delete`

This preserves correctness because the existing storage/index mutation logic remains unchanged after admission is granted.

### 3. Batch Worker Clamping

Batch insert currently allows caller-requested worker concurrency. In Phase 1:
- `BatchOptions.MaxConcurrency` becomes an upper bound
- actual batch worker count is clamped by the collection’s write controller

This avoids spinning up many internal workers for a collection that is intentionally write-limited.

### 4. Streaming Worker Clamping

Streaming batch insert should use the same effective write concurrency as batch insert so it cannot bypass the write admission model.

## Configuration

Add database options:
- `WithMaxConcurrentWrites(int)`
- `WithMaxWriteQueueDepth(int)`

Initial recommended defaults:
- `MaxConcurrentWrites = min(GOMAXPROCS, 2)`
- `MaxWriteQueueDepth = 32`

These defaults aim to keep plugin and desktop usage safe while preserving enough throughput for normal ingestion.

## Expected Behavior

### Direct writes

If the collection is idle:
- write starts immediately

If write capacity is busy:
- write waits if queue space is available
- write fails with `ErrWriteQueueFull` if queue is full
- write returns context error if the caller cancels while waiting

### Batch writes

Batch execution:
- uses the same write admission control through collection mutation calls
- does not spawn more internal workers than the collection is configured to handle

### Streaming writes

Streaming workers:
- are capped to the same effective concurrency
- remain backpressure-aware

## Files to Change

### New

- `libravdb/write_controller.go`
- `docs/implementation-plan-concurrency-phase1.md`
- focused Phase 1 tests

### Existing

- `libravdb/database.go`
- `libravdb/options.go`
- `libravdb/collection.go`
- `libravdb/batch.go`
- `libravdb/streaming.go`
- `libravdb/errors.go`

## Test Plan

Add focused tests for:
- queued write succeeds once the active writer releases
- write queue full returns `ErrWriteQueueFull`
- waiting writer respects context timeout/cancellation
- batch worker count is clamped by collection write concurrency

## Rollout Notes

This phase is intentionally small and safe:
- it does not redesign the storage or index core
- it does not change public write semantics
- it provides immediate protection for plugin/subagent usage

After Phase 1 lands, the next logical step is Phase 2:
- adaptive concurrency
- memory-aware admission
- richer backpressure and scheduling policy
