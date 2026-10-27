# SQL scan allocation profile and fix

A 14,001-record mneme fixture (8,000 tasks, 4,000 gotchas, 2,000 typed
notes, one spec) exposed approximately 411,267 allocations and 47.25 MB per
one-term title/body search. CPU/allocation profiles identify scan-time FTS
vectors/query parsing, storage metadata cloning and full-list materialization.
The CLI's small returned result set was not the source of that growth.

The optimized simple lexical path measured approximately 594 allocations and
69 KB per operation on the same persisted fixture. This measures warm query
execution, not store open or model inference. See mneme's retrieval benchmarks
for end-to-end callers and repeated comparison data.

## Changes

- Internal read-only storage scans borrow records only while a read lock is held.
  Locks are released between 1,024-ordinal windows; the initial ordinal frontier
  excludes later inserts. Public `Get` and `Iterate` still return owned copies.
- Live relational scans apply predicates before cloning any result data.
  Transactions, epoch overlays and historical snapshots retain their prior paths.
- FTS predicates parse the query once per execution. Simple positive conjunctions
  test normalized lexeme membership without allocating a document position map.
  Other dictionaries/query forms retain their existing matching/ranking semantics.
- Bounded id/simple-rank ordering retains K owned snapshots, including OFFSET.
  Replaced snapshots reuse scalar maps; mutable nested values are deep-copied.
  DISTINCT, unsupported ordering/ranking and mixed numeric id comparisons use
  the general path. Existing descending-id tie behavior is preserved.
- Projections use the retained snapshot instead of fetching/cloning it again.
  String ordering avoids formatting and failed numeric parsing for ordinary text.

No persisted format, public ownership contract, vector index, result cache or
query language is added. Work can still scale with scanned bytes and requested
candidate/result counts; this removes avoidable allocations per rejected row.

## Regression coverage

The tests compare bounded selection against unbounded ordering, preserve numeric
special values and full-text dictionary/query semantics, reject storage aliasing
through returned nested JSON, and check cancellation/callback errors. Allocation
guards assert zero allocation for prepared simple membership and constant query
allocation cost when the number of rejected records grows from 32 to 512.

The affected SQL/FTS/temporal/iteration/scan suite passes under `-race`, as do the
storage and pgwire packages in the wider run. Analysis of unchanged v1.6.20 reports
the same 134 staticcheck diagnostics and existing vet unsafe-pointer/self-assignment
findings; changed-package vet is clean. The full race run also exceeded its default
10-minute timeout in vector stress/benchmark workloads outside this patch. Those
results are not represented as a clean upstream full gate.
