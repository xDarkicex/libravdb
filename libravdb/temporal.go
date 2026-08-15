package libravdb

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/xDarkicex/libravdb/internal/storage/singlefile"
)

// TemporalStats is a defensive, point-in-time observability value. It never
// owns or changes a snapshot lease and never triggers compaction.
type TemporalStats struct {
	ActiveLeaseCount       uint64
	DistinctPinnedLSNCount uint64
	MinimumPinnedLSN       uint64
	OldestPinAge           time.Duration
	// RetainedVersionCount is the number of archived historical record
	// versions still available for exact-LSN reads. It excludes each
	// collection's current live record mirror.
	RetainedVersionCount uint64
	// RetainedBytes is the deterministic decoded payload estimate for the
	// archived historical versions: temporal fields, vectors, and metadata.
	// It is not process RSS, index memory, or WAL size.
	RetainedBytes          uint64
	OldestRetainedLSN      uint64
	OldestRetainedAt       time.Time
	ConfiguredRetention    time.Duration
	LastCompactionBoundary uint64
}

// TemporalSnapshot is a pinned historical snapshot handle. It registers an
// active LSN with the database to prevent compaction from evicting history
// needed by in-flight queries. Callers must Close() the handle after use.
type TemporalSnapshot struct {
	LSN       uint64
	Timestamp time.Time
	db        *Database
	closed    atomic.Bool
}

// Close releases the snapshot's pin on the retention boundary. After close,
// the handle must not be used.
func (s *TemporalSnapshot) Close() {
	if s.closed.Swap(true) {
		return
	}
	if s.db != nil {
		s.db.unpinSnapshot(s.LSN)
	}
}

// TemporalConfig holds retention policy configuration.
type TemporalConfig struct {
	// RetainDuration is the minimum history to retain. Commits older than
	// time.Now().UTC().Add(-RetainDuration) may be compacted. Zero means
	// no automatic pruning (default: conservative).
	RetainDuration time.Duration
}

// activeSnapshots tracks LSNs pinned by in-flight temporal queries.
// Compaction cannot advance past the minimum pinned LSN.
type activeSnapshots struct {
	mu     sync.Mutex
	pins   map[uint64]snapshotPin
	minLSN uint64 // cached minimum; 0 if no pins
}

type snapshotPin struct {
	refs          uint64
	firstPinnedAt time.Time
}

func (a *activeSnapshots) pin(lsn uint64) {
	a.mu.Lock()
	if a.pins == nil {
		a.pins = make(map[uint64]snapshotPin)
	}
	pin := a.pins[lsn]
	if pin.refs == 0 {
		pin.firstPinnedAt = time.Now()
	}
	pin.refs++
	a.pins[lsn] = pin
	if a.minLSN == 0 || lsn < a.minLSN {
		a.minLSN = lsn
	}
	a.mu.Unlock()
}

func (a *activeSnapshots) unpin(lsn uint64) {
	a.mu.Lock()
	if pin, ok := a.pins[lsn]; ok && pin.refs > 0 {
		if pin.refs == 1 {
			delete(a.pins, lsn)
		} else {
			pin.refs--
			a.pins[lsn] = pin
		}
	}
	// Recompute min.
	a.minLSN = 0
	for l := range a.pins {
		if a.minLSN == 0 || l < a.minLSN {
			a.minLSN = l
		}
	}
	a.mu.Unlock()
}

// stats returns a point-in-time reduction of active snapshot leases. The
// mutex is held for the complete reduction, so callers never observe a
// partially pinned or partially released LSN.
func (a *activeSnapshots) stats(now time.Time) (leases, distinct, minLSN uint64, oldestAge time.Duration) {
	a.mu.Lock()
	defer a.mu.Unlock()

	minLSN = a.minLSN
	for lsn, pin := range a.pins {
		if pin.refs == 0 {
			continue
		}
		leases += pin.refs
		distinct++
		age := now.Sub(pin.firstPinnedAt)
		if age < 0 {
			age = 0
		}
		if distinct == 1 || age > oldestAge {
			oldestAge = age
		}
		if minLSN == 0 || lsn < minLSN {
			minLSN = lsn
		}
	}
	return leases, distinct, minLSN, oldestAge
}

// safeRetentionBoundary returns the highest LSN that is safe to compact
// (i.e., the minimum pinned LSN, or if no pins, 0 meaning no restriction).
func (a *activeSnapshots) safeRetentionBoundary() uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.minLSN
}

// SnapshotAt returns the latest committed state at or before t (UTC). If t
// is newer than the latest commit, the latest commit is returned. If t is
// older than the earliest retained commit, ErrRetentionExpired is returned.
// The returned handle must be Close()d after use to release the snapshot pin.
func (db *Database) SnapshotAt(ctx context.Context, t time.Time) (*TemporalSnapshot, error) {
	t = t.UTC()
	type temporalEngine interface {
		ResolveTimestamp(t time.Time) (uint64, time.Time, error)
	}
	eng, ok := db.storage.(temporalEngine)
	if !ok {
		return nil, fmt.Errorf("storage engine does not support temporal resolution")
	}
	lsn, ts, err := eng.ResolveTimestamp(t)
	if err != nil {
		return nil, err
	}
	snap := &TemporalSnapshot{LSN: lsn, Timestamp: ts, db: db}
	db.activeSnaps.pin(lsn)
	return snap, nil
}

// SnapshotAtLSN returns the commit timestamp for an exact LSN. The returned
// handle must be Close()d after use.
func (db *Database) SnapshotAtLSN(ctx context.Context, lsn uint64) (*TemporalSnapshot, error) {
	type temporalEngine interface {
		ResolveLSN(lsn uint64) (time.Time, error)
	}
	eng, ok := db.storage.(temporalEngine)
	if !ok {
		return nil, fmt.Errorf("storage engine does not support temporal resolution")
	}
	ts, err := eng.ResolveLSN(lsn)
	if err != nil {
		return nil, err
	}
	snap := &TemporalSnapshot{LSN: lsn, Timestamp: ts, db: db}
	db.activeSnaps.pin(lsn)
	return snap, nil
}

// LatestCommitLSN returns the exact latest durable transaction LSN. It reads
// the storage commit catalog and never derives a boundary from wall-clock time
// or an allocated-but-uncommitted WAL sequence.
func (db *Database) LatestCommitLSN(ctx context.Context) (uint64, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
	}
	db.mu.RLock()
	closed := db.closed
	storageEngine := db.storage
	db.mu.RUnlock()
	if closed {
		return 0, ErrDatabaseClosed
	}
	provider, ok := storageEngine.(interface{ LatestCommitLSN() (uint64, error) })
	if !ok {
		return 0, fmt.Errorf("storage engine does not expose latest committed LSN")
	}
	return provider.LatestCommitLSN()
}

// TemporalStats returns a defensive, point-in-time view of active snapshot
// leases and retained temporal history. It never creates a temporary lease,
// triggers compaction, or exposes mutable engine state.
func (db *Database) TemporalStats(ctx context.Context) (TemporalStats, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return TemporalStats{}, err
		}
	}

	db.mu.RLock()
	if db.closed {
		db.mu.RUnlock()
		return TemporalStats{}, ErrDatabaseClosed
	}
	storageEngine := db.storage
	configuredRetention := db.config.Temporal.RetainDuration
	db.mu.RUnlock()

	leases, distinct, minLSN, oldestAge := db.activeSnaps.stats(time.Now())
	stats := TemporalStats{
		ActiveLeaseCount:       leases,
		DistinctPinnedLSNCount: distinct,
		MinimumPinnedLSN:       minLSN,
		OldestPinAge:           oldestAge,
		ConfiguredRetention:    configuredRetention,
	}
	historyInfo, ok := storageEngine.(interface {
		RetainedHistoryStats() (uint64, uint64, error)
	})
	if !ok {
		return TemporalStats{}, fmt.Errorf("storage engine does not expose retained history stats")
	}
	retainedVersions, retainedBytes, err := historyInfo.RetainedHistoryStats()
	if err != nil {
		return TemporalStats{}, err
	}
	stats.RetainedVersionCount = retainedVersions
	stats.RetainedBytes = retainedBytes

	retentionInfo, ok := storageEngine.(interface {
		OldestRetained() (time.Time, uint64, error)
	})
	if !ok {
		return TemporalStats{}, fmt.Errorf("storage engine does not expose retention info")
	}
	oldestAt, oldestLSN, err := retentionInfo.OldestRetained()
	if err != nil {
		if errors.Is(err, singlefile.ErrNoCommits) {
			return stats, nil
		}
		return TemporalStats{}, err
	}
	stats.OldestRetainedAt = oldestAt
	stats.OldestRetainedLSN = oldestLSN
	if oldestLSN != 0 {
		stats.LastCompactionBoundary = oldestLSN
	}
	return stats, nil
}

func (db *Database) unpinSnapshot(lsn uint64) {
	db.activeSnaps.unpin(lsn)
}

// WithTemporalRetention configures the minimum history duration to retain.
// Commits older than duration ago may be compacted. Zero (default) means
// no automatic pruning — all history is retained.
func WithTemporalRetention(d time.Duration) Option {
	return func(c *Config) error {
		c.Temporal.RetainDuration = d
		return nil
	}
}

// CompactHistory prunes record/vector and graph edge history older than the
// configured retention duration. Active pinned snapshots are respected —
// compaction never removes data needed by in-flight queries. Callers should
// ensure no long-running snapshots are pinned unnecessarily.
//
// Returns the new oldestRetainedLSN after compaction, or 0 if nothing was
// pruned.
func (db *Database) CompactHistory(ctx context.Context) (uint64, error) {
	type compactor interface {
		CompactTemporalHistory(retainLSN uint64) (uint64, error)
	}
	eng, ok := db.storage.(compactor)
	if !ok {
		return 0, fmt.Errorf("storage engine does not support temporal compaction")
	}

	// Compute safe boundary: max(config boundary, oldest pinned snapshot).
	boundary := db.computeRetentionBoundary()
	pinned := db.activeSnaps.safeRetentionBoundary()
	if pinned > 0 && pinned < boundary {
		boundary = pinned
	}
	if boundary == 0 {
		return 0, nil // nothing to compact
	}

	newBoundary, err := eng.CompactTemporalHistory(boundary)
	if err != nil {
		return 0, err
	}
	return newBoundary, nil
}

func (db *Database) computeRetentionBoundary() uint64 {
	if db.config.Temporal.RetainDuration <= 0 {
		return 0 // no pruning configured
	}
	cutoff := time.Now().UTC().Add(-db.config.Temporal.RetainDuration)
	snap, err := db.SnapshotAt(context.Background(), cutoff)
	if err != nil {
		return 0
	}
	lsn := snap.LSN
	snap.Close()
	return lsn
}

// OldestRetained returns the oldest timestamp and LSN still available for
// temporal queries.
func (db *Database) OldestRetained(ctx context.Context) (time.Time, uint64, error) {
	type retentionInfo interface {
		OldestRetained() (time.Time, uint64, error)
	}
	eng, ok := db.storage.(retentionInfo)
	if !ok {
		return time.Time{}, 0, fmt.Errorf("storage engine does not expose retention info")
	}
	return eng.OldestRetained()
}
