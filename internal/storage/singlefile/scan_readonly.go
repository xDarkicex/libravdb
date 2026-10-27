package singlefile

import (
	"context"
	"fmt"
)

// ScanReadOnly borrows each record under a read lock. This internal executor
// primitive must not mutate/retain its inputs or reenter storage from fn. Public
// Get/Iterate still return owned copies. Releasing the lock between ordinal
// windows preserves Iterate's frontier semantics and lets commits make progress.
func (c *Collection) ScanReadOnly(ctx context.Context, fn func(string, uint32, uint64, []float32, map[string]interface{}) error) error {
	if fn == nil {
		return fmt.Errorf("read-only scan callback cannot be nil")
	}
	limit, err := c.readOnlyScanFrontier()
	if err != nil {
		return err
	}
	const windowSize uint32 = 1024
	for start := uint32(0); start < limit; {
		end := start + min(windowSize, limit-start)
		if err := c.scanReadOnlyWindow(ctx, start, end, fn); err != nil {
			return err
		}
		start = end
	}
	return ctx.Err()
}
func (c *Collection) readOnlyCollectionLocked() (*persistedCollection, error) {
	if c.closed.Load() || c.engine.closed.Load() {
		return nil, fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		return nil, fmt.Errorf("collection %s not found", c.name)
	}
	return persisted, nil
}
func (c *Collection) readOnlyScanFrontier() (uint32, error) {
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	persisted, err := c.readOnlyCollectionLocked()
	if err != nil {
		return 0, err
	}
	return persisted.NextOrdinal, nil
}
func (c *Collection) scanReadOnlyWindow(ctx context.Context, start, end uint32, fn func(string, uint32, uint64, []float32, map[string]interface{}) error) error {
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	persisted, err := c.readOnlyCollectionLocked()
	if err != nil {
		return err
	}
	end = min(end, uint32(len(persisted.ordinalToID)))
	for ordinal := start; ordinal < end; ordinal++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := visitReadOnlyOrdinal(persisted, ordinal, fn); err != nil {
			return err
		}
	}
	return nil
}
func visitReadOnlyOrdinal(persisted *persistedCollection, ordinal uint32, fn func(string, uint32, uint64, []float32, map[string]interface{}) error) error {
	id := persisted.ordinalToID[ordinal]
	if id == "" {
		return nil
	}
	record := persisted.Records[id]
	if record == nil || record.Deleted || record.Ordinal != ordinal {
		return nil
	}
	return fn(id, ordinal, record.Version, record.Vector, record.Metadata)
}
