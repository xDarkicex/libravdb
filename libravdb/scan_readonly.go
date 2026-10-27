package libravdb

import (
	"context"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
)

// The capability is private to the executor: callers of Get/Iterate must retain
// their owned-copy contract. Fallback engines keep using their existing iterator.
type readOnlyStorageScanner interface {
	ScanReadOnly(context.Context, func(string, uint32, uint64, []float32, map[string]interface{}) error) error
}

func (c *Collection) scanReadOnly(ctx context.Context, fn func(Record) error) error {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return ErrCollectionClosed
	}
	if c.shards == nil {
		return scanReadOnlyStorage(ctx, c.storage, fn)
	}
	for i := range c.shards {
		if err := scanReadOnlyStorage(ctx, c.shards[i].storage, fn); err != nil {
			return err
		}
	}
	return nil
}
func scanReadOnlyStorage(ctx context.Context, source storage.Collection, fn func(Record) error) error {
	if scanner, ok := source.(readOnlyStorageScanner); ok {
		return scanner.ScanReadOnly(ctx, func(id string, ordinal uint32, version uint64, vector []float32, metadata map[string]interface{}) error {
			return fn(Record{ID: id, Ordinal: ordinal, Version: version, Vector: vector, Metadata: metadata})
		})
	}
	return source.Iterate(ctx, func(entry *index.VectorEntry) error { return fn(recordFromIndexEntry(entry)) })
}
