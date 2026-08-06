package libravdb

import (
	"context"
	"fmt"

	"github.com/xDarkicex/libravdb/internal/storage"
)

// GetAtLSN returns the record version visible at snapshotLSN. Returns nil
// if the record did not exist or was deleted at that LSN.
func (c *Collection) GetAtLSN(ctx context.Context, id string, snapshotLSN uint64) (*Record, error) {
	if c.db == nil {
		return nil, fmt.Errorf("collection is not attached to a database")
	}
	te, ok := c.db.storage.(storage.TemporalReader)
	if !ok {
		return nil, fmt.Errorf("storage engine does not support temporal reads")
	}
	tr, err := te.GetRecordAtLSN(c.name, id, snapshotLSN)
	if err != nil {
		return nil, err
	}
	if tr == nil {
		return nil, nil
	}
	return &Record{
		ID: tr.ID, Vector: tr.Vector, Metadata: tr.Metadata,
		Ordinal: tr.Ordinal, Version: tr.Version,
	}, nil
}

// ListVisibleAtLSN iterates all records visible at snapshotLSN, calling fn
// for each. Iteration stops early if fn returns false.
func (c *Collection) ListVisibleAtLSN(ctx context.Context, snapshotLSN uint64, fn func(*Record) bool) error {
	if c.db == nil {
		return fmt.Errorf("collection is not attached to a database")
	}
	te, ok := c.db.storage.(storage.TemporalReader)
	if !ok {
		return fmt.Errorf("storage engine does not support temporal reads")
	}
	return te.ListVisibleAtLSN(c.name, snapshotLSN, func(tr *storage.TemporalRecord) bool {
		return fn(&Record{
			ID: tr.ID, Vector: tr.Vector, Metadata: tr.Metadata,
			Ordinal: tr.Ordinal, Version: tr.Version,
		})
	})
}
