package libravdb

import (
	"context"
	"fmt"

	"github.com/xDarkicex/libravdb/internal/storage"
)

// ErrChangeCursor means the cursor does not belong to the available history.
var ErrChangeCursor = storage.ErrChangeCursor

// ChangeRows contains owned, columnar references to committed mutations.
type ChangeRows = storage.ChangeRows

// ChangePage is a bounded page; consumers must inspect TransactionComplete
// before treating a split transaction as completely delivered.
type ChangePage = storage.ChangePage

type changeReader interface {
	ChangeCheckpoint(context.Context) (string, error)
	ReadChanges(context.Context, string, int) (storage.ChangePage, error)
}

func (db *Database) changeReader() (changeReader, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()
	if db.closed {
		return nil, ErrDatabaseClosed
	}
	reader, ok := db.storage.(changeReader)
	if !ok {
		return nil, fmt.Errorf("storage engine does not expose committed changes")
	}
	return reader, nil
}

// ChangeCheckpoint returns an opaque position after currently committed
// mutations. Store it with the consumer's baseline, then pass it to ReadChanges.
// Remote commits are refreshed before capturing the checkpoint.
func (db *Database) ChangeCheckpoint(ctx context.Context) (string, error) {
	reader, err := db.changeReader()
	if err != nil {
		return "", err
	}
	return reader.ChangeCheckpoint(ctx)
}

// ReadChanges enumerates at most limit compact mutations in commit order.
// An empty after starts at the available feed baseline, reported in StartLSN;
// upgraded databases do not invent changes for their checkpointed past.
// A page may split a transaction: buffer its rows until TransactionComplete
// or Positions/TransactionSizes indicate completion. Record payloads are not
// retained here; historical reads remain subject to temporal retention.
func (db *Database) ReadChanges(ctx context.Context, after string, limit int) (ChangePage, error) {
	reader, err := db.changeReader()
	if err != nil {
		return ChangePage{}, err
	}
	return reader.ReadChanges(ctx, after, limit)
}
