package singlefile

import (
	"context"
	"fmt"
	"os"

	"github.com/xDarkicex/libravdb/internal/storage"
)

// OpenV1 opens a v1-formatted database in a read-only compatibility mode
// to facilitate migration to v2. It skips loading corrupted v1 binary indexes
// and relies entirely on recovering records from the snapshot and WAL.
func OpenV1(path string) (*Engine, error) {
	resolved, err := resolveDatabasePath(path)
	if err != nil {
		return nil, err
	}

	// Open read-only to be safe
	file, err := os.OpenFile(resolved, os.O_RDONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("open v1 database file: %w", err)
	}

	stat, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("stat v1 database file: %w", err)
	}

	if stat.Size() == 0 {
		file.Close()
		return nil, fmt.Errorf("v1 database is empty")
	}

	engine := &Engine{
		path:        resolved,
		file:        file,
		state:       &persistedState{NextCollectionID: 1, Collections: make(map[string]*persistedCollection)},
		collections: make(map[string]*Collection),
	}
	engine.ctx, engine.cancel = context.WithCancel(context.Background())
	engine.status.Store(int32(storage.StatusStarting))

	header, err := engine.readHeader()
	if err != nil {
		engine.fail(fmt.Errorf("read header: %w", err))
		return nil, err
	}

	// ENFORCE version 1
	if header.FormatVersion != 1 {
		engine.fail(fmt.Errorf("expected format version 1 for migration, got %d", header.FormatVersion))
		return nil, fmt.Errorf("not a v1 database")
	}

	engine.fileID = header.FileID
	chosen, err := engine.selectRecoveryCandidate()
	if err != nil {
		engine.fail(err)
		return nil, err
	}
	engine.metaEpoch = chosen.meta.MetaEpoch
	engine.activeMetaPage = metaPageNumber(chosen.meta)
	engine.lastLSN = chosen.meta.LastAppliedLSN
	engine.state = chosen.state
	engine.replayedTxs = 0
	engine.discardedTxs = 0

	// SKIP loadIndexes entirely for v1compat!
	// Phase 2: rebuild indexes from records
	engine.status.Store(int32(storage.StatusRecoveringIndexes))
	if err := engine.rebuildIndexesFromRecords(); err != nil {
		engine.fail(fmt.Errorf("rebuild indexes: %w", err))
		return nil, err
	}

	// Phase 3: replay WAL
	engine.status.Store(int32(storage.StatusReplayingWAL))
	if err := engine.replayWAL(chosen.meta.LastAppliedLSN); err != nil {
		engine.fail(fmt.Errorf("replay WAL: %w", err))
		return nil, err
	}

	engine.status.Store(int32(storage.StatusReady))
	return engine, nil
}
