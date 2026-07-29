package singlefile

import (
	"os"

	"github.com/xDarkicex/libravdb/internal/storage"
)

// installTestFaultHooks injects failure hooks into the Engine's WAL pipeline.
// This is strictly a test-only utility to simulate append or sync failures
// without exposing test hooks in the public or production API boundaries.
func installTestFaultHooks(eng storage.Engine, writeFn func([]byte) (int, error), syncFn func(*os.File) error) {
	e, ok := eng.(*Engine)
	if !ok {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.walWriteFn = writeFn
	e.walSyncFn = syncFn
}

// withCheckpointFaultHooks returns an Option that installs checkpoint write
// and sync failure hooks before recovery runs. Test-only: same-package
// callers use it to verify that a checkpoint write/sync failure during V2
// migration prevents the engine from publishing StatusReady with nondurable
// generated IDs.
func withCheckpointFaultHooks(writeFn func(offset int64, data []byte) (int, error), syncFn func(*os.File) error) Option {
	return func(e *Engine) error {
		e.checkpointWriteFn = writeFn
		e.checkpointSyncFn = syncFn
		return nil
	}
}
