package graph

import (
	"sync/atomic"
	"time"

	"github.com/xDarkicex/libravdb/internal/storage/wal"
)

// CheckpointCoordinator polls vector and edge subsystems to safely advance the database checkpoint.
type CheckpointCoordinator struct {
	vectorLastFlushedGen *uint32
	edgeLastFlushedGen   *uint32
	checkpointGen        *uint32
	w                    *wal.WAL
	ticker               *time.Ticker
	quit                 chan struct{}
}

// NewCheckpointCoordinator creates a new CheckpointCoordinator.
func NewCheckpointCoordinator(vectorGen, edgeGen, chkGen *uint32, w *wal.WAL) *CheckpointCoordinator {
	return &CheckpointCoordinator{
		vectorLastFlushedGen: vectorGen,
		edgeLastFlushedGen:   edgeGen,
		checkpointGen:        chkGen,
		w:                    w,
		quit:                 make(chan struct{}),
	}
}

// Start begins the polling loop in a background goroutine.
func (c *CheckpointCoordinator) Start() {
	c.ticker = time.NewTicker(100 * time.Millisecond)
	go c.run()
}

// Stop halts the polling loop.
func (c *CheckpointCoordinator) Stop() {
	if c.ticker != nil {
		c.ticker.Stop()
	}
	close(c.quit)
}

func (c *CheckpointCoordinator) run() {
	for {
		select {
		case <-c.ticker.C:
			vGen := atomic.LoadUint32(c.vectorLastFlushedGen)
			eGen := atomic.LoadUint32(c.edgeLastFlushedGen)
			
			minGen := vGen
			if eGen < minGen {
				minGen = eGen
			}

			current := atomic.LoadUint32(c.checkpointGen)
			if minGen > current {
				atomic.StoreUint32(c.checkpointGen, minGen)
				// In a full implementation with partial WAL truncation, we would discard entries <= minGen.
				// Since wal.Truncate() removes all entries, we only truncate if everything is flushed.
				// Assuming vector subsystem handles actual WAL file rotation/truncation.
			}
		case <-c.quit:
			return
		}
	}
}
