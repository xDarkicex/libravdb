package graph

import (
	"sync"
	"sync/atomic"
	"time"
)

// CheckpointCoordinator polls vector and edge subsystems to safely advance the database checkpoint.
type CheckpointCoordinator struct {
	vectorLastFlushedGen *uint32
	edgeLastFlushedGen   *uint32
	checkpointGen        *uint32
	ticker               *time.Ticker
	quit                 chan struct{}
	startOnce            sync.Once
	stopOnce             sync.Once
}

// NewCheckpointCoordinator creates a new CheckpointCoordinator.
func NewCheckpointCoordinator(vectorGen, edgeGen, chkGen *uint32) *CheckpointCoordinator {
	return &CheckpointCoordinator{
		vectorLastFlushedGen: vectorGen,
		edgeLastFlushedGen:   edgeGen,
		checkpointGen:        chkGen,
		quit:                 make(chan struct{}),
	}
}

// Start begins the polling loop in a background goroutine.
func (c *CheckpointCoordinator) Start() {
	c.startOnce.Do(func() {
		c.ticker = time.NewTicker(100 * time.Millisecond)
		go c.run()
	})
}

// Stop halts the polling loop.
func (c *CheckpointCoordinator) Stop() {
	c.stopOnce.Do(func() {
		if c.ticker != nil {
			c.ticker.Stop()
		}
		close(c.quit)
	})
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
			}
		case <-c.quit:
			return
		}
	}
}
