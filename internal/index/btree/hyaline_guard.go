package btree

import (
	"errors"

	"github.com/xDarkicex/memory"
)

// readGuard protects every page pool while a lock-free reader resolves and
// traverses page pointers. A tree can grow a new segment concurrently with a
// search, so the pool membership lock remains held for the whole interval.
type readGuard struct {
	tree        *BTree
	guards      [128]poolGuard
	extraGuards []poolGuard
	guardCount  int
}

type poolGuard struct {
	pool   *memory.ShardedFreeList
	handle memory.HyalineHandle
}

func (t *BTree) enterRead() (*readGuard, error) {
	t.poolsMu.RLock()
	guard := &readGuard{tree: t}
	for _, pool := range t.pagePools {
		handle, err := pool.HyalineEnter(0)
		if err != nil {
			var leaveErr error
			for i := guard.guardCount - 1; i >= 0; i-- {
				entry := guard.guardAt(i)
				leaveErr = errors.Join(leaveErr, entry.pool.HyalineLeave(entry.handle))
			}
			t.poolsMu.RUnlock()
			return nil, errors.Join(err, leaveErr)
		}
		guard.add(poolGuard{pool: pool, handle: handle})
	}
	return guard, nil
}

func (g *readGuard) leave() error {
	if g == nil || g.tree == nil {
		return nil
	}
	var err error
	for i := g.guardCount - 1; i >= 0; i-- {
		entry := g.guardAt(i)
		err = errors.Join(err, entry.pool.HyalineLeave(entry.handle))
	}
	g.tree.poolsMu.RUnlock()
	g.tree = nil
	return err
}

func (g *readGuard) add(entry poolGuard) {
	if g.guardCount < len(g.guards) {
		g.guards[g.guardCount] = entry
	} else {
		g.extraGuards = append(g.extraGuards, entry)
	}
	g.guardCount++
}

func (g *readGuard) guardAt(index int) poolGuard {
	if index < len(g.guards) {
		return g.guards[index]
	}
	return g.extraGuards[index-len(g.guards)]
}
