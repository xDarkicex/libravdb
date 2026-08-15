package graph

import (
	"errors"

	"github.com/xDarkicex/memory"
)

// graphHyalineGuard keeps the page-pool membership lock while the protected
// pointers are being traversed. A forward page chain may span any of the
// lazily-created page pools, so one handle per live pool is required.
// Holding pagePoolsMu prevents a reader from observing a page published from
// a newly-created pool after its protection set was captured.
type graphHyalineGuard struct {
	store          *graphStore
	handles        [128]graphHyalineHandle
	extraHandles   []graphHyalineHandle
	handleCount    int
	poolLockHeld   bool
	writerLockHeld bool
}

type graphHyalineHandle struct {
	pool   *memory.ShardedFreeList
	handle memory.HyalineHandle
}

func (g *graphStore) enterHyaline(pool *memory.ShardedFreeList, preferredShard int) (graphHyalineGuard, error) {
	if pool == nil {
		return graphHyalineGuard{}, errors.New("graph: nil Hyaline pool")
	}

	g.pagePoolsMu.RLock()
	poolCount := 1
	if len(g.pagePools) > 0 && pool == g.pagePools[0] {
		poolCount = len(g.pagePools)
	}
	guard := graphHyalineGuard{store: g, poolLockHeld: true}
	for i := 0; i < poolCount; i++ {
		candidate := pool
		if poolCount > 1 {
			candidate = g.pagePools[i]
		}
		handle, err := candidate.HyalineEnter(preferredShard)
		if err != nil {
			var leaveErr error
			for j := guard.handleCount - 1; j >= 0; j-- {
				h := guard.handleAt(j)
				leaveErr = errors.Join(leaveErr, h.pool.HyalineLeave(h.handle))
			}
			g.pagePoolsMu.RUnlock()
			return graphHyalineGuard{}, errors.Join(err, leaveErr)
		}
		guard.addHandle(graphHyalineHandle{pool: candidate, handle: handle})
	}
	return guard, nil
}

func (g *graphStore) enterHyalineWrite(pool *memory.ShardedFreeList, preferredShard int) (graphHyalineGuard, error) {
	if pool == nil {
		return graphHyalineGuard{}, errors.New("graph: nil Hyaline pool")
	}
	g.writeMu.Lock()

	g.pagePoolsMu.RLock()
	poolCount := 1
	if len(g.pagePools) > 0 && pool == g.pagePools[0] {
		poolCount = len(g.pagePools)
	}
	g.pagePoolsMu.RUnlock()

	guard := graphHyalineGuard{
		store:          g,
		writerLockHeld: true,
	}
	for i := 0; i < poolCount; i++ {
		candidate := pool
		if poolCount > 1 {
			candidate = g.pagePools[i]
		}
		handle, err := candidate.HyalineEnter(preferredShard)
		if err != nil {
			var leaveErr error
			for j := guard.handleCount - 1; j >= 0; j-- {
				h := guard.handleAt(j)
				leaveErr = errors.Join(leaveErr, h.pool.HyalineLeave(h.handle))
			}
			g.writeMu.Unlock()
			return graphHyalineGuard{}, errors.Join(err, leaveErr)
		}
		guard.addHandle(graphHyalineHandle{pool: candidate, handle: handle})
	}
	return guard, nil
}

func (g *graphHyalineGuard) leave() error {
	if g == nil || g.store == nil {
		return nil
	}
	var err error
	for i := g.handleCount - 1; i >= 0; i-- {
		h := g.handleAt(i)
		err = errors.Join(err, h.pool.HyalineLeave(h.handle))
	}
	if g.poolLockHeld {
		g.store.pagePoolsMu.RUnlock()
	}
	if g.writerLockHeld {
		g.store.writeMu.Unlock()
	}
	g.store = nil
	return err
}

func (g *graphHyalineGuard) addHandle(handle graphHyalineHandle) {
	if g.handleCount < len(g.handles) {
		g.handles[g.handleCount] = handle
	} else {
		g.extraHandles = append(g.extraHandles, handle)
	}
	g.handleCount++
}

func (g *graphHyalineGuard) handleAt(index int) graphHyalineHandle {
	if index < len(g.handles) {
		return g.handles[index]
	}
	return g.extraHandles[index-len(g.handles)]
}

// withHyaline runs fn while all pages reachable through pool are protected.
// Leave errors are returned as part of the same error chain; callers therefore
// cannot accidentally hide an unbalanced or invalid Hyaline interval.
func (g *graphStore) withHyaline(pool *memory.ShardedFreeList, preferredShard int, fn func() error) error {
	guard, err := g.enterHyaline(pool, preferredShard)
	if err != nil {
		return err
	}
	fnErr := fn()
	return joinGraphErrors(fnErr, guard.leave())
}

func (g *graphStore) withHyalineWrite(pool *memory.ShardedFreeList, preferredShard int, fn func() error) error {
	guard, err := g.enterHyalineWrite(pool, preferredShard)
	if err != nil {
		return err
	}
	fnErr := fn()
	return joinGraphErrors(fnErr, guard.leave())
}

func joinGraphErrors(primary, secondary error) error {
	if primary == nil {
		return secondary
	}
	if secondary == nil {
		return primary
	}
	return errors.Join(primary, secondary)
}

func (g *graphStore) rememberPageOwner(page *EdgeTablePage, pool *memory.ShardedFreeList) {
	if page == nil || pool == nil {
		return
	}
	g.ownersMu.Lock()
	g.pageOwners[page] = pool
	g.ownersMu.Unlock()
}

func (g *graphStore) forgetPageOwner(page *EdgeTablePage) *memory.ShardedFreeList {
	if page == nil {
		return nil
	}
	g.ownersMu.Lock()
	pool := g.pageOwners[page]
	delete(g.pageOwners, page)
	delete(g.pageSegments, page)
	g.ownersMu.Unlock()
	return pool
}

func (g *graphStore) rememberPropertyOwner(page *EdgePropertyPage, pool *memory.ShardedFreeList) {
	if page == nil || pool == nil {
		return
	}
	g.ownersMu.Lock()
	g.propertyOwners[page] = pool
	g.ownersMu.Unlock()
}

func (g *graphStore) forgetPropertyOwner(page *EdgePropertyPage) *memory.ShardedFreeList {
	if page == nil {
		return nil
	}
	g.ownersMu.Lock()
	pool := g.propertyOwners[page]
	delete(g.propertyOwners, page)
	g.ownersMu.Unlock()
	return pool
}
