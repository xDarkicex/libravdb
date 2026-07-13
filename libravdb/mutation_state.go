package libravdb

import (
	"runtime"
	"sync/atomic"

	"github.com/xDarkicex/libravdb/internal/index"
	offheap "github.com/xDarkicex/memory"
)

const mutationStateSlotCount = 4096

type mutationStateSlot struct {
	state uint64
	_     [56]byte
}

type mutationStateTable struct {
	arena *offheap.Arena
	slots []mutationStateSlot
}

type mutationGuard struct {
	table *mutationStateTable
	token uint64
	slot  uint32
}

type mutationBatchGuard struct {
	table   *mutationStateTable
	entries []*index.VectorEntry
}

func newMutationStateTable() (*mutationStateTable, error) {
	arena, err := offheap.NewArena(mutationStateSlotCount*64, 64)
	if err != nil {
		return nil, err
	}
	slots, err := offheap.ArenaSlice[mutationStateSlot](arena, mutationStateSlotCount)
	if err != nil {
		_ = arena.Free()
		return nil, err
	}
	return &mutationStateTable{
		arena: arena,
		slots: slots[:mutationStateSlotCount],
	}, nil
}

func (t *mutationStateTable) close() {
	if t != nil && t.arena != nil {
		_ = t.arena.Free()
		t.arena = nil
		t.slots = nil
	}
}

func mutationToken(id string) uint64 {
	const (
		offset = uint64(14695981039346656037)
		prime  = uint64(1099511628211)
	)
	hash := offset
	for i := 0; i < len(id); i++ {
		hash ^= uint64(id[i])
		hash *= prime
	}
	if hash == 0 {
		return 1
	}
	return hash
}

func mutationSlotFor(token uint64) uint32 {
	return uint32(token) & (mutationStateSlotCount - 1)
}

func (t *mutationStateTable) tryAcquire(slot uint32, token uint64) bool {
	return atomic.CompareAndSwapUint64(&t.slots[slot].state, 0, token)
}

func (t *mutationStateTable) release(slot uint32, token uint64) {
	if !atomic.CompareAndSwapUint64(&t.slots[slot].state, token, 0) {
		panic("libravdb: mutation state released by non-owner")
	}
}

func (c *Collection) getMutationState() *mutationStateTable {
	if table := c.mutationState.Load(); table != nil {
		return table
	}
	table, err := newMutationStateTable()
	if err != nil {
		panic("libravdb: allocate off-heap mutation state: " + err.Error())
	}
	if c.mutationState.CompareAndSwap(nil, table) {
		return table
	}
	table.close()
	return c.mutationState.Load()
}

func (c *Collection) lockMutationID(id string) mutationGuard {
	table := c.getMutationState()
	token := mutationToken(id)
	slot := mutationSlotFor(token)
	for !table.tryAcquire(slot, token) {
		runtime.Gosched()
	}
	return mutationGuard{table: table, token: token, slot: slot}
}

func (guard mutationGuard) unlock() {
	if guard.table != nil {
		guard.table.release(guard.slot, guard.token)
	}
}

func (c *Collection) lockMutationEntries(entries []*index.VectorEntry) mutationBatchGuard {
	if len(entries) == 0 {
		return mutationBatchGuard{}
	}
	table := c.getMutationState()
	for {
		acquired := 0
		for i, entry := range entries {
			token := mutationToken(entry.ID)
			slot := mutationSlotFor(token)
			if mutationEntrySlotSeen(entries, i, slot) {
				continue
			}
			if table.tryAcquire(slot, token) {
				acquired = i + 1
				continue
			}
			for j := acquired - 1; j >= 0; j-- {
				priorToken := mutationToken(entries[j].ID)
				priorSlot := mutationSlotFor(priorToken)
				if mutationEntrySlotSeen(entries, j, priorSlot) {
					continue
				}
				table.release(priorSlot, priorToken)
			}
			acquired = -1
			break
		}
		if acquired >= 0 {
			return mutationBatchGuard{table: table, entries: entries}
		}
		runtime.Gosched()
	}
}

func mutationEntrySlotSeen(entries []*index.VectorEntry, end int, slot uint32) bool {
	for i := 0; i < end; i++ {
		if mutationSlotFor(mutationToken(entries[i].ID)) == slot {
			return true
		}
	}
	return false
}

func (guard mutationBatchGuard) unlock() {
	if guard.table == nil {
		return
	}
	for i := len(guard.entries) - 1; i >= 0; i-- {
		token := mutationToken(guard.entries[i].ID)
		slot := mutationSlotFor(token)
		if mutationEntrySlotSeen(guard.entries, i, slot) {
			continue
		}
		guard.table.release(slot, token)
	}
}
