package btree

import (
	"context"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/xDarkicex/memory"
)

// Config holds B-tree creation parameters.
type Config struct {
	PageSlots  int
	PageShards int
}

// DefaultConfig returns a Config with sensible defaults (~512MB page pool).
func DefaultConfig() Config {
	return Config{
		PageSlots:  131072,
		PageShards: 64,
	}
}

// BTree is an off-heap, lock-free-read B-link tree backed by ShardedFreeList pages.
type BTree struct {
	cfg       Config
	pageReg   *pageRegistry
	pagePool  *memory.ShardedFreeList
	rootID    atomic.Uint32
	gen       atomic.Uint32
	mu        sync.Mutex
	nodeCount atomic.Int64
}

// New creates a B-tree with an off-heap page pool.
func New(cfg Config) (*BTree, error) {
	if cfg.PageSlots <= 0 {
		cfg = DefaultConfig()
	}
	if cfg.PageShards <= 0 {
		cfg.PageShards = 64
	}

	pool, err := memory.NewShardedFreeList(memory.FreeListConfig{
		PoolSize:  uint64(cfg.PageSlots) * PageSize,
		SlotSize:  PageSize,
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 32,
		Prealloc:  false,
	}, 64, cfg.PageShards)
	if err != nil {
		return nil, err
	}

	reg := newPageRegistry()
	_, rootID, err := allocPage(pool, reg, P_LEAF, 0)
	if err != nil {
		pool.Free()
		return nil, err
	}

	t := &BTree{cfg: cfg, pageReg: reg, pagePool: pool}
	t.rootID.Store(rootID)
	return t, nil
}

func allocPage(pool *memory.ShardedFreeList, reg *pageRegistry, flags uint16, shard int) (*BTreePage, uint32, error) {
	slot, err := pool.Allocate()
	if err != nil {
		return nil, 0, err
	}
	page := (*BTreePage)(unsafe.Pointer(&slot[UserDataOffset]))
	slotID := reg.register(page)
	page.initPage(flags, slotID, shard)
	return page, slotID, nil
}

func freePage(pool *memory.ShardedFreeList, reg *pageRegistry, slotID uint32) {
	if slotID == 0 {
		return
	}
	page := reg.get(slotID)
	reg.unregister(slotID)
	if page != nil {
		slotBytes := unsafe.Slice((*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(page))-UserDataOffset)), PageSize)
		pool.Deallocate(slotBytes)
	}
}

// Insert inserts or replaces a key-value pair using COW semantics.
// Pattern: LMDB _mdb_cursor_put (line 8556) — walk root→leaf, copy pages, bubble splits.
func (t *BTree) Insert(ctx context.Context, key, value []byte) error {
	if len(key) > MaxKeyLen {
		return errKeyTooLarge
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	rootID := t.rootID.Load()
	root := t.pageReg.get(rootID)
	if root == nil {
		return errTreeClosed
	}

	txn := &writeTxn{
		tree:       t,
		dirtyPages: make(map[uint32]*BTreePage),
		oldRoot:    rootID,
	}

	newChildID, splitKey, err := t.insertRecursiveCOW(ctx, txn, root, key, value)
	if err != nil {
		txn.abort()
		return err
	}

	if newChildID != 0 {
		if err := txn.newRootCOW(rootID, newChildID, splitKey); err != nil {
			txn.abort()
			return err
		}
	}

	txn.commit()
	t.nodeCount.Add(1)
	return nil
}

// insertRecursive inserts into a subtree. Returns (newRightChild, splitKey, error).
// newRightChild=0 means no split occurred.
func (t *BTree) insertRecursive(ctx context.Context, page *BTreePage, key, value []byte) (uint32, []byte, error) {
	if page.IsLeaf() {
		return t.insertLeaf(ctx, page, key, value)
	}
	return t.insertBranch(ctx, page, key, value)
}

// insertLeaf handles insertion into a leaf page. May trigger a split.
func (t *BTree) insertLeaf(ctx context.Context, page *BTreePage, key, value []byte) (uint32, []byte, error) {
	idx, found := page.findKey(key)
	if found {
		return 0, nil, nil
	}

	need := page.SpaceNeeded(len(key), len(value))
	if need <= page.FreeSpace() {
		insertCell(page, idx, key, value, 0)
		page.Header.Generation++
		return 0, nil, nil
	}

	rightID, err := t.splitLeaf(page, idx, key, value)
	if err != nil {
		return 0, nil, err
	}
	rightPage := t.pageReg.get(rightID)
	return rightID, rightPage.HighKey(), nil
}

// insertBranch handles insertion into a branch page. Follows child, handles split.
func (t *BTree) insertBranch(ctx context.Context, page *BTreePage, key, value []byte) (uint32, []byte, error) {
	idx, childID := page.findChild(key)

	child := t.pageReg.get(childID)
	if child == nil {
		return 0, nil, errTreeCorrupt
	}

	newChildID, splitKey, err := t.insertRecursive(ctx, child, key, value)
	if err != nil {
		return 0, nil, err
	}
	if newChildID == 0 {
		return 0, nil, nil
	}

	need := page.SpaceNeeded(len(splitKey), 0)
	if need <= page.FreeSpace() {
		insertCell(page, idx, splitKey, nil, newChildID)
		page.Header.Generation++
		return 0, nil, nil
	}

	return t.splitBranch(page, idx, splitKey, newChildID)
}

// splitLeaf splits a full leaf page, distributes keys, links right sibling.
func (t *BTree) splitLeaf(page *BTreePage, insertIdx int, key, value []byte) (uint32, error) {
	rightPage, rightID, err := allocPage(t.pagePool, t.pageReg, P_LEAF, 0)
	if err != nil {
		return 0, err
	}

	all := collectLeafKVs(page, insertIdx, key, value)
	splitPoint := len(all) / 2

	page.resetPage(P_LEAF)
	for i := 0; i < splitPoint; i++ {
		insertCell(page, i, all[i].key, all[i].value, 0)
	}

	for i := splitPoint; i < len(all); i++ {
		insertCell(rightPage, i-splitPoint, all[i].key, all[i].value, 0)
	}

	rightPage.Header.RightSibling = page.Header.RightSibling
	page.Header.RightSibling = rightID
	page.Header.Generation++
	rightPage.Header.Generation++

	return rightID, nil
}

// splitBranch splits a full branch page. Returns (rightPageID, promotedKey, error).
func (t *BTree) splitBranch(page *BTreePage, insertIdx int, sepKey []byte, newChildID uint32) (uint32, []byte, error) {
	rightPage, rightID, err := allocPage(t.pagePool, t.pageReg, P_BRANCH, 0)
	if err != nil {
		return 0, nil, err
	}

	all := collectBranchEntries(page, insertIdx, sepKey, newChildID)
	splitPoint := len(all) / 2

	promotedKey := cloneBytes(all[splitPoint].key)

	// Left page: entries 0..splitPoint-1
	// all[0] = FirstChild (no key), all[1..splitPoint-1] = (key, child) pairs
	page.resetPage(P_BRANCH)
	page.Header.FirstChild = all[0].childID
	for i := 0; i < splitPoint-1; i++ {
		insertCell(page, i, all[i+1].key, nil, all[i+1].childID)
	}

	// Right page: entries splitPoint..len(all)-1
	// all[splitPoint] = promoted (goes to parent), all[splitPoint+1..] = remaining entries
	rightPage.Header.FirstChild = all[splitPoint].childID
	rightCount := len(all) - splitPoint - 1
	for i := 0; i < rightCount; i++ {
		insertCell(rightPage, i, all[splitPoint+1+i].key, nil, all[splitPoint+1+i].childID)
	}

	rightPage.Header.RightSibling = page.Header.RightSibling
	page.Header.RightSibling = rightID
	rightPage.Header.LeftSibling = page.Header.PageSlot
	page.Header.Generation++
	rightPage.Header.Generation++

	return rightID, promotedKey, nil
}

// newRoot creates a branch root with two children separated by sepKey.
func (t *BTree) newRoot(leftID, rightID uint32, sepKey []byte) (uint32, error) {
	root, rootID, err := allocPage(t.pagePool, t.pageReg, P_BRANCH, 0)
	if err != nil {
		return 0, err
	}

	root.Header.FirstChild = leftID
	insertCell(root, 0, sepKey, nil, rightID)
	root.Header.Generation++

	return rootID, nil
}

// Search finds the value for an exact key. Lock-free read with generation check.
func (t *BTree) Search(ctx context.Context, key []byte) ([]byte, error) {
	if len(key) > MaxKeyLen {
		return nil, errKeyTooLarge
	}

	for {
		gen := t.gen.Load()
		rootID := t.rootID.Load()
		page := t.pageReg.get(rootID)
		if page == nil {
			return nil, errTreeClosed
		}

		// Descend the tree within one generation snapshot
		for {
			val, childID, idx, found := searchPage(page, key)

			// Right-link traversal: if key is beyond this page's range, follow sibling
			for !found && page.IsLeaf() && idx >= int(page.Header.Count) && page.Header.RightSibling != 0 {
				page = t.pageReg.get(page.Header.RightSibling)
				if page == nil {
					return nil, errTreeCorrupt
				}
				val, childID, idx, found = searchPage(page, key)
			}

			if val != nil {
				result := cloneBytes(val)
				if t.gen.Load() == gen {
					return result, nil
				}
				break // generation changed, retry outer loop
			}

			if childID != 0 {
				page = t.pageReg.get(childID)
				if page == nil {
					return nil, errTreeCorrupt
				}
				continue // descend
			}

			// Not found in leaf — verify generation before reporting
			if t.gen.Load() == gen {
				return nil, errKeyNotFound
			}
			break // retry
		}
		// Generation changed — retry outer loop
	}
}

// searchPage searches within a single page. Returns:
//   - value if found in leaf
//   - childPageID if this is a branch and we need to descend
//   - idx = insertion/found index
//   - found = exact match found
func searchPage(page *BTreePage, key []byte) (value []byte, childPageID uint32, idx int, found bool) {
	if page.IsLeaf() {
		i, ok := page.findKey(key)
		if ok {
			return page.NodeAt(i).Value(), 0, i, true
		}
		return nil, 0, i, false
	}

	// Branch page
	i, childID := page.findChild(key)
	return nil, childID, i, false
}

// findChild returns the child page ID for a key in a branch page.
// Uses the convention:
//   - FirstChild for keys < first key
//   - Node[i].Child for keys >= Node[i].Key and < Node[i+1].Key
func (p *BTreePage) findChild(key []byte) (idx int, childID uint32) {
	i, _ := p.findKey(key)
	if i == 0 {
		return 0, p.Header.FirstChild
	}
	// Child pointer is on node[i-1].Child
	// (node[i-1] is the last node with key <= search key)
	return i, p.NodeAt(i - 1).Child
}

// Close frees all pages.
func (t *BTree) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	for _, id := range t.pageReg.snapshotIDs() {
		freePage(t.pagePool, t.pageReg, id)
	}
	return t.pagePool.Free()
}

// BatchInsert inserts multiple key-value pairs within a single COW transaction.
// Pages freed by early splits are reused by later inserts in the same batch.
// This avoids the per-insert overhead of root copies and commit cycles.
func (t *BTree) BatchInsert(ctx context.Context, pairs []KVPair) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	rootID := t.rootID.Load()
	root := t.pageReg.get(rootID)
	if root == nil {
		return errTreeClosed
	}

	txn := &writeTxn{
		tree:       t,
		dirtyPages: make(map[uint32]*BTreePage),
		oldRoot:    rootID,
	}

	var err error
	inserted := 0
	effectiveRootID := rootID
	effectiveRoot := root
	for _, p := range pairs {
		if len(p.Key) > MaxKeyLen {
			err = errKeyTooLarge
			break
		}
		if p.Key == nil {
			continue
		}

		// Resolve effective root: if a previous split created a new root, use it
		if txn.newRoot != nil {
			effectiveRoot = txn.newRoot
			effectiveRootID = effectiveRoot.Header.PageSlot
		}

		newChildID, splitKey, e := t.insertRecursiveCOW(ctx, txn, effectiveRoot, p.Key, p.Value)
		if e != nil {
			if e == errKeyExists {
				continue
			}
			err = e
			break
		}

		if newChildID != 0 {
			if e := txn.newRootCOW(effectiveRootID, newChildID, splitKey); e != nil {
				err = e
				break
			}
		}
		inserted++
	}

	if err != nil {
		txn.abort()
		return err
	}

	txn.commit()
	t.nodeCount.Add(int64(inserted))
	return nil
}

// KVPair is a key-value pair for batch operations.
type KVPair struct {
	Key, Value []byte
}

// Delete removes a key from the tree. COW path: copies leaf, removes key.
// Phase 1: leaf deletion only, no page merge (underflow deferred per Lehman-Yao).
func (t *BTree) Delete(ctx context.Context, key []byte) error {
	if len(key) > MaxKeyLen {
		return errKeyTooLarge
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	rootID := t.rootID.Load()
	root := t.pageReg.get(rootID)
	if root == nil {
		return errTreeClosed
	}

	txn := &writeTxn{
		tree:       t,
		dirtyPages: make(map[uint32]*BTreePage),
		oldRoot:    rootID,
	}

	if err := t.deleteCOW(ctx, txn, root, key); err != nil {
		txn.abort()
		return err
	}

	txn.commit()
	t.nodeCount.Add(-1)
	return nil
}

func (t *BTree) deleteCOW(ctx context.Context, txn *writeTxn, page *BTreePage, key []byte) error {
	_ = ctx
	if page.IsLeaf() {
		idx, found := page.findKey(key)
		if !found {
			return errKeyNotFound
		}
		copy := txn.copyPage(page)
		deleteCell(copy, idx)
		// Check fill threshold and rebalance if needed
		return t.rebalance(ctx, txn, copy)
	}

	// Branch — descend to child
	_, childID := page.findChild(key)
	child := t.pageReg.get(childID)
	if child == nil {
		return errTreeCorrupt
	}
	if err := t.deleteCOW(ctx, txn, child, key); err != nil {
		return err
	}
	// After child deletion, check if this branch page needs rebalancing
	return t.rebalance(ctx, txn, page)
}

// deleteCell removes a node at index idx from a page.
func deleteCell(page *BTreePage, idx int) {
	n := int(page.Header.Count)
	if idx < 0 || idx >= n {
		return
	}

	// Shift ptrs: move ptrs[idx+1..n-1] → ptrs[idx..n-2]
	// Ptrs grow downward starting at Lower. Removing idx means
	// the ptr array shrinks by 2 bytes (Lower increases).
	ptrs := page.Ptrs()
	removedOffset := ptrs[idx]

	// Shift ptrs left (toward lower addresses = higher byte offsets since Lower decreases)
	copy(ptrs[idx:], ptrs[idx+1:])

	// Increase Lower (shrink ptr array upward)
	page.Header.Lower += 2
	page.Header.Count--

	// TODO: compact node data (shift nodes after the removed one down)
	// For Phase 1: leave the deleted node in place (wasted space).
	// NodeAt(idx) now returns the old node that was at idx+1.
	// The old node at removedOffset is orphaned (its bytes are still in the page).
	_ = removedOffset
}

// fillThreshold is the minimum fill percentage before a page is merged.
// Pattern: LMDB FILL_THRESHOLD (mdb.c:1139) — pages below 25% are candidates.
const fillThreshold = 0.25

// fillPercent returns the fill percentage (0.0–1.0) of a page.
func fillPercent(page *BTreePage) float64 {
	maxKeys := UserDataSize / (nodeHeaderSize + 4) // ~4-byte minimum KV pair
	if maxKeys == 0 {
		return 1.0
	}
	return float64(page.Header.Count) / float64(maxKeys)
}

// findChildIndex returns the index of the separator in a branch page
// that points to childSlotID. Returns -1 if not found.
func findChildIndex(page *BTreePage, childSlotID uint32) int {
	if childSlotID == 0 {
		return -1
	}
	for i := 0; i < int(page.Header.Count); i++ {
		if page.NodeAt(i).Child == childSlotID {
			return i
		}
	}
	return -1
}

// rebalance checks if a page is below fill threshold and merges/redistributes.
// Pattern: LMDB mdb_rebalance (mdb.c:10318).
func (t *BTree) rebalance(ctx context.Context, txn *writeTxn, page *BTreePage) error {
	_ = ctx

	// Case 1: above threshold — nothing to do
	if fillPercent(page) >= fillThreshold {
		return nil
	}

	// Case 2 & 3: root page
	if page.Header.PageSlot == t.rootID.Load() {
		if page.Header.Count == 0 {
			t.rootID.Store(0)
			return nil
		}
		if page.IsBranch() && page.Header.Count == 1 {
			// Collapse: single-child root → child becomes root
			childID := page.Header.FirstChild
			if childID == 0 {
				childID = page.NodeAt(0).Child
			}
			t.rootID.Store(childID)
			loosePage(txn, page)
			return nil
		}
		return nil
	}

	// Case 4: merge with right sibling
	if page.Header.RightSibling != 0 {
		right := t.pageReg.get(page.Header.RightSibling)
		if right != nil && fillPercent(right) >= fillThreshold && right.Header.Count > 1 {
			return redistribRight(txn, page, right)
		}
		return mergeRight(ctx, txn, page, right)
	}

	// Case 4b: merge with left sibling
	if page.Header.LeftSibling != 0 {
		left := t.pageReg.get(page.Header.LeftSibling)
		if left != nil && fillPercent(left) >= fillThreshold && left.Header.Count > 1 {
			return redistribLeft(txn, left, page)
		}
		return mergeLeft(ctx, txn, left, page)
	}

	return nil
}

// mergeRight merges the right sibling into the left page.
// Pattern: LMDB mdb_page_merge (mdb.c:10151).
func mergeRight(ctx context.Context, txn *writeTxn, left, right *BTreePage) error {
	_ = ctx
	leftDirty := ensureDirty(txn, left)

	// Move all nodes from right to left
	for i := 0; i < int(right.Header.Count); i++ {
		rnode := right.NodeAt(i)
		insertCell(leftDirty, int(leftDirty.Header.Count), cloneBytes(rnode.Key()), cloneBytes(rnode.Value()), rnode.Child)
	}

	// Update sibling chain: skip right
	leftDirty.Header.RightSibling = right.Header.RightSibling
	if right.Header.RightSibling != 0 {
		nextRight := txn.tree.pageReg.get(right.Header.RightSibling)
		if nextRight != nil {
			nextDirty := ensureDirty(txn, nextRight)
			nextDirty.Header.LeftSibling = leftDirty.Header.PageSlot
		}
	}

	// Free right page
	loosePage(txn, right)

	return nil
}

// mergeLeft merges the left sibling into the right page.
func mergeLeft(ctx context.Context, txn *writeTxn, left, right *BTreePage) error {
	_ = ctx
	rightDirty := ensureDirty(txn, right)

	// Move all nodes from left to right (prepend)
	// Insert left nodes first, then right nodes are already in place
	leftNodes := make([]struct {
		key, value []byte
		child      uint32
	}, left.Header.Count)
	for i := 0; i < int(left.Header.Count); i++ {
		n := left.NodeAt(i)
		leftNodes[i].key = cloneBytes(n.Key())
		leftNodes[i].value = cloneBytes(n.Value())
		leftNodes[i].child = n.Child
	}
	// Reset right and re-insert: left nodes first, then old right nodes
	oldRightNodes := make([]struct {
		key, value []byte
		child      uint32
	}, rightDirty.Header.Count)
	oldCount := int(rightDirty.Header.Count)
	for i := 0; i < oldCount; i++ {
		n := rightDirty.NodeAt(i)
		oldRightNodes[i].key = cloneBytes(n.Key())
		oldRightNodes[i].value = cloneBytes(n.Value())
		oldRightNodes[i].child = n.Child
	}
	rightDirty.resetPage(rightDirty.Header.Flags)
	for i := 0; i < len(leftNodes); i++ {
		insertCell(rightDirty, i, leftNodes[i].key, leftNodes[i].value, leftNodes[i].child)
	}
	for i := 0; i < len(oldRightNodes); i++ {
		insertCell(rightDirty, len(leftNodes)+i, oldRightNodes[i].key, oldRightNodes[i].value, oldRightNodes[i].child)
	}

	// Update sibling chain
	rightDirty.Header.LeftSibling = left.Header.LeftSibling
	if left.Header.LeftSibling != 0 {
		prevLeft := txn.tree.pageReg.get(left.Header.LeftSibling)
		if prevLeft != nil {
			prevDirty := ensureDirty(txn, prevLeft)
			prevDirty.Header.RightSibling = rightDirty.Header.PageSlot
		}
	}

	loosePage(txn, left)
	return nil
}

// redistribRight moves one key from the right sibling to the left page.
// Pattern: LMDB mdb_node_move (mdb.c).
func redistribRight(txn *writeTxn, left, right *BTreePage) error {
	leftDirty := ensureDirty(txn, left)
	rightDirty := ensureDirty(txn, right)
	// Move first node from right to left
	n := rightDirty.NodeAt(0)
	insertCell(leftDirty, int(leftDirty.Header.Count), cloneBytes(n.Key()), cloneBytes(n.Value()), n.Child)
	deleteCell(rightDirty, 0)
	return nil
}

// redistribLeft moves one key from the left sibling to the right page.
func redistribLeft(txn *writeTxn, left, right *BTreePage) error {
	leftDirty := ensureDirty(txn, left)
	rightDirty := ensureDirty(txn, right)
	// Move last node from left to front of right
	n := leftDirty.NodeAt(int(leftDirty.Header.Count) - 1)
	insertCell(rightDirty, 0, cloneBytes(n.Key()), cloneBytes(n.Value()), n.Child)
	deleteCell(leftDirty, int(leftDirty.Header.Count)-1)
	return nil
}

// loosePage adds a page to the txn's loose list for deferred reclamation.
// Pattern: LMDB mdb_page_loose (mdb.c:2417).
func loosePage(txn *writeTxn, page *BTreePage) {
	txn.loosePages = append(txn.loosePages, page.Header.PageSlot)
}

// Len returns the approximate key count.
func (t *BTree) Len() int { return int(t.nodeCount.Load()) }

// --- Internal helpers ---

type kvPair struct {
	key, value []byte
}

type branchEntry struct {
	key     []byte
	childID uint32
}

func collectLeafKVs(page *BTreePage, insertIdx int, key, value []byte) []kvPair {
	n := int(page.Header.Count)
	all := make([]kvPair, n+1)
	j := 0
	for i := 0; i < n; i++ {
		if i == insertIdx {
			all[j] = kvPair{cloneBytes(key), cloneBytes(value)}
			j++
		}
		node := page.NodeAt(i)
		all[j] = kvPair{cloneBytes(node.Key()), cloneBytes(node.Value())}
		j++
	}
	if insertIdx == n {
		all[j] = kvPair{cloneBytes(key), cloneBytes(value)}
	}
	return all
}

func collectBranchEntries(page *BTreePage, insertIdx int, sepKey []byte, newChildID uint32) []branchEntry {
	n := int(page.Header.Count)
	// Collect n existing entries + FirstChild + 1 inserted separator = n + 2 entries
	total := n + 2

	all := make([]branchEntry, total)
	all[0] = branchEntry{childID: page.Header.FirstChild}

	j := 1 // index in all
	for i := 0; i < n; i++ {
		if i == insertIdx {
			all[j] = branchEntry{key: cloneBytes(sepKey), childID: newChildID}
			j++
		}
		node := page.NodeAt(i)
		all[j] = branchEntry{key: cloneBytes(node.Key()), childID: node.Child}
		j++
	}
	if insertIdx == n {
		all[j] = branchEntry{key: cloneBytes(sepKey), childID: newChildID}
	}
	return all
}

// insertCell writes a key-value pair (or key+child for branch) into a page at index idx.
func insertCell(page *BTreePage, idx int, key, value []byte, child uint32) {
	nodeOff := int(page.Header.Upper)
	page.Header.Lower -= 2 // grow ptr array downward (makes room at low end)
	page.Header.Count++

	ptrs := page.Ptrs()

	// Lower decreased by 2 bytes — the ptr array now starts 2 bytes earlier.
	// The old ptr values are still at the SAME byte positions, but their slice
	// indices shifted up by 1 (ptrs[i+1] = old ptrs[i]).
	// For insertion at idx, we need to shift old ptrs[0..idx-1] from positions
	// [1..idx] down to [0..idx-1]. We shift right-to-left to avoid overwrites.
	if idx > 0 {
		// Shift old ptrs[0..idx-1] from positions [1..idx] down to [0..idx-1].
		// Iterate LEFT-TO-RIGHT so each source is read before its destination is written.
		for i := 0; i < idx; i++ {
			ptrs[i] = ptrs[i+1]
		}
	}
	ptrs[idx] = uint16(nodeOff)

	writeNodeAt(page, nodeOff, key, value, child)
	page.Header.Upper += uint16(nodeSize(len(key), len(value)))
}

func writeNodeAt(page *BTreePage, offset int, key, value []byte, child uint32) {
	n := (*BTreeNode)(unsafe.Pointer(uintptr(unsafe.Pointer(page)) + uintptr(offset)))
	n.KeyLen = uint16(len(key))
	n.ValLen = uint16(len(value))
	n.Child = child
	copy(n.Key(), key)
	if value != nil {
		copy(n.Value(), value)
	}
}

func cloneBytes(b []byte) []byte {
	if b == nil {
		return nil
	}
	out := make([]byte, len(b))
	copy(out, b)
	return out
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
