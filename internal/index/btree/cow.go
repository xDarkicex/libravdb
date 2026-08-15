package btree

import (
	"bytes"
	"context"
	"unsafe"
)

// writeTxn tracks dirty pages for one COW mutation.
// Pattern: LMDB MDB_txn — mt_dirty_list (dirtyPages) + mt_loose_pgs (loosePages).
type writeTxn struct {
	tree       *BTree
	dirtyPages map[uint32]*BTreePage // original slot ID → dirty copy (mdb_page_dirty)
	loosePages []uint32              // freed slot IDs, reusable this txn (mdb_page_loose)
	newRoot    *BTreePage            // new root page, if root was split (nil = no split)
	oldRoot    uint32                // original root slot ID
}

// copyPage creates a writable copy of src.
// Pattern: LMDB mdb_page_copy (line 2934) — copies only used bytes, not free space.
// The copy is NOT registered in pageReg until commit (avoiding double registration).
func (txn *writeTxn) copyPage(src *BTreePage) *BTreePage {
	pool, _, slot, _ := txn.tree.allocateSlot()
	dst := (*BTreePage)(unsafe.Pointer(&slot[UserDataOffset]))

	srcData := src.pageData()
	dstData := dst.pageData()

	// Copy used portions: header + nodes [0..Upper) and ptrs [Lower..end)
	copy(dstData[:src.Header.Upper], srcData[:src.Header.Upper])
	copy(dstData[src.Header.Lower:], srcData[src.Header.Lower:])

	dst.Header.RightSibling = src.Header.RightSibling
	dst.Header.LeftSibling = src.Header.LeftSibling
	dst.Header.FirstChild = src.Header.FirstChild
	dst.Header.PageSlot = src.Header.PageSlot // preserve original slot for dirty lookup during txn
	dst.Header.Generation = src.Header.Generation + 1
	dst.Header.HyalineSlot = src.Header.HyalineSlot
	txn.tree.pageOwners[dst] = pool

	// Track: original → copy (mdb_page_dirty pattern).
	// The original page memory will be retired, but the slot ID is reused by the copy.
	txn.dirtyPages[src.Header.PageSlot] = dst
	return dst
}

// allocatePage gets a page, reusing loose pages first.
// Pattern: LMDB mdb_page_alloc (line 2734).
// The page is registered immediately so child pointers have valid slot IDs.
func (txn *writeTxn) allocatePage(flags uint16) (*BTreePage, uint32) {
	if len(txn.loosePages) > 0 {
		id := txn.loosePages[len(txn.loosePages)-1]
		txn.loosePages = txn.loosePages[:len(txn.loosePages)-1]
		page := txn.tree.pageReg.get(id)
		if page != nil {
			page.resetPage(flags)
			return page, id
		}
	}
	pool, segIdx, slot, _ := txn.tree.allocateSlot()
	page := (*BTreePage)(unsafe.Pointer(&slot[UserDataOffset]))
	id := txn.tree.pageReg.register(page)
	page.initPage(flags, id, 0)
	txn.tree.pageSegments[id] = uint8(segIdx)
	txn.tree.pageOwners[page] = pool
	return page, id
}

// commit publishes the COW mutation. Pattern: LMDB mdb_txn_commit.
func (txn *writeTxn) commit() {
	// 1. Replace dirty copies in-place (slot IDs unchanged).
	for origID, dirty := range txn.dirtyPages {
		old := txn.tree.pageReg.get(origID)
		txn.tree.pageReg.replace(origID, dirty)
		if old != nil {
			txn.tree.retirePage(old)
			delete(txn.tree.pageSegments, origID)
		}
	}

	// 2. Remove loose pages (pages no longer in the tree)
	for _, origID := range txn.loosePages {
		txn.tree.retireSlot(origID)
		txn.tree.pageReg.unregister(origID)
	}

	// 3. Publish root
	if txn.newRoot != nil {
		txn.tree.rootID.Store(txn.newRoot.Header.PageSlot)
	}

	txn.tree.gen.Add(1)
}

// abort discards the COW mutation. Pattern: LMDB mdb_txn_abort.
func (txn *writeTxn) abort() {
	// Free dirty copies — never published, no readers can see them
	for _, dirty := range txn.dirtyPages {
		txn.tree.deallocatePage(dirty)
	}
	// Loose pages are still valid originals — keep them registered
}

// insertRecursive performs a COW insert into a subtree.
func (txn *writeTxn) insertRecursive(page *BTreePage, key, value []byte) (uint32, []byte, error) {
	if page.IsLeaf() {
		return txn.insertLeaf(page, key, value)
	}
	return txn.insertBranch(page, key, value)
}

// insertLeaf handles COW insertion into a leaf page.
func (txn *writeTxn) insertLeaf(page *BTreePage, key, value []byte) (uint32, []byte, error) {
	idx, found := page.findKey(key)
	if found {
		return 0, nil, nil
	}

	need := page.SpaceNeeded(len(key), len(value))
	if need <= page.FreeSpace() {
		// Copy the page, insert into the copy
		copy := txn.copyPage(page)
		insertCell(copy, idx, key, value, 0)
		return 0, nil, nil
	}

	// Split — all pages involved are copied
	return txn.splitLeaf(page, idx, key, value)
}

// insertBranch handles COW insertion into a branch page.
func (txn *writeTxn) insertBranch(page *BTreePage, key, value []byte) (uint32, []byte, error) {
	idx, childID := page.findChild(key)

	child := txn.tree.pageReg.get(childID)
	if child == nil {
		return 0, nil, errTreeCorrupt
	}

	newChildID, splitKey, err := txn.insertRecursive(child, key, value)
	if err != nil {
		return 0, nil, err
	}
	if newChildID == 0 {
		return 0, nil, nil
	}

	need := page.SpaceNeeded(len(splitKey), 0)
	if need <= page.FreeSpace() {
		copy := txn.copyPage(page)
		insertCell(copy, idx, splitKey, nil, newChildID)
		return 0, nil, nil
	}

	return txn.splitBranch(page, idx, splitKey, newChildID)
}

// splitLeaf splits a full leaf page with COW.
// The original page is copied, the copy is modified, and the original is retired.
func (txn *writeTxn) splitLeaf(page *BTreePage, insertIdx int, key, value []byte) (uint32, []byte, error) {
	// Copy the original page first (COW: never mutate originals in-place)
	var left *BTreePage
	if _, ok := txn.dirtyPages[page.Header.PageSlot]; ok {
		left = txn.dirtyPages[page.Header.PageSlot]
	} else {
		left = txn.copyPage(page)
	}

	rightPage, rightID := txn.allocatePage(P_LEAF)
	leftSibling := left.Header.LeftSibling
	rightSibling := left.Header.RightSibling

	all := collectLeafKVs(page, insertIdx, key, value)
	splitPoint := len(all) / 2

	left.resetPage(P_LEAF)
	left.Header.LeftSibling = leftSibling
	for i := 0; i < splitPoint; i++ {
		insertCell(left, i, all[i].key, all[i].value, 0)
	}

	for i := splitPoint; i < len(all); i++ {
		insertCell(rightPage, i-splitPoint, all[i].key, all[i].value, 0)
	}

	rightPage.Header.RightSibling = rightSibling
	left.Header.RightSibling = rightID
	rightPage.Header.LeftSibling = left.Header.PageSlot

	// The old right neighbor used to point back to left. Keep the doubly
	// linked leaf chain coherent after inserting right between them. The
	// neighbor is copied through the same COW transaction so readers never
	// observe an in-place mutation of a published page.
	if rightSibling != 0 {
		if neighbor := txn.tree.pageReg.get(rightSibling); neighbor != nil {
			neighborDirty := ensureDirty(txn, neighbor)
			neighborDirty.Header.LeftSibling = rightID
		}
	}

	return rightID, rightPage.HighKey(), nil
}

// splitBranch splits a full branch page with COW.
// Returns (rightPageID, promotedKey, error). The left page is the COW copy.
func (txn *writeTxn) splitBranch(page *BTreePage, insertIdx int, sepKey []byte, newChildID uint32) (uint32, []byte, error) {
	// Copy the original page first (COW: never mutate originals in-place)
	var left *BTreePage
	if _, ok := txn.dirtyPages[page.Header.PageSlot]; ok {
		left = txn.dirtyPages[page.Header.PageSlot]
	} else {
		left = txn.copyPage(page)
	}

	rightPage, rightID := txn.allocatePage(P_BRANCH)
	leftSibling := left.Header.LeftSibling
	rightSibling := left.Header.RightSibling

	all := collectBranchEntries(page, insertIdx, sepKey, newChildID)
	splitPoint := len(all) / 2

	promotedKey := cloneBytes(all[splitPoint].key)

	left.resetPage(P_BRANCH)
	left.Header.LeftSibling = leftSibling
	left.Header.FirstChild = all[0].childID
	for i := 0; i < splitPoint-1; i++ {
		insertCell(left, i, all[i+1].key, nil, all[i+1].childID)
	}

	rightPage.Header.FirstChild = all[splitPoint].childID
	rightCount := len(all) - splitPoint - 1
	for i := 0; i < rightCount; i++ {
		insertCell(rightPage, i, all[splitPoint+1+i].key, nil, all[splitPoint+1+i].childID)
	}

	rightPage.Header.RightSibling = rightSibling
	left.Header.RightSibling = rightID
	rightPage.Header.LeftSibling = left.Header.PageSlot
	if rightSibling != 0 {
		if neighbor := txn.tree.pageReg.get(rightSibling); neighbor != nil {
			neighborDirty := ensureDirty(txn, neighbor)
			neighborDirty.Header.LeftSibling = rightID
		}
	}

	return rightID, promotedKey, nil
}

// newRootCOW creates a new branch root with two children (COW-safe).
func (txn *writeTxn) newRootCOW(leftID, rightID uint32, sepKey []byte) error {
	root, _ := txn.allocatePage(P_BRANCH)
	root.Header.FirstChild = leftID
	insertCell(root, 0, sepKey, nil, rightID)
	txn.newRoot = root
	return nil
}

// insertRecursiveCOW is the public entry point for COW insert, used by BTree.Insert.
// Copies pages along the path; reuses existing dirty copies for batch inserts.
func (t *BTree) insertRecursiveCOW(ctx context.Context, txn *writeTxn, page *BTreePage, key, value []byte) (uint32, []byte, error) {
	_ = ctx
	// Resolve page to dirty copy if it was modified earlier in this txn
	page = resolveChild(t, txn, page.Header.PageSlot)

	if page.IsLeaf() {
		// MDB_APPEND fast path: if key > max key in this page, skip binary search.
		// Pattern: LMDB _mdb_cursor_put (mdb.c:8631).
		var idx int
		if page.Header.Count > 0 && bytes.Compare(key, page.MaxKey()) > 0 {
			idx = int(page.Header.Count)
		} else {
			var found bool
			idx, found = page.findKey(key)
			if found {
				return 0, nil, errKeyExists
			}
		}
		need := page.SpaceNeeded(len(key), len(value))
		if need <= page.FreeSpace() {
			dirty := ensureDirty(txn, page)
			insertCell(dirty, idx, key, value, 0)
			return 0, nil, nil
		}
		return txn.splitLeaf(page, idx, key, value)
	}

	// Branch — ensure dirty copy, descend, handle split
	idx, childID := page.findChild(key)
	branchDirty := ensureDirty(txn, page)

	child := resolveChild(t, txn, childID)
	if child == nil {
		return 0, nil, errTreeCorrupt
	}

	newChildID, splitKey, err := t.insertRecursiveCOW(ctx, txn, child, key, value)
	if err != nil {
		return 0, nil, err
	}
	if newChildID == 0 {
		return 0, nil, nil
	}

	need := branchDirty.SpaceNeeded(len(splitKey), 0)
	if need <= branchDirty.FreeSpace() {
		insertCell(branchDirty, idx, splitKey, nil, newChildID)
		return 0, nil, nil
	}

	return txn.splitBranch(branchDirty, idx, splitKey, newChildID)
}

// ensureDirty returns a writable copy of page, reusing an existing dirty copy.
func ensureDirty(txn *writeTxn, page *BTreePage) *BTreePage {
	if dirty, ok := txn.dirtyPages[page.Header.PageSlot]; ok {
		return dirty
	}
	return txn.copyPage(page)
}

// resolveChild looks up a child page, checking dirty copies first.
func resolveChild(t *BTree, txn *writeTxn, slotID uint32) *BTreePage {
	if dirty, ok := txn.dirtyPages[slotID]; ok {
		return dirty
	}
	return t.pageReg.get(slotID)
}
