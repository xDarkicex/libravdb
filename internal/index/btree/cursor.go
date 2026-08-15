package btree

import "bytes"

// Cursor provides ordered iteration over B-tree keys.
// Intra-page advance is O(1). Page-boundary crossing re-descents from root.
//
// Read safety: the cursor validates page pointers against the registry on every
// access. If commit() replaced a page via COW, the stale pointer is detected
// and the cursor re-descents from root using lastKey as pivot.
type Cursor struct {
	tree    *BTree
	stack   []cursorLevel
	valid   bool
	lastKey []byte
}

type cursorLevel struct {
	pageID uint32
	page   *BTreePage
	index  int
}

// Seek positions the cursor at the first key >= target.
func (t *BTree) Seek(target []byte) *Cursor {
	c := &Cursor{tree: t, stack: make([]cursorLevel, 0, 8)}
	guard, err := t.enterRead()
	if err != nil {
		return c
	}
	c.descend(target)
	if err := guard.leave(); err != nil {
		c.valid = false
	}
	return c
}

// SeekFirst positions the cursor at the smallest key.
func (t *BTree) SeekFirst() *Cursor {
	return t.Seek(nil)
}

func (c *Cursor) descend(target []byte) {
	c.stack = c.stack[:0]
	c.valid = false

	rootID := c.tree.rootID.Load()
	page := c.tree.pageReg.get(rootID)
	if page == nil || page.Header.Count == 0 {
		return
	}

	pageID := rootID
	for page.IsBranch() {
		var idx int
		var childID uint32
		if target != nil {
			idx, childID = page.findChild(target)
		} else {
			idx, childID = 0, page.Header.FirstChild
		}
		c.stack = append(c.stack, cursorLevel{pageID: pageID, page: page, index: idx})
		page = c.tree.pageReg.get(childID)
		pageID = childID
		if page == nil {
			return
		}
	}

	var idx int
	if target != nil {
		idx, _ = page.findKey(target)
	} else {
		idx = 0
	}

	// B-link: if target is past this leaf's last key, follow right sibling
	for idx >= int(page.Header.Count) && page.Header.RightSibling != 0 {
		nextID := page.Header.RightSibling
		page = c.tree.pageReg.get(nextID)
		pageID = nextID
		if page == nil {
			c.valid = false
			return
		}
		if target != nil {
			idx, _ = page.findKey(target)
		} else {
			idx = 0
		}
	}

	c.stack = append(c.stack, cursorLevel{pageID: pageID, page: page, index: idx})
	c.valid = idx < int(page.Header.Count)
	if c.valid {
		c.lastKey = cloneBytes(page.NodeAt(idx).Key())
	}
}

// Valid returns true if the cursor points to a valid key.
func (c *Cursor) Valid() bool { return c.valid }

// Key returns the current key. Validates page pointer safety.
func (c *Cursor) Key() []byte {
	if !c.valid {
		return nil
	}
	guard, err := c.tree.enterRead()
	if err != nil {
		c.valid = false
		return nil
	}
	valid := c.checkPage()
	key := cloneBytes(c.lastKey)
	if err := guard.leave(); err != nil {
		c.valid = false
		return nil
	}
	if !valid {
		return nil
	}
	return key
}

// Value returns the current value. Validates page pointer safety first.
func (c *Cursor) Value() []byte {
	if !c.valid {
		return nil
	}
	guard, err := c.tree.enterRead()
	if err != nil {
		c.valid = false
		return nil
	}
	if !c.checkPage() || c.stack[len(c.stack)-1].index >= int(c.stack[len(c.stack)-1].page.Header.Count) {
		_ = guard.leave()
		return nil
	}
	top := &c.stack[len(c.stack)-1]
	value := cloneBytes(top.page.NodeAt(top.index).Value())
	if err := guard.leave(); err != nil {
		c.valid = false
		return nil
	}
	return value
}

// Next advances to the next key in order.
func (c *Cursor) Next() (ok bool) {
	if !c.valid {
		return false
	}
	guard, err := c.tree.enterRead()
	if err != nil {
		c.valid = false
		return false
	}
	defer func() {
		if err := guard.leave(); err != nil {
			c.valid = false
			ok = false
		}
	}()

	top := &c.stack[len(c.stack)-1]

	// Validate leaf hasn't been replaced by COW commit
	if !c.checkPage() {
		return c.valid
	}

	top.index++
	if top.index < int(top.page.Header.Count) {
		c.lastKey = cloneBytes(top.page.NodeAt(top.index).Key())
		return true
	}

	// Past end of page — try right sibling (B-link) before giving up
	if top.page.Header.RightSibling != 0 {
		sib := c.tree.pageReg.get(top.page.Header.RightSibling)
		if sib != nil && sib.Header.Count > 0 {
			top.page = sib
			top.pageID = top.page.Header.RightSibling
			top.index = 0
			c.lastKey = cloneBytes(sib.NodeAt(0).Key())
			return true
		}
	}

	// Re-descent from root for next key
	nextKey := make([]byte, len(c.lastKey)+1)
	copy(nextKey, c.lastKey)
	c.descend(nextKey)
	return c.valid
}

// checkPage validates that the leaf page hasn't been replaced by a COW commit.
// Does NOT invalidate the cursor when index is past Count (that's handled by Next).
func (c *Cursor) checkPage() bool {
	if len(c.stack) == 0 {
		return false
	}
	top := &c.stack[len(c.stack)-1]
	current := c.tree.pageReg.get(top.pageID)
	if current != top.page {
		// Page was replaced — re-descend
		if c.lastKey != nil {
			c.descend(c.lastKey)
		}
		return c.valid
	}
	return true
}

// SeekLast positions the cursor at the largest key.
func (t *BTree) SeekLast() *Cursor {
	c := &Cursor{tree: t, stack: make([]cursorLevel, 0, 8)}
	guard, err := t.enterRead()
	if err != nil {
		return c
	}
	c.descendLast()
	if err := guard.leave(); err != nil {
		c.valid = false
	}
	return c
}

func (c *Cursor) descendLast() {
	c.stack = c.stack[:0]
	c.valid = false

	rootID := c.tree.rootID.Load()
	page := c.tree.pageReg.get(rootID)
	if page == nil || page.Header.Count == 0 {
		return
	}

	pageID := rootID
	for page.IsBranch() {
		// Follow the rightmost child at each level
		idx := int(page.Header.Count) // index of last child
		var childID uint32
		if idx == 0 {
			childID = page.Header.FirstChild
		} else {
			childID = page.NodeAt(idx - 1).Child
		}
		c.stack = append(c.stack, cursorLevel{pageID: pageID, page: page, index: idx})
		page = c.tree.pageReg.get(childID)
		pageID = childID
		if page == nil {
			return
		}
	}

	// At leaf — position at last key
	lastIdx := int(page.Header.Count) - 1
	if lastIdx < 0 {
		return
	}
	c.stack = append(c.stack, cursorLevel{pageID: pageID, page: page, index: lastIdx})
	c.valid = true
	c.lastKey = cloneBytes(page.NodeAt(lastIdx).Key())
}

// Prev moves to the previous key in order.
func (c *Cursor) Prev() (ok bool) {
	if !c.valid {
		return false
	}
	guard, err := c.tree.enterRead()
	if err != nil {
		c.valid = false
		return false
	}
	defer func() {
		if err := guard.leave(); err != nil {
			c.valid = false
			ok = false
		}
	}()

	top := &c.stack[len(c.stack)-1]

	if !c.checkPage() {
		return c.valid
	}

	top.index--
	if top.index >= 0 {
		c.lastKey = cloneBytes(top.page.NodeAt(top.index).Key())
		return true
	}

	// Past start of page — try left sibling
	if top.page.Header.LeftSibling != 0 {
		sib := c.tree.pageReg.get(top.page.Header.LeftSibling)
		if sib != nil && sib.Header.Count > 0 {
			top.page = sib
			top.pageID = top.page.Header.LeftSibling
			top.index = int(sib.Header.Count) - 1
			c.lastKey = cloneBytes(sib.NodeAt(top.index).Key())
			return true
		}
	}

	// No left sibling — this is the leftmost leaf. Re-descent to predecessor.
	if c.lastKey != nil {
		c.descendPrev(c.lastKey)
		return c.valid
	}

	c.valid = false
	return false
}

// descendPrev descends to the leaf containing the largest key < target.
func (c *Cursor) descendPrev(target []byte) {
	c.stack = c.stack[:0]
	c.valid = false

	if target == nil {
		return
	}

	rootID := c.tree.rootID.Load()
	page := c.tree.pageReg.get(rootID)
	if page == nil || page.Header.Count == 0 {
		return
	}

	// Check if target is <= the smallest key in the tree
	if page.IsLeaf() {
		idx, found := page.findKey(target)
		if idx == 0 && found {
			return // target is the first key, no predecessor
		}
	} else {
		firstLeaf := c.tree.pageReg.get(page.Header.FirstChild)
		if firstLeaf != nil && firstLeaf.Header.Count > 0 {
			firstKey := firstLeaf.NodeAt(0).Key()
			if bytes.Compare(target, firstKey) <= 0 {
				return // target is at or before the first key
			}
		}
	}

	pageID := rootID
	for page.IsBranch() {
		idx, childID := page.findChild(target)
		c.stack = append(c.stack, cursorLevel{pageID: pageID, page: page, index: idx})
		page = c.tree.pageReg.get(childID)
		pageID = childID
		if page == nil {
			return
		}
	}

	// At leaf — find insertion index for target, go one before
	idx, _ := page.findKey(target)
	if idx > 0 {
		idx--
	}

	// B-link: follow left siblings until we find a non-empty page
	for idx < 0 && page.Header.LeftSibling != 0 {
		previousID := page.Header.LeftSibling
		page = c.tree.pageReg.get(previousID)
		pageID = previousID
		if page == nil {
			return
		}
		idx = int(page.Header.Count) - 1
	}

	c.stack = append(c.stack, cursorLevel{pageID: pageID, page: page, index: idx})
	c.valid = idx >= 0 && idx < int(page.Header.Count)
	if c.valid {
		c.lastKey = cloneBytes(page.NodeAt(idx).Key())
	}
}

// Close releases the cursor.
func (c *Cursor) Close() {
	c.stack = nil
	c.valid = false
	c.lastKey = nil
}
