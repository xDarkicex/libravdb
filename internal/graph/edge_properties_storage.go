package graph

import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/xDarkicex/memory"
)

// EdgeView is a materialized edge plus its immutable property envelope. The
// ordinary Edge API remains allocation-light and preserves compatibility;
// traversal paths that need arbitrary properties use EdgeView.
type EdgeView struct {
	Edge       Edge
	Properties []byte
}

func makePropertyRef(root, offset uint32) uint64 {
	return uint64(root)<<32 | uint64(offset)
}

func propertyRefRoot(ref uint64) uint32   { return uint32(ref >> 32) }
func propertyRefOffset(ref uint64) uint32 { return uint32(ref) }

func (g *graphStore) newPropertyPage(pool *memory.ShardedFreeList, shard int) (*EdgePropertyPage, uint32, error) {
	actualPool, slotBytes, err := g.allocatePageSlot(pool, shard)
	if err != nil {
		return nil, 0, err
	}
	page := (*EdgePropertyPage)(unsafe.Pointer(&slotBytes[64]))
	*page = EdgePropertyPage{}
	g.rememberPropertyOwner(page, actualPool)
	id := g.propertyReg.Register(page)
	return page, id, nil
}

func propertyPageSlot(page *EdgePropertyPage) []byte {
	return unsafe.Slice((*byte)(unsafe.Add(unsafe.Pointer(page), -64)), 4096)
}

// appendPropertyBytes appends one length-prefixed property envelope to the
// page-owned logical stream and returns a ref whose low word is the logical
// offset of that record. Property values may span overflow pages.
func (g *graphStore) appendPropertyBytes(head *EdgeTablePage, properties []byte, pool *memory.ShardedFreeList, shard int) (uint64, error) {
	if len(properties) == 0 {
		return 0, nil
	}
	if uint64(len(properties)) > uint64(^uint32(0))-4 {
		return 0, fmt.Errorf("edge properties exceed uint32 storage limit")
	}

	root := head.Header.PropertyRoot
	var last *EdgePropertyPage
	var lastID uint32
	var logical uint64
	if root != 0 {
		lastID = root
		last = g.propertyReg.Get(lastID)
		if last == nil {
			return 0, fmt.Errorf("missing edge property root page %d", root)
		}
		for last != nil {
			logical += uint64(last.Used)
			if last.Next == 0 {
				break
			}
			lastID = last.Next
			last = g.propertyReg.Get(lastID)
			if last == nil {
				return 0, fmt.Errorf("missing edge property page %d", lastID)
			}
		}
	}

	if last == nil {
		var err error
		last, root, err = g.newPropertyPage(pool, shard)
		if err != nil {
			return 0, err
		}
		head.Header.PropertyRoot = root
		lastID = root
	}

	entryOffset := logical
	remaining := make([]byte, 4+len(properties))
	binary.LittleEndian.PutUint32(remaining[:4], uint32(len(properties)))
	copy(remaining[4:], properties)
	for len(remaining) > 0 {
		available := EdgePropertyPageDataSize - int(last.Used)
		if available == 0 {
			next, nextID, err := g.newPropertyPage(pool, shard)
			if err != nil {
				return 0, err
			}
			last.Next = nextID
			last = next
			lastID = nextID
			available = EdgePropertyPageDataSize
		}
		n := len(remaining)
		if n > available {
			n = available
		}
		copy(last.Data[last.Used:], remaining[:n])
		last.Used += uint32(n)
		remaining = remaining[n:]
	}
	_ = lastID
	return makePropertyRef(root, uint32(entryOffset)), nil
}

func (g *graphStore) readPropertyStream(root, offset uint32, length int) ([]byte, error) {
	if root == 0 {
		return nil, nil
	}
	page := g.propertyReg.Get(root)
	if page == nil {
		return nil, fmt.Errorf("missing edge property root page %d", root)
	}
	skip := uint64(offset)
	for page != nil && skip >= uint64(page.Used) {
		skip -= uint64(page.Used)
		if page.Next == 0 {
			page = nil
			break
		}
		page = g.propertyReg.Get(page.Next)
	}
	if page == nil {
		return nil, fmt.Errorf("edge property offset %d is outside property chain", offset)
	}
	out := make([]byte, 0, length)
	for len(out) < length && page != nil {
		available := int(page.Used) - int(skip)
		if available > 0 {
			n := length - len(out)
			if n > available {
				n = available
			}
			out = append(out, page.Data[int(skip):int(skip)+n]...)
		}
		skip = 0
		if len(out) < length {
			if page.Next == 0 {
				return nil, fmt.Errorf("short edge property payload")
			}
			page = g.propertyReg.Get(page.Next)
		}
	}
	return out, nil
}

func (g *graphStore) propertyBytes(ref uint64) ([]byte, error) {
	if ref == 0 {
		return nil, nil
	}
	root, offset := propertyRefRoot(ref), propertyRefOffset(ref)
	header, err := g.readPropertyStream(root, offset, 4)
	if err != nil {
		return nil, err
	}
	length := binary.LittleEndian.Uint32(header)
	return g.readPropertyStream(root, offset+4, int(length))
}

func (g *graphStore) clonePropertyChain(root uint32, pool *memory.ShardedFreeList, shard int) (uint32, error) {
	if root == 0 {
		return 0, nil
	}
	old := g.propertyReg.Get(root)
	if old == nil {
		return 0, fmt.Errorf("missing edge property root page %d", root)
	}
	var newRoot, previous uint32
	for old != nil {
		page, id, err := g.newPropertyPage(pool, shard)
		if err != nil {
			return 0, err
		}
		*page = *old
		page.Next = 0
		if newRoot == 0 {
			newRoot = id
		} else if prev := g.propertyReg.Get(previous); prev != nil {
			prev.Next = id
		}
		previous = id
		if old.Next == 0 {
			break
		}
		old = g.propertyReg.Get(old.Next)
		if old == nil {
			return 0, fmt.Errorf("missing edge property overflow page")
		}
	}
	return newRoot, nil
}

func (g *graphStore) retirePropertyChain(root uint32) error {
	for root != 0 {
		page := g.propertyReg.Get(root)
		if page == nil {
			return fmt.Errorf("missing edge property page %d", root)
		}
		next := page.Next
		g.propertyReg.Unregister(root)
		owner := g.forgetPropertyOwner(page)
		if owner == nil {
			return fmt.Errorf("missing owner for edge property page %d", root)
		}
		if err := owner.Retire(propertyPageSlot(page)); err != nil {
			return fmt.Errorf("retire edge property page %d: %w", root, err)
		}
		root = next
	}
	return nil
}

func (g *graphStore) rewritePropertyRefs(head *EdgeTablePage, oldRoot, newRoot uint32) {
	if oldRoot == 0 || oldRoot == newRoot {
		return
	}
	remaining := head.Header.Count
	page := head
	for page != nil && remaining > 0 {
		count := remaining
		if count > EdgePageCapacity {
			count = EdgePageCapacity
		}
		inline := count
		if inline > EdgePageInlineCapacity {
			inline = EdgePageInlineCapacity
		}
		for i := 0; i < int(inline); i++ {
			if propertyRefRoot(page.Inline[i].PropertyRef) == oldRoot {
				page.Inline[i].PropertyRef = makePropertyRef(newRoot, propertyRefOffset(page.Inline[i].PropertyRef))
			}
		}
		if count > EdgePageInlineCapacity {
			extra := unsafe.Slice((*Edge)(unsafe.Pointer(&page.Padding[0])), EdgePageOverflowCapacity)
			for i := 0; i < int(count-EdgePageInlineCapacity); i++ {
				if propertyRefRoot(extra[i].PropertyRef) == oldRoot {
					extra[i].PropertyRef = makePropertyRef(newRoot, propertyRefOffset(extra[i].PropertyRef))
				}
			}
		}
		remaining -= count
		if page.Header.Overflow == 0 {
			break
		}
		page = g.pageReg.Get(page.Header.Overflow)
	}
}
