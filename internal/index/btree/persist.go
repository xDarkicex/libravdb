package btree

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"sort"
)

const btreeMagic = "BTREEV01"
const btreeVersion uint32 = 1

// SerializeToBytes encodes the entire tree as a binary blob.
// Format: [magic:8][version:4][page_count:4][pages...][crc32:4]
func (t *BTree) SerializeToBytes() ([]byte, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	guard, err := t.enterRead()
	if err != nil {
		return nil, err
	}
	defer guard.leave()

	ids := t.pageReg.snapshotIDs()
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })

	var buf bytes.Buffer
	buf.WriteString(btreeMagic)
	binary.Write(&buf, binary.LittleEndian, btreeVersion)
	binary.Write(&buf, binary.LittleEndian, uint32(len(ids)))
	binary.Write(&buf, binary.LittleEndian, t.rootID.Load())

	for _, id := range ids {
		page := t.pageReg.get(id)
		if page == nil {
			continue
		}
		serializePage(&buf, page, id)
	}

	// CRC32 of body (everything after magic)
	body := buf.Bytes()[len(btreeMagic):]
	cksum := crc32.ChecksumIEEE(body)
	binary.Write(&buf, binary.LittleEndian, cksum)

	return buf.Bytes(), nil
}

func serializePage(w *bytes.Buffer, page *BTreePage, id uint32) {
	h := &page.Header
	binary.Write(w, binary.LittleEndian, id)
	binary.Write(w, binary.LittleEndian, h.Flags)
	binary.Write(w, binary.LittleEndian, h.Count)
	binary.Write(w, binary.LittleEndian, h.RightSibling)
	binary.Write(w, binary.LittleEndian, h.LeftSibling)
	binary.Write(w, binary.LittleEndian, h.FirstChild)
	binary.Write(w, binary.LittleEndian, h.Lower)
	binary.Write(w, binary.LittleEndian, h.Upper)
	binary.Write(w, binary.LittleEndian, h.Generation)

	for i := uint16(0); i < h.Count; i++ {
		n := page.NodeAt(int(i))
		key := n.Key()
		val := n.Value()
		binary.Write(w, binary.LittleEndian, n.KeyLen)
		binary.Write(w, binary.LittleEndian, n.ValLen)
		binary.Write(w, binary.LittleEndian, n.Child)
		w.Write(key)
		w.Write(val)
	}
}

// DeserializeFromBytes reconstructs the tree from a binary blob.
// Uses a two-pass approach: allocate all pages first, then remap child pointers
// since slot IDs change during deserialization.
func (t *BTree) DeserializeFromBytes(ctx context.Context, data []byte) error {
	if len(data) < len(btreeMagic)+4+4+4+4 {
		return fmt.Errorf("btree: serialized data too short (%d bytes)", len(data))
	}

	if string(data[:len(btreeMagic)]) != btreeMagic {
		return fmt.Errorf("btree: invalid magic %q", string(data[:len(btreeMagic)]))
	}
	off := len(btreeMagic)

	version := binary.LittleEndian.Uint32(data[off:])
	off += 4
	if version != btreeVersion {
		return fmt.Errorf("btree: unsupported version %d", version)
	}

	body := data[len(btreeMagic):]
	if len(body) < 4 {
		return fmt.Errorf("btree: missing checksum")
	}
	expected := binary.LittleEndian.Uint32(body[len(body)-4:])
	actual := crc32.ChecksumIEEE(body[:len(body)-4])
	if expected != actual {
		return fmt.Errorf("btree: checksum mismatch (got 0x%08x, want 0x%08x)", actual, expected)
	}

	pageCount := binary.LittleEndian.Uint32(data[off:])
	off += 4
	oldRootID := binary.LittleEndian.Uint32(data[off:])
	off += 4

	t.mu.Lock()
	defer t.mu.Unlock()
	t.poolsMu.Lock()
	for _, id := range t.pageReg.snapshotIDs() {
		freePage(t, t.pageReg, id)
	}
	t.poolsMu.Unlock()
	t.nodeCount.Store(0)

	type pageRec struct {
		page         *BTreePage
		newID        uint32
		flags        uint16
		rightSibling uint32
		leftSibling  uint32
		firstChild   uint32
		generation   uint32
		count        uint16
		origID       uint32
	}
	records := make([]pageRec, pageCount)
	oldToNew := make(map[uint32]uint32)

	dataEnd := len(data) - 4 // position of CRC32

	// Pass 1: allocate pages, deserialize nodes
	for i := uint32(0); i < pageCount; i++ {
		if off+28 > dataEnd {
			return fmt.Errorf("btree: truncated page header at page %d", i)
		}

		origID := binary.LittleEndian.Uint32(data[off:])
		off += 4
		flags := binary.LittleEndian.Uint16(data[off:])
		off += 2
		count := binary.LittleEndian.Uint16(data[off:])
		off += 2
		rightSibling := binary.LittleEndian.Uint32(data[off:])
		off += 4
		leftSibling := binary.LittleEndian.Uint32(data[off:])
		off += 4
		firstChild := binary.LittleEndian.Uint32(data[off:])
		off += 4
		_ = binary.LittleEndian.Uint16(data[off:]) // lower
		off += 2
		_ = binary.LittleEndian.Uint16(data[off:]) // upper
		off += 2
		generation := binary.LittleEndian.Uint32(data[off:])
		off += 4

		page, newID, err := allocPage(t, t.pageReg, flags, 0)
		if err != nil {
			return fmt.Errorf("btree: alloc page %d: %w", origID, err)
		}
		oldToNew[origID] = newID

		page.Header.Generation = generation

		for j := uint16(0); j < count; j++ {
			if off+8 > dataEnd {
				return fmt.Errorf("btree: truncated node %d in page %d", j, origID)
			}
			keyLen := binary.LittleEndian.Uint16(data[off:])
			off += 2
			valLen := binary.LittleEndian.Uint16(data[off:])
			off += 2
			child := binary.LittleEndian.Uint32(data[off:])
			off += 4

			if off+int(keyLen)+int(valLen) > dataEnd {
				return fmt.Errorf("btree: truncated key/value at node %d in page %d", j, origID)
			}
			key := data[off : off+int(keyLen)]
			off += int(keyLen)
			val := data[off : off+int(valLen)]
			off += int(valLen)

			insertCell(page, int(j), key, val, child)
		}

		records[i] = pageRec{
			page: page, newID: newID, flags: flags,
			rightSibling: rightSibling, leftSibling: leftSibling, firstChild: firstChild,
			generation: generation, count: count, origID: origID,
		}
		if page.IsLeaf() {
			t.nodeCount.Add(int64(page.Header.Count))
		}
	}

	// Pass 2: remap child pointers
	for _, rec := range records {
		rec.page.Header.RightSibling = remapID(oldToNew, rec.rightSibling)
		rec.page.Header.LeftSibling = remapID(oldToNew, rec.leftSibling)
		if rec.page.IsBranch() {
			rec.page.Header.FirstChild = remapID(oldToNew, rec.firstChild)
			for j := uint16(0); j < rec.page.Header.Count; j++ {
				n := rec.page.NodeAt(int(j))
				n.Child = remapID(oldToNew, n.Child)
			}
		}
		rec.page.Header.Generation = rec.generation
	}

	t.rootID.Store(remapID(oldToNew, oldRootID))
	return nil
}

func remapID(m map[uint32]uint32, old uint32) uint32 {
	if old == 0 {
		return 0
	}
	if newID, ok := m[old]; ok {
		return newID
	}
	return old // unchanged (shouldn't happen)
}

// SaveToDisk writes the serialized tree to a file atomically.
func (t *BTree) SaveToDisk(ctx context.Context, path string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	data, err := t.SerializeToBytes()
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// LoadFromDisk reads the serialized tree from a file.
func (t *BTree) LoadFromDisk(ctx context.Context, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return t.DeserializeFromBytes(ctx, data)
}
