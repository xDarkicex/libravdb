package graph

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"sort"
	"sync/atomic"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/storage/wal"
	"github.com/xDarkicex/memory"
)

const SegmentVersion uint32 = 1
const SegmentHeaderSize = 32

// SegmentHeader represents the 32-byte header of a segment file.
type SegmentHeader struct {
	Version        uint32
	NodeCount      uint64
	EdgeCount      uint64
	CRC32          uint32
	ManifestOffset uint32
	ManifestLength uint32
}

// Serialize encodes the header into 32 bytes.
func (h *SegmentHeader) Serialize() []byte {
	buf := make([]byte, SegmentHeaderSize)
	binary.LittleEndian.PutUint32(buf[0:4], h.Version)
	binary.LittleEndian.PutUint64(buf[4:12], h.NodeCount)
	binary.LittleEndian.PutUint64(buf[12:20], h.EdgeCount)
	binary.LittleEndian.PutUint32(buf[20:24], h.CRC32)
	binary.LittleEndian.PutUint32(buf[24:28], h.ManifestOffset)
	binary.LittleEndian.PutUint32(buf[28:32], h.ManifestLength)
	return buf
}

// DeserializeSegmentHeader decodes a 32-byte header.
func DeserializeSegmentHeader(data []byte) (*SegmentHeader, error) {
	if len(data) < SegmentHeaderSize {
		return nil, fmt.Errorf("data too short for segment header")
	}
	h := &SegmentHeader{
		Version:        binary.LittleEndian.Uint32(data[0:4]),
		NodeCount:      binary.LittleEndian.Uint64(data[4:12]),
		EdgeCount:      binary.LittleEndian.Uint64(data[12:20]),
		CRC32:          binary.LittleEndian.Uint32(data[20:24]),
		ManifestOffset: binary.LittleEndian.Uint32(data[24:28]),
		ManifestLength: binary.LittleEndian.Uint32(data[28:32]),
	}
	return h, nil
}

// FlushToSegment writes the current forward EdgeTable to an LSM-style segment file on disk.
func (g *graphStore) FlushToSegment(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	f, err := os.Create(path + ".tmp")
	if err != nil {
		return err
	}
	defer f.Close()

	var nodeIDs []uint64
	for i := uint64(0); i < g.index.capacity; i++ {
		slot := atomic.LoadUint32(&g.index.table[i].PageSlot)
		if slot != 0 && slot != Tombstone {
			nodeIDs = append(nodeIDs, g.index.table[i].NodeID)
		}
	}

	sort.Slice(nodeIDs, func(i, j int) bool { return nodeIDs[i] < nodeIDs[j] })

	manifestBytes := g.manifest.Serialize()
	header := &SegmentHeader{
		Version:        SegmentVersion,
		NodeCount:      uint64(len(nodeIDs)),
		EdgeCount:      0,
		ManifestOffset: SegmentHeaderSize,
		ManifestLength: uint32(len(manifestBytes)),
	}

	// Seek past header
	if _, err = f.Seek(SegmentHeaderSize, 0); err != nil {
		return err
	}

	hashWriter := crc32.NewIEEE()
	if _, err = f.Write(manifestBytes); err != nil {
		return err
	}
	hashWriter.Write(manifestBytes)
	var totalEdges uint64

	for _, nodeID := range nodeIDs {
		edges, err := g.Neighbors(nodeID)
		if err != nil {
			return err
		}

		if len(edges) > 65535 {
			return fmt.Errorf("node %d has %d edges, exceeding uint16 limit for segment serialization", nodeID, len(edges))
		}

		totalEdges += uint64(len(edges))

		buf := make([]byte, 16)
		binary.LittleEndian.PutUint64(buf[0:8], nodeID)
		binary.LittleEndian.PutUint16(buf[8:10], uint16(len(edges)))

		if _, err = f.Write(buf); err != nil {
			return err
		}
		hashWriter.Write(buf)

		if len(edges) > 0 {
			edgesBytes := unsafe.Slice((*byte)(unsafe.Pointer(&edges[0])), len(edges)*int(unsafe.Sizeof(Edge{})))
			if _, err = f.Write(edgesBytes); err != nil {
				return err
			}
			hashWriter.Write(edgesBytes)
		}
	}

	// Write magic footer "SGMT"
	footer := [4]byte{'S', 'G', 'M', 'T'}
	if _, err = f.Write(footer[:]); err != nil {
		return err
	}
	hashWriter.Write(footer[:])

	header.EdgeCount = totalEdges
	header.CRC32 = hashWriter.Sum32()

	if _, err = f.Seek(0, 0); err != nil {
		return err
	}
	headerBytes := header.Serialize()
	if _, err = f.Write(headerBytes); err != nil {
		return err
	}

	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}

	if err := os.Rename(path+".tmp", path); err != nil {
		return err
	}

	if err := syncDir(filepath.Dir(path)); err != nil {
		return err
	}

	atomic.StoreUint32(&g.lastFlushedGen, g.globalStamp.Load())
	return nil
}

// LoadFromSegment mmaps a segment file and populates the graph, then triggers WAL replay.
func (g *graphStore) LoadFromSegment(path string, w *wal.WAL) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			if w != nil {
				return ReplayWAL(w, g)
			}
			return nil
		}
		return err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return err
	}

	if info.Size() < SegmentHeaderSize {
		return fmt.Errorf("segment file too small")
	}

	data, err := memory.MmapFileReadOnly(int(f.Fd()), 0, int(info.Size()))
	if err != nil {
		return fmt.Errorf("mmap failed: %w", err)
	}
	defer memory.Munmap(data)

	header, err := DeserializeSegmentHeader(data)
	if err != nil {
		return err
	}

	if header.Version != SegmentVersion {
		return fmt.Errorf("unsupported segment version: %d", header.Version)
	}

	// Check Magic Footer
	hasFooter := false

	// Overflow-safe bounds check
	manifestEnd := int64(header.ManifestOffset) + int64(header.ManifestLength)
	if manifestEnd < 0 || manifestEnd > info.Size() {
		return fmt.Errorf("manifest bounds out of range")
	}

	if info.Size() >= manifestEnd+4 {
		footer := data[len(data)-4:]
		if footer[0] == 'S' && footer[1] == 'G' && footer[2] == 'M' && footer[3] == 'T' {
			hasFooter = true
		}
	}

	if !hasFooter {
		return fmt.Errorf("segment %s is missing SGMT footer", path)
	}

	// Validate CRC
	hashWriter := crc32.NewIEEE()
	if header.ManifestLength > 0 {
		hashWriter.Write(data[header.ManifestOffset:manifestEnd])
	}

	dataEnd := len(data)
	if hasFooter {
		dataEnd -= 4
	}

	if manifestEnd > int64(dataEnd) {
		return fmt.Errorf("payload bounds out of range")
	}
	hashWriter.Write(data[manifestEnd:dataEnd])
	if hasFooter {
		hashWriter.Write([]byte{'S', 'G', 'M', 'T'})
	}

	if hashWriter.Sum32() != header.CRC32 {
		return fmt.Errorf("segment CRC mismatch: expected %d, got %d", hashWriter.Sum32(), header.CRC32)
	}

	if header.ManifestLength > 0 {
		end := int(header.ManifestOffset + header.ManifestLength)
		// Bounds already checked above
		manifest, err := DeserializeManifest(data[header.ManifestOffset:end])
		if err != nil {
			return fmt.Errorf("failed to load manifest: %w", err)
		}
		// Implementation reader version is 2.
		if manifest.MinReaderVersion > 2 {
			return fmt.Errorf("database requires reader version >= %d, but implementation is 2", manifest.MinReaderVersion)
		}
		g.manifest = manifest
	}

	offset := int(header.ManifestOffset + header.ManifestLength)
	if header.ManifestLength == 0 {
		offset = SegmentHeaderSize
	}

	var maxStamp uint32

	for i := uint64(0); i < header.NodeCount; i++ {
		if offset+16 > len(data) {
			return fmt.Errorf("unexpected EOF reading node record")
		}

		nodeID := binary.LittleEndian.Uint64(data[offset : offset+8])
		edgeCount := binary.LittleEndian.Uint16(data[offset+8 : offset+10])
		offset += 16

		edgesBytesSize := int(edgeCount) * int(unsafe.Sizeof(Edge{}))
		if offset+edgesBytesSize > len(data) {
			return fmt.Errorf("unexpected EOF reading edges")
		}

		if edgeCount > 0 {
			edges := unsafe.Slice((*Edge)(unsafe.Pointer(&data[offset])), edgeCount)
			offset += edgesBytesSize

			for j := 0; j < int(edgeCount); j++ {
				stamp := edges[j].GetStamp()
				if stamp > maxStamp {
					maxStamp = stamp
				}
				err := g.appendEdgeToTable(nodeID, edges[j], g.index, g.pagePool)
				if err != nil {
					return err
				}
			}
		}
	}

	// Restore gen
	g.globalStamp.Store(maxStamp)
	atomic.StoreUint32(&g.lastFlushedGen, maxStamp)

	// Rebuild reverse index
	g.rebuildReverseIndex()

	if w != nil {
		if err := ReplayWAL(w, g); err != nil {
			return fmt.Errorf("WAL replay failed: %w", err)
		}
	}

	return nil
}

// syncDir fsyncs a directory to ensure rename durability
func syncDir(dirPath string) error {
	d, err := os.Open(dirPath)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
