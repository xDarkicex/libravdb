package graph

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/xDarkicex/memory"
)

// CompactSegment reads a segment, validates its CRC and integrity,
// and rewrites it sequentially to outPath. This serves as a migration
// and defragmentation pass.
func CompactSegment(inPath, outPath string) error {
	fIn, err := os.Open(inPath)
	if err != nil {
		return err
	}
	defer fIn.Close()

	info, err := fIn.Stat()
	if err != nil {
		return err
	}
	if info.Size() < SegmentHeaderSize {
		return fmt.Errorf("input segment too small")
	}

	data, err := memory.MmapFileReadOnly(int(fIn.Fd()), 0, int(info.Size()))
	if err != nil {
		return err
	}
	defer memory.Munmap(data)

	header, err := DeserializeSegmentHeader(data)
	if err != nil {
		return err
	}

	// Overflow-safe bounds check
	manifestEnd := int64(header.ManifestOffset) + int64(header.ManifestLength)
	if manifestEnd < 0 || manifestEnd > info.Size() {
		return fmt.Errorf("manifest bounds out of range")
	}

	// Check Magic Footer
	hasFooter := false
	if info.Size() >= manifestEnd+4 {
		footer := data[len(data)-4:]
		if footer[0] == 'S' && footer[1] == 'G' && footer[2] == 'M' && footer[3] == 'T' {
			hasFooter = true
		}
	}

	if !hasFooter {
		return fmt.Errorf("segment %s is missing SGMT footer", inPath)
	}

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

	var manifest *DBManifest
	if header.ManifestLength > 0 {
		if manifestEnd > int64(len(data)) {
			return fmt.Errorf("manifest bounds out of range")
		}
		manifest, err = DeserializeManifest(data[header.ManifestOffset:manifestEnd])
		if err != nil {
			return err
		}
	} else {
		manifest = NewDBManifest()
	}

	// Opportunistic migration: bump reader version to 2 if it's 1.
	if manifest.MinReaderVersion < 2 {
		manifest.MinReaderVersion = 2
	}

	if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
		return err
	}

	fOut, err := os.Create(outPath + ".tmp")
	if err != nil {
		return err
	}
	defer fOut.Close()

	manifestBytes := manifest.Serialize()
	outHeader := &SegmentHeader{
		Version:        SegmentVersion,
		NodeCount:      header.NodeCount,
		EdgeCount:      header.EdgeCount,
		ManifestOffset: SegmentHeaderSize,
		ManifestLength: uint32(len(manifestBytes)),
	}

	if _, err = fOut.Seek(SegmentHeaderSize, 0); err != nil {
		return err
	}

	crc := crc32.NewIEEE()
	if _, err = fOut.Write(manifestBytes); err != nil {
		return err
	}
	crc.Write(manifestBytes)

	offset := int(manifestEnd)
	if header.ManifestLength == 0 {
		offset = SegmentHeaderSize
	}

	for i := uint64(0); i < header.NodeCount; i++ {
		if offset+16 > len(data) {
			return fmt.Errorf("unexpected EOF reading node record")
		}

		nodeBytes := data[offset : offset+16]
		edgeCount := binary.LittleEndian.Uint16(nodeBytes[8:10])
		offset += 16

		edgesBytesSize := int(edgeCount) * int(unsafe.Sizeof(Edge{}))
		if offset+edgesBytesSize > len(data) {
			return fmt.Errorf("unexpected EOF reading edges")
		}

		if _, err = fOut.Write(nodeBytes); err != nil {
			return err
		}
		crc.Write(nodeBytes)

		if edgeCount > 0 {
			edgesBytes := data[offset : offset+edgesBytesSize]
			if _, err = fOut.Write(edgesBytes); err != nil {
				return err
			}
			crc.Write(edgesBytes)
			offset += edgesBytesSize
		}
	}
	// Write magic footer "SGMT"
	footer := [4]byte{'S', 'G', 'M', 'T'}
	if _, err = fOut.Write(footer[:]); err != nil {
		return err
	}
	crc.Write(footer[:])

	outHeader.CRC32 = crc.Sum32()

	if _, err = fOut.Seek(0, 0); err != nil {
		return err
	}
	headerBytes := outHeader.Serialize()
	if _, err = fOut.Write(headerBytes); err != nil {
		return err
	}

	if err := fOut.Sync(); err != nil {
		return err
	}
	if err := fOut.Close(); err != nil {
		return err
	}

	if err := os.Rename(outPath+".tmp", outPath); err != nil {
		return err
	}

	return syncDir(filepath.Dir(outPath))
}
