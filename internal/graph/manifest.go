package graph

import (
	"encoding/binary"
	"fmt"
)

// DBManifest holds forward-compatibility metadata for the graph segment.
type DBManifest struct {
	MinReaderVersion uint32
	KindManifest     map[uint8]string
}

// NewDBManifest creates an empty manifest.
func NewDBManifest() *DBManifest {
	return &DBManifest{
		MinReaderVersion: 1,
		KindManifest:     make(map[uint8]string),
	}
}

// RegisterKind adds a semantic kind code. Returns an error on conflict.
func (m *DBManifest) RegisterKind(code uint8, name string) error {
	if existing, ok := m.KindManifest[code]; ok && existing != name {
		return fmt.Errorf("kind code %d already registered as %q", code, existing)
	}
	if len(name) > 255 {
		return fmt.Errorf("kind name %q exceeds 255 bytes", name)
	}
	m.KindManifest[code] = name
	return nil
}

// Serialize converts the manifest to its binary representation.
func (m *DBManifest) Serialize() []byte {
	// MinReaderVersion (4) + KindCount (2)
	size := 6
	for _, name := range m.KindManifest {
		size += 1 + 1 + len(name) // Code (1) + NameLen (1) + Name
	}

	buf := make([]byte, size)
	binary.LittleEndian.PutUint32(buf[0:4], m.MinReaderVersion)
	binary.LittleEndian.PutUint16(buf[4:6], uint16(len(m.KindManifest)))

	offset := 6
	for code, name := range m.KindManifest {
		buf[offset] = code
		buf[offset+1] = uint8(len(name))
		copy(buf[offset+2:], name)
		offset += 2 + len(name)
	}

	return buf
}

// DeserializeManifest reads the binary manifest.
// A zero-length buffer assumes no manifest (defaults).
func DeserializeManifest(data []byte) (*DBManifest, error) {
	if len(data) == 0 {
		return NewDBManifest(), nil
	}
	if len(data) < 6 {
		return nil, fmt.Errorf("manifest data too short")
	}

	m := &DBManifest{
		MinReaderVersion: binary.LittleEndian.Uint32(data[0:4]),
		KindManifest:     make(map[uint8]string),
	}

	kindCount := binary.LittleEndian.Uint16(data[4:6])
	offset := 6

	for i := uint16(0); i < kindCount; i++ {
		if offset+2 > len(data) {
			return nil, fmt.Errorf("unexpected EOF reading manifest kind header")
		}
		code := data[offset]
		nameLen := int(data[offset+1])
		offset += 2

		if offset+nameLen > len(data) {
			return nil, fmt.Errorf("unexpected EOF reading manifest kind name")
		}
		name := string(data[offset : offset+nameLen])
		m.KindManifest[code] = name
		offset += nameLen
	}

	return m, nil
}
