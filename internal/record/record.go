// Package record defines the off-heap record ABI shared by storage and indexes.
//
// A RecordRef never owns Go heap payload. Its bytes, vector, metadata, and
// header are allocated from the generation arena that produced it. A caller
// holding a RecordRef must also hold that generation's read lease.
package record

import (
	"encoding/binary"
	"errors"
	"math"
	"time"
	"unsafe"

	"github.com/xDarkicex/memory"
)

var (
	ErrInvalidID       = errors.New("record ID cannot be empty")
	ErrInvalidVector   = errors.New("record vector cannot be empty")
	ErrMetadataFull    = errors.New("metadata builder capacity exhausted")
	ErrUnsupportedType = errors.New("unsupported metadata type")
)

// RecordRef is a stable pointer into an immutable generation arena.
// It is deliberately a pointer-sized value so it can be stored in memory.IDMap.
type RecordRef struct{ ptr unsafe.Pointer }

// VectorView is a borrowed, immutable float32 span. It contains no Go-owned
// payload; its backing storage belongs to a request or generation arena.
type VectorView struct {
	ptr unsafe.Pointer
	len uint32
}

// BytesView is a borrowed immutable byte span.
type BytesView struct {
	ptr unsafe.Pointer
	len uint32
}

// BorrowBytes and BorrowVector create request-scoped views. They never retain
// caller memory: RecordBuilder.Seal always copies their payload into its
// generation arena before a record can be published.
func BorrowBytes(value []byte) BytesView {
	if len(value) == 0 {
		return BytesView{}
	}
	return BytesView{ptr: unsafe.Pointer(unsafe.SliceData(value)), len: uint32(len(value))}
}

func BorrowVector(value []float32) VectorView {
	if len(value) == 0 {
		return VectorView{}
	}
	return VectorView{ptr: unsafe.Pointer(unsafe.SliceData(value)), len: uint32(len(value))}
}

func (v VectorView) Len() int { return int(v.len) }

func (v VectorView) At(i int) float32 {
	if i < 0 || i >= int(v.len) {
		panic("record: vector index out of range")
	}
	return unsafe.Slice((*float32)(v.ptr), v.len)[i]
}

func (v VectorView) Float32s() []float32 {
	if v.ptr == nil || v.len == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(v.ptr), v.len)
}

func (v BytesView) Len() int { return int(v.len) }

func (v BytesView) Bytes() []byte {
	if v.ptr == nil || v.len == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(v.ptr), v.len)
}

// MetadataType is the canonical on-arena metadata wire type.
type MetadataType uint8

const (
	MetadataNull MetadataType = iota
	MetadataBool
	MetadataInt64
	MetadataUint64
	MetadataFloat64
	MetadataString
	MetadataTimeUnixNano
	MetadataBytes
)

const metadataFieldHeaderBytes = 1 + 2 + 4

// MetadataBuilder writes canonical typed fields into a caller supplied arena.
// Fields are appended in call order; callers that need canonical ordering must
// submit keys in lexical byte order. This makes writes allocation-free and
// keeps ordering policy explicit instead of hiding a Go map behind the API.
type MetadataBuilder struct {
	buf []byte
	off uint32
}

func NewMetadataBuilder(arena *memory.Arena, capacity uint32) (*MetadataBuilder, error) {
	if capacity == 0 {
		return &MetadataBuilder{}, nil
	}
	ptr, err := arena.Alloc(uint64(capacity))
	if err != nil {
		return nil, err
	}
	return &MetadataBuilder{buf: unsafe.Slice((*byte)(ptr), capacity)}, nil
}

func (b *MetadataBuilder) appendField(key []byte, typ MetadataType, value []byte) error {
	if len(key) == 0 || len(key) > math.MaxUint16 || len(value) > math.MaxUint32 {
		return ErrUnsupportedType
	}
	n := metadataFieldHeaderBytes + len(key) + len(value)
	if uint64(b.off)+uint64(n) > uint64(len(b.buf)) {
		return ErrMetadataFull
	}
	p := b.buf[b.off : b.off+uint32(n)]
	p[0] = byte(typ)
	binary.LittleEndian.PutUint16(p[1:], uint16(len(key)))
	binary.LittleEndian.PutUint32(p[3:], uint32(len(value)))
	copy(p[metadataFieldHeaderBytes:], key)
	copy(p[metadataFieldHeaderBytes+len(key):], value)
	b.off += uint32(n)
	return nil
}

func (b *MetadataBuilder) Null(key []byte) error {
	return b.appendField(key, MetadataNull, nil)
}

func (b *MetadataBuilder) Bool(key []byte, value bool) error {
	var raw [1]byte
	if value {
		raw[0] = 1
	}
	return b.appendField(key, MetadataBool, raw[:])
}

func (b *MetadataBuilder) Int64(key []byte, value int64) error {
	var raw [8]byte
	binary.LittleEndian.PutUint64(raw[:], uint64(value))
	return b.appendField(key, MetadataInt64, raw[:])
}

func (b *MetadataBuilder) Uint64(key []byte, value uint64) error {
	var raw [8]byte
	binary.LittleEndian.PutUint64(raw[:], value)
	return b.appendField(key, MetadataUint64, raw[:])
}

func (b *MetadataBuilder) Float64(key []byte, value float64) error {
	var raw [8]byte
	binary.LittleEndian.PutUint64(raw[:], math.Float64bits(value))
	return b.appendField(key, MetadataFloat64, raw[:])
}

func (b *MetadataBuilder) String(key, value []byte) error {
	return b.appendField(key, MetadataString, value)
}

func (b *MetadataBuilder) Time(key []byte, value time.Time) error {
	var raw [8]byte
	binary.LittleEndian.PutUint64(raw[:], uint64(value.UnixNano()))
	return b.appendField(key, MetadataTimeUnixNano, raw[:])
}

func (b *MetadataBuilder) Bytes(key, value []byte) error {
	return b.appendField(key, MetadataBytes, value)
}

func (b *MetadataBuilder) View() MetadataView {
	if b.off == 0 || len(b.buf) == 0 {
		return MetadataView{}
	}
	return MetadataView{data: BytesView{ptr: unsafe.Pointer(unsafe.SliceData(b.buf)), len: b.off}}
}

// MetadataView exposes allocation-free typed field lookup over canonical bytes.
type MetadataView struct{ data BytesView }

func (m MetadataView) Bytes() BytesView { return m.data }

func (m MetadataView) Find(key []byte) (MetadataType, BytesView, bool) {
	data := m.data.Bytes()
	for off := 0; off < len(data); {
		if len(data)-off < metadataFieldHeaderBytes {
			return MetadataNull, BytesView{}, false
		}
		typ := MetadataType(data[off])
		keyLen := int(binary.LittleEndian.Uint16(data[off+1:]))
		valueLen := int(binary.LittleEndian.Uint32(data[off+3:]))
		start := off + metadataFieldHeaderBytes
		end := start + keyLen + valueLen
		if keyLen < 0 || valueLen < 0 || end < start || end > len(data) {
			return MetadataNull, BytesView{}, false
		}
		if len(key) == keyLen && string(data[start:start+keyLen]) == string(key) {
			value := data[start+keyLen : end]
			return typ, BytesView{ptr: unsafe.Pointer(unsafe.SliceData(value)), len: uint32(len(value))}, true
		}
		off = end
	}
	return MetadataNull, BytesView{}, false
}

// RecordBuilder owns only borrowed request-arena views until Seal copies the
// record into a persistent generation arena.
type RecordBuilder struct {
	ID       BytesView
	Vector   VectorView
	Metadata MetadataView
	Version  uint64
	Ordinal  uint32
}

type recordHeader struct {
	idPtr       unsafe.Pointer
	vectorPtr   unsafe.Pointer
	metadataPtr unsafe.Pointer
	version     uint64
	ordinal     uint32
	idLen       uint32
	vectorLen   uint32
	metadataLen uint32
	flags       uint32
	_           uint32
}

const recordHeaderBytes = uint64(unsafe.Sizeof(recordHeader{}))

const recordFlagTombstone uint32 = 1

func (b RecordBuilder) Seal(arena *memory.Arena) (RecordRef, error) {
	if b.ID.ptr == nil || b.ID.len == 0 {
		return RecordRef{}, ErrInvalidID
	}
	if b.Vector.ptr == nil || b.Vector.len == 0 {
		return RecordRef{}, ErrInvalidVector
	}
	bytes := recordHeaderBytes + uint64(b.ID.len) + uint64(b.Vector.len)*4 + uint64(b.Metadata.data.len)
	base, err := arena.Alloc(bytes)
	if err != nil {
		return RecordRef{}, err
	}
	header := (*recordHeader)(base)
	cursor := unsafe.Add(base, recordHeaderBytes)
	header.idPtr = cursor
	header.idLen = b.ID.len
	copy(unsafe.Slice((*byte)(cursor), b.ID.len), b.ID.Bytes())
	cursor = unsafe.Add(cursor, b.ID.len)
	header.vectorPtr = cursor
	header.vectorLen = b.Vector.len
	copy(unsafe.Slice((*float32)(cursor), b.Vector.len), b.Vector.Float32s())
	cursor = unsafe.Add(cursor, uintptr(b.Vector.len)*4)
	header.metadataPtr = cursor
	header.metadataLen = b.Metadata.data.len
	copy(unsafe.Slice((*byte)(cursor), b.Metadata.data.len), b.Metadata.data.Bytes())
	header.version = b.Version
	header.ordinal = b.Ordinal
	return RecordRef{ptr: base}, nil
}

// SealTombstone creates an immutable deletion marker. Tombstones retain the
// stable ordinal so a newer generation can suppress an older segment record
// without scanning an ID map in the search loop.
func SealTombstone(arena *memory.Arena, id BytesView, ordinal uint32, version uint64) (RecordRef, error) {
	if id.ptr == nil || id.len == 0 {
		return RecordRef{}, ErrInvalidID
	}
	base, err := arena.Alloc(recordHeaderBytes + uint64(id.len))
	if err != nil {
		return RecordRef{}, err
	}
	header := (*recordHeader)(base)
	header.idPtr = unsafe.Add(base, recordHeaderBytes)
	header.idLen = id.len
	header.ordinal = ordinal
	header.version = version
	header.flags = recordFlagTombstone
	copy(unsafe.Slice((*byte)(header.idPtr), id.len), id.Bytes())
	return RecordRef{ptr: base}, nil
}

func (r RecordRef) Valid() bool { return r.ptr != nil }

func (r RecordRef) header() *recordHeader { return (*recordHeader)(r.ptr) }

func (r RecordRef) ID() BytesView {
	if r.ptr == nil {
		return BytesView{}
	}
	h := r.header()
	return BytesView{ptr: h.idPtr, len: h.idLen}
}

func (r RecordRef) Vector() VectorView {
	if r.ptr == nil {
		return VectorView{}
	}
	h := r.header()
	return VectorView{ptr: h.vectorPtr, len: h.vectorLen}
}

func (r RecordRef) Metadata() MetadataView {
	if r.ptr == nil {
		return MetadataView{}
	}
	h := r.header()
	return MetadataView{data: BytesView{ptr: h.metadataPtr, len: h.metadataLen}}
}

func (r RecordRef) Version() uint64 {
	if r.ptr == nil {
		return 0
	}
	return r.header().version
}

func (r RecordRef) Ordinal() uint32 {
	if r.ptr == nil {
		return 0
	}
	return r.header().ordinal
}

func (r RecordRef) Tombstone() bool {
	return r.ptr != nil && r.header().flags&recordFlagTombstone != 0
}

// Footprint reports the immutable record payload bytes owned by its generation
// arena. It excludes allocator/page slack and directory capacity.
func (r RecordRef) Footprint() uint64 {
	if r.ptr == nil {
		return 0
	}
	h := r.header()
	return recordHeaderBytes + uint64(h.idLen) + uint64(h.vectorLen)*4 + uint64(h.metadataLen)
}

// Directory maps copied off-heap ID bytes to off-heap RecordRef headers.
// It is a mutable construction helper only; published generations expose it
// behind an immutable generation root.
type Directory struct{ ids *memory.IDMap }

func NewDirectory(capacity, keyBytes uint64) (*Directory, error) {
	ids, err := memory.NewIDMap(memory.IDMapConfig{Capacity: capacity, KeyBytes: keyBytes, Alignment: 128})
	if err != nil {
		return nil, err
	}
	return &Directory{ids: ids}, nil
}

func (d *Directory) Put(id BytesView, ref RecordRef) error {
	if !ref.Valid() {
		return ErrInvalidID
	}
	return d.ids.PutBytes(id.Bytes(), ref.ptr)
}

func (d *Directory) Get(id BytesView) (RecordRef, bool) {
	ptr, ok := d.ids.GetBytes(id.Bytes())
	return RecordRef{ptr: ptr}, ok
}

func (d *Directory) Delete(id BytesView) bool { return d.ids.DeleteBytes(id.Bytes()) }

func (d *Directory) Free() error {
	if d == nil || d.ids == nil {
		return nil
	}
	err := d.ids.Free()
	d.ids = nil
	return err
}
