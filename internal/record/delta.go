package record

import (
	"errors"
	"unsafe"

	"github.com/xDarkicex/memory"
)

var ErrDeltaFull = errors.New("transaction mutation capacity exhausted")

// MutationKind is the logical operation performed on one stable record ID.
type MutationKind uint8

const (
	MutationInsert MutationKind = iota
	MutationUpdate
	MutationDelete
	MutationUpsert
)

// DeltaConfig bounds every transaction-owned allocation. No operation in a
// live delta is permitted to grow a Go slice or map beyond these limits.
type DeltaConfig struct {
	ArenaBytes   uint64
	MaxMutations uint32
	IDCapacity   uint64
	IDKeyBytes   uint64
}

type mutationCell struct {
	idPtr           unsafe.Pointer
	before          unsafe.Pointer
	after           unsafe.Pointer
	expectedVersion uint64
	idLen           uint32
	sequence        uint32
	kind            MutationKind
	hasExpected     bool
	_               [2]byte
}

// Mutation is a borrowed view over an off-heap mutation cell.
type Mutation struct{ cell *mutationCell }

func (m Mutation) ID() BytesView {
	if m.cell == nil {
		return BytesView{}
	}
	return BytesView{ptr: m.cell.idPtr, len: m.cell.idLen}
}

func (m Mutation) Before() RecordRef {
	if m.cell == nil {
		return RecordRef{}
	}
	return RecordRef{ptr: m.cell.before}
}

func (m Mutation) After() RecordRef {
	if m.cell == nil {
		return RecordRef{}
	}
	return RecordRef{ptr: m.cell.after}
}

func (m Mutation) Kind() MutationKind {
	if m.cell == nil {
		return MutationDelete
	}
	return m.cell.kind
}

func (m Mutation) ExpectedVersion() (uint64, bool) {
	if m.cell == nil {
		return 0, false
	}
	return m.cell.expectedVersion, m.cell.hasExpected
}

// Delta is an isolated, off-heap transaction write-set for one collection.
// The directory is a dedupe accelerator; entries preserve first-touch order in
// the arena-backed log for deterministic durable encoding.
type Delta struct {
	arena      *memory.Arena
	ids        *Directory
	entries    []*mutationCell
	count      uint32
	idCapacity uint64
	idKeyBytes uint64
}

func NewDelta(cfg DeltaConfig) (*Delta, error) {
	if cfg.MaxMutations == 0 {
		return nil, ErrDeltaFull
	}
	if cfg.ArenaBytes == 0 {
		return nil, ErrDeltaFull
	}
	if cfg.IDCapacity == 0 {
		cfg.IDCapacity = uint64(cfg.MaxMutations) * 2
	}
	arena, err := memory.NewArena(cfg.ArenaBytes, 64)
	if err != nil {
		return nil, err
	}
	ids, err := NewDirectory(cfg.IDCapacity, cfg.IDKeyBytes)
	if err != nil {
		_ = arena.Free()
		return nil, err
	}
	entries, err := memory.ArenaSlice[*mutationCell](arena, int(cfg.MaxMutations))
	if err != nil {
		_ = ids.Free()
		_ = arena.Free()
		return nil, err
	}
	return &Delta{arena: arena, ids: ids, entries: entries, idCapacity: cfg.IDCapacity, idKeyBytes: cfg.IDKeyBytes}, nil
}

func (d *Delta) Arena() *memory.Arena { return d.arena }

func (d *Delta) Len() int { return int(d.count) }

func (d *Delta) Lookup(id BytesView) (Mutation, bool) {
	ref, ok := d.ids.Get(id)
	return Mutation{cell: (*mutationCell)(ref.ptr)}, ok
}

func (d *Delta) stageCell(id BytesView, before RecordRef, kind MutationKind, expectedVersion uint64, hasExpected bool) (*mutationCell, bool, error) {
	if existing, ok := d.Lookup(id); ok {
		cell := existing.cell
		if hasExpected {
			if cell.hasExpected && cell.expectedVersion != expectedVersion {
				return nil, false, ErrUnsupportedType
			}
			cell.hasExpected = true
			cell.expectedVersion = expectedVersion
		}
		cell.kind = kind
		return cell, false, nil
	}
	if d.count >= uint32(cap(d.entries)) {
		return nil, false, ErrDeltaFull
	}
	idCopy, err := d.arena.Alloc(uint64(id.len))
	if err != nil {
		return nil, false, err
	}
	copy(unsafe.Slice((*byte)(idCopy), id.len), id.Bytes())
	cell, err := memory.ArenaAlloc[mutationCell](d.arena)
	if err != nil {
		return nil, false, err
	}
	cell.idPtr = idCopy
	cell.idLen = id.len
	cell.before = before.ptr
	cell.expectedVersion = expectedVersion
	cell.hasExpected = hasExpected
	cell.kind = kind
	cell.sequence = d.count
	if err := d.ids.Put(BytesView{ptr: idCopy, len: id.len}, RecordRef{ptr: unsafe.Pointer(cell)}); err != nil {
		return nil, false, err
	}
	d.entries = d.entries[:d.count+1]
	d.entries[d.count] = cell
	d.count++
	return cell, true, nil
}

// StagePut adds or replaces a record. before must be the record visible in the
// published snapshot on first touch; it is ignored on subsequent mutations of
// the same ID so transaction semantics remain sequential.
func (d *Delta) StagePut(kind MutationKind, before RecordRef, builder RecordBuilder, expectedVersion uint64, hasExpected bool) (Mutation, bool, error) {
	if kind != MutationInsert && kind != MutationUpdate && kind != MutationUpsert {
		return Mutation{}, false, ErrUnsupportedType
	}
	cell, first, err := d.stageCell(builder.ID, before, kind, expectedVersion, hasExpected)
	if err != nil {
		return Mutation{}, false, err
	}
	ref, err := builder.Seal(d.arena)
	if err != nil {
		return Mutation{}, false, err
	}
	cell.after = ref.ptr
	return Mutation{cell: cell}, first, nil
}

// StageDelete writes an explicit tombstone. Its Before field retains the
// first-touch published record so CAS and delete/no-op rules can be decided by
// the caller without scanning the collection.
func (d *Delta) StageDelete(before RecordRef, id BytesView, expectedVersion uint64, hasExpected bool) (Mutation, bool, error) {
	if id.ptr == nil || id.len == 0 {
		return Mutation{}, false, ErrInvalidID
	}
	cell, first, err := d.stageCell(id, before, MutationDelete, expectedVersion, hasExpected)
	if err != nil {
		return Mutation{}, false, err
	}
	// An insert followed by delete has no published effect. A deletion of an
	// existing record receives an explicit off-heap tombstone so it masks the
	// parent generation in both ID and ordinal lookups.
	if cell.before != nil {
		tombstone, err := SealTombstone(d.arena, id, before.Ordinal(), before.Version())
		if err != nil {
			return Mutation{}, false, err
		}
		cell.after = tombstone.ptr
	} else {
		cell.after = nil
	}
	return Mutation{cell: cell}, first, nil
}

// At returns mutations in deterministic first-touch order. It never exposes a
// Go heap slice to the caller.
func (d *Delta) At(i int) Mutation {
	if i < 0 || i >= int(d.count) {
		panic("record: mutation index out of range")
	}
	return Mutation{cell: d.entries[i]}
}

func (d *Delta) Close() error {
	if d == nil {
		return nil
	}
	var firstErr error
	if d.ids != nil {
		firstErr = d.ids.Free()
		d.ids = nil
	}
	if d.arena != nil {
		if err := d.arena.Free(); firstErr == nil {
			firstErr = err
		}
		d.arena = nil
	}
	d.entries = nil
	d.count = 0
	return firstErr
}
