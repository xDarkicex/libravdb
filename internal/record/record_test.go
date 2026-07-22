package record

import (
	"testing"
	"time"
	"unsafe"

	"github.com/xDarkicex/memory"
)

func bytesView(data []byte) BytesView {
	return BytesView{ptr: unsafe.Pointer(unsafe.SliceData(data)), len: uint32(len(data))}
}

func vectorView(data []float32) VectorView {
	return VectorView{ptr: unsafe.Pointer(unsafe.SliceData(data)), len: uint32(len(data))}
}

func TestRecordSealCopiesAllPayloadOffHeap(t *testing.T) {
	request, err := memory.NewArena(4096, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer request.Free()
	persistent, err := memory.NewArena(4096, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer persistent.Free()

	metadata, err := NewMetadataBuilder(request, 128)
	if err != nil {
		t.Fatal(err)
	}
	if err := metadata.String([]byte("kind"), []byte("fact")); err != nil {
		t.Fatal(err)
	}
	if err := metadata.Time([]byte("at"), time.Unix(1, 7)); err != nil {
		t.Fatal(err)
	}
	id := []byte("fact-1")
	vector := []float32{1, 2, 3}
	ref, err := (RecordBuilder{ID: bytesView(id), Vector: vectorView(vector), Metadata: metadata.View(), Version: 9, Ordinal: 4}).Seal(persistent)
	if err != nil {
		t.Fatal(err)
	}
	id[0] = 'x'
	vector[0] = 99
	if got := string(ref.ID().Bytes()); got != "fact-1" {
		t.Fatalf("ID = %q", got)
	}
	if got := ref.Vector().At(0); got != 1 {
		t.Fatalf("vector[0] = %v", got)
	}
	typ, value, ok := ref.Metadata().Find([]byte("kind"))
	if !ok || typ != MetadataString || string(value.Bytes()) != "fact" {
		t.Fatalf("metadata = type %v value %q ok %v", typ, value.Bytes(), ok)
	}
	if ref.Version() != 9 || ref.Ordinal() != 4 {
		t.Fatalf("header = version %d ordinal %d", ref.Version(), ref.Ordinal())
	}
}

func TestDirectoryStoresOnlyOffHeapRecordPointers(t *testing.T) {
	arena, err := memory.NewArena(4096, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer arena.Free()
	directory, err := NewDirectory(8, 512)
	if err != nil {
		t.Fatal(err)
	}
	defer directory.Free()
	id := []byte("r")
	vector := []float32{1}
	ref, err := (RecordBuilder{ID: bytesView(id), Vector: vectorView(vector)}).Seal(arena)
	if err != nil {
		t.Fatal(err)
	}
	if err := directory.Put(bytesView(id), ref); err != nil {
		t.Fatal(err)
	}
	got, ok := directory.Get(bytesView(id))
	if !ok || got.ptr != ref.ptr {
		t.Fatalf("directory lookup = %#v, %v", got, ok)
	}
}

func TestDeltaDeduplicatesWithoutLosingFirstTouchState(t *testing.T) {
	delta, err := NewDelta(DeltaConfig{ArenaBytes: 8192, MaxMutations: 4, IDCapacity: 8, IDKeyBytes: 512})
	if err != nil {
		t.Fatal(err)
	}
	defer delta.Close()
	id := []byte("r")
	first, wasFirst, err := delta.StagePut(MutationUpsert, RecordRef{}, RecordBuilder{ID: bytesView(id), Vector: vectorView([]float32{1})}, 7, true)
	if err != nil || !wasFirst {
		t.Fatalf("first stage: mutation=%#v first=%v err=%v", first, wasFirst, err)
	}
	second, wasFirst, err := delta.StagePut(MutationUpdate, RecordRef{}, RecordBuilder{ID: bytesView(id), Vector: vectorView([]float32{2})}, 7, true)
	if err != nil || wasFirst {
		t.Fatalf("second stage: mutation=%#v first=%v err=%v", second, wasFirst, err)
	}
	if delta.Len() != 1 || delta.At(0).Kind() != MutationUpdate || delta.At(0).After().Vector().At(0) != 2 {
		t.Fatalf("delta did not retain latest state: len=%d kind=%v value=%v", delta.Len(), delta.At(0).Kind(), delta.At(0).After().Vector().At(0))
	}
	if version, ok := delta.At(0).ExpectedVersion(); !ok || version != 7 {
		t.Fatalf("CAS state = %d, %v", version, ok)
	}
}

func TestGenerationCopyOnWritePreservesParentAndTombstones(t *testing.T) {
	baseDelta, err := NewDelta(DeltaConfig{ArenaBytes: 16 << 10, MaxMutations: 4, IDCapacity: 8, IDKeyBytes: 512})
	if err != nil {
		t.Fatal(err)
	}
	id := []byte("r")
	_, _, err = baseDelta.StagePut(MutationInsert, RecordRef{}, RecordBuilder{ID: bytesView(id), Vector: vectorView([]float32{1}), Ordinal: 13, Version: 1}, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	base, err := NewGeneration(nil, baseDelta)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Release()
	old, found := base.Lookup(bytesView(id))
	if !found || old.Tombstone() || old.Vector().At(0) != 1 {
		t.Fatalf("base record = %#v found=%v", old, found)
	}

	childDelta, err := NewDelta(DeltaConfig{ArenaBytes: 16 << 10, MaxMutations: 4, IDCapacity: 8, IDKeyBytes: 512})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = childDelta.StagePut(MutationUpdate, old, RecordBuilder{ID: bytesView(id), Vector: vectorView([]float32{2}), Ordinal: old.Ordinal(), Version: 2}, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	child, err := NewGeneration(base, childDelta)
	if err != nil {
		t.Fatal(err)
	}
	updated, found := child.Lookup(bytesView(id))
	if !found || updated.Vector().At(0) != 2 || !child.Visible(updated) {
		t.Fatalf("child record = %#v found=%v", updated, found)
	}
	if base.ByOrdinal(13).ptr != old.ptr || old.Vector().At(0) != 1 {
		t.Fatal("child mutation changed parent generation")
	}

	deleteDelta, err := NewDelta(DeltaConfig{ArenaBytes: 16 << 10, MaxMutations: 4, IDCapacity: 8, IDKeyBytes: 512})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = deleteDelta.StageDelete(updated, bytesView(id), 0, false)
	if err != nil {
		t.Fatal(err)
	}
	deleted, err := NewGeneration(child, deleteDelta)
	if err != nil {
		t.Fatal(err)
	}
	defer deleted.Release()
	tombstone, found := deleted.Lookup(bytesView(id))
	if !found || !tombstone.Tombstone() || deleted.Visible(updated) {
		t.Fatalf("tombstone = %#v found=%v visible-old=%v", tombstone, found, deleted.Visible(updated))
	}
	child.Release()
}
