package flat

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/record"
	"github.com/xDarkicex/libravdb/internal/util"
)

func TestCoreUsesImmutableOffHeapDeltas(t *testing.T) {
	ctx := context.Background()
	core, err := NewCore(&Config{Dimension: 2, Metric: util.L2Distance})
	if err != nil {
		t.Fatal(err)
	}
	defer core.Close()
	id := []byte("r")
	firstVector := []float32{1, 0}
	delta, err := core.NewDelta(16<<10, 2, 512)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = delta.StagePut(record.MutationInsert, record.RecordRef{}, record.RecordBuilder{
		ID:      record.BorrowBytes(id),
		Vector:  record.BorrowVector(firstVector),
		Ordinal: 5,
		Version: 1,
	}, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := core.CommitDelta(delta); err != nil {
		t.Fatal(err)
	}

	previous, found := core.CurrentRecord(record.BorrowBytes(id))
	if !found || previous.Vector().At(0) != 1 {
		t.Fatalf("initial record found=%v value=%v", found, previous.Vector().At(0))
	}
	secondVector := []float32{0, 1}
	update, err := core.NewDelta(16<<10, 2, 512)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = update.StagePut(record.MutationUpdate, previous, record.RecordBuilder{
		ID:      record.BorrowBytes(id),
		Vector:  record.BorrowVector(secondVector),
		Ordinal: previous.Ordinal(),
		Version: 2,
	}, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := core.CommitDelta(update); err != nil {
		t.Fatal(err)
	}
	set, err := core.SearchBorrowed(ctx, record.BorrowVector([]float32{0, 1}), 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	if set.Len() != 1 {
		t.Fatalf("visible results = %d, want 1", set.Len())
	}
	got, _ := set.At(0)
	if got.Version() != 2 || got.Vector().At(1) != 1 {
		t.Fatalf("result = version %d vector=%v", got.Version(), got.Vector().Float32s())
	}

	deleteDelta, err := core.NewDelta(16<<10, 2, 512)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = deleteDelta.StageDelete(got, record.BorrowBytes(id), 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := core.CommitDelta(deleteDelta); err != nil {
		t.Fatal(err)
	}
	deleted, err := core.SearchBorrowed(ctx, record.BorrowVector([]float32{0, 1}), 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer deleted.Close()
	if deleted.Len() != 0 {
		t.Fatalf("deleted record remained visible: %d results", deleted.Len())
	}
}
