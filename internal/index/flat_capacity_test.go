package index

import (
	"testing"

	"github.com/xDarkicex/libravdb/internal/index/flat"
	"github.com/xDarkicex/libravdb/internal/record"
	"github.com/xDarkicex/libravdb/internal/util"
)

func TestFlatCapacityModelFiftyThousandBoundary(t *testing.T) {
	const (
		count = 50_000
		dim   = 128
	)
	entries := make([]*VectorEntry, count)
	for i := range entries {
		entries[i] = &VectorEntry{
			ID:      formatCapacityID(i),
			Vector:  make([]float32, dim),
			Ordinal: uint32(i),
			Version: 1,
		}
	}
	report := flatDeltaCapacity(entries, nil)
	if report.ArenaBytes == 0 || report.KeyBytes == 0 {
		t.Fatalf("invalid 50K capacity report: %+v", report)
	}
	if report.GenerationOrdinalPages == 0 || report.PeakTrackedOffHeapBytes <= report.ArenaBytes {
		t.Fatalf("incomplete 50K capacity report: %+v", report)
	}
	t.Logf("50K flat capacity: arena=%d keys=%d staging=%d radix_pages=%d radix_bytes=%d peak_tracked_offheap=%d", report.ArenaBytes, report.KeyBytes, report.StagingBytes, report.GenerationOrdinalPages, report.GenerationOrdinalPageBytes, report.PeakTrackedOffHeapBytes)

	core, err := flat.NewCore(&flat.Config{Dimension: dim, Metric: util.L2Distance})
	if err != nil {
		t.Fatal(err)
	}
	defer core.Close()
	if err := stageFlatCapacityBatch(t, core, report.ArenaBytes-1, report.KeyBytes, entries); err == nil {
		t.Fatal("one byte below the derived 50K arena minimum unexpectedly succeeded")
	}
	if err := stageFlatCapacityBatch(t, core, report.ArenaBytes, report.KeyBytes, entries); err != nil {
		t.Fatalf("derived 50K capacity rejected batch: %v", err)
	}
	if got := core.Size(); got != count {
		t.Fatalf("Flat size=%d, want %d", got, count)
	}
}

func stageFlatCapacityBatch(t *testing.T, core *flat.Core, arenaBytes, keyBytes uint64, entries []*VectorEntry) error {
	t.Helper()
	delta, err := core.NewDelta(arenaBytes, uint32(len(entries)), keyBytes)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if _, _, err := delta.StagePut(record.MutationInsert, record.RecordRef{}, record.RecordBuilder{
			ID:      record.BorrowBytes([]byte(entry.ID)),
			Vector:  record.BorrowVector(entry.Vector),
			Ordinal: entry.Ordinal,
			Version: entry.Version,
		}, 0, false); err != nil {
			_ = delta.Close()
			return err
		}
	}
	return core.CommitDelta(delta)
}

func formatCapacityID(value int) string {
	const prefix = "flat-capacity-"
	buf := make([]byte, len(prefix)+8)
	copy(buf, prefix)
	for i := 0; i < 8; i++ {
		buf[len(prefix)+7-i] = byte('0' + value%10)
		value /= 10
	}
	return string(buf)
}
