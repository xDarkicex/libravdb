package record

import (
	"testing"

	"github.com/xDarkicex/memory"
)

func capacityPut(idBytes, vectorBytes uint64, ordinal uint32) DeltaCapacityMutation {
	return DeltaCapacityMutation{
		IDBytes:       idBytes,
		VectorBytes:   vectorBytes,
		Ordinal:       ordinal,
		OrdinalKnown:  true,
		ProducesAfter: true,
	}
}

func stageCapacityPlan(t *testing.T, capacity uint64, plan []DeltaCapacityMutation) error {
	t.Helper()
	report := EstimateDeltaCapacity(uint64(len(plan)), plan)
	delta, err := NewDelta(DeltaConfig{
		ArenaBytes:   capacity,
		MaxMutations: uint32(len(plan)),
		IDCapacity:   uint64(len(plan)) * 2,
		IDKeyBytes:   report.KeyBytes,
	})
	if err != nil {
		return err
	}
	defer delta.Close()
	for i, mutation := range plan {
		id := make([]byte, mutation.IDBytes)
		for j := range id {
			id[j] = byte('a' + (i+j)%26)
		}
		vector := make([]float32, mutation.VectorBytes/4)
		if _, _, err := delta.StagePut(MutationInsert, RecordRef{}, RecordBuilder{
			ID:      bytesView(id),
			Vector:  vectorView(vector),
			Ordinal: mutation.Ordinal,
		}, 0, false); err != nil {
			return err
		}
	}
	generation, err := NewGeneration(nil, delta)
	if err != nil {
		return err
	}
	generation.Release()
	return nil
}

func TestEstimateDeltaCapacityExactBoundary(t *testing.T) {
	plan := []DeltaCapacityMutation{
		capacityPut(7, 128, 1),
		capacityPut(129, 512, 0x01020304),
		capacityPut(31, 256, 0x05060708),
	}
	report := EstimateDeltaCapacity(uint64(len(plan)), plan)
	if report.ArenaBytes == 0 || report.KeyBytes == 0 {
		t.Fatalf("invalid capacity report: %+v", report)
	}
	if err := stageCapacityPlan(t, report.ArenaBytes, plan); err != nil {
		t.Fatalf("derived capacity rejected valid plan: report=%+v err=%v", report, err)
	}
	if report.ArenaBytes > 1 {
		if err := stageCapacityPlan(t, report.ArenaBytes-1, plan); err == nil {
			t.Fatalf("capacity below derived minimum unexpectedly succeeded: report=%+v", report)
		}
	}
	if report.GenerationOrdinalPages != 9 {
		t.Fatalf("ordinal pages=%d, want 9 distinct path prefixes", report.GenerationOrdinalPages)
	}
	t.Logf("flat delta capacity: arena=%d keys=%d staging=%d root=%d segment=%d radix_pages=%d radix_bytes=%d peak_tracked_offheap=%d", report.ArenaBytes, report.KeyBytes, report.StagingBytes, report.GenerationRootBytes, report.GenerationSegmentBytes, report.GenerationOrdinalPages, report.GenerationOrdinalPageBytes, report.PeakTrackedOffHeapBytes)
}

func TestEstimateDeltaCapacityDenseAndUnknownDeleteBounds(t *testing.T) {
	dense := []DeltaCapacityMutation{
		capacityPut(16, 128, 0),
		capacityPut(16, 128, 1),
		capacityPut(16, 128, 2),
		capacityPut(16, 128, 255),
	}
	sparse := []DeltaCapacityMutation{
		capacityPut(16, 128, 0),
		capacityPut(16, 128, 1<<24),
		capacityPut(16, 128, 2<<24),
		capacityPut(16, 128, 3<<24),
	}
	denseReport := EstimateDeltaCapacity(uint64(len(dense)), dense)
	sparseReport := EstimateDeltaCapacity(uint64(len(sparse)), sparse)
	if denseReport.GenerationOrdinalPages >= sparseReport.GenerationOrdinalPages {
		t.Fatalf("dense pages=%d should be below sparse pages=%d", denseReport.GenerationOrdinalPages, sparseReport.GenerationOrdinalPages)
	}
	unknownDelete := DeltaCapacityMutation{IDBytes: 16, ProducesAfter: true, Tombstone: true}
	deleteReport := EstimateDeltaCapacity(1, []DeltaCapacityMutation{unknownDelete})
	if deleteReport.GenerationOrdinalPages != ordinalPathLevels {
		t.Fatalf("unknown delete pages=%d, want conservative %d", deleteReport.GenerationOrdinalPages, ordinalPathLevels)
	}
	if err := stageCapacityPlan(t, denseReport.ArenaBytes, dense); err != nil {
		t.Fatalf("dense derived capacity failed: %v", err)
	}
	if err := stageCapacityPlan(t, sparseReport.ArenaBytes, sparse); err != nil {
		t.Fatalf("sparse derived capacity failed: %v", err)
	}
}

func TestEstimateDeltaCapacityLongIDsAndConfiguredDimensions(t *testing.T) {
	plan := []DeltaCapacityMutation{capacityPut(4096, 1536*4, 99)}
	report := EstimateDeltaCapacity(1, plan)
	if report.KeyBytes < 4096 {
		t.Fatalf("key capacity=%d, want at least long ID bytes", report.KeyBytes)
	}
	if err := stageCapacityPlan(t, report.ArenaBytes, plan); err != nil {
		t.Fatalf("long-ID/configured-dimension capacity failed: %v", err)
	}
	if report.ArenaBytes <= report.StagingBytes {
		t.Fatalf("generation allocations missing from report: %+v", report)
	}
}

func TestEstimateDeltaCapacityUpdateAndDelete(t *testing.T) {
	plan := []DeltaCapacityMutation{
		{IDBytes: 24, VectorBytes: 256, Ordinal: 11, OrdinalKnown: true, ProducesAfter: true},
		{IDBytes: 24, Ordinal: 22, OrdinalKnown: true, ProducesAfter: true, Tombstone: true},
	}
	report := EstimateDeltaCapacity(uint64(len(plan)), plan)
	delta, err := NewDelta(DeltaConfig{
		ArenaBytes:   report.ArenaBytes,
		MaxMutations: uint32(len(plan)),
		IDCapacity:   4,
		IDKeyBytes:   report.KeyBytes,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer delta.Close()
	baseArena, err := memory.NewArena(4096, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer baseArena.Free()
	base, err := (RecordBuilder{ID: bytesView([]byte("deleted-record")), Vector: vectorView([]float32{1}), Ordinal: 22, Version: 1}).Seal(baseArena)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := delta.StagePut(MutationUpdate, RecordRef{}, RecordBuilder{
		ID: bytesView([]byte("updated-record")), Vector: vectorView([]float32{1, 2, 3, 4}), Ordinal: 11, Version: 2,
	}, 0, false); err != nil {
		t.Fatalf("stage update: %v", err)
	}
	if _, _, err := delta.StageDelete(base, bytesView([]byte("deleted-record")), 0, false); err != nil {
		t.Fatalf("stage delete: %v", err)
	}
	generation, err := NewGeneration(nil, delta)
	if err != nil {
		t.Fatalf("derived update/delete capacity rejected: %v", err)
	}
	defer generation.Release()
	if updated, ok := generation.Lookup(bytesView([]byte("updated-record"))); !ok || updated.Tombstone() {
		t.Fatal("updated record missing from generated state")
	}
	if deleted, ok := generation.Lookup(bytesView([]byte("deleted-record"))); !ok || !deleted.Tombstone() {
		t.Fatal("delete tombstone missing from generated state")
	}
}
