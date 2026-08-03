package hmgi

import (
	"testing"
	"unsafe"
)

func TestPageHeaderLayout(t *testing.T) {
	var header PageHeader
	
	// The struct must be exactly 64 bytes (cache line aligned)
	if size := unsafe.Sizeof(header); size != 64 {
		t.Fatalf("PageHeader expected to be 64 bytes, got %d", size)
	}
	
	// Check exact offsets
	if off := unsafe.Offsetof(header.CommunityID); off != 0 {
		t.Fatalf("CommunityID expected at offset 0, got %d", off)
	}
	if off := unsafe.Offsetof(header.Spinlock); off != 4 {
		t.Fatalf("Spinlock expected at offset 4, got %d", off)
	}
	if off := unsafe.Offsetof(header.Padding); off != 8 {
		t.Fatalf("Padding expected at offset 8, got %d", off)
	}
	if off := unsafe.Offsetof(header.ECBBPayload); off != 16 {
		t.Fatalf("ECBBPayload expected at offset 16, got %d", off)
	}
}

func TestECBBViews(t *testing.T) {
	var cap ECBBSphericalCap
	if unsafe.Sizeof(cap) > 48 {
		t.Fatalf("ECBBSphericalCap exceeds 48 bytes: %d", unsafe.Sizeof(cap))
	}
	
	var rect ECBBHyperrectangle
	if unsafe.Sizeof(rect) > 48 {
		t.Fatalf("ECBBHyperrectangle exceeds 48 bytes: %d", unsafe.Sizeof(rect))
	}
}

func TestDimensionAwareCapacity(t *testing.T) {
	// 128 dimensions * 4 = 512 bytes + 64 bytes overhead = 576 bytes
	// 4032 / 576 = 7
	cap128 := DimensionAwareCapacity(128)
	if cap128 != 7 {
		t.Fatalf("Expected 7 nodes for 128-dim, got %d", cap128)
	}
	
	// 1536 dimensions * 4 = 6144 bytes
	// > 4032, so it should return 1 (multi-page span)
	cap1536 := DimensionAwareCapacity(1536)
	if cap1536 != 1 {
		t.Fatalf("Expected 1 node span for 1536-dim, got %d", cap1536)
	}
}

func BenchmarkECBBIntersection(b *testing.B) {
	var payload [48]byte
	pruner := NewPruner(MetricCosine, &payload, nil)
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		_ = pruner.Intersects(123, 0.5)
	}
}
