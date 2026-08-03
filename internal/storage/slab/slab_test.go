package slab

import (
	"sync"
	"testing"
)

func TestRoutingPointerPack(t *testing.T) {
	offset := uint64(0x123456789ABC)
	degree := uint16(0xABCD)
	
	ptr := Pack(offset, degree)
	if ptr.Offset() != offset {
		t.Fatalf("Expected offset %x, got %x", offset, ptr.Offset())
	}
	if ptr.Degree() != degree {
		t.Fatalf("Expected degree %x, got %x", degree, ptr.Degree())
	}
}

func TestSlabVisibility(t *testing.T) {
	// Created at 100, active
	s := NewSlabHeader(100, 0, 1)
	
	if !s.IsVisible(101) {
		t.Fatal("Should be visible to xid 101")
	}
	if s.IsVisible(99) {
		t.Fatal("Should not be visible to xid 99 (created in future)")
	}
	
	// Tombstoned at 150
	s.Xmax = 150
	if !s.IsVisible(149) {
		t.Fatal("Should be visible to xid 149 (deleted in future)")
	}
	if s.IsVisible(151) {
		t.Fatal("Should not be visible to xid 151 (already deleted)")
	}
	if !s.IsTombstoned(151) {
		t.Fatal("Should be marked as tombstoned for xid 151")
	}
}

func TestConcurrentCASUpdate(t *testing.T) {
	node := &Node{Ptr: Pack(0, 0)}
	var wg sync.WaitGroup
	
	const goroutines = 100
	const increments = 500
	
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < increments; j++ {
				for {
					curr := node.Read()
					deg := curr.Degree()
					next := Pack(curr.Offset(), deg+1)
					if node.Update(curr, next) {
						break
					}
				}
			}
		}()
	}
	
	wg.Wait()
	final := node.Read()
	expected := uint16(goroutines * increments)
	if final.Degree() != expected {
		t.Fatalf("Expected degree %d, got %d", expected, final.Degree())
	}
}

func TestEBRReclamation(t *testing.T) {
	registry := NewEpochRegistry()
	manager := NewLimboManager(registry, 1024*1024) // 1MB budget
	
	// Reader 1 starts at xid 100
	slot1 := registry.Acquire(100)
	
	// Reader 2 starts at xid 150
	slot2 := registry.Acquire(150)
	
	// Writer deletes slabs at 120 and 160
	manager.Retire(120, 0, 1024)
	manager.Retire(160, 1024, 1024)
	
	// Current xid is 200
	reclaimed := manager.Reclaim(200)
	if len(reclaimed) != 0 {
		t.Fatalf("Should not reclaim anything, oldest reader is at 100, which is < 120")
	}
	
	// Reader 1 finishes
	registry.Release(slot1)
	
	// Now oldest reader is 150
	reclaimed = manager.Reclaim(200)
	if len(reclaimed) != 1 || reclaimed[0].DeletionXid != 120 {
		t.Fatalf("Should reclaim slab deleted at 120")
	}
	
	// Reader 2 finishes
	registry.Release(slot2)
	
	// Now no active readers, oldest is 200 (currentXid)
	reclaimed = manager.Reclaim(200)
	if len(reclaimed) != 1 || reclaimed[0].DeletionXid != 160 {
		t.Fatalf("Should reclaim slab deleted at 160")
	}
}

func BenchmarkNodeRead(b *testing.B) {
	node := &Node{Ptr: Pack(0xABCDEF, 10)}
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		_ = node.Read()
	}
}

func BenchmarkNodeUpdateCAS(b *testing.B) {
	node := &Node{Ptr: Pack(0, 0)}
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		curr := node.Read()
		next := Pack(curr.Offset(), curr.Degree()+1)
		node.Update(curr, next)
	}
}

func BenchmarkEBRAcquireRelease(b *testing.B) {
	registry := NewEpochRegistry()
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		slot := registry.Acquire(uint64(i))
		registry.Release(slot)
	}
}
