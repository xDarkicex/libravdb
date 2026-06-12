package graph

import (
	"testing"
)

func TestManifestSerialization(t *testing.T) {
	m := NewDBManifest()
	m.MinReaderVersion = 2
	
	err := m.RegisterKind(1, "causes")
	if err != nil {
		t.Fatal(err)
	}
	err = m.RegisterKind(2, "inhibits")
	if err != nil {
		t.Fatal(err)
	}
	
	// Test rejection of duplicate kind
	err = m.RegisterKind(1, "different_causes")
	if err == nil {
		t.Fatal("expected error registering duplicate kind")
	}
	
	data := m.Serialize()
	
	m2, err := DeserializeManifest(data)
	if err != nil {
		t.Fatal(err)
	}
	
	if m2.MinReaderVersion != 2 {
		t.Fatalf("expected MinReaderVersion 2, got %d", m2.MinReaderVersion)
	}
	
	if len(m2.KindManifest) != 2 {
		t.Fatalf("expected 2 kinds, got %d", len(m2.KindManifest))
	}
	
	if name := m2.KindManifest[1]; name != "causes" {
		t.Fatalf("expected 'causes', got %q", name)
	}
	if name := m2.KindManifest[2]; name != "inhibits" {
		t.Fatalf("expected 'inhibits', got %q", name)
	}
}

func TestDeserializeManifestEmpty(t *testing.T) {
	m, err := DeserializeManifest([]byte{})
	if err != nil {
		t.Fatal(err)
	}
	
	if m.MinReaderVersion != 1 {
		t.Fatalf("expected MinReaderVersion 1, got %d", m.MinReaderVersion)
	}
	if len(m.KindManifest) != 0 {
		t.Fatalf("expected 0 kinds, got %d", len(m.KindManifest))
	}
}
