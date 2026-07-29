package libravdb

import (
	"testing"
	"unsafe"
)

func TestGraphNodeIDSize(t *testing.T) {
	if s := unsafe.Sizeof(GraphNodeID(0)); s != 8 {
		t.Errorf("GraphNodeID size = %d, want 8", s)
	}
}

func TestGraphNodeIDIsValid(t *testing.T) {
	if GraphNodeID(0).IsValid() {
		t.Error("zero GraphNodeID must be invalid")
	}
	if !GraphNodeID(1).IsValid() {
		t.Error("GraphNodeID(1) must be valid")
	}
	if !GraphNodeID(^uint64(0)).IsValid() {
		t.Error("max uint64 GraphNodeID must be valid")
	}
}

func TestGraphNodeIDValidate(t *testing.T) {
	if GraphNodeID(0).Validate() != ErrInvalidGraphNodeID {
		t.Error("zero GraphNodeID must return sentinel error")
	}
	if GraphNodeID(42).Validate() != nil {
		t.Error("nonzero GraphNodeID Validate must return nil")
	}
}

func TestGraphNodeIDIsValidNoAlloc(t *testing.T) {
	// Both valid and invalid paths must be allocation-free.
	if n := testing.AllocsPerRun(100, func() { GraphNodeID(0).IsValid() }); n > 0 {
		t.Errorf("IsValid(invalid) allocated %f per call", n)
	}
	if n := testing.AllocsPerRun(100, func() { GraphNodeID(42).IsValid() }); n > 0 {
		t.Errorf("IsValid(valid) allocated %f per call", n)
	}
}

func TestGraphNodeIDValidateNoAlloc(t *testing.T) {
	// Both valid and invalid Validate paths must be allocation-free.
	if n := testing.AllocsPerRun(100, func() { GraphNodeID(0).Validate() }); n > 0 {
		t.Errorf("Validate(invalid) allocated %f per call", n)
	}
	if n := testing.AllocsPerRun(100, func() { GraphNodeID(42).Validate() }); n > 0 {
		t.Errorf("Validate(valid) allocated %f per call", n)
	}
}
