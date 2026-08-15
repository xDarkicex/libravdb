package graph

import (
	"testing"
	"unsafe"
)

func TestEdgeSize(t *testing.T) {
	if sz := unsafe.Sizeof(Edge{}); sz != 24 {
		t.Fatalf("Edge size = %d, want 24", sz)
	}
}
