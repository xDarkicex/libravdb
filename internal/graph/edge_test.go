package graph

import (
	"testing"
	"unsafe"
)

func TestEdgeSize(t *testing.T) {
	if sz := unsafe.Sizeof(Edge{}); sz != 16 {
		t.Fatalf("Edge size = %d, want 16", sz)
	}
}
