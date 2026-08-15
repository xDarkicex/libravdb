package libravdb

import "testing"

func TestPublicEdgeKindRegistry(t *testing.T) {
	const name = "PUBLIC_API_EDGE_KIND_251"
	if !RegisterEdgeKind(name, 251) {
		t.Fatal("RegisterEdgeKind rejected a fresh public registration")
	}
	if got := ResolveEdgeKind(name); got != 251 {
		t.Fatalf("ResolveEdgeKind(%q) = %d, want 251", name, got)
	}
	if !RegisterEdgeKind(name, 251) {
		t.Fatal("idempotent RegisterEdgeKind was rejected")
	}
}
