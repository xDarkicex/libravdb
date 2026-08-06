package graph

// KindSet is a 256-bit mask for branch-free edge kind filtering
type KindSet [4]uint64

// Has checks if a kind is in the set (branch-free)
func (ks KindSet) Has(kind uint8) bool {
	word := kind / 64
	bit := kind % 64
	return (ks[word] & (1 << bit)) != 0
}

// Set marks a kind as present
func (ks *KindSet) Set(kind uint8) {
	word := kind / 64
	bit := kind % 64
	ks[word] |= (1 << bit)
}

// NewKindSet creates a set from kind values
func NewKindSet(kinds ...uint8) KindSet {
	var ks KindSet
	for _, k := range kinds {
		ks.Set(k)
	}
	return ks
}

// EdgeKindRegistry maps edge type names to their assigned uint8 kind values.
// Register kinds before using them in graph queries with typed edges.
var EdgeKindRegistry = struct {
	byName map[string]uint8
	byKind map[uint8]string
}{
	byName: make(map[string]uint8),
	byKind: make(map[uint8]string),
}

// RegisterEdgeKind assigns a kind number to an edge type name.
// Kind 0 is reserved (treated as "no filter").
//
// Idempotent: re-registering the same (name, kind) pair returns true.
// Returns false when the kind number is 0, when the name is already
// mapped to a different kind number, or when the kind number is already
// claimed by a different name.
func RegisterEdgeKind(name string, kind uint8) bool {
	if kind == 0 {
		return false
	}
	if existing, ok := EdgeKindRegistry.byName[name]; ok {
		return existing == kind
	}
	if _, ok := EdgeKindRegistry.byKind[kind]; ok {
		// kind number already claimed by a different name — hard conflict.
		return false
	}
	EdgeKindRegistry.byName[name] = kind
	EdgeKindRegistry.byKind[kind] = name
	return true
}

// ResolveEdgeKind returns the kind value for an edge type name, or 0 if not found.
func ResolveEdgeKind(name string) uint8 {
	return EdgeKindRegistry.byName[name]
}

// EdgeKindName returns the name for a kind value, or empty string if not found.
func EdgeKindName(kind uint8) string {
	return EdgeKindRegistry.byKind[kind]
}
