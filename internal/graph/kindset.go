package graph

import "sync"

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

// Clear removes a kind from the set.
func (ks *KindSet) Clear(kind uint8) {
	word := kind / 64
	bit := kind % 64
	ks[word] &^= 1 << bit
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
	mu         sync.RWMutex
	byName     map[string]uint8
	byKind     map[uint8]string
	undirected map[string]bool
}{
	byName:     make(map[string]uint8),
	byKind:     make(map[uint8]string),
	undirected: make(map[string]bool),
}

// RegisterEdgeKind assigns a kind number to an edge type name.
// Kind 0 is reserved (treated as "no filter").
//
// Idempotent: re-registering the same (name, kind) pair returns true.
// A numeric kind may have more than one registered name.  The graph format
// stores the numeric kind, while names are a SQL-facing registry; allowing
// aliases is important for independent embedded databases and tests that
// choose the same small numeric kind for different local labels.  The first
// registered name remains the canonical name returned by EdgeKindName.
// Returns false when the kind number is 0 or when the name is already mapped
// to a different kind number.
func RegisterEdgeKind(name string, kind uint8) bool {
	return registerEdgeKind(name, kind, false, false)
}

// RegisterEdgeKindWithDirection registers an edge type and its traversal
// direction. The direction is part of the edge-kind definition, not of an
// individual edge, so all aliases of the same numeric kind share it.
//
// Unlike RegisterEdgeKind, this function treats the direction as explicit:
// attempting to reuse a kind with a conflicting direction is rejected. This
// prevents two durable SQL databases in one process from silently changing
// the meaning of an already registered kind.
func RegisterEdgeKindWithDirection(name string, kind uint8, undirected bool) bool {
	return registerEdgeKind(name, kind, undirected, true)
}

// RegisterUndirectedEdgeKind is the concise public form for a bidirectional
// edge type.
func RegisterUndirectedEdgeKind(name string, kind uint8) bool {
	return RegisterEdgeKindWithDirection(name, kind, true)
}

func registerEdgeKind(name string, kind uint8, undirected, explicit bool) bool {
	if kind == 0 {
		return false
	}
	EdgeKindRegistry.mu.Lock()
	defer EdgeKindRegistry.mu.Unlock()
	if existing, ok := EdgeKindRegistry.byName[name]; ok {
		if existing != kind {
			return false
		}
		if explicit && EdgeKindRegistry.undirected[name] != undirected {
			return false
		}
		return true
	}
	if explicit {
		EdgeKindRegistry.undirected[name] = undirected
	} else if _, ok := EdgeKindRegistry.undirected[name]; !ok {
		// Legacy registrations are directed unless a durable definition has
		// already established the kind as undirected.
		EdgeKindRegistry.undirected[name] = false
	}
	EdgeKindRegistry.byName[name] = kind
	if _, ok := EdgeKindRegistry.byKind[kind]; !ok {
		EdgeKindRegistry.byKind[kind] = name
	}
	return true
}

// IsUndirectedEdgeKind reports whether a registered kind has bidirectional
// traversal semantics. Unknown kinds remain directed for compatibility.
func IsUndirectedEdgeKind(kind uint8) bool {
	EdgeKindRegistry.mu.RLock()
	defer EdgeKindRegistry.mu.RUnlock()
	for name, registeredKind := range EdgeKindRegistry.byName {
		if registeredKind == kind && EdgeKindRegistry.undirected[name] {
			return true
		}
	}
	return false
}

// IsUndirectedEdgeType reports direction metadata for a name without the
// ambiguity that a process-wide registry necessarily has for reused numeric
// kinds across independent database instances.
func IsUndirectedEdgeType(name string) bool {
	EdgeKindRegistry.mu.RLock()
	defer EdgeKindRegistry.mu.RUnlock()
	return EdgeKindRegistry.undirected[name]
}

// ResolveEdgeKind returns the kind value for an edge type name, or 0 if not found.
func ResolveEdgeKind(name string) uint8 {
	EdgeKindRegistry.mu.RLock()
	defer EdgeKindRegistry.mu.RUnlock()
	return EdgeKindRegistry.byName[name]
}

// EdgeKindName returns the name for a kind value, or empty string if not found.
func EdgeKindName(kind uint8) string {
	EdgeKindRegistry.mu.RLock()
	defer EdgeKindRegistry.mu.RUnlock()
	return EdgeKindRegistry.byKind[kind]
}
