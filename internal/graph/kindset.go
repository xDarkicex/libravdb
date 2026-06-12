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
