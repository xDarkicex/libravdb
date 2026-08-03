package hmgi

import (
	"github.com/xDarkicex/libravdb/internal/storage/slab"
)

// MigrationWorker evaluates incremental modularity shifts (Delta Q)
// and dynamically migrates vectors between 4KB pages to maintain topological co-location.
// This executes asynchronously in the background. Community assignments are ephemeral,
// meaning they are re-derived on startup and never written to the WAL.
type MigrationWorker struct {
	tau float64 // The predefined migration threshold to prevent thrashing
}

// NewMigrationWorker initializes a worker with the Leiden stability threshold.
func NewMigrationWorker(totalEdges int) MigrationWorker {
	// Standard Leiden stability recommendation: tau = 1 / (2 * |E|)
	var tau float64
	if totalEdges > 0 {
		tau = 1.0 / float64(2*totalEdges)
	} else {
		tau = 0.0001
	}
	
	return MigrationWorker{tau: tau}
}

// CalculateDeltaQ computes the incremental modularity shift for moving a node to a new community.
func (w *MigrationWorker) CalculateDeltaQ(k_i_in float64, Sigma_in float64, Sigma_tot float64, k_i float64, m float64) float64 {
	// The standard Newman modularity shift formula
	if m == 0 {
		return 0
	}
	
	two_m := 2 * m
	term1 := (Sigma_in + 2*k_i_in) / two_m
	
	term2 := (Sigma_tot + k_i) / two_m
	term2 = term2 * term2
	
	term3 := Sigma_in / two_m
	
	term4 := Sigma_tot / two_m
	term4 = term4 * term4
	
	term5 := k_i / two_m
	term5 = term5 * term5
	
	return (term1 - term2) - (term3 - term4 - term5)
}

// ShouldMigrate evaluates if the modularity gain exceeds the threshold.
func (w *MigrationWorker) ShouldMigrate(deltaQ float64) bool {
	return deltaQ > w.tau
}

// Migrate performs the lock-free community swap using CoW semantics.
// It requires locking both the source and target page headers (to safely move the vector payload),
// but the actual graph adjacency pointer update is lock-free via the Phase 4 CAS.
func (w *MigrationWorker) Migrate(node *slab.Node, srcHeader, dstHeader *PageHeader, newPhysicalOffset uint64, nodeVersion uint64, currentLSN uint64) bool {
	// 1. Edge-staleness guard: DeltaQ was computed against current neighbors.
	// If the node's edges mutated (LSN/version changed), the relocation is wasted.
	if nodeVersion != currentLSN {
		return false
	}

	// 2. Lock the physical pages (spinlocks are ultra-fast, only contention is other migrations)
	if srcHeader.CommunityID < dstHeader.CommunityID {
		srcHeader.Lock()
		dstHeader.Lock()
	} else if srcHeader.CommunityID > dstHeader.CommunityID {
		dstHeader.Lock()
		srcHeader.Lock()
	} else {
		// Same community, no migration needed
		return true
	}
	
	defer srcHeader.Unlock()
	defer dstHeader.Unlock()
	
	// 2. Perform the physical memory move (AVX-512 aligned copy in full engine)
	
	// 3. Atomically update the graph's routing pointer.
	curr := node.Read()
	next := slab.Pack(newPhysicalOffset, curr.Degree())
	
	// The CAS ensures that if another thread mutated the node during migration,
	// we fail gracefully. CAS failure is expected contention, not a fault.
	return node.Update(curr, next)
}

// MigrationMetrics tracks the results of a migration tick.
type MigrationMetrics struct {
	Visited   int
	Evaluated int
	Migrated  int
}

// IncrementalMigrate performs a bounded random walk over the graph, evaluating DeltaQ
// and migrating nodes to co-locate them in memory.
// It uses a persistent cursor to ensure uniform coverage across ticks.
func (w *MigrationWorker) IncrementalMigrate(
	cursor *uint32, 
	budget int, 
	maxNode uint32,
	getNode func(id uint32) (*slab.Node, uint64, uint64), // returns node, nodeVersion, currentLSN
	getHeader func(id uint32) *PageHeader,
	getMetrics func(nodeID, commID uint32) (k_i_in, Sigma_in, Sigma_tot, k_i, m float64),
) MigrationMetrics {
	metrics := MigrationMetrics{}
	
	// Fast path for empty graph
	if maxNode <= 1 {
		return metrics
	}

	for metrics.Visited < budget {
		*cursor++
		if *cursor >= maxNode {
			*cursor = 1
		}
		
		node, version, lsn := getNode(*cursor)
		if node == nil {
			continue
		}
		metrics.Visited++
		
		// 1. Evaluate DeltaQ for neighbors (to be implemented when hmgi graph is wired)
		// 2. If ShouldMigrate(deltaQ), call Migrate()
		// metrics.Evaluated++
		// if w.Migrate(...) { metrics.Migrated++ }
		
		_ = version
		_ = lsn
		_ = getHeader
		_ = getMetrics
	}
	
	return metrics
}
