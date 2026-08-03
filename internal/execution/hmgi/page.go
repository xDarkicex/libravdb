package hmgi

import (
	"sync/atomic"
	"unsafe"
)

// MetricType indicates the geometry for the ECBB payload.
// Matches the catalog definitions.
type MetricType uint8

const (
	MetricL2     MetricType = 1
	MetricCosine MetricType = 2
	MetricIP     MetricType = 3
)

// PageHeader represents the 64-byte cache-line aligned prefix of a 4KB Leiden page.
// The community ID dictates co-location. The Spinlock protects vector migration.
// The 48-byte ECBBPayload is an inline bounding shape used for algebraic pruning.
type PageHeader struct {
	CommunityID uint32
	Spinlock    uint32
	Padding     uint64
	ECBBPayload [48]byte
}

// Lock acquires the spinlock. Readers in HNSW are wait-free and bypass this.
// Only Leiden migrations use the lock to safely swap vectors between pages.
func (h *PageHeader) Lock() {
	for !atomic.CompareAndSwapUint32(&h.Spinlock, 0, 1) {
		// spin loop, expected to be extremely short
	}
}

// Unlock releases the spinlock.
func (h *PageHeader) Unlock() {
	atomic.StoreUint32(&h.Spinlock, 0)
}

// ECBBSphericalCap is the Cosine/IP distance view of the 48-byte payload.
// Uses a quantized centroid and a radial threshold to bound the page's contents.
type ECBBSphericalCap struct {
	CentroidID uint32
	Radius     float32
	// Extensible space for higher-order moments or multi-centroid bounds
}

// ECBBHyperrectangle is the L2 distance view of the 48-byte payload.
type ECBBHyperrectangle struct {
	// Compressed scalar bounds (e.g., Min/Max for the most variant principal components)
	MinBound [5]float32
	MaxBound [5]float32
	// 5 * 4 + 5 * 4 = 40 bytes. Fits in 48.
}

// Pruner provides zero-allocation intersection checks against the 48-byte ECBB.
type Pruner struct {
	metric   MetricType
	payload  *[48]byte
	distFunc func(c1, c2 uint32) float32
}

// NewPruner binds the metric to the page's raw ECBB memory.
func NewPruner(metric MetricType, payload *[48]byte, distFunc func(c1, c2 uint32) float32) Pruner {
	return Pruner{metric: metric, payload: payload, distFunc: distFunc}
}

// Intersects checks if a given query hypersphere overlaps with this page's ECBB.
// This is used by ECQO to algebraically eliminate nested loops over entire communities.
// It is guaranteed to be strictly conservative (stale bounds = false positive, never false negative).
func (p Pruner) Intersects(queryCentroidID uint32, queryRadius float32) bool {
	if p.metric == MetricCosine {
		cap := (*ECBBSphericalCap)(unsafe.Pointer(p.payload))
		
		if cap.Radius == 0 {
			return true // Uninitialized or infinite bounds
		}
		
		if p.distFunc != nil {
			dist := p.distFunc(cap.CentroidID, queryCentroidID)
			// If distance between centroids > sum of their radii, they don't intersect.
			if dist > (cap.Radius + queryRadius) {
				return false
			}
		}
		
		return true
	}
	
	// Fallback for L2 / unhandled metric:
	return true
}

// DimensionAwareCapacity computes how many vectors fit into a 4KB page.
// 4096 bytes total - 64 bytes header = 4032 bytes for payload.
// A node contains factorized metadata (say, 64 bytes for routing/slab data) + D*4 bytes (float32).
func DimensionAwareCapacity(dims uint32) int {
	// We assume a 64-byte overhead per vector (for slab pointer / local graph routing inside the page)
	nodeSize := 64 + (dims * 4)
	
	if nodeSize > 4032 {
		// Multi-page vector span required (D > 992)
		// Return 1, meaning it consumes the whole page (and spills over to subsequent contiguous pages).
		return 1
	}
	
	return 4032 / int(nodeSize)
}
