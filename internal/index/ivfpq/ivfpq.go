package ivfpq

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

// Config holds configuration for IVF-PQ index
type Config struct {
	Dimension     int                       // Vector dimension
	NClusters     int                       // Number of clusters (inverted lists)
	NProbes       int                       // Number of clusters to probe during search
	Metric        util.DistanceMetric       // Distance metric
	Quantization  *quant.QuantizationConfig // Product quantization config
	MaxIterations int                       // Max k-means iterations
	Tolerance     float64                   // K-means convergence tolerance
	RandomSeed    int64                     // Random seed for reproducibility
}

// VectorEntry represents a vector entry
type VectorEntry struct {
	ID       string
	Vector   []float32
	Metadata map[string]interface{}
}

// SearchResult represents a search result
type SearchResult struct {
	ID       string
	Score    float32
	Vector   []float32
	Metadata map[string]interface{}
}

// DefaultConfig returns a default IVF-PQ configuration
func DefaultConfig(dimension int) *Config {
	// Rule of thumb: sqrt(N) clusters for N vectors, but start with reasonable defaults
	nClusters := int(math.Max(64, math.Min(4096, float64(dimension))))

	return &Config{
		Dimension:     dimension,
		NClusters:     nClusters,
		NProbes:       min(16, nClusters/4), // Probe 25% of clusters by default
		Metric:        util.L2Distance,
		Quantization:  quant.DefaultConfig(quant.ProductQuantization),
		MaxIterations: 100,
		Tolerance:     1e-4,
		RandomSeed:    time.Now().UnixNano(),
	}
}

// AutoTuneConfig automatically tunes IVF-PQ parameters based on dataset characteristics
func AutoTuneConfig(dimension int, estimatedVectors int, targetMemoryMB int) *Config {
	// Automatic cluster count tuning based on dataset size
	var nClusters int
	if estimatedVectors < 1000 {
		nClusters = max(4, estimatedVectors/50) // Small datasets: fewer clusters
	} else if estimatedVectors < 100000 {
		nClusters = int(math.Sqrt(float64(estimatedVectors))) // Medium datasets: sqrt(N)
	} else {
		nClusters = int(math.Pow(float64(estimatedVectors), 0.4)) // Large datasets: N^0.4
	}

	// Clamp cluster count to reasonable bounds
	nClusters = max(4, min(nClusters, 16384))

	// Automatic probe count tuning for accuracy vs speed trade-off
	var nProbes int
	if estimatedVectors < 10000 {
		nProbes = max(1, nClusters/2) // Small datasets: probe more clusters for accuracy
	} else if estimatedVectors < 1000000 {
		nProbes = max(1, nClusters/4) // Medium datasets: balanced approach
	} else {
		nProbes = max(1, nClusters/8) // Large datasets: probe fewer for speed
	}

	// Clamp probe count
	nProbes = max(1, min(nProbes, nClusters))

	// Auto-tune quantization based on memory constraints
	var quantConfig *quant.QuantizationConfig
	if targetMemoryMB > 0 {
		// Estimate memory usage and adjust quantization accordingly
		estimatedMemoryMB := (estimatedVectors * dimension * 4) / (1024 * 1024) // 4 bytes per float32

		if estimatedMemoryMB > targetMemoryMB {
			// Need aggressive quantization
			quantConfig = &quant.QuantizationConfig{
				Type:       quant.ProductQuantization,
				Codebooks:  max(4, dimension/16), // More aggressive subspace division
				Bits:       4,                    // Lower bits for more compression
				TrainRatio: 0.1,
				CacheSize:  1000,
			}
		} else if estimatedMemoryMB > targetMemoryMB/2 {
			// Moderate quantization
			quantConfig = &quant.QuantizationConfig{
				Type:       quant.ProductQuantization,
				Codebooks:  max(4, dimension/8),
				Bits:       6,
				TrainRatio: 0.15,
				CacheSize:  2000,
			}
		} else {
			// Light quantization for better accuracy
			quantConfig = &quant.QuantizationConfig{
				Type:       quant.ProductQuantization,
				Codebooks:  max(4, dimension/4),
				Bits:       8,
				TrainRatio: 0.2,
				CacheSize:  5000,
			}
		}
	} else {
		// Default quantization when no memory constraint
		quantConfig = quant.DefaultConfig(quant.ProductQuantization)
	}

	return &Config{
		Dimension:     dimension,
		NClusters:     nClusters,
		NProbes:       nProbes,
		Metric:        util.L2Distance,
		Quantization:  quantConfig,
		MaxIterations: 100,
		Tolerance:     1e-4,
		RandomSeed:    time.Now().UnixNano(),
	}
}

// Cluster represents a single inverted list cluster
type Cluster struct {
	ID                int               // Cluster ID
	Centroid          []float32         // Cluster centroid
	Entries           []*VectorEntry    // Vectors assigned to this cluster
	CompressedVectors map[string][]byte // Compressed vectors for quantized storage
	mutex             sync.RWMutex      // Protects concurrent access
}

// Index implements IVF-PQ (Inverted File with Product Quantization) index
type Index struct {
	config       *Config
	clusters     []*Cluster
	quantizer    quant.Quantizer
	distanceFunc util.DistanceFunc
	trained      bool
	size         int
	mutex        sync.RWMutex
	rand         *rand.Rand

	// Adaptive search parameters
	searchStats  *SearchStats
	adaptiveMode bool
}

// SearchStats tracks search performance for adaptive optimization
type SearchStats struct {
	mutex          sync.RWMutex
	totalSearches  int64
	totalLatencyMs int64
	accuracySum    float64
	currentProbes  int
	lastAdjustment time.Time
}

// NewIVFPQ creates a new IVF-PQ index
func NewIVFPQ(config *Config) (*Index, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	if config.Dimension <= 0 {
		return nil, fmt.Errorf("dimension must be positive, got %d", config.Dimension)
	}

	if config.NClusters <= 0 {
		return nil, fmt.Errorf("number of clusters must be positive, got %d", config.NClusters)
	}

	if config.NProbes <= 0 || config.NProbes > config.NClusters {
		return nil, fmt.Errorf("number of probes must be between 1 and %d, got %d", config.NClusters, config.NProbes)
	}

	// Create quantizer if quantization is enabled
	var quantizer quant.Quantizer
	if config.Quantization != nil {
		var err error
		quantizer, err = quant.Create(config.Quantization)
		if err != nil {
			return nil, fmt.Errorf("failed to create quantizer: %w", err)
		}
	}

	// Get distance function
	distanceFunc, err := util.GetDistanceFunc(config.Metric)
	if err != nil {
		return nil, fmt.Errorf("failed to get distance function: %w", err)
	}

	// Initialize clusters
	clusters := make([]*Cluster, config.NClusters)
	for i := 0; i < config.NClusters; i++ {
		clusters[i] = &Cluster{
			ID:                i,
			Centroid:          make([]float32, config.Dimension),
			Entries:           make([]*VectorEntry, 0),
			CompressedVectors: make(map[string][]byte),
		}
	}

	return &Index{
		config:       config,
		clusters:     clusters,
		quantizer:    quantizer,
		distanceFunc: distanceFunc,
		trained:      false,
		size:         0,
		rand:         rand.New(rand.NewSource(config.RandomSeed)),
		searchStats: &SearchStats{
			currentProbes:  config.NProbes,
			lastAdjustment: time.Now(),
		},
		adaptiveMode: false,
	}, nil
}

// Train trains the IVF-PQ index using k-means clustering
func (idx *Index) Train(ctx context.Context, vectors [][]float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no training vectors provided")
	}

	if len(vectors) < idx.config.NClusters {
		return fmt.Errorf("need at least %d training vectors for %d clusters, got %d",
			idx.config.NClusters, idx.config.NClusters, len(vectors))
	}

	// Validate vector dimensions
	for i, vec := range vectors {
		if len(vec) != idx.config.Dimension {
			return fmt.Errorf("vector %d has dimension %d, expected %d", i, len(vec), idx.config.Dimension)
		}
	}

	// Perform k-means clustering to train coarse quantizer
	if err := idx.trainCoarseQuantizer(ctx, vectors); err != nil {
		return fmt.Errorf("failed to train coarse quantizer: %w", err)
	}

	// Train fine quantizer (PQ) if enabled
	if idx.quantizer != nil {
		if err := idx.quantizer.Train(ctx, vectors); err != nil {
			return fmt.Errorf("failed to train fine quantizer: %w", err)
		}
	}

	idx.trained = true
	return nil
}

// trainCoarseQuantizer performs k-means clustering to create cluster centroids
func (idx *Index) trainCoarseQuantizer(ctx context.Context, vectors [][]float32) error {
	// Initialize centroids using k-means++
	if err := idx.initializeCentroids(vectors); err != nil {
		return fmt.Errorf("failed to initialize centroids: %w", err)
	}

	prevInertia := math.Inf(1)

	for iter := 0; iter < idx.config.MaxIterations; iter++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Assignment step: assign each vector to nearest centroid
		assignments := make([]int, len(vectors))
		totalInertia := float64(0)

		for i, vec := range vectors {
			bestCluster := 0
			bestDistance := float32(math.Inf(1))

			for j, cluster := range idx.clusters {
				distance := idx.distanceFunc(vec, cluster.Centroid)

				if distance < bestDistance {
					bestDistance = distance
					bestCluster = j
				}
			}

			assignments[i] = bestCluster
			totalInertia += float64(bestDistance)
		}

		// Check for convergence
		if math.Abs(prevInertia-totalInertia)/prevInertia < idx.config.Tolerance {
			break
		}
		prevInertia = totalInertia

		// Update step: recompute centroids
		if err := idx.updateCentroids(vectors, assignments); err != nil {
			return fmt.Errorf("failed to update centroids: %w", err)
		}
	}

	return nil
}

// initializeCentroids initializes cluster centroids using k-means++
func (idx *Index) initializeCentroids(vectors [][]float32) error {
	if len(vectors) < idx.config.NClusters {
		return fmt.Errorf("not enough vectors for initialization")
	}

	// Choose first centroid randomly
	firstIdx := idx.rand.Intn(len(vectors))
	copy(idx.clusters[0].Centroid, vectors[firstIdx])

	// Choose remaining centroids using k-means++ (proportional to squared distance)
	for k := 1; k < idx.config.NClusters; k++ {
		distances := make([]float64, len(vectors))
		totalDistance := float64(0)

		// Compute distance to nearest existing centroid for each vector
		for i, vec := range vectors {
			minDistance := float32(math.Inf(1))

			for j := 0; j < k; j++ {
				distance := idx.distanceFunc(vec, idx.clusters[j].Centroid)
				if distance < minDistance {
					minDistance = distance
				}
			}

			distances[i] = float64(minDistance * minDistance) // Squared distance
			totalDistance += distances[i]
		}

		// Choose next centroid with probability proportional to squared distance
		target := idx.rand.Float64() * totalDistance
		cumulative := float64(0)

		for i, distance := range distances {
			cumulative += distance
			if cumulative >= target {
				copy(idx.clusters[k].Centroid, vectors[i])
				break
			}
		}
	}

	return nil
}

// updateCentroids recomputes cluster centroids based on current assignments
func (idx *Index) updateCentroids(vectors [][]float32, assignments []int) error {
	// Reset centroids
	for _, cluster := range idx.clusters {
		for i := range cluster.Centroid {
			cluster.Centroid[i] = 0
		}
	}

	// Count vectors per cluster
	counts := make([]int, idx.config.NClusters)

	// Sum vectors for each cluster
	for i, vec := range vectors {
		clusterID := assignments[i]
		counts[clusterID]++

		for j, val := range vec {
			idx.clusters[clusterID].Centroid[j] += val
		}
	}

	// Compute averages (avoid division by zero)
	for i, cluster := range idx.clusters {
		if counts[i] > 0 {
			for j := range cluster.Centroid {
				cluster.Centroid[j] /= float32(counts[i])
			}
		} else {
			// Reinitialize empty clusters randomly
			randomIdx := idx.rand.Intn(len(vectors))
			copy(cluster.Centroid, vectors[randomIdx])
		}
	}

	return nil
}

// assignToCluster finds the best cluster for a vector
func (idx *Index) assignToCluster(vector []float32) (int, error) {
	if !idx.trained {
		return 0, fmt.Errorf("index must be trained before assignment")
	}

	bestCluster := 0
	bestDistance := float32(math.Inf(1))

	for i, cluster := range idx.clusters {
		distance := idx.distanceFunc(vector, cluster.Centroid)

		if distance < bestDistance {
			bestDistance = distance
			bestCluster = i
		}
	}

	return bestCluster, nil
}

// findProbeClusters finds the top-k closest clusters for search probing
func (idx *Index) findProbeClusters(query []float32) ([]int, error) {
	if !idx.trained {
		return nil, fmt.Errorf("index must be trained before search")
	}

	type clusterDistance struct {
		id       int
		distance float32
	}

	distances := make([]clusterDistance, len(idx.clusters))

	for i, cluster := range idx.clusters {
		distance := idx.distanceFunc(query, cluster.Centroid)
		distances[i] = clusterDistance{id: i, distance: distance}
	}

	// Sort by distance and take top NProbes
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	probes := make([]int, min(idx.config.NProbes, len(distances)))
	for i := range probes {
		probes[i] = distances[i].id
	}

	return probes, nil
}

// IsTrained returns whether the index has been trained
func (idx *Index) IsTrained() bool {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	return idx.trained
}

// GetConfig returns the index configuration
func (idx *Index) GetConfig() *Config {
	return idx.config
}

// GetClusterInfo returns information about clusters
func (idx *Index) GetClusterInfo() []ClusterInfo {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()

	info := make([]ClusterInfo, len(idx.clusters))
	for i, cluster := range idx.clusters {
		cluster.mutex.RLock()
		info[i] = ClusterInfo{
			ID:       cluster.ID,
			Size:     len(cluster.Entries),
			Centroid: make([]float32, len(cluster.Centroid)),
		}
		copy(info[i].Centroid, cluster.Centroid)
		cluster.mutex.RUnlock()
	}

	return info
}

// ClusterInfo provides information about a cluster
type ClusterInfo struct {
	ID       int       // Cluster ID
	Size     int       // Number of vectors in cluster
	Centroid []float32 // Cluster centroid
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Insert adds a vector entry to the index with enhanced quantization support
func (idx *Index) Insert(ctx context.Context, entry *VectorEntry) error {
	if entry == nil {
		return fmt.Errorf("entry cannot be nil")
	}

	if len(entry.Vector) != idx.config.Dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d",
			len(entry.Vector), idx.config.Dimension)
	}

	if !idx.trained {
		return fmt.Errorf("index must be trained before insertion")
	}

	// Find the best cluster for this vector
	clusterID, err := idx.assignToCluster(entry.Vector)
	if err != nil {
		return fmt.Errorf("failed to assign vector to cluster: %w", err)
	}

	// Compress vector if quantization is enabled
	var compressed []byte
	if idx.quantizer != nil && idx.quantizer.IsTrained() {
		compressed, err = idx.quantizer.Compress(entry.Vector)
		if err != nil {
			return fmt.Errorf("failed to compress vector: %w", err)
		}
	}

	// Add to cluster
	cluster := idx.clusters[clusterID]
	cluster.mutex.Lock()
	cluster.Entries = append(cluster.Entries, entry)

	// Store compressed version if available
	if compressed != nil {
		cluster.CompressedVectors[entry.ID] = compressed
	}

	cluster.mutex.Unlock()

	// Update size
	idx.mutex.Lock()
	idx.size++
	idx.mutex.Unlock()

	return nil
}

// Search performs k-NN search using IVF-PQ with enhanced multi-probe strategy
func (idx *Index) Search(ctx context.Context, query []float32, k int) ([]*SearchResult, error) {
	startTime := time.Now()

	if len(query) != idx.config.Dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d",
			len(query), idx.config.Dimension)
	}

	if k <= 0 {
		return nil, fmt.Errorf("k must be positive, got %d", k)
	}

	if !idx.trained {
		return nil, fmt.Errorf("index must be trained before search")
	}

	// Adjust probe count if adaptive mode is enabled
	idx.adjustProbeCount()

	// Find clusters to probe with enhanced strategy
	probeClusters, clusterDistances, err := idx.findProbeClustersWithDistances(query)
	if err != nil {
		return nil, fmt.Errorf("failed to find probe clusters: %w", err)
	}

	// Collect candidates from probe clusters with early termination optimization
	type candidate struct {
		entry       *VectorEntry
		distance    float32
		clusterDist float32 // Distance to cluster centroid for pruning
	}

	var candidates []candidate
	candidateCount := 0
	maxCandidates := k * 10 // Collect more candidates than needed for better accuracy

	for i, clusterID := range probeClusters {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		cluster := idx.clusters[clusterID]
		cluster.mutex.RLock()

		clusterDist := clusterDistances[i]

		// Search within this cluster with quantization optimization
		for _, entry := range cluster.Entries {
			var distance float32

			if idx.quantizer != nil && idx.quantizer.IsTrained() {
				// Use pre-compressed vector if available, otherwise compress on-the-fly
				compressed, exists := cluster.CompressedVectors[entry.ID]
				if !exists {
					var err error
					compressed, err = idx.quantizer.Compress(entry.Vector)
					if err != nil {
						cluster.mutex.RUnlock()
						return nil, fmt.Errorf("failed to compress vector: %w", err)
					}
					// Cache the compressed vector for future use
					cluster.CompressedVectors[entry.ID] = compressed
				}

				var err error
				distance, err = idx.quantizer.DistanceToQuery(compressed, query)
				if err != nil {
					cluster.mutex.RUnlock()
					return nil, fmt.Errorf("failed to compute quantized distance: %w", err)
				}
			} else {
				// Use full precision distance
				distance = idx.distanceFunc(query, entry.Vector)
			}

			candidates = append(candidates, candidate{
				entry:       entry,
				distance:    distance,
				clusterDist: clusterDist,
			})
			candidateCount++

			// Early termination if we have enough candidates and current cluster is far
			if candidateCount >= maxCandidates && i < len(probeClusters)-1 {
				// Sort current candidates to check if we can terminate early
				sort.Slice(candidates, func(i, j int) bool {
					return candidates[i].distance < candidates[j].distance
				})

				// If the k-th best candidate is much better than the next cluster's centroid distance,
				// we can potentially terminate early
				if len(candidates) >= k {
					kthDistance := candidates[k-1].distance
					nextClusterDist := clusterDistances[i+1]

					// Conservative early termination: only if k-th distance is significantly better
					if kthDistance < nextClusterDist*0.8 {
						cluster.mutex.RUnlock()
						break
					}
				}
			}
		}

		cluster.mutex.RUnlock()
	}

	// Sort all candidates by distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].distance < candidates[j].distance
	})

	// Convert to search results
	resultCount := min(k, len(candidates))
	results := make([]*SearchResult, resultCount)

	for i := 0; i < resultCount; i++ {
		results[i] = &SearchResult{
			ID:       candidates[i].entry.ID,
			Score:    candidates[i].distance,
			Vector:   candidates[i].entry.Vector,
			Metadata: candidates[i].entry.Metadata,
		}
	}

	// Record search statistics for adaptive optimization
	latencyMs := time.Since(startTime).Milliseconds()

	// Estimate accuracy based on result quality (simplified metric)
	accuracy := 1.0
	if len(results) > 0 {
		// Simple accuracy estimation: if we found results, assume good accuracy
		// In practice, this could be more sophisticated
		accuracy = math.Min(1.0, float64(len(results))/float64(k))
	}

	idx.recordSearchStats(latencyMs, accuracy)

	return results, nil
}

// findProbeClustersWithDistances finds probe clusters and returns their distances to query
func (idx *Index) findProbeClustersWithDistances(query []float32) ([]int, []float32, error) {
	if !idx.trained {
		return nil, nil, fmt.Errorf("index must be trained before search")
	}

	type clusterDistance struct {
		id       int
		distance float32
	}

	distances := make([]clusterDistance, len(idx.clusters))

	for i, cluster := range idx.clusters {
		distance := idx.distanceFunc(query, cluster.Centroid)
		distances[i] = clusterDistance{id: i, distance: distance}
	}

	// Sort by distance and take top NProbes
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	// Use adaptive probe count if enabled, otherwise use configured value
	var probeCount int
	if idx.adaptiveMode {
		idx.searchStats.mutex.RLock()
		probeCount = min(idx.searchStats.currentProbes, len(distances))
		idx.searchStats.mutex.RUnlock()
	} else {
		probeCount = min(idx.config.NProbes, len(distances))
	}

	probes := make([]int, probeCount)
	probeDists := make([]float32, probeCount)

	for i := 0; i < probeCount; i++ {
		probes[i] = distances[i].id
		probeDists[i] = distances[i].distance
	}

	return probes, probeDists, nil
}

// EnableAdaptiveSearch enables adaptive probe count adjustment based on performance
func (idx *Index) EnableAdaptiveSearch() {
	idx.mutex.Lock()
	defer idx.mutex.Unlock()
	idx.adaptiveMode = true
}

// DisableAdaptiveSearch disables adaptive probe count adjustment
func (idx *Index) DisableAdaptiveSearch() {
	idx.mutex.Lock()
	defer idx.mutex.Unlock()
	idx.adaptiveMode = false
}

// GetSearchStats returns current search statistics
func (idx *Index) GetSearchStats() SearchStats {
	idx.searchStats.mutex.RLock()
	defer idx.searchStats.mutex.RUnlock()
	return *idx.searchStats
}

// adjustProbeCount adaptively adjusts the number of probes based on search performance
func (idx *Index) adjustProbeCount() {
	if !idx.adaptiveMode {
		return
	}

	idx.searchStats.mutex.Lock()
	defer idx.searchStats.mutex.Unlock()

	// Only adjust every 100 searches or after 30 seconds
	if idx.searchStats.totalSearches%100 != 0 &&
		time.Since(idx.searchStats.lastAdjustment) < 30*time.Second {
		return
	}

	if idx.searchStats.totalSearches < 10 {
		return // Need more data
	}

	avgLatencyMs := float64(idx.searchStats.totalLatencyMs) / float64(idx.searchStats.totalSearches)
	avgAccuracy := idx.searchStats.accuracySum / float64(idx.searchStats.totalSearches)

	// Target: < 50ms latency, > 0.9 accuracy
	targetLatencyMs := 50.0
	targetAccuracy := 0.9

	currentProbes := idx.searchStats.currentProbes
	newProbes := currentProbes

	if avgLatencyMs > targetLatencyMs && avgAccuracy > targetAccuracy {
		// Too slow but accurate - reduce probes
		newProbes = max(1, currentProbes-1)
	} else if avgLatencyMs < targetLatencyMs*0.5 && avgAccuracy < targetAccuracy {
		// Fast but inaccurate - increase probes
		newProbes = min(idx.config.NClusters, currentProbes+1)
	}

	if newProbes != currentProbes {
		idx.searchStats.currentProbes = newProbes
		idx.searchStats.lastAdjustment = time.Now()

		// Reset stats for next adjustment period
		idx.searchStats.totalSearches = 0
		idx.searchStats.totalLatencyMs = 0
		idx.searchStats.accuracySum = 0
	}
}

// recordSearchStats records statistics for a search operation
func (idx *Index) recordSearchStats(latencyMs int64, accuracy float64) {
	if !idx.adaptiveMode {
		return
	}

	idx.searchStats.mutex.Lock()
	defer idx.searchStats.mutex.Unlock()

	idx.searchStats.totalSearches++
	idx.searchStats.totalLatencyMs += latencyMs
	idx.searchStats.accuracySum += accuracy
}

// Delete removes a vector entry from the index
func (idx *Index) Delete(ctx context.Context, id string) error {
	if id == "" {
		return fmt.Errorf("id cannot be empty")
	}

	idx.mutex.Lock()
	defer idx.mutex.Unlock()

	// Search all clusters for the entry
	for _, cluster := range idx.clusters {
		cluster.mutex.Lock()

		for i, entry := range cluster.Entries {
			if entry.ID == id {
				// Remove entry by swapping with last element
				cluster.Entries[i] = cluster.Entries[len(cluster.Entries)-1]
				cluster.Entries = cluster.Entries[:len(cluster.Entries)-1]

				// Remove compressed vector if it exists
				delete(cluster.CompressedVectors, id)

				cluster.mutex.Unlock()

				idx.size--
				return nil
			}
		}

		cluster.mutex.Unlock()
	}

	return fmt.Errorf("entry with id %s not found", id)
}

// Size returns the number of vectors in the index
func (idx *Index) Size() int {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	return idx.size
}

// MemoryUsage returns the estimated memory usage in bytes
func (idx *Index) MemoryUsage() int64 {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()

	var usage int64

	// Cluster centroids
	usage += int64(len(idx.clusters) * idx.config.Dimension * 4) // float32 = 4 bytes

	// Vector entries and compressed vectors
	for _, cluster := range idx.clusters {
		cluster.mutex.RLock()
		for _, entry := range cluster.Entries {
			// Vector data (original)
			usage += int64(len(entry.Vector) * 4) // float32 = 4 bytes
			// ID string (approximate)
			usage += int64(len(entry.ID))
			// Metadata (approximate)
			usage += int64(len(entry.Metadata) * 50) // Rough estimate
		}

		// Compressed vectors storage
		for _, compressed := range cluster.CompressedVectors {
			usage += int64(len(compressed))
		}

		cluster.mutex.RUnlock()
	}

	// Quantizer memory usage
	if idx.quantizer != nil {
		usage += idx.quantizer.MemoryUsage()
	}

	return usage
}

// Close closes the index and releases resources
func (idx *Index) Close() error {
	idx.mutex.Lock()
	defer idx.mutex.Unlock()

	// Clear all clusters
	for _, cluster := range idx.clusters {
		cluster.mutex.Lock()
		cluster.Entries = nil
		cluster.Centroid = nil
		cluster.CompressedVectors = nil
		cluster.mutex.Unlock()
	}

	idx.clusters = nil
	idx.quantizer = nil
	idx.trained = false
	idx.size = 0

	return nil
}

// SaveToDisk saves the index to disk (placeholder implementation)
func (idx *Index) SaveToDisk(ctx context.Context, path string) error {
	// TODO: Implement persistence for IVF-PQ index
	return fmt.Errorf("persistence not yet implemented for IVF-PQ index")
}

// LoadFromDisk loads the index from disk (placeholder implementation)
func (idx *Index) LoadFromDisk(ctx context.Context, path string) error {
	// TODO: Implement persistence for IVF-PQ index
	return fmt.Errorf("persistence not yet implemented for IVF-PQ index")
}

// PersistenceMetadata holds metadata about persisted index (placeholder)
type PersistenceMetadata struct {
	// TODO: Define persistence metadata structure
}

// GetPersistenceMetadata returns persistence metadata (placeholder implementation)
func (idx *Index) GetPersistenceMetadata() *PersistenceMetadata {
	// TODO: Implement persistence metadata for IVF-PQ index
	return nil
}
