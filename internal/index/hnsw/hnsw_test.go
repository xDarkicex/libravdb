package hnsw

import (
	"context"
	"fmt"
	"math"
	"reflect"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/memory"
)

func TestAppendUniqueLinkPreventsDuplicates(t *testing.T) {
	arena, err := memory.NewArena(4096, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer arena.Free()

	const baseM = 8
	capacity := linkArrayCapacity(baseM, 0)
	data, err := arena.Alloc(uint64(capacity) * uint64(unsafe.Sizeof(uint32(0))))
	if err != nil {
		t.Fatal(err)
	}
	links := unsafe.Slice((*uint32)(data), capacity)
	for i := range links {
		links[i] = SentinelNodeID
	}
	links[0], links[1] = 1, 2

	node := &Node{}
	node.Links[0] = &links[0]
	atomic.StoreUint32(&node.LinkCounts[0], 2)

	idx := &Index{config: &Config{M: baseM}}
	if added := idx.appendUniqueLink(node, baseM, 0, 2); added {
		t.Fatal("expected duplicate link append to be skipped")
	}

	if count := len(idx.getNodeLinks(node, 0)); count != 2 {
		t.Errorf("Expected 2 links, got %d", count)
	}

	if added := idx.appendUniqueLink(node, baseM, 0, 3); !added {
		t.Fatal("expected unique link to be appended")
	}

	if count := len(idx.getNodeLinks(node, 0)); count != 3 {
		t.Errorf("Expected 3 links after duplicate append, got %d", count)
	}
}

func TestIDMapSizingSupportsGrowthBeyondInitialCapacity(t *testing.T) {
	config := &Config{}
	if got := config.idMapCapacity(); got != defaultIDMapCapacity {
		t.Fatalf("default ID map capacity = %d, want %d", got, defaultIDMapCapacity)
	}
	if got := config.idMapKeyBytes(); got != defaultIDMapKeyBytes {
		t.Fatalf("default ID key bytes = %d, want %d", got, defaultIDMapKeyBytes)
	}

	config.IDMapCapacity = 50_000
	if got, want := config.idMapKeyBytes(), uint64(50_000*64); got != want {
		t.Fatalf("configured ID key bytes = %d, want %d", got, want)
	}

	config.Dimension = 4
	config.M = 4
	config.EfConstruction = 8
	config.EfSearch = 8
	config.Metric = util.L2Distance
	config.IDMapCapacity = maxNodeCapacity + 1
	if err := config.validate(); err == nil {
		t.Fatal("ID map capacity beyond the node registry limit was accepted")
	}
}

func TestGlobalStateStoresZeroOrdinalZeroLevelEntryPoint(t *testing.T) {
	idx := &Index{nodes: newSegmentedNodeArray()}
	node := &Node{Ordinal: 0, Level: 0}
	idx.nodes.Set(0, node)

	if !idx.updateEntryPointCAS(node) {
		t.Fatal("expected node 0 level 0 to become entry point")
	}
	if got := idx.getEntryPoint(); got != node {
		t.Fatalf("entry point mismatch: got %p want %p", got, node)
	}
	if got := idx.getMaxLevel(); got != 0 {
		t.Fatalf("max level mismatch: got %d want 0", got)
	}
}

func TestInitializeEntryPointCASDoesNotReplaceExistingEntry(t *testing.T) {
	idx := &Index{nodes: newSegmentedNodeArray()}
	defer idx.nodes.Close()
	first := &Node{Ordinal: 0, Level: 0}
	higher := &Node{Ordinal: 1, Level: 4}
	idx.nodes.Set(first.Ordinal, first)
	idx.nodes.Set(higher.Ordinal, higher)

	if !idx.initializeEntryPointCAS(first) {
		t.Fatal("first node did not initialize the entry point")
	}
	if idx.initializeEntryPointCAS(higher) {
		t.Fatal("empty-state initialization replaced an existing entry point")
	}
	if got := idx.getEntryPoint(); got != first {
		t.Fatalf("entry point = %p, want first node %p", got, first)
	}
}

func TestGenerateLevelUsesExponentialDistribution(t *testing.T) {
	config := &Config{
		Dimension:      4,
		M:              16,
		EfConstruction: 16,
		EfSearch:       8,
		ML:             1.0,
		Metric:         util.L2Distance,
		RandomSeed:     42,
	}
	idx, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("NewHNSW: %v", err)
	}
	defer idx.Close()

	const samples = 1000
	level0 := 0
	maxLevel := 0
	for i := range samples {
		level := idx.generateLevel(uint32(i))
		if level == 0 {
			level0++
		}
		if level > maxLevel {
			maxLevel = level
		}
	}

	if level0 < samples/2 {
		t.Fatalf("level generator collapsed upward: level0=%d/%d", level0, samples)
	}
	if maxLevel >= MaxLevel-1 {
		t.Fatalf("unexpected max-level saturation with ML=1.0: maxLevel=%d", maxLevel)
	}

	expectedLevel0 := 1 - math.Exp(-1/config.ML)
	gotLevel0 := float64(level0) / samples
	if math.Abs(gotLevel0-expectedLevel0) > 0.10 {
		t.Fatalf("level0 frequency got %.3f want roughly %.3f", gotLevel0, expectedLevel0)
	}
}

func TestInsertedEdgesOnlyTargetNodesPresentAtLevel(t *testing.T) {
	config := &Config{
		Dimension:      16,
		M:              8,
		EfConstruction: 32,
		EfSearch:       16,
		ML:             1.0,
		Metric:         util.L2Distance,
		RandomSeed:     7,
	}
	index, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("NewHNSW: %v", err)
	}
	defer index.Close()

	ctx := context.Background()
	for i, vec := range generateTestVectors(300, 16) {
		if err := index.Insert(ctx, &VectorEntry{ID: fmt.Sprintf("vec_%d", i), Vector: vec}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	for i := 0; i < index.nodes.Len(); i++ {
		node := index.nodes.Get(uint32(i))
		if node == nil {
			continue
		}
		for level := 0; level <= node.Level; level++ {
			for _, neighborID := range index.getNodeLinks(node, level) {
				neighbor := index.nodes.Get(neighborID)
				if neighbor == nil {
					t.Fatalf("node %d level %d links missing neighbor %d", i, level, neighborID)
				}
				if neighbor.Level < level {
					t.Fatalf("node %d level %d links neighbor %d with level %d", i, level, neighborID, neighbor.Level)
				}
			}
		}
	}
}

func TestPruneDroppedBacklinkRemovesForwardEdgeOnly(t *testing.T) {
	index := &Index{
		config: &Config{M: 4},
		nodes:  newSegmentedNodeArray(),
	}
	oldNode := &Node{Ordinal: 1, Level: 0}
	newNode := &Node{Ordinal: 2, Level: 0}
	oldLinks := sentinelLinks(levelMaxLinks(index.config.M, 0) + levelOverflowSlack(levelMaxLinks(index.config.M, 0)))
	oldBacklinks := sentinelLinks(levelMaxLinks(index.config.M, 0) + levelOverflowSlack(levelMaxLinks(index.config.M, 0)))
	newLinks := sentinelLinks(levelMaxLinks(index.config.M, 0) + levelOverflowSlack(levelMaxLinks(index.config.M, 0)))
	newBacklinks := sentinelLinks(levelMaxLinks(index.config.M, 0) + levelOverflowSlack(levelMaxLinks(index.config.M, 0)))
	oldLinks[0] = newNode.Ordinal
	oldBacklinks[0] = newNode.Ordinal
	newLinks[0] = oldNode.Ordinal
	newBacklinks[0] = oldNode.Ordinal
	oldNode.Links[0] = &oldLinks[0]
	oldNode.Backlinks[0] = &oldBacklinks[0]
	newNode.Links[0] = &newLinks[0]
	newNode.Backlinks[0] = &newBacklinks[0]
	atomic.StoreUint32(&oldNode.LinkCounts[0], 1)
	atomic.StoreUint32(&oldNode.BacklinkCounts[0], 1)
	atomic.StoreUint32(&newNode.LinkCounts[0], 1)
	atomic.StoreUint32(&newNode.BacklinkCounts[0], 1)
	index.nodes.Set(oldNode.Ordinal, oldNode)
	index.nodes.Set(newNode.Ordinal, newNode)

	selector := NewNeighborSelector(index.config.M, 2.0)
	selector.removeDroppedBacklinks(oldNode.Ordinal, 0, []uint32{newNode.Ordinal}, nil, index)

	if containsUint32(index.getNodeLinks(oldNode, 0), newNode.Ordinal) {
		t.Fatal("expected pruned forward edge old -> new to be removed")
	}
	if containsUint32(index.getNodeBacklinks(newNode, 0), oldNode.Ordinal) {
		t.Fatal("expected new node backlink from old node to be removed")
	}
	if !containsUint32(index.getNodeLinks(newNode, 0), oldNode.Ordinal) {
		t.Fatal("new node's construction edge new -> old was removed")
	}
}

func sentinelLinks(size int) []uint32 {
	links := make([]uint32, size)
	for i := range links {
		links[i] = SentinelNodeID
	}
	return links
}

func containsUint32(values []uint32, want uint32) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func TestHNSWDeterministicFixedSeedSearchResults(t *testing.T) {
	buildAndSearch := func() []string {
		t.Helper()

		config := &Config{
			Dimension:      64,
			M:              16,
			EfConstruction: 100,
			EfSearch:       50,
			ML:             1.0,
			Metric:         util.L2Distance,
			RandomSeed:     42,
		}

		index, err := NewHNSW(config)
		if err != nil {
			t.Fatalf("failed to create HNSW index: %v", err)
		}
		defer index.Close()

		vectors := generateTestVectors(200, 64)
		for i, vec := range vectors {
			err := index.Insert(context.Background(), &VectorEntry{
				ID:       fmt.Sprintf("vec_%d", i),
				Vector:   vec,
				Metadata: map[string]interface{}{"index": i},
			})
			if err != nil {
				t.Fatalf("failed to insert vector %d: %v", i, err)
			}
		}

		results, err := index.Search(context.Background(), vectors[0], 10, nil)
		if err != nil {
			t.Fatalf("failed to search deterministic index: %v", err)
		}

		ids := make([]string, len(results))
		for i, result := range results {
			ids[i] = result.ID
		}
		return ids
	}

	first := buildAndSearch()
	second := buildAndSearch()
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("expected deterministic results with fixed seed, got %v and %v", first, second)
	}
}
