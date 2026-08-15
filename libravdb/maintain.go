package libravdb

import (
	"context"
	"fmt"
	"github.com/xDarkicex/libravdb/internal/index/hnsw"
	"hash/crc32"
)

// Maintain sweeps the graph topology of the specified collection to cluster nodes into
// logical communities. It operates statelessly using the Louvain method, bounded by the given budget.
// The computed community bounds are persisted in their own chunk.
func (db *Database) Maintain(ctx context.Context, collectionName string, budget int) error {
	col, err := db.GetCollection(collectionName)
	if err != nil {
		return fmt.Errorf("collection %q not found: %w", collectionName, err)
	}

	// Access the underlying index — maintain currently only supports HNSW
	idx := col.GetIndex()
	if idx == nil {
		return fmt.Errorf("collection %q has no index", collectionName)
	}

	type communityIndex interface {
		ComputeCommunities(ctx context.Context, budget int) (*hnsw.CommunityRegistry, error)
		SetCommunities(registry *hnsw.CommunityRegistry)
	}
	hnswIdx, ok := idx.(communityIndex)
	if !ok {
		return fmt.Errorf("maintain currently only supports HNSW indexes")
	}

	// Compute communities via stateless bounded refinement
	bounds, err := hnswIdx.ComputeCommunities(ctx, budget)
	if err != nil {
		return fmt.Errorf("failed to compute communities: %w", err)
	}

	// Serialize for the engine
	chunk, err := bounds.Serialize()
	if err != nil {
		return fmt.Errorf("failed to serialize community registry: %w", err)
	}

	// Compute real checksum
	checksum := crc32.ChecksumIEEE(chunk)

	// 1. Give it to the engine for the next checkpoint
	if sfEngine, ok := db.storage.(interface {
		SetPendingCommunityState(chunk []byte, checksum uint32)
	}); ok {
		sfEngine.SetPendingCommunityState(chunk, checksum)
	}

	// 2. Set it on the live index for queries
	hnswIdx.SetCommunities(bounds)

	return nil
}
