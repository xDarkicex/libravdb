package libravdb

import (
	"fmt"
	"hash/fnv"
	"strings"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
)

// shardCount is the number of internal write lanes per collection.
// This is a private constant; not user-configurable in this patch.
const shardCount = 4

// shardSeparator is the delimiter used to create hidden shard collection names.
const shardSeparator = "__shard__"

// shard represents one independent write lane with its own storage and index.
// Each shard owns its own storage collection and index, providing true parallelism.
type shard struct {
	name    string
	storage storage.Collection
	index   index.Index
}

// shardForID returns the shard index for a given vector ID.
// Uses a stable FNV-1a hash so the same ID always routes to the same shard.
func shardForID(id string) int {
	h := fnv.New32a()
	h.Write([]byte(id))
	return int(h.Sum32() % shardCount)
}

// shardName constructs the hidden storage collection name for a given parent and shard index.
// Format: parent + "__shard__" + index
func shardName(parent string, shardIdx int) string {
	return fmt.Sprintf("%s%s%d", parent, shardSeparator, shardIdx)
}

// isShardName checks if a name is a hidden shard collection name.
// Returns the parent collection name, shard index, and ok=true if it's a shard name.
func isShardName(name string) (parent string, shardIdx int, ok bool) {
	prefix := shardSeparator
	if !strings.HasPrefix(name, prefix) {
		return "", 0, false
	}

	// Find the shard index after the separator
	idxStr := strings.TrimPrefix(name, prefix)
	if idxStr == "" {
		return "", 0, false
	}

	// Parse the shard index
	var idx int
	for _, c := range idxStr {
		if c < '0' || c > '9' {
			return "", 0, false
		}
		idx = idx*10 + int(c-'0')
	}

	return "", idx, false
}

// parseShardName parses a shard collection name and returns parent and shard index.
// Returns empty parent if not a valid shard name.
func parseShardName(name string) (parent string, shardIdx int, ok bool) {
	parts := strings.Split(name, shardSeparator)
	if len(parts) != 2 {
		return "", 0, false
	}
	parent = parts[0]
	var idx int
	for _, c := range parts[1] {
		if c < '0' || c > '9' {
			return "", 0, false
		}
		idx = idx*10 + int(c-'0')
	}
	if idx < 0 || idx >= shardCount {
		return "", 0, false
	}
	return parent, idx, true
}

// groupEntriesByShard groups entries by their target shard index.
func groupEntriesByShard(entries []*index.VectorEntry) map[int][]*index.VectorEntry {
	groups := make(map[int][]*index.VectorEntry, shardCount)
	for _, entry := range entries {
		si := shardForID(entry.ID)
		groups[si] = append(groups[si], entry)
	}
	return groups
}

// initShards creates N independent shard storage collections and indexes for a collection.
// Each shard gets its own storage collection and index for true parallelism.
func (c *Collection) initShards(storageEngine storage.Engine, shardStorageNames []string, engineConfig *storage.CollectionConfig) error {
	c.shards = make([]shard, shardCount)

	for i := 0; i < shardCount; i++ {
		// Create shard storage collection
		shardStorage, err := storageEngine.CreateCollection(shardStorageNames[i], engineConfig)
		if err != nil {
			return fmt.Errorf("failed to create shard %d storage: %w", i, err)
		}

		// Create index provider for this shard's storage
		provider, _ := shardStorage.(interface {
			GetByOrdinal(uint32) ([]float32, error)
			Distance([]float32, uint32) (float32, error)
		})

		// Create index for this shard
		idx, err := createIndexForCollection(c.config, provider)
		if err != nil {
			shardStorage.Close()
			return fmt.Errorf("failed to create shard %d index: %w", i, err)
		}

		c.shards[i] = shard{
			name:    shardStorageNames[i],
			storage: shardStorage,
			index:   idx,
		}
	}

	return nil
}

// closeShards closes all shard resources.
func (c *Collection) closeShards() error {
	var errs []error
	for i := range c.shards {
		if c.shards[i].index != nil {
			if err := c.shards[i].index.Close(); err != nil {
				errs = append(errs, fmt.Errorf("shard %d index close: %w", i, err))
			}
		}
		if c.shards[i].storage != nil {
			if err := c.shards[i].storage.Close(); err != nil {
				errs = append(errs, fmt.Errorf("shard %d storage close: %w", i, err))
			}
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("errors closing shards: %v", errs)
	}
	return nil
}

// getShard returns the shard for a given vector ID.
func (c *Collection) getShard(id string) *shard {
	return &c.shards[shardForID(id)]
}

// getShardByIndex returns the shard at a given index.
func (c *Collection) getShardByIndex(idx int) *shard {
	if idx < 0 || idx >= shardCount {
		return nil
	}
	return &c.shards[idx]
}

// shardStorageNames returns the storage collection names for all shards of a parent.
func shardStorageNames(parent string) []string {
	names := make([]string, shardCount)
	for i := 0; i < shardCount; i++ {
		names[i] = shardName(parent, i)
	}
	return names
}

// isShardedCollectionName returns true if the name is a public collection name (not a shard).
func isShardedCollectionName(name string) bool {
	_, _, ok := parseShardName(name)
	return !ok
}

// collectionOwnsShard returns true if the given shard name belongs to this collection.
func (c *Collection) collectionOwnsShard(shardName string) bool {
	parent, _, ok := parseShardName(shardName)
	return ok && parent == c.name
}