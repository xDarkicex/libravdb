package libravdb

import (
	"context"
	"fmt"
	"os"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage/singlefile"
)

// Migrate converts a v1 database file to the new v2 format.
// It creates a .v1.bak backup of the original database to prevent data loss.
func Migrate(ctx context.Context, path string) error {
	v1Engine, err := singlefile.OpenV1(path)
	if err != nil {
		return fmt.Errorf("failed to open v1 database: %w", err)
	}

	migratingPath := path + ".migrating"
	stagedPath := path + ".staged"
	backupPath := path + ".v1.bak"

	// Clean up any stale leftovers from a previous interrupted migration.
	os.Remove(migratingPath)
	os.Remove(stagedPath)

	v2DB, err := Open(WithStoragePath(migratingPath))
	if err != nil {
		v1Engine.Close()
		return fmt.Errorf("failed to create v2 database: %w", err)
	}

	colNames, err := v1Engine.ListCollections()
	if err != nil {
		v1Engine.Close()
		v2DB.Close()
		return fmt.Errorf("failed to list collections: %w", err)
	}

	for _, name := range colNames {
		err := func() error {
			colInfo, config, err := v1Engine.GetCollectionWithConfig(name)
			if err != nil {
				return fmt.Errorf("failed to read collection info %s: %w", name, err)
			}
			defer colInfo.Close()

			v2Col, err := v2DB.CreateCollection(ctx, name, func(c *CollectionConfig) error {
				c.Dimension = config.Dimension
				c.Metric = DistanceMetric(config.Metric)
				c.IndexType = IndexType(config.IndexType)
				c.M = config.M
				c.EfConstruction = config.EfConstruction
				c.EfSearch = config.EfSearch
				c.NClusters = config.NClusters
				c.NProbes = config.NProbes
				c.ML = config.ML
				c.Version = config.Version
				c.RawVectorStore = config.RawVectorStore
				c.RawStoreCap = config.RawStoreCap
				return nil
			})
			if err != nil {
				return fmt.Errorf("failed to create v2 collection %s: %w", name, err)
			}

			var batch []VectorEntry
			err = colInfo.Iterate(ctx, func(entry *index.VectorEntry) error {
				batch = append(batch, VectorEntry{
					ID:       entry.ID,
					Vector:   entry.Vector,
					Metadata: entry.Metadata,
				})
				if len(batch) >= 1000 {
					if err := v2Col.InsertBatch(ctx, batch); err != nil {
						return err
					}
					batch = batch[:0]
				}
				return nil
			})
			if err != nil {
				return fmt.Errorf("failed to iterate collection %s: %w", name, err)
			}

			if len(batch) > 0 {
				if err := v2Col.InsertBatch(ctx, batch); err != nil {
					return fmt.Errorf("failed to insert final batch into collection %s: %w", name, err)
				}
			}
			return nil
		}()
		if err != nil {
			v1Engine.Close()
			v2DB.Close()
			return err
		}
	}

	if err := v1Engine.Close(); err != nil {
		v2DB.Close()
		return fmt.Errorf("failed to close v1 engine: %w", err)
	}

	if err := v2DB.Close(); err != nil {
		return fmt.Errorf("failed to close v2 database: %w", err)
	}

	// Three-step atomic swap so an interrupted migration is resumable:
	// 1. migrating → staged (mark ready)
	// 2. path → backup (preserve original)
	// 3. staged → path (activate)
	if err := os.Rename(migratingPath, stagedPath); err != nil {
		return fmt.Errorf("failed to stage migration: %w", err)
	}
	if err := os.Rename(path, backupPath); err != nil {
		os.Rename(stagedPath, migratingPath)
		return fmt.Errorf("failed to backup v1 database: %w", err)
	}
	if err := os.Rename(stagedPath, path); err != nil {
		os.Rename(backupPath, path)
		return fmt.Errorf("failed to activate migration: %w", err)
	}
	return nil
}

// recoverMigrate cleans up leftover staged/migrating files from a previous
// interrupted migration. Called by Open when detecting a v1 file.
func recoverMigrate(path string) {
	os.Remove(path + ".staged")
	os.Remove(path + ".migrating")
}
