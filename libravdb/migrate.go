package libravdb

import (
	"context"
	"fmt"
	"os"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage/fsdurability"
	"github.com/xDarkicex/libravdb/internal/storage/singlefile"
)

// Migrate converts a v1 database file to the new v2 format.
// It creates a .v1.bak backup of the original database to prevent data loss.
func Migrate(ctx context.Context, path string) error {
	v1Engine, err := singlefile.OpenV1(path)
	if err != nil {
		return fmt.Errorf("failed to open v1 database: %w", err)
	}
	defer v1Engine.Close()

	migratingPath := path + ".migrating"
	stagedPath := path + ".staged"
	backupPath := path + ".v1.bak"

	// Clean up any stale leftovers from a previous interrupted migration.
	if err := os.Remove(migratingPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("cleanup stale migrating file: %w", err)
	}
	if err := os.Remove(stagedPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("cleanup stale staged file: %w", err)
	}

	// Count collections before creating the v2 database so the limit
	// can be sized to fit. Production databases routinely exceed the
	// default 100-collection cap, especially session-scoped stores.
	colNames, err := v1Engine.ListCollections()
	if err != nil {
		return fmt.Errorf("failed to list collections: %w", err)
	}
	limit := len(colNames) + 1000 // headroom for future collections

	v2DB, err := Open(WithStoragePath(migratingPath), WithMaxCollections(limit))
	if err != nil {
		return fmt.Errorf("failed to create v2 database: %w", err)
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
				c.IDMapCapacity = config.IDMapCapacity
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
			v2DB.Close()
			return err
		}
	}

	if err := v2DB.Close(); err != nil {
		return fmt.Errorf("failed to close v2 database: %w", err)
	}

	// Three-step atomic swap so an interrupted migration is resumable:
	// 1. migrating → staged (mark ready)
	// 2. path → backup (preserve original)
	// 3. staged → path (activate)
	if err := replaceFileDurably(migratingPath, stagedPath); err != nil {
		return fmt.Errorf("failed to stage migration: %w", err)
	}
	if err := replaceFileDurably(path, backupPath); err != nil {
		_ = replaceFileDurably(stagedPath, migratingPath)
		return fmt.Errorf("failed to backup v1 database: %w", err)
	}
	if err := replaceFileDurably(stagedPath, path); err != nil {
		_ = replaceFileDurably(backupPath, path)
		return fmt.Errorf("failed to activate migration: %w", err)
	}
	return nil
}

func replaceFileDurably(oldPath, newPath string) error {
	if err := fsdurability.ReplaceFile(oldPath, newPath); err != nil {
		return err
	}
	return fsdurability.SyncParent(newPath)
}

// recoverMigrate cleans up leftover files from a previous interrupted
// migration. If the swap was interrupted after backup but before activation
// (staged + backup both exist), it finishes the activation by renaming
// staged → path. Otherwise it just removes stale temp files.
func recoverMigrate(path string) {
	stagedPath := path + ".staged"
	backupPath := path + ".v1.bak"
	migratingPath := path + ".migrating"

	stagedExists := fileExists(stagedPath)
	backupExists := fileExists(backupPath)

	if stagedExists && backupExists {
		// Interrupted after backup was created but before staged→path
		// activation completed. Finish the migration.
		_ = replaceFileDurably(stagedPath, path)
		os.Remove(migratingPath)
		return
	}
	os.Remove(stagedPath)
	os.Remove(migratingPath)
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
