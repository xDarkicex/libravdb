//go:build !windows

package fsdurability

import (
	"errors"
	"os"
	"path/filepath"
)

// ReplaceFile atomically replaces newPath with oldPath. The caller must sync
// the parent directory after the replacement has been incorporated into its
// live state.
func ReplaceFile(oldPath, newPath string) error {
	return os.Rename(oldPath, newPath)
}

// SyncParent makes a previously synced file creation, removal, or rename
// durable in the containing directory.
func SyncParent(path string) error {
	dir, err := os.Open(filepath.Dir(filepath.Clean(path)))
	if err != nil {
		return err
	}
	syncErr := dir.Sync()
	closeErr := dir.Close()
	return errors.Join(syncErr, closeErr)
}
