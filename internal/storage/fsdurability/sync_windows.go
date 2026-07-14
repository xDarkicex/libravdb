//go:build windows

package fsdurability

import "golang.org/x/sys/windows"

// ReplaceFile asks Windows to flush the move before returning. Windows does
// not expose the POSIX directory-fsync model through os.File.Sync.
func ReplaceFile(oldPath, newPath string) error {
	oldName, err := windows.UTF16PtrFromString(oldPath)
	if err != nil {
		return err
	}
	newName, err := windows.UTF16PtrFromString(newPath)
	if err != nil {
		return err
	}
	return windows.MoveFileEx(
		oldName,
		newName,
		windows.MOVEFILE_REPLACE_EXISTING|windows.MOVEFILE_WRITE_THROUGH,
	)
}

// SyncParent is covered by ReplaceFile's write-through semantics for renames.
// File.Sync already flushes newly created file contents; there is no portable
// directory-handle equivalent to POSIX fsync through Go's os package.
func SyncParent(string) error { return nil }
