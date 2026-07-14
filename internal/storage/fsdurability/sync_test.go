package fsdurability

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReplaceFilePublishesSyncedReplacement(t *testing.T) {
	dir := t.TempDir()
	destination := filepath.Join(dir, "database.libravdb")
	temporary := destination + ".compact"

	if err := os.WriteFile(destination, []byte("old"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(temporary, []byte("new"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := ReplaceFile(temporary, destination); err != nil {
		t.Fatal(err)
	}
	if err := SyncParent(destination); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(destination)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "new" {
		t.Fatalf("replacement contents = %q, want new", data)
	}
	if _, err := os.Stat(temporary); !os.IsNotExist(err) {
		t.Fatalf("temporary file still exists after replacement: %v", err)
	}
}
