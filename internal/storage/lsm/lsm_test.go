package lsm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNormalizeBasePathForMemoryPrefix(t *testing.T) {
	got := normalizeBasePath(":memory:basic")

	if strings.Contains(got, ":") {
		t.Fatalf("expected normalized path to exclude colons, got %q", got)
	}
	if !strings.Contains(got, filepath.Join("libravdb-memory", "basic")) {
		t.Fatalf("expected normalized path to target temp libravdb-memory dir, got %q", got)
	}
}

func TestNormalizeBasePathLeavesNormalPathsAlone(t *testing.T) {
	input := filepath.Join(".", "testdata")
	if got := normalizeBasePath(input); got != input {
		t.Fatalf("expected %q to remain unchanged, got %q", input, got)
	}
}

func TestNewCreatesSafeMemoryBackedStorage(t *testing.T) {
	engine, err := New(":memory:perf")
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer engine.Close()

	lsmEngine, ok := engine.(*Engine)
	if !ok {
		t.Fatalf("expected *Engine, got %T", engine)
	}
	if strings.Contains(lsmEngine.basePath, ":") {
		t.Fatalf("expected safe basePath without colons, got %q", lsmEngine.basePath)
	}
	if _, err := os.Stat(lsmEngine.basePath); err != nil {
		t.Fatalf("expected normalized storage path to exist: %v", err)
	}
}
