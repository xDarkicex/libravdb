package memory

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMemoryMap_Basic(t *testing.T) {
	// Create temporary directory
	tmpDir, err := os.MkdirTemp("", "mmap_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Test file path
	testPath := filepath.Join(tmpDir, "test.mmap")
	testSize := int64(1024)

	// Create memory map
	mmap, err := NewMemoryMap(testPath, testSize, false)
	if err != nil {
		t.Fatalf("Failed to create memory map: %v", err)
	}
	defer mmap.Close()

	// Test basic properties
	if mmap.Size() != testSize {
		t.Errorf("Expected size %d, got %d", testSize, mmap.Size())
	}

	if mmap.Path() != testPath {
		t.Errorf("Expected path %s, got %s", testPath, mmap.Path())
	}

	if mmap.IsReadOnly() {
		t.Error("Expected read-write mapping, got read-only")
	}

	// Test data access
	data := mmap.Data()
	if len(data) != int(testSize) {
		t.Errorf("Expected data length %d, got %d", testSize, len(data))
	}

	// Write some data
	copy(data[:4], []byte("test"))

	// Sync to disk
	if err := mmap.Sync(); err != nil {
		t.Errorf("Failed to sync: %v", err)
	}
}

func TestMemoryMap_ReadOnly(t *testing.T) {
	// Create temporary directory
	tmpDir, err := os.MkdirTemp("", "mmap_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create test file with data
	testPath := filepath.Join(tmpDir, "test.mmap")
	testData := []byte("Hello, World!")

	if err := os.WriteFile(testPath, testData, 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Create read-only memory map
	mmap, err := NewMemoryMap(testPath, 0, true) // Size 0 means use file size
	if err != nil {
		t.Fatalf("Failed to create memory map: %v", err)
	}
	defer mmap.Close()

	// Test properties
	if !mmap.IsReadOnly() {
		t.Error("Expected read-only mapping, got read-write")
	}

	if mmap.Size() != int64(len(testData)) {
		t.Errorf("Expected size %d, got %d", len(testData), mmap.Size())
	}

	// Test data reading
	data := mmap.Data()
	if string(data) != string(testData) {
		t.Errorf("Expected data %s, got %s", testData, data)
	}
}

func TestMemoryMap_Resize(t *testing.T) {
	// Create temporary directory
	tmpDir, err := os.MkdirTemp("", "mmap_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Test file path
	testPath := filepath.Join(tmpDir, "test.mmap")
	initialSize := int64(1024)
	newSize := int64(2048)

	// Create memory map
	mmap, err := NewMemoryMap(testPath, initialSize, false)
	if err != nil {
		t.Fatalf("Failed to create memory map: %v", err)
	}
	defer mmap.Close()

	// Write some data
	data := mmap.Data()
	copy(data[:4], []byte("test"))

	// Resize
	if err := mmap.Resize(newSize); err != nil {
		t.Fatalf("Failed to resize: %v", err)
	}

	// Check new size
	if mmap.Size() != newSize {
		t.Errorf("Expected size %d after resize, got %d", newSize, mmap.Size())
	}

	// Check data is still there
	data = mmap.Data()
	if string(data[:4]) != "test" {
		t.Errorf("Data lost after resize: expected 'test', got %s", data[:4])
	}
}

func TestMemoryMapManager(t *testing.T) {
	// Create temporary directory
	tmpDir, err := os.MkdirTemp("", "mmap_manager_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create manager
	manager := NewMemoryMapManager(tmpDir)
	defer manager.Close()

	// Create mapping
	mapping1, err := manager.CreateMapping("test1", 1024, false)
	if err != nil {
		t.Fatalf("Failed to create mapping: %v", err)
	}

	// Test retrieval
	retrieved, exists := manager.GetMapping("test1")
	if !exists {
		t.Error("Mapping not found")
	}
	if retrieved != mapping1 {
		t.Error("Retrieved mapping is different")
	}

	// Create another mapping
	mapping2, err := manager.CreateMapping("test2", 2048, false)
	if err != nil {
		t.Fatalf("Failed to create second mapping: %v", err)
	}

	// Test total size
	expectedTotal := mapping1.Size() + mapping2.Size()
	if manager.TotalSize() != expectedTotal {
		t.Errorf("Expected total size %d, got %d", expectedTotal, manager.TotalSize())
	}

	// Remove mapping
	if err := manager.RemoveMapping("test1"); err != nil {
		t.Errorf("Failed to remove mapping: %v", err)
	}

	// Check it's gone
	_, exists = manager.GetMapping("test1")
	if exists {
		t.Error("Mapping should have been removed")
	}

	// Test duplicate creation
	_, err = manager.CreateMapping("test2", 1024, false)
	if err == nil {
		t.Error("Expected error for duplicate mapping name")
	}
}

func TestMemoryMap_ErrorCases(t *testing.T) {
	// Test invalid path
	_, err := NewMemoryMap("/invalid/path/test.mmap", 1024, false)
	if err == nil {
		t.Error("Expected error for invalid path")
	}

	// Test zero size for new file
	tmpDir, err := os.MkdirTemp("", "mmap_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testPath := filepath.Join(tmpDir, "empty.mmap")
	_, err = NewMemoryMap(testPath, 0, false)
	if err == nil {
		t.Error("Expected error for zero size new file")
	}
}

func TestMemoryMap_CloseOperations(t *testing.T) {
	// Create temporary directory
	tmpDir, err := os.MkdirTemp("", "mmap_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create memory map
	testPath := filepath.Join(tmpDir, "test.mmap")
	mmap, err := NewMemoryMap(testPath, 1024, false)
	if err != nil {
		t.Fatalf("Failed to create memory map: %v", err)
	}

	// Close it
	if err := mmap.Close(); err != nil {
		t.Errorf("Failed to close memory map: %v", err)
	}

	// Test operations on closed map
	if err := mmap.Sync(); err == nil {
		t.Error("Expected error for sync on closed map")
	}

	if err := mmap.Resize(2048); err == nil {
		t.Error("Expected error for resize on closed map")
	}
}
