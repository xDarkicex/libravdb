//go:build !linux && !darwin && !freebsd && !netbsd && !openbsd && !windows

package memory

import (
	"errors"
	"fmt"
)

// ErrPlatformUnsupported is returned when mmap is not supported on the platform
var ErrPlatformUnsupported = errors.New("memory mapping is not supported on this platform (Linux, macOS, BSD, and Windows only)")

// MemoryMap represents a memory-mapped file (stub for unsupported platforms)
type MemoryMap struct {
	mu       struct{}
	data     []byte
	size     int64
	path     string
	readOnly bool
}

// NewMemoryMap returns an error on unsupported platforms
func NewMemoryMap(path string, size int64, readOnly bool) (*MemoryMap, error) {
	return nil, ErrPlatformUnsupported
}

// Data returns nil data
func (m *MemoryMap) Data() []byte {
	return nil
}

// Size returns 0
func (m *MemoryMap) Size() int64 {
	return 0
}

// Path returns empty string
func (m *MemoryMap) Path() string {
	return ""
}

// IsReadOnly returns the read-only flag
func (m *MemoryMap) IsReadOnly() bool {
	return m.readOnly
}

// Sync returns unsupported error
func (m *MemoryMap) Sync() error {
	return ErrPlatformUnsupported
}

// Close returns nil
func (m *MemoryMap) Close() error {
	return nil
}

// Resize returns unsupported error
func (m *MemoryMap) Resize(newSize int64) error {
	return ErrPlatformUnsupported
}

// MemoryMapManager manages multiple memory mappings (stub)
type MemoryMapManager struct {
	mappings map[string]*MemoryMap
	basePath string
}

// NewMemoryMapManager creates a new manager
func NewMemoryMapManager(basePath string) *MemoryMapManager {
	return &MemoryMapManager{
		mappings: make(map[string]*MemoryMap),
		basePath: basePath,
	}
}

// CreateMapping returns unsupported error
func (m *MemoryMapManager) CreateMapping(name string, size int64, readOnly bool) (*MemoryMap, error) {
	return nil, ErrPlatformUnsupported
}

// GetMapping returns nil, false
func (m *MemoryMapManager) GetMapping(name string) (*MemoryMap, bool) {
	return nil, false
}

// RemoveMapping returns unsupported error
func (m *MemoryMapManager) RemoveMapping(name string) error {
	return ErrPlatformUnsupported
}

// TotalSize returns 0
func (m *MemoryMapManager) TotalSize() int64 {
	return 0
}

// Close returns nil
func (m *MemoryMapManager) Close() error {
	return nil
}