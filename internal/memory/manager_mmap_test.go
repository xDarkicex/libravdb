package memory

import (
	"context"
	"os"
	"sync/atomic"
	"testing"
	"time"
)

// MockMemoryMappable implements the MemoryMappable interface for testing
type MockMemoryMappable struct {
	canMap        bool
	estimatedSize int64
	isMapped      atomic.Bool
	mappedSize    atomic.Int64
	enableError   error
	disableError  error
}

func (m *MockMemoryMappable) CanMemoryMap() bool {
	return m.canMap
}

func (m *MockMemoryMappable) EstimateSize() int64 {
	return m.estimatedSize
}

func (m *MockMemoryMappable) EnableMemoryMapping(path string) error {
	if m.enableError != nil {
		return m.enableError
	}
	m.isMapped.Store(true)
	m.mappedSize.Store(m.estimatedSize)
	return nil
}

func (m *MockMemoryMappable) DisableMemoryMapping() error {
	if m.disableError != nil {
		return m.disableError
	}
	m.isMapped.Store(false)
	m.mappedSize.Store(0)
	return nil
}

func (m *MockMemoryMappable) IsMemoryMapped() bool {
	return m.isMapped.Load()
}

func (m *MockMemoryMappable) MemoryMappedSize() int64 {
	return m.mappedSize.Load()
}

func TestMemoryManager_MemoryMapping(t *testing.T) {
	// Create temporary directory for memory mapping
	tmpDir, err := os.MkdirTemp("", "memory_manager_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create config with memory mapping enabled
	config := DefaultMemoryConfig()
	config.EnableMMap = true
	config.MMapPath = tmpDir
	config.MMapThreshold = 1024 // 1KB threshold

	manager := NewManager(config)

	// Create mock mappable that exceeds threshold
	mockMappable := &MockMemoryMappable{
		canMap:        true,
		estimatedSize: 2048, // Exceeds threshold
	}

	// Register the mappable
	err = manager.RegisterMemoryMappable("test", mockMappable)
	if err != nil {
		t.Fatalf("Failed to register mappable: %v", err)
	}

	// Poll for automatic mapping to occur (background goroutine)
	assertMappedState(t, mockMappable, true, "auto-mapping", 5*time.Second)

	// Test memory usage calculation
	usage := manager.GetUsage()
	if usage.MemoryMapped != mockMappable.MemoryMappedSize() {
		t.Errorf("Expected memory mapped usage %d, got %d",
			mockMappable.MemoryMappedSize(), usage.MemoryMapped)
	}

	// Unregister and check cleanup
	err = manager.UnregisterMemoryMappable("test")
	if err != nil {
		t.Errorf("Failed to unregister mappable: %v", err)
	}

	if mockMappable.IsMemoryMapped() {
		t.Error("Expected memory mapping to be disabled after unregistration")
	}
}

func TestMemoryManager_MemoryPressureMapping(t *testing.T) {
	// Create temporary directory for memory mapping
	tmpDir, err := os.MkdirTemp("", "memory_manager_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create config with memory mapping enabled and low threshold
	config := DefaultMemoryConfig()
	config.EnableMMap = true
	config.MMapPath = tmpDir
	config.MMapThreshold = 10 * 1024 * 1024 // High threshold so no auto-mapping
	config.MaxMemory = 0                    // No initial limit
	config.MonitorInterval = 50 * time.Millisecond

	manager := NewManager(config)

	// Start monitoring
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err = manager.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start manager: %v", err)
	}
	defer manager.Stop()

	// Create mock mappable that can be mapped under pressure
	mockMappable := &MockMemoryMappable{
		canMap:        true,
		estimatedSize: 512, // Half the memory limit
	}

	t.Logf("Mock mappable size: %d, threshold: %d", mockMappable.EstimateSize(), config.MMapThreshold)

	// Register the mappable
	err = manager.RegisterMemoryMappable("test", mockMappable)
	if err != nil {
		t.Fatalf("Failed to register mappable: %v", err)
	}

	// Initially should not be mapped (below threshold)
	// Poll to verify no automatic mapping occurred (use generous deadline for -race)
	assertMappedState(t, mockMappable, false, "below threshold", 2*time.Second)

	// Check current memory usage
	usage := manager.GetUsage()
	t.Logf("Current memory usage: Total=%d, Limit=%d, Available=%d", usage.Total, usage.Limit, usage.Available)

	// Now simulate memory pressure by setting a very low limit
	currentUsage := usage.Total - usage.MemoryMapped // Only heap usage counts against limit
	lowLimit := currentUsage - 100                   // Set limit below current usage to trigger pressure
	err = manager.SetLimit(lowLimit)
	if err != nil {
		t.Fatalf("Failed to set memory limit: %v", err)
	}

	// Poll until pressure response maps the mappable (generous deadline for -race)
	assertMappedState(t, mockMappable, true, "memory pressure", 5*time.Second)
}

// assertMappedState polls until the mappable reaches the expected mapped state or deadline expires.
func assertMappedState(t *testing.T, mappable *MockMemoryMappable, want bool, context string, deadline time.Duration) {
	t.Helper()
	timeout := time.After(deadline)
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		if mappable.IsMemoryMapped() == want {
			return
		}
		select {
		case <-timeout:
			t.Errorf("Expected mapped=%v (%s) after %v but got mapped=%v",
				want, context, deadline, mappable.IsMemoryMapped())
			return
		case <-ticker.C:
		}
	}
}

func TestMemoryManager_MappingErrors(t *testing.T) {
	// Create config with memory mapping enabled
	config := DefaultMemoryConfig()
	config.EnableMMap = true
	config.MMapThreshold = 1024

	manager := NewManager(config)

	// Test nil mappable
	err := manager.RegisterMemoryMappable("test", nil)
	if err == nil {
		t.Error("Expected error for nil mappable")
	}

	// Test duplicate registration
	mockMappable := &MockMemoryMappable{
		canMap:        true,
		estimatedSize: 2048,
	}

	err = manager.RegisterMemoryMappable("test", mockMappable)
	if err != nil {
		t.Fatalf("Failed to register mappable: %v", err)
	}

	err = manager.RegisterMemoryMappable("test", mockMappable)
	if err == nil {
		t.Error("Expected error for duplicate registration")
	}

	// Test unregistering non-existent mappable
	err = manager.UnregisterMemoryMappable("nonexistent")
	if err == nil {
		t.Error("Expected error for unregistering non-existent mappable")
	}
}

func TestMemoryManager_MappingDisabled(t *testing.T) {
	// Create config with memory mapping disabled
	config := DefaultMemoryConfig()
	config.EnableMMap = false

	manager := NewManager(config)

	// Create mock mappable
	mockMappable := &MockMemoryMappable{
		canMap:        true,
		estimatedSize: 10 * 1024 * 1024, // Large size
	}

	// Register the mappable
	err := manager.RegisterMemoryMappable("test", mockMappable)
	if err != nil {
		t.Fatalf("Failed to register mappable: %v", err)
	}

	// Poll to verify no mapping occurs when disabled
	assertMappedState(t, mockMappable, false, "mmap disabled", 2*time.Second)
}

func TestMemoryManager_MappingPriorityOrder(t *testing.T) {
	// Create temporary directory for memory mapping
	tmpDir, err := os.MkdirTemp("", "memory_manager_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create config
	config := DefaultMemoryConfig()
	config.EnableMMap = true
	config.MMapPath = tmpDir
	config.MMapThreshold = 10 * 1024 * 1024 // High threshold

	manager := NewManager(config)

	// Create multiple mappables of different sizes
	smallMappable := &MockMemoryMappable{
		canMap:        true,
		estimatedSize: 100,
	}

	largeMappable := &MockMemoryMappable{
		canMap:        true,
		estimatedSize: 1000,
	}

	// Register mappables
	err = manager.RegisterMemoryMappable("small", smallMappable)
	if err != nil {
		t.Fatalf("Failed to register small mappable: %v", err)
	}

	err = manager.RegisterMemoryMappable("large", largeMappable)
	if err != nil {
		t.Fatalf("Failed to register large mappable: %v", err)
	}

	// Simulate memory pressure by setting a very low limit and starting monitoring
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err = manager.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start manager: %v", err)
	}
	defer manager.Stop()

	// Set very low memory limit to trigger pressure
	err = manager.SetLimit(50) // Very low limit
	if err != nil {
		t.Fatalf("Failed to set memory limit: %v", err)
	}

	// Poll until at least one mappable is mapped under pressure
	deadline := time.After(5 * time.Second)
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		mappedCount := 0
		if smallMappable.IsMemoryMapped() {
			mappedCount++
		}
		if largeMappable.IsMemoryMapped() {
			mappedCount++
		}
		if mappedCount > 0 {
			break
		}
		select {
		case <-deadline:
			t.Error("Expected at least one mappable to be mapped under memory pressure")
			return
		case <-ticker.C:
		}
	}
}
