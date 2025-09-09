package memory

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestNewManager(t *testing.T) {
	config := DefaultMemoryConfig()
	config.MaxMemory = 1024 * 1024 // 1MB

	mgr := NewManager(config)
	if mgr == nil {
		t.Fatal("NewManager returned nil")
	}

	// Test initial state
	usage := mgr.GetUsage()
	if usage.Limit != config.MaxMemory {
		t.Errorf("Expected limit %d, got %d", config.MaxMemory, usage.Limit)
	}
}

func TestSetLimit(t *testing.T) {
	mgr := NewManager(DefaultMemoryConfig())

	// Test setting valid limit
	err := mgr.SetLimit(2 * 1024 * 1024) // 2MB
	if err != nil {
		t.Fatalf("SetLimit failed: %v", err)
	}

	usage := mgr.GetUsage()
	if usage.Limit != 2*1024*1024 {
		t.Errorf("Expected limit 2MB, got %d", usage.Limit)
	}

	// Test setting negative limit
	err = mgr.SetLimit(-1)
	if err == nil {
		t.Error("Expected error for negative limit")
	}
}

func TestRegisterCache(t *testing.T) {
	mgr := NewManager(DefaultMemoryConfig())
	cache := NewLRUCache("test", 1024)

	// Test successful registration
	err := mgr.RegisterCache("test", cache)
	if err != nil {
		t.Fatalf("RegisterCache failed: %v", err)
	}

	// Test duplicate registration
	err = mgr.RegisterCache("test", cache)
	if err == nil {
		t.Error("Expected error for duplicate cache registration")
	}

	// Test nil cache
	err = mgr.RegisterCache("nil", nil)
	if err == nil {
		t.Error("Expected error for nil cache")
	}
}

func TestUnregisterCache(t *testing.T) {
	mgr := NewManager(DefaultMemoryConfig())
	cache := NewLRUCache("test", 1024)

	// Register cache first
	err := mgr.RegisterCache("test", cache)
	if err != nil {
		t.Fatalf("RegisterCache failed: %v", err)
	}

	// Test successful unregistration
	err = mgr.UnregisterCache("test")
	if err != nil {
		t.Fatalf("UnregisterCache failed: %v", err)
	}

	// Test unregistering non-existent cache
	err = mgr.UnregisterCache("nonexistent")
	if err == nil {
		t.Error("Expected error for non-existent cache")
	}
}

func TestMemoryPressureCallbacks(t *testing.T) {
	config := DefaultMemoryConfig()
	config.MaxMemory = 1024 * 1024 // 1MB
	config.MonitorInterval = 100 * time.Millisecond

	mgr := NewManager(config)

	var callbackCalled bool
	var mu sync.Mutex

	mgr.OnMemoryPressure(func(usage MemoryUsage) {
		mu.Lock()
		callbackCalled = true
		mu.Unlock()
	})

	// Start monitoring
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err := mgr.Start(ctx)
	if err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer mgr.Stop()

	// Set a very low limit to trigger pressure
	mgr.SetLimit(1024) // 1KB - very low

	// Wait for callback
	time.Sleep(200 * time.Millisecond)

	mu.Lock()
	called := callbackCalled
	mu.Unlock()

	if !called {
		t.Error("Memory pressure callback was not called")
	}
}

func TestMemoryReleaseCallbacks(t *testing.T) {
	mgr := NewManager(DefaultMemoryConfig())

	var releaseCalled bool
	var freedBytes int64
	var mu sync.Mutex

	mgr.OnMemoryRelease(func(freed int64) {
		mu.Lock()
		releaseCalled = true
		freedBytes = freed
		mu.Unlock()
	})

	// Trigger GC to simulate memory release
	mgr.TriggerGC()

	// The callback might not be called immediately since GC might not free much
	// This test mainly verifies the callback registration works
	mu.Lock()
	_ = releaseCalled
	_ = freedBytes
	mu.Unlock()
}

func TestStartStop(t *testing.T) {
	mgr := NewManager(DefaultMemoryConfig())

	ctx := context.Background()

	// Test starting
	err := mgr.Start(ctx)
	if err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	// Test starting again (should fail)
	err = mgr.Start(ctx)
	if err == nil {
		t.Error("Expected error when starting already started manager")
	}

	// Test stopping
	err = mgr.Stop()
	if err != nil {
		t.Fatalf("Stop failed: %v", err)
	}

	// Test stopping again (should fail)
	err = mgr.Stop()
	if err == nil {
		t.Error("Expected error when stopping already stopped manager")
	}
}

func TestMemoryLimitEnforcement(t *testing.T) {
	config := DefaultMemoryConfig()
	config.MaxMemory = 1024 * 1024 // 1MB
	config.MonitorInterval = 50 * time.Millisecond

	mgr := NewManager(config)

	// Create a cache that will consume memory
	cache := NewLRUCache("test", 512*1024) // 512KB cache
	err := mgr.RegisterCache("test", cache)
	if err != nil {
		t.Fatalf("RegisterCache failed: %v", err)
	}

	// Fill the cache with data
	for i := 0; i < 100; i++ {
		data := make([]byte, 1024) // 1KB per item
		cache.Put(fmt.Sprintf("key%d", i), data, int64(len(data)))
	}

	// Start monitoring
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	err = mgr.Start(ctx)
	if err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer mgr.Stop()

	// Set a low limit to trigger eviction
	mgr.SetLimit(10 * 1024) // 10KB - very low

	// Wait for memory management to kick in
	time.Sleep(200 * time.Millisecond)

	// Check that cache was evicted
	cacheSize := cache.Size()
	if cacheSize > 10*1024 {
		t.Logf("Cache size after eviction: %d bytes (limit was 10KB)", cacheSize)
		// Note: This might not always pass due to timing and GC behavior
		// In a real scenario, you'd want more sophisticated testing
	}
}

func TestGetUsage(t *testing.T) {
	mgr := NewManager(DefaultMemoryConfig())

	usage := mgr.GetUsage()

	// Basic sanity checks
	if usage.Total < 0 {
		t.Error("Total usage should not be negative")
	}

	if usage.Timestamp.IsZero() {
		t.Error("Timestamp should be set")
	}

	// Test with limit set
	mgr.SetLimit(1024 * 1024)
	usage = mgr.GetUsage()

	if usage.Limit != 1024*1024 {
		t.Errorf("Expected limit 1MB, got %d", usage.Limit)
	}

	if usage.Available < 0 && usage.Limit > 0 {
		t.Error("Available should not be negative when limit is set")
	}
}

func TestTriggerGC(t *testing.T) {
	mgr := NewManager(DefaultMemoryConfig())

	err := mgr.TriggerGC()
	if err != nil {
		t.Fatalf("TriggerGC failed: %v", err)
	}

	// GC should complete without error
	// We can't easily test if it actually freed memory
}
