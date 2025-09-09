package memory

import (
	"fmt"
	"testing"
)

func TestNewLRUCache(t *testing.T) {
	cache := NewLRUCache("test", 1024)

	if cache.Name() != "test" {
		t.Errorf("Expected name 'test', got '%s'", cache.Name())
	}

	if cache.Capacity() != 1024 {
		t.Errorf("Expected capacity 1024, got %d", cache.Capacity())
	}

	if cache.Size() != 0 {
		t.Errorf("Expected initial size 0, got %d", cache.Size())
	}

	if cache.Len() != 0 {
		t.Errorf("Expected initial length 0, got %d", cache.Len())
	}
}

func TestLRUCachePutGet(t *testing.T) {
	cache := NewLRUCache("test", 1024)

	// Test putting and getting a value
	success := cache.Put("key1", "value1", 100)
	if !success {
		t.Error("Put should succeed")
	}

	value, exists := cache.Get("key1")
	if !exists {
		t.Error("Key should exist")
	}

	if value != "value1" {
		t.Errorf("Expected 'value1', got '%v'", value)
	}

	if cache.Size() != 100 {
		t.Errorf("Expected size 100, got %d", cache.Size())
	}

	if cache.Len() != 1 {
		t.Errorf("Expected length 1, got %d", cache.Len())
	}
}

func TestLRUCacheUpdate(t *testing.T) {
	cache := NewLRUCache("test", 1024)

	// Put initial value
	cache.Put("key1", "value1", 100)

	// Update with new value and size
	success := cache.Put("key1", "value2", 200)
	if !success {
		t.Error("Update should succeed")
	}

	value, exists := cache.Get("key1")
	if !exists {
		t.Error("Key should exist")
	}

	if value != "value2" {
		t.Errorf("Expected 'value2', got '%v'", value)
	}

	if cache.Size() != 200 {
		t.Errorf("Expected size 200, got %d", cache.Size())
	}

	if cache.Len() != 1 {
		t.Errorf("Expected length 1, got %d", cache.Len())
	}
}

func TestLRUCacheEviction(t *testing.T) {
	cache := NewLRUCache("test", 250) // Small capacity

	// Add items that will exceed capacity
	cache.Put("key1", "value1", 100)
	cache.Put("key2", "value2", 100)
	cache.Put("key3", "value3", 100) // This should evict key1

	// key1 should be evicted (least recently used)
	_, exists := cache.Get("key1")
	if exists {
		t.Error("key1 should have been evicted")
	}

	// key2 and key3 should still exist
	_, exists = cache.Get("key2")
	if !exists {
		t.Error("key2 should still exist")
	}

	_, exists = cache.Get("key3")
	if !exists {
		t.Error("key3 should still exist")
	}

	if cache.Len() != 2 {
		t.Errorf("Expected length 2, got %d", cache.Len())
	}
}

func TestLRUCacheLRUOrder(t *testing.T) {
	cache := NewLRUCache("test", 250)

	// Add items
	cache.Put("key1", "value1", 100)
	cache.Put("key2", "value2", 100)

	// Access key1 to make it more recently used
	cache.Get("key1")

	// Add key3, which should evict key2 (now least recently used)
	cache.Put("key3", "value3", 100)

	// key2 should be evicted
	_, exists := cache.Get("key2")
	if exists {
		t.Error("key2 should have been evicted")
	}

	// key1 and key3 should still exist
	_, exists = cache.Get("key1")
	if !exists {
		t.Error("key1 should still exist")
	}

	_, exists = cache.Get("key3")
	if !exists {
		t.Error("key3 should still exist")
	}
}

func TestLRUCacheRemove(t *testing.T) {
	cache := NewLRUCache("test", 1024)

	cache.Put("key1", "value1", 100)
	cache.Put("key2", "value2", 100)

	// Remove existing key
	removed := cache.Remove("key1")
	if !removed {
		t.Error("Remove should return true for existing key")
	}

	_, exists := cache.Get("key1")
	if exists {
		t.Error("key1 should not exist after removal")
	}

	if cache.Size() != 100 {
		t.Errorf("Expected size 100 after removal, got %d", cache.Size())
	}

	if cache.Len() != 1 {
		t.Errorf("Expected length 1 after removal, got %d", cache.Len())
	}

	// Remove non-existent key
	removed = cache.Remove("nonexistent")
	if removed {
		t.Error("Remove should return false for non-existent key")
	}
}

func TestLRUCacheClear(t *testing.T) {
	cache := NewLRUCache("test", 1024)

	cache.Put("key1", "value1", 100)
	cache.Put("key2", "value2", 100)

	cache.Clear()

	if cache.Size() != 0 {
		t.Errorf("Expected size 0 after clear, got %d", cache.Size())
	}

	if cache.Len() != 0 {
		t.Errorf("Expected length 0 after clear, got %d", cache.Len())
	}

	_, exists := cache.Get("key1")
	if exists {
		t.Error("No keys should exist after clear")
	}
}

func TestLRUCacheEvictMethod(t *testing.T) {
	cache := NewLRUCache("test", 1024)

	// Add several items
	cache.Put("key1", "value1", 100)
	cache.Put("key2", "value2", 150)
	cache.Put("key3", "value3", 200)

	initialSize := cache.Size()

	// Evict 200 bytes
	freed := cache.Evict(200)

	if freed < 200 {
		t.Errorf("Expected to free at least 200 bytes, freed %d", freed)
	}

	newSize := cache.Size()
	if newSize >= initialSize {
		t.Error("Cache size should have decreased after eviction")
	}

	// Evict more than available
	freed = cache.Evict(10000)
	if cache.Size() != 0 {
		t.Errorf("Expected cache to be empty after large eviction, size is %d", cache.Size())
	}
}

func TestLRUCacheOversizedItem(t *testing.T) {
	cache := NewLRUCache("test", 100)

	// Try to put an item larger than capacity
	success := cache.Put("big", "value", 200)
	if success {
		t.Error("Put should fail for oversized item")
	}

	if cache.Size() != 0 {
		t.Errorf("Cache should be empty after failed oversized put, size is %d", cache.Size())
	}
}

func TestLRUCacheKeys(t *testing.T) {
	cache := NewLRUCache("test", 1024)

	cache.Put("key1", "value1", 100)
	cache.Put("key2", "value2", 100)
	cache.Put("key3", "value3", 100)

	keys := cache.Keys()
	if len(keys) != 3 {
		t.Errorf("Expected 3 keys, got %d", len(keys))
	}

	// Check that all keys are present
	keyMap := make(map[string]bool)
	for _, key := range keys {
		keyMap[key] = true
	}

	expectedKeys := []string{"key1", "key2", "key3"}
	for _, expectedKey := range expectedKeys {
		if !keyMap[expectedKey] {
			t.Errorf("Expected key %s not found", expectedKey)
		}
	}
}

func TestLRUCacheStats(t *testing.T) {
	cache := NewLRUCache("test", 1024)

	cache.Put("key1", "value1", 100)
	cache.Put("key2", "value2", 150)

	stats := cache.Stats()

	if stats.Name != "test" {
		t.Errorf("Expected name 'test', got '%s'", stats.Name)
	}

	if stats.Capacity != 1024 {
		t.Errorf("Expected capacity 1024, got %d", stats.Capacity)
	}

	if stats.Size != 250 {
		t.Errorf("Expected size 250, got %d", stats.Size)
	}

	if stats.Items != 2 {
		t.Errorf("Expected 2 items, got %d", stats.Items)
	}

	// Test string representation
	str := stats.String()
	if str == "" {
		t.Error("Stats string should not be empty")
	}
}

func TestLRUCacheConcurrency(t *testing.T) {
	cache := NewLRUCache("test", 10000)

	// Test concurrent access
	done := make(chan bool, 10)

	// Start multiple goroutines doing cache operations
	for i := 0; i < 10; i++ {
		go func(id int) {
			defer func() { done <- true }()

			// Each goroutine does multiple operations
			for j := 0; j < 100; j++ {
				key := fmt.Sprintf("key_%d_%d", id, j)
				value := fmt.Sprintf("value_%d_%d", id, j)

				cache.Put(key, value, 10)
				cache.Get(key)

				if j%10 == 0 {
					cache.Remove(key)
				}
			}
		}(i)
	}

	// Wait for all goroutines to complete
	for i := 0; i < 10; i++ {
		<-done
	}

	// Cache should still be functional
	cache.Put("final", "test", 10)
	value, exists := cache.Get("final")
	if !exists || value != "test" {
		t.Error("Cache should still be functional after concurrent access")
	}
}
