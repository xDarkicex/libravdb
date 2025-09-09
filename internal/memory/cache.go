package memory

import (
	"container/list"
	"fmt"
	"sync"
)

// LRUCache implements a thread-safe LRU (Least Recently Used) cache
type LRUCache struct {
	name     string
	capacity int64
	size     int64

	mu    sync.RWMutex
	items map[string]*list.Element
	order *list.List
}

// cacheItem represents an item in the LRU cache
type cacheItem struct {
	key   string
	value interface{}
	size  int64
}

// NewLRUCache creates a new LRU cache with the specified capacity in bytes
func NewLRUCache(name string, capacity int64) *LRUCache {
	return &LRUCache{
		name:     name,
		capacity: capacity,
		items:    make(map[string]*list.Element),
		order:    list.New(),
	}
}

// Name returns the cache identifier
func (c *LRUCache) Name() string {
	return c.name
}

// Size returns current cache size in bytes
func (c *LRUCache) Size() int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.size
}

// Capacity returns the maximum cache capacity in bytes
func (c *LRUCache) Capacity() int64 {
	return c.capacity
}

// Get retrieves a value from the cache and marks it as recently used
func (c *LRUCache) Get(key string) (interface{}, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, exists := c.items[key]; exists {
		// Move to front (most recently used)
		c.order.MoveToFront(elem)
		item := elem.Value.(*cacheItem)
		return item.value, true
	}

	return nil, false
}

// Put adds or updates a value in the cache
func (c *LRUCache) Put(key string, value interface{}, size int64) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if item already exists
	if elem, exists := c.items[key]; exists {
		// Update existing item
		item := elem.Value.(*cacheItem)
		oldSize := item.size
		item.value = value
		item.size = size
		c.size = c.size - oldSize + size
		c.order.MoveToFront(elem)

		// Evict if necessary
		c.evictIfNeeded()
		return true
	}

	// Check if we have space for the new item
	if size > c.capacity {
		// Item is too large for cache
		return false
	}

	// Make space if needed
	for c.size+size > c.capacity && c.order.Len() > 0 {
		c.removeLRU()
	}

	// Add new item
	item := &cacheItem{
		key:   key,
		value: value,
		size:  size,
	}

	elem := c.order.PushFront(item)
	c.items[key] = elem
	c.size += size

	return true
}

// Remove removes an item from the cache
func (c *LRUCache) Remove(key string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, exists := c.items[key]; exists {
		c.removeElement(elem)
		return true
	}

	return false
}

// Clear removes all items from the cache
func (c *LRUCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.items = make(map[string]*list.Element)
	c.order = list.New()
	c.size = 0
}

// Evict removes items to free the specified number of bytes
// Returns the actual number of bytes freed
func (c *LRUCache) Evict(bytes int64) int64 {
	c.mu.Lock()
	defer c.mu.Unlock()

	var freed int64

	// Remove items from least recently used until we've freed enough space
	for freed < bytes && c.order.Len() > 0 {
		elem := c.order.Back()
		if elem == nil {
			break
		}

		item := elem.Value.(*cacheItem)
		freed += item.size
		c.removeElement(elem)
	}

	return freed
}

// Len returns the number of items in the cache
func (c *LRUCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.items)
}

// Keys returns all keys in the cache (for testing/debugging)
func (c *LRUCache) Keys() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	keys := make([]string, 0, len(c.items))
	for key := range c.items {
		keys = append(keys, key)
	}
	return keys
}

// removeLRU removes the least recently used item
func (c *LRUCache) removeLRU() {
	elem := c.order.Back()
	if elem != nil {
		c.removeElement(elem)
	}
}

// removeElement removes a specific element from the cache
func (c *LRUCache) removeElement(elem *list.Element) {
	item := elem.Value.(*cacheItem)
	delete(c.items, item.key)
	c.order.Remove(elem)
	c.size -= item.size
}

// evictIfNeeded evicts items if the cache exceeds capacity
func (c *LRUCache) evictIfNeeded() {
	for c.size > c.capacity && c.order.Len() > 0 {
		c.removeLRU()
	}
}

// Stats returns cache statistics
func (c *LRUCache) Stats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return CacheStats{
		Name:     c.name,
		Size:     c.size,
		Capacity: c.capacity,
		Items:    len(c.items),
	}
}

// CacheStats represents cache statistics
type CacheStats struct {
	Name     string
	Size     int64
	Capacity int64
	Items    int
}

// String returns a string representation of cache stats
func (s CacheStats) String() string {
	return fmt.Sprintf("Cache{name=%s, size=%d, capacity=%d, items=%d}",
		s.Name, s.Size, s.Capacity, s.Items)
}
