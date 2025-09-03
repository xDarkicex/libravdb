package libravdb

import (
	"time"
)

// VectorEntry represents a vector with metadata for storage/indexing
type VectorEntry struct {
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// SearchResult represents a single search result
type SearchResult struct {
	ID       string                 `json:"id"`
	Score    float32                `json:"score"`
	Vector   []float32              `json:"vector,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// SearchResults represents the complete search response
type SearchResults struct {
	Results []*SearchResult `json:"results"`
	Took    time.Duration   `json:"took"`
	Total   int             `json:"total"`
}

// DatabaseStats represents database-wide statistics
type DatabaseStats struct {
	CollectionCount int                         `json:"collection_count"`
	Collections     map[string]*CollectionStats `json:"collections"`
	MemoryUsage     int64                       `json:"memory_usage"`
	Uptime          time.Duration               `json:"uptime"`
}

// CollectionStats represents collection-specific statistics
type CollectionStats struct {
	Name        string `json:"name"`
	VectorCount int    `json:"vector_count"`
	Dimension   int    `json:"dimension"`
	IndexType   string `json:"index_type"`
	MemoryUsage int64  `json:"memory_usage"`
}

// HealthStatus represents the health status of the database
type HealthStatus struct {
	Status string                  `json:"status"`
	Checks map[string]*CheckResult `json:"checks"`
}

// CheckResult represents the result of a single health check
type CheckResult struct {
	Healthy bool   `json:"healthy"`
	Message string `json:"message"`
}
