package index

import (
	"fmt"
)

// IndexFactory creates index instances based on configuration
type IndexFactory struct{}

// NewIndexFactory creates a new index factory
func NewIndexFactory() *IndexFactory {
	return &IndexFactory{}
}

// CreateIndex creates an index based on the provided configuration
func (f *IndexFactory) CreateIndex(indexType IndexType, config interface{}) (Index, error) {
	switch indexType {
	case IndexTypeHNSW:
		hnswConfig, ok := config.(*HNSWConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for HNSW index")
		}
		return NewHNSW(hnswConfig)

	case IndexTypeIVFPQ:
		ivfpqConfig, ok := config.(*IVFPQConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for IVF-PQ index")
		}
		return NewIVFPQ(ivfpqConfig)

	case IndexTypeFlat:
		flatConfig, ok := config.(*FlatConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for Flat index")
		}
		return NewFlat(flatConfig)

	default:
		return nil, fmt.Errorf("unsupported index type: %v", indexType)
	}
}

// SupportedIndexTypes returns a list of supported index types
func (f *IndexFactory) SupportedIndexTypes() []IndexType {
	return []IndexType{
		IndexTypeHNSW,
		IndexTypeIVFPQ,
		IndexTypeFlat,
	}
}

// DefaultIndexFactory is the global index factory instance
var DefaultIndexFactory = NewIndexFactory()
