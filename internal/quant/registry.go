package quant

import (
	"fmt"
	"sync"
)

// Registry manages quantizer factories
type Registry struct {
	mu        sync.RWMutex
	factories map[QuantizationType]QuantizerFactory
}

// NewRegistry creates a new quantizer registry
func NewRegistry() *Registry {
	return &Registry{
		factories: make(map[QuantizationType]QuantizerFactory),
	}
}

// Register registers a quantizer factory for a specific type
func (r *Registry) Register(qType QuantizationType, factory QuantizerFactory) error {
	if factory == nil {
		return fmt.Errorf("factory cannot be nil")
	}

	if !factory.Supports(qType) {
		return fmt.Errorf("factory %s does not support quantization type %s",
			factory.Name(), qType.String())
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.factories[qType]; exists {
		return fmt.Errorf("factory for quantization type %s already registered", qType.String())
	}

	r.factories[qType] = factory
	return nil
}

// Unregister removes a quantizer factory
func (r *Registry) Unregister(qType QuantizationType) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.factories, qType)
}

// Create creates a quantizer instance using the registered factory
func (r *Registry) Create(config *QuantizationConfig) (Quantizer, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	r.mu.RLock()
	factory, exists := r.factories[config.Type]
	r.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("no factory registered for quantization type %s", config.Type.String())
	}

	return factory.Create(config)
}

// IsSupported returns true if the quantization type is supported
func (r *Registry) IsSupported(qType QuantizationType) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, exists := r.factories[qType]
	return exists
}

// SupportedTypes returns all supported quantization types
func (r *Registry) SupportedTypes() []QuantizationType {
	r.mu.RLock()
	defer r.mu.RUnlock()

	types := make([]QuantizationType, 0, len(r.factories))
	for qType := range r.factories {
		types = append(types, qType)
	}
	return types
}

// GetFactory returns the factory for a specific quantization type
func (r *Registry) GetFactory(qType QuantizationType) (QuantizerFactory, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	factory, exists := r.factories[qType]
	if !exists {
		return nil, fmt.Errorf("no factory registered for quantization type %s", qType.String())
	}

	return factory, nil
}

// Clear removes all registered factories
func (r *Registry) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.factories = make(map[QuantizationType]QuantizerFactory)
}

// Global registry instance
var globalRegistry = NewRegistry()

// Register registers a quantizer factory globally
func Register(qType QuantizationType, factory QuantizerFactory) error {
	return globalRegistry.Register(qType, factory)
}

// Create creates a quantizer using the global registry
func Create(config *QuantizationConfig) (Quantizer, error) {
	return globalRegistry.Create(config)
}

// IsSupported checks if a quantization type is supported globally
func IsSupported(qType QuantizationType) bool {
	return globalRegistry.IsSupported(qType)
}

// SupportedTypes returns all globally supported quantization types
func SupportedTypes() []QuantizationType {
	return globalRegistry.SupportedTypes()
}

// init registers default quantizer factories
func init() {
	// Register Product Quantization factory
	pqFactory := NewProductQuantizerFactory()
	if err := Register(ProductQuantization, pqFactory); err != nil {
		panic(fmt.Sprintf("Failed to register ProductQuantizer factory: %v", err))
	}

	// Register Scalar Quantization factory
	sqFactory := NewScalarQuantizerFactory()
	if err := Register(ScalarQuantization, sqFactory); err != nil {
		panic(fmt.Sprintf("Failed to register ScalarQuantizer factory: %v", err))
	}
}
