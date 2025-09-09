package quant

import (
	"context"
	"testing"
)

// Mock quantizer for testing
type mockQuantizer struct {
	config *QuantizationConfig
}

func (m *mockQuantizer) Train(ctx context.Context, vectors [][]float32) error {
	return nil
}

func (m *mockQuantizer) Configure(config *QuantizationConfig) error {
	m.config = config
	return nil
}

func (m *mockQuantizer) Compress(vector []float32) ([]byte, error) {
	return []byte{1, 2, 3}, nil
}

func (m *mockQuantizer) Decompress(data []byte) ([]float32, error) {
	return []float32{1.0, 2.0, 3.0}, nil
}

func (m *mockQuantizer) Distance(compressed1, compressed2 []byte) (float32, error) {
	return 0.5, nil
}

func (m *mockQuantizer) DistanceToQuery(compressed []byte, query []float32) (float32, error) {
	return 0.3, nil
}

func (m *mockQuantizer) CompressionRatio() float32 {
	return 8.0
}

func (m *mockQuantizer) MemoryUsage() int64 {
	return 1024
}

func (m *mockQuantizer) IsTrained() bool {
	return true
}

func (m *mockQuantizer) Config() *QuantizationConfig {
	return m.config
}

// Mock factory for testing
type mockFactory struct {
	name          string
	supportedType QuantizationType
}

func (f *mockFactory) Create(config *QuantizationConfig) (Quantizer, error) {
	q := &mockQuantizer{}
	q.Configure(config)
	return q, nil
}

func (f *mockFactory) Supports(qType QuantizationType) bool {
	return qType == f.supportedType
}

func (f *mockFactory) Name() string {
	return f.name
}

func TestRegistry_Register(t *testing.T) {
	registry := NewRegistry()
	factory := &mockFactory{
		name:          "test-factory",
		supportedType: ProductQuantization,
	}

	// Test successful registration
	err := registry.Register(ProductQuantization, factory)
	if err != nil {
		t.Errorf("Registry.Register() error = %v, want nil", err)
	}

	// Test duplicate registration
	err = registry.Register(ProductQuantization, factory)
	if err == nil {
		t.Error("Registry.Register() expected error for duplicate registration")
	}

	// Test nil factory
	err = registry.Register(ScalarQuantization, nil)
	if err == nil {
		t.Error("Registry.Register() expected error for nil factory")
	}

	// Test unsupported type
	unsupportedFactory := &mockFactory{
		name:          "unsupported-factory",
		supportedType: ScalarQuantization,
	}
	err = registry.Register(ProductQuantization, unsupportedFactory)
	if err == nil {
		t.Error("Registry.Register() expected error for unsupported type")
	}
}

func TestRegistry_Create(t *testing.T) {
	registry := NewRegistry()
	factory := &mockFactory{
		name:          "test-factory",
		supportedType: ProductQuantization,
	}

	// Register factory
	err := registry.Register(ProductQuantization, factory)
	if err != nil {
		t.Fatalf("Failed to register factory: %v", err)
	}

	// Test successful creation
	config := DefaultConfig(ProductQuantization)
	quantizer, err := registry.Create(config)
	if err != nil {
		t.Errorf("Registry.Create() error = %v, want nil", err)
	}
	if quantizer == nil {
		t.Error("Registry.Create() returned nil quantizer")
	}

	// Test nil config
	_, err = registry.Create(nil)
	if err == nil {
		t.Error("Registry.Create() expected error for nil config")
	}

	// Test invalid config
	invalidConfig := &QuantizationConfig{
		Type:       ProductQuantization,
		Bits:       0, // Invalid
		TrainRatio: 0.1,
	}
	_, err = registry.Create(invalidConfig)
	if err == nil {
		t.Error("Registry.Create() expected error for invalid config")
	}

	// Test unsupported type
	unsupportedConfig := DefaultConfig(ScalarQuantization)
	_, err = registry.Create(unsupportedConfig)
	if err == nil {
		t.Error("Registry.Create() expected error for unsupported type")
	}
}

func TestRegistry_IsSupported(t *testing.T) {
	registry := NewRegistry()
	factory := &mockFactory{
		name:          "test-factory",
		supportedType: ProductQuantization,
	}

	// Test unsupported type
	if registry.IsSupported(ProductQuantization) {
		t.Error("Registry.IsSupported() should return false for unregistered type")
	}

	// Register factory
	err := registry.Register(ProductQuantization, factory)
	if err != nil {
		t.Fatalf("Failed to register factory: %v", err)
	}

	// Test supported type
	if !registry.IsSupported(ProductQuantization) {
		t.Error("Registry.IsSupported() should return true for registered type")
	}

	// Test still unsupported type
	if registry.IsSupported(ScalarQuantization) {
		t.Error("Registry.IsSupported() should return false for unregistered type")
	}
}

func TestRegistry_SupportedTypes(t *testing.T) {
	registry := NewRegistry()

	// Test empty registry
	types := registry.SupportedTypes()
	if len(types) != 0 {
		t.Errorf("Registry.SupportedTypes() = %v, want empty slice", types)
	}

	// Register factories
	factory1 := &mockFactory{
		name:          "factory1",
		supportedType: ProductQuantization,
	}
	factory2 := &mockFactory{
		name:          "factory2",
		supportedType: ScalarQuantization,
	}

	registry.Register(ProductQuantization, factory1)
	registry.Register(ScalarQuantization, factory2)

	types = registry.SupportedTypes()
	if len(types) != 2 {
		t.Errorf("Registry.SupportedTypes() length = %d, want 2", len(types))
	}

	// Check that both types are present
	found := make(map[QuantizationType]bool)
	for _, qType := range types {
		found[qType] = true
	}

	if !found[ProductQuantization] || !found[ScalarQuantization] {
		t.Errorf("Registry.SupportedTypes() = %v, missing expected types", types)
	}
}

func TestRegistry_Unregister(t *testing.T) {
	registry := NewRegistry()
	factory := &mockFactory{
		name:          "test-factory",
		supportedType: ProductQuantization,
	}

	// Register and verify
	registry.Register(ProductQuantization, factory)
	if !registry.IsSupported(ProductQuantization) {
		t.Error("Factory should be registered")
	}

	// Unregister and verify
	registry.Unregister(ProductQuantization)
	if registry.IsSupported(ProductQuantization) {
		t.Error("Factory should be unregistered")
	}

	// Unregistering non-existent type should not panic
	registry.Unregister(ScalarQuantization)
}

func TestRegistry_Clear(t *testing.T) {
	registry := NewRegistry()
	factory1 := &mockFactory{
		name:          "factory1",
		supportedType: ProductQuantization,
	}
	factory2 := &mockFactory{
		name:          "factory2",
		supportedType: ScalarQuantization,
	}

	// Register factories
	registry.Register(ProductQuantization, factory1)
	registry.Register(ScalarQuantization, factory2)

	// Verify registration
	if len(registry.SupportedTypes()) != 2 {
		t.Error("Should have 2 registered factories")
	}

	// Clear and verify
	registry.Clear()
	if len(registry.SupportedTypes()) != 0 {
		t.Error("Should have no registered factories after clear")
	}
}

func TestGlobalRegistry(t *testing.T) {
	// Clear global registry to start fresh
	globalRegistry.Clear()

	factory := &mockFactory{
		name:          "global-factory",
		supportedType: ProductQuantization,
	}

	// Test global registration
	err := Register(ProductQuantization, factory)
	if err != nil {
		t.Errorf("Global Register() error = %v, want nil", err)
	}

	// Test global IsSupported
	if !IsSupported(ProductQuantization) {
		t.Error("Global IsSupported() should return true")
	}

	// Test global SupportedTypes
	types := SupportedTypes()
	if len(types) != 1 || types[0] != ProductQuantization {
		t.Errorf("Global SupportedTypes() = %v, want [ProductQuantization]", types)
	}

	// Test global Create
	config := DefaultConfig(ProductQuantization)
	quantizer, err := Create(config)
	if err != nil {
		t.Errorf("Global Create() error = %v, want nil", err)
	}
	if quantizer == nil {
		t.Error("Global Create() returned nil quantizer")
	}

	// Clean up
	globalRegistry.Clear()
}
