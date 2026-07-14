package libravdb

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/memory"
	"github.com/xDarkicex/libravdb/internal/quant"
)

func TestCollectionConfigValidation(t *testing.T) {
	tests := []struct {
		config      *CollectionConfig
		name        string
		errorMsg    string
		expectError bool
	}{
		{
			name: "valid basic config",
			config: &CollectionConfig{
				Dimension:      128,
				Metric:         CosineDistance,
				IndexType:      HNSW,
				M:              16,
				EfConstruction: 200,
				EfSearch:       50,
				BatchConfig:    DefaultBatchConfig(),
			},
			expectError: false,
		},
		{
			name: "invalid dimension",
			config: &CollectionConfig{
				Dimension:      -1,
				Metric:         CosineDistance,
				IndexType:      HNSW,
				M:              16,
				EfConstruction: 200,
				EfSearch:       50,
				BatchConfig:    DefaultBatchConfig(),
			},
			expectError: true,
			errorMsg:    "dimension must be positive",
		},
		{
			name: "invalid memory limit",
			config: &CollectionConfig{
				Dimension:      128,
				Metric:         CosineDistance,
				IndexType:      HNSW,
				M:              16,
				EfConstruction: 200,
				EfSearch:       50,
				MemoryLimit:    -100,
				BatchConfig:    DefaultBatchConfig(),
			},
			expectError: true,
			errorMsg:    "memory limit must be non-negative",
		},
		{
			name: "valid quantization config",
			config: &CollectionConfig{
				Dimension:      128,
				Metric:         CosineDistance,
				IndexType:      HNSW,
				M:              16,
				EfConstruction: 200,
				EfSearch:       50,
				Quantization: &quant.QuantizationConfig{
					Type:       quant.ProductQuantization,
					Codebooks:  8,
					Bits:       8,
					TrainRatio: 0.1,
					CacheSize:  1000,
				},
				BatchConfig: DefaultBatchConfig(),
			},
			expectError: false,
		},
		{
			name: "invalid quantization config",
			config: &CollectionConfig{
				Dimension:      128,
				Metric:         CosineDistance,
				IndexType:      HNSW,
				M:              16,
				EfConstruction: 200,
				EfSearch:       50,
				Quantization: &quant.QuantizationConfig{
					Type:       quant.ProductQuantization,
					Codebooks:  0, // Invalid
					Bits:       8,
					TrainRatio: 0.1,
					CacheSize:  1000,
				},
				BatchConfig: DefaultBatchConfig(),
			},
			expectError: true,
			errorMsg:    "invalid quantization config",
		},
		{
			name: "valid metadata schema",
			config: &CollectionConfig{
				Dimension:      128,
				Metric:         CosineDistance,
				IndexType:      HNSW,
				M:              16,
				EfConstruction: 200,
				EfSearch:       50,
				MetadataSchema: MetadataSchema{
					"category": StringField,
					"price":    FloatField,
					"tags":     StringArrayField,
				},
				IndexedFields: []string{"category", "price"},
				BatchConfig:   DefaultBatchConfig(),
			},
			expectError: false,
		},
		{
			name: "invalid indexed field not in schema",
			config: &CollectionConfig{
				Dimension:      128,
				Metric:         CosineDistance,
				IndexType:      HNSW,
				M:              16,
				EfConstruction: 200,
				EfSearch:       50,
				MetadataSchema: MetadataSchema{
					"category": StringField,
					"price":    FloatField,
				},
				IndexedFields: []string{"category", "nonexistent"},
				BatchConfig:   DefaultBatchConfig(),
			},
			expectError: true,
			errorMsg:    "indexed field 'nonexistent' not found in metadata schema",
		},
		{
			name: "invalid batch config - negative chunk size",
			config: &CollectionConfig{
				Dimension:      128,
				Metric:         CosineDistance,
				IndexType:      HNSW,
				M:              16,
				EfConstruction: 200,
				EfSearch:       50,
				BatchConfig: BatchConfig{
					ChunkSize:       -1, // Invalid
					MaxConcurrency:  4,
					TimeoutPerChunk: 30 * time.Second,
				},
			},
			expectError: true,
			errorMsg:    "batch chunk size must be positive",
		},
		{
			name: "valid memory config",
			config: &CollectionConfig{
				Dimension:      128,
				Metric:         CosineDistance,
				IndexType:      HNSW,
				M:              16,
				EfConstruction: 200,
				EfSearch:       50,
				MemoryConfig: &memory.MemoryConfig{
					MaxMemory:       1024 * 1024 * 1024, // 1GB
					MonitorInterval: 5 * time.Second,
					EnableGC:        true,
					GCThreshold:     0.8,
					EnableMMap:      true,
					MMapThreshold:   100 * 1024 * 1024, // 100MB
				},
				BatchConfig: DefaultBatchConfig(),
			},
			expectError: false,
		},
		{
			name: "invalid memory config - negative max memory",
			config: &CollectionConfig{
				Dimension:      128,
				Metric:         CosineDistance,
				IndexType:      HNSW,
				M:              16,
				EfConstruction: 200,
				EfSearch:       50,
				MemoryConfig: &memory.MemoryConfig{
					MaxMemory:       -1, // Invalid
					MonitorInterval: 5 * time.Second,
					EnableGC:        true,
					GCThreshold:     0.8,
				},
				BatchConfig: DefaultBatchConfig(),
			},
			expectError: true,
			errorMsg:    "max memory must be non-negative",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.validate()
			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
				} else if tt.errorMsg != "" && !strings.Contains(err.Error(), tt.errorMsg) {
					t.Errorf("expected error message to contain '%s', got '%s'", tt.errorMsg, err.Error())
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			}
		})
	}
}

func TestMetadataSchemaValidation(t *testing.T) {
	tests := []struct {
		schema      MetadataSchema
		name        string
		errorMsg    string
		expectError bool
	}{
		{
			name: "valid schema",
			schema: MetadataSchema{
				"category": StringField,
				"price":    FloatField,
				"count":    IntField,
				"active":   BoolField,
				"tags":     StringArrayField,
			},
			expectError: false,
		},
		{
			name: "empty field name",
			schema: MetadataSchema{
				"":         StringField,
				"category": StringField,
			},
			expectError: true,
			errorMsg:    "field name cannot be empty",
		},
		{
			name:        "empty schema",
			schema:      MetadataSchema{},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.schema.Validate()
			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
				} else if tt.errorMsg != "" && !strings.Contains(err.Error(), tt.errorMsg) {
					t.Errorf("expected error message to contain '%s', got '%s'", tt.errorMsg, err.Error())
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			}
		})
	}
}

func TestCollectionOptionsValidation(t *testing.T) {
	tests := []struct {
		option      CollectionOption
		name        string
		errorMsg    string
		expectError bool
	}{
		{
			name:        "valid memory limit",
			option:      WithMemoryLimit(1024 * 1024 * 1024),
			expectError: false,
		},
		{
			name:        "invalid negative memory limit",
			option:      WithMemoryLimit(-1),
			expectError: true,
			errorMsg:    "memory limit must be non-negative",
		},
		{
			name:        "valid cache policy",
			option:      WithCachePolicy(LRUCache),
			expectError: false,
		},
		{
			name: "valid metadata schema",
			option: WithMetadataSchema(MetadataSchema{
				"category": StringField,
				"price":    FloatField,
			}),
			expectError: false,
		},
		{
			name: "invalid metadata schema",
			option: WithMetadataSchema(MetadataSchema{
				"": StringField, // Empty field name
			}),
			expectError: true,
			errorMsg:    "invalid metadata schema",
		},
		{
			name:        "valid batch chunk size",
			option:      WithBatchChunkSize(500),
			expectError: false,
		},
		{
			name:        "invalid batch chunk size",
			option:      WithBatchChunkSize(0),
			expectError: true,
			errorMsg:    "batch chunk size must be positive",
		},
		{
			name:        "valid batch concurrency",
			option:      WithBatchConcurrency(8),
			expectError: false,
		},
		{
			name:        "invalid batch concurrency",
			option:      WithBatchConcurrency(-1),
			expectError: true,
			errorMsg:    "batch concurrency must be positive",
		},
		{
			name: "valid memory config",
			option: WithMemoryConfig(&memory.MemoryConfig{
				MaxMemory:       1024 * 1024 * 1024,
				MonitorInterval: 5 * time.Second,
				EnableGC:        true,
				GCThreshold:     0.8,
			}),
			expectError: false,
		},
		{
			name:        "nil memory config",
			option:      WithMemoryConfig(nil),
			expectError: true,
			errorMsg:    "memory config cannot be nil",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &CollectionConfig{
				Dimension:      128,
				Metric:         CosineDistance,
				IndexType:      HNSW,
				M:              16,
				EfConstruction: 200,
				EfSearch:       50,
				BatchConfig:    DefaultBatchConfig(),
			}

			err := tt.option(config)
			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
				} else if tt.errorMsg != "" && !strings.Contains(err.Error(), tt.errorMsg) {
					t.Errorf("expected error message to contain '%s', got '%s'", tt.errorMsg, err.Error())
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			}
		})
	}
}

func TestWithHNSWUpdatesLevelMultiplierFromM(t *testing.T) {
	config := &CollectionConfig{
		Dimension:      128,
		Metric:         CosineDistance,
		IndexType:      HNSW,
		M:              32,
		EfConstruction: 200,
		EfSearch:       50,
		ML:             1.0,
		BatchConfig:    DefaultBatchConfig(),
	}

	if err := WithHNSW(16, 100, 64)(config); err != nil {
		t.Fatalf("WithHNSW: %v", err)
	}
	want := 1.0 / math.Log(16)
	if math.Abs(config.ML-want) > 1e-12 {
		t.Fatalf("ML got %.12f want %.12f", config.ML, want)
	}
}

func TestWithFSQQuantization(t *testing.T) {
	config := &CollectionConfig{
		Dimension:      128,
		Metric:         CosineDistance,
		IndexType:      HNSW,
		M:              16,
		EfConstruction: 100,
		EfSearch:       64,
		BatchConfig:    DefaultBatchConfig(),
	}

	if err := WithFSQQuantization(6, 0.25, 8, 8, 8, 6, 5)(config); err != nil {
		t.Fatalf("WithFSQQuantization: %v", err)
	}
	if config.Quantization == nil {
		t.Fatal("expected quantization config")
	}
	if got := config.Quantization.Type.String(); got != "fsq" {
		t.Fatalf("quantization type got %s want fsq", got)
	}
	wantLevels := []int{8, 8, 8, 6, 5}
	if len(config.Quantization.Levels) != len(wantLevels) {
		t.Fatalf("levels got %v", config.Quantization.Levels)
	}
	for i, want := range wantLevels {
		if config.Quantization.Levels[i] != want {
			t.Fatalf("levels got %v", config.Quantization.Levels)
		}
	}
}

func TestBackwardCompatibility(t *testing.T) {
	// Test that existing configurations still work without new fields
	config := &CollectionConfig{
		Dimension:      768,
		Metric:         CosineDistance,
		IndexType:      HNSW,
		M:              32,
		EfConstruction: 200,
		EfSearch:       50,
		ML:             1.0,
		Version:        1,
		AutoSave:       true,
		SaveInterval:   5 * time.Minute,
		SavePath:       "/tmp/test",
		// Note: No new fields set - should use defaults
	}

	// Should validate successfully even without new fields
	err := config.validate()
	if err != nil {
		t.Errorf("backward compatibility test failed: %v", err)
	}

	// Test that default batch config is applied
	if config.BatchConfig.ChunkSize == 0 {
		t.Error("expected default batch config to be applied")
	}
}

func TestFieldTypeString(t *testing.T) {
	tests := []struct {
		expected  string
		fieldType FieldType
	}{
		{expected: "string", fieldType: StringField},
		{expected: "int", fieldType: IntField},
		{expected: "float", fieldType: FloatField},
		{expected: "bool", fieldType: BoolField},
		{expected: "time", fieldType: TimeField},
		{expected: "string_array", fieldType: StringArrayField},
		{expected: "int_array", fieldType: IntArrayField},
		{expected: "float_array", fieldType: FloatArrayField},
		{expected: "unknown", fieldType: FieldType(999)},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			result := tt.fieldType.String()
			if result != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, result)
			}
		})
	}
}

func TestCachePolicyString(t *testing.T) {
	tests := []struct {
		expected string
		policy   CachePolicy
	}{
		{expected: "lru", policy: LRUCache},
		{expected: "lfu", policy: LFUCache},
		{expected: "fifo", policy: FIFOCache},
		{expected: "unknown", policy: CachePolicy(999)},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			result := tt.policy.String()
			if result != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, result)
			}
		})
	}
}

func TestDefaultBatchConfig(t *testing.T) {
	config := DefaultBatchConfig()

	if config.ChunkSize <= 0 {
		t.Errorf("expected positive chunk size, got %d", config.ChunkSize)
	}
	if config.MaxConcurrency <= 0 {
		t.Errorf("expected positive max concurrency, got %d", config.MaxConcurrency)
	}
	if config.TimeoutPerChunk <= 0 {
		t.Errorf("expected positive timeout per chunk, got %v", config.TimeoutPerChunk)
	}
}
