package quant

import (
	"testing"
)

func TestQuantizationType_String(t *testing.T) {
	tests := []struct {
		qType    QuantizationType
		expected string
	}{
		{ProductQuantization, "product"},
		{ScalarQuantization, "scalar"},
		{QuantizationType(999), "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			if got := tt.qType.String(); got != tt.expected {
				t.Errorf("QuantizationType.String() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestQuantizationConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		config  *QuantizationConfig
		wantErr bool
	}{
		{
			name: "valid product quantization config",
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  8,
				Bits:       8,
				TrainRatio: 0.1,
				CacheSize:  1000,
			},
			wantErr: false,
		},
		{
			name: "valid scalar quantization config",
			config: &QuantizationConfig{
				Type:       ScalarQuantization,
				Bits:       8,
				TrainRatio: 0.1,
			},
			wantErr: false,
		},
		{
			name: "invalid bits - too low",
			config: &QuantizationConfig{
				Type:       ScalarQuantization,
				Bits:       0,
				TrainRatio: 0.1,
			},
			wantErr: true,
		},
		{
			name: "invalid bits - too high",
			config: &QuantizationConfig{
				Type:       ScalarQuantization,
				Bits:       33,
				TrainRatio: 0.1,
			},
			wantErr: true,
		},
		{
			name: "invalid train ratio - too low",
			config: &QuantizationConfig{
				Type:       ScalarQuantization,
				Bits:       8,
				TrainRatio: 0.0,
			},
			wantErr: true,
		},
		{
			name: "invalid train ratio - too high",
			config: &QuantizationConfig{
				Type:       ScalarQuantization,
				Bits:       8,
				TrainRatio: 1.1,
			},
			wantErr: true,
		},
		{
			name: "invalid codebooks for product quantization",
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  0,
				Bits:       8,
				TrainRatio: 0.1,
			},
			wantErr: true,
		},
		{
			name: "invalid cache size",
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  8,
				Bits:       8,
				TrainRatio: 0.1,
				CacheSize:  -1,
			},
			wantErr: true,
		},
		{
			name: "unsupported quantization type",
			config: &QuantizationConfig{
				Type:       QuantizationType(999),
				Bits:       8,
				TrainRatio: 0.1,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("QuantizationConfig.Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestDefaultConfig(t *testing.T) {
	tests := []struct {
		name  string
		qType QuantizationType
		want  *QuantizationConfig
	}{
		{
			name:  "product quantization default",
			qType: ProductQuantization,
			want: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  8,
				Bits:       8,
				TrainRatio: 0.1,
				CacheSize:  1000,
			},
		},
		{
			name:  "scalar quantization default",
			qType: ScalarQuantization,
			want: &QuantizationConfig{
				Type:       ScalarQuantization,
				Bits:       8,
				TrainRatio: 0.1,
			},
		},
		{
			name:  "unsupported type",
			qType: QuantizationType(999),
			want:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DefaultConfig(tt.qType)
			if tt.want == nil {
				if got != nil {
					t.Errorf("DefaultConfig() = %v, want nil", got)
				}
				return
			}

			if got == nil {
				t.Errorf("DefaultConfig() = nil, want %v", tt.want)
				return
			}

			if got.Type != tt.want.Type ||
				got.Codebooks != tt.want.Codebooks ||
				got.Bits != tt.want.Bits ||
				got.TrainRatio != tt.want.TrainRatio ||
				got.CacheSize != tt.want.CacheSize {
				t.Errorf("DefaultConfig() = %v, want %v", got, tt.want)
			}
		})
	}
}
