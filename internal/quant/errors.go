package quant

import (
	"context"
	"fmt"
	"time"
)

// QuantizationErrorCode represents specific quantization error types
type QuantizationErrorCode int

const (
	ErrQuantUnknown QuantizationErrorCode = iota
	ErrQuantConfigInvalid
	ErrQuantTrainingFailed
	ErrQuantTrainingDataInsufficient
	ErrQuantCompressionFailed
	ErrQuantDecompressionFailed
	ErrQuantDistanceComputationFailed
	ErrQuantCodebookCorrupted
	ErrQuantDimensionMismatch
	ErrQuantNotTrained
	ErrQuantMemoryExhausted
	ErrQuantConvergenceFailed
)

// QuantizationError represents a quantization-specific error
type QuantizationError struct {
	Code        QuantizationErrorCode  `json:"code"`
	Message     string                 `json:"message"`
	Component   string                 `json:"component"`
	Operation   string                 `json:"operation"`
	Retryable   bool                   `json:"retryable"`
	Recoverable bool                   `json:"recoverable"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	Cause       error                  `json:"cause,omitempty"`
	Timestamp   time.Time              `json:"timestamp"`
}

func (qe *QuantizationError) Error() string {
	if qe.Cause != nil {
		return fmt.Sprintf("quantization error in %s.%s: %s (caused by: %v)",
			qe.Component, qe.Operation, qe.Message, qe.Cause)
	}
	return fmt.Sprintf("quantization error in %s.%s: %s",
		qe.Component, qe.Operation, qe.Message)
}

// Unwrap returns the underlying cause error
func (qe *QuantizationError) Unwrap() error {
	return qe.Cause
}

// NewQuantizationError creates a new quantization error
func NewQuantizationError(code QuantizationErrorCode, component, operation, message string) *QuantizationError {
	return &QuantizationError{
		Code:      code,
		Message:   message,
		Component: component,
		Operation: operation,
		Metadata:  make(map[string]interface{}),
		Timestamp: time.Now(),
	}
}

// WithCause adds a cause error
func (qe *QuantizationError) WithCause(cause error) *QuantizationError {
	qe.Cause = cause
	return qe
}

// WithRetryable sets whether the error is retryable
func (qe *QuantizationError) WithRetryable(retryable bool) *QuantizationError {
	qe.Retryable = retryable
	return qe
}

// WithRecoverable sets whether the error is recoverable
func (qe *QuantizationError) WithRecoverable(recoverable bool) *QuantizationError {
	qe.Recoverable = recoverable
	return qe
}

// WithMetadata adds metadata to the error
func (qe *QuantizationError) WithMetadata(key string, value interface{}) *QuantizationError {
	if qe.Metadata == nil {
		qe.Metadata = make(map[string]interface{})
	}
	qe.Metadata[key] = value
	return qe
}

// QuantizationRecoveryManager handles quantization error recovery
type QuantizationRecoveryManager struct {
	fallbackToUncompressed bool
	maxTrainingRetries     int
	trainingRetryBackoff   time.Duration
}

// NewQuantizationRecoveryManager creates a new quantization recovery manager
func NewQuantizationRecoveryManager(fallbackToUncompressed bool) *QuantizationRecoveryManager {
	return &QuantizationRecoveryManager{
		fallbackToUncompressed: fallbackToUncompressed,
		maxTrainingRetries:     3,
		trainingRetryBackoff:   time.Second * 2,
	}
}

// RecoverFromTrainingFailure attempts to recover from training failures
func (qrm *QuantizationRecoveryManager) RecoverFromTrainingFailure(
	ctx context.Context,
	quantizer Quantizer,
	vectors [][]float32,
	err *QuantizationError,
) error {
	if !err.Retryable {
		return err
	}

	// Try different recovery strategies based on the error code
	switch err.Code {
	case ErrQuantTrainingDataInsufficient:
		return qrm.recoverFromInsufficientData(ctx, quantizer, vectors, err)
	case ErrQuantConvergenceFailed:
		return qrm.recoverFromConvergenceFailure(ctx, quantizer, vectors, err)
	case ErrQuantMemoryExhausted:
		return qrm.recoverFromMemoryExhaustion(ctx, quantizer, vectors, err)
	default:
		return qrm.retryTraining(ctx, quantizer, vectors, err)
	}
}

// recoverFromInsufficientData handles insufficient training data
func (qrm *QuantizationRecoveryManager) recoverFromInsufficientData(
	ctx context.Context,
	quantizer Quantizer,
	vectors [][]float32,
	err *QuantizationError,
) error {
	// Strategy 1: Use all available vectors for training
	if len(vectors) > 0 {
		config := quantizer.Config()
		if config != nil {
			// Temporarily increase train ratio to use more data
			originalRatio := config.TrainRatio
			config.TrainRatio = 1.0 // Use all vectors

			if configErr := quantizer.Configure(config); configErr != nil {
				return fmt.Errorf("failed to adjust training ratio: %w", configErr)
			}

			// Try training again
			if trainErr := quantizer.Train(ctx, vectors); trainErr == nil {
				return nil // Success
			}

			// Restore original ratio
			config.TrainRatio = originalRatio
			quantizer.Configure(config)
		}
	}

	// Strategy 2: Reduce quantization complexity
	return qrm.reduceQuantizationComplexity(ctx, quantizer, vectors)
}

// recoverFromConvergenceFailure handles convergence failures
func (qrm *QuantizationRecoveryManager) recoverFromConvergenceFailure(
	ctx context.Context,
	quantizer Quantizer,
	vectors [][]float32,
	err *QuantizationError,
) error {
	config := quantizer.Config()
	if config == nil {
		return fmt.Errorf("cannot recover: no configuration available")
	}

	// Strategy 1: Reduce number of codebooks for Product Quantization
	if config.Type == ProductQuantization && config.Codebooks > 2 {
		originalCodebooks := config.Codebooks
		config.Codebooks = config.Codebooks / 2

		if configErr := quantizer.Configure(config); configErr != nil {
			return fmt.Errorf("failed to reduce codebooks: %w", configErr)
		}

		if trainErr := quantizer.Train(ctx, vectors); trainErr == nil {
			return nil // Success
		}

		// Restore original configuration
		config.Codebooks = originalCodebooks
		quantizer.Configure(config)
	}

	// Strategy 2: Reduce bit precision
	if config.Bits > 4 {
		originalBits := config.Bits
		config.Bits = config.Bits / 2

		if configErr := quantizer.Configure(config); configErr != nil {
			return fmt.Errorf("failed to reduce bits: %w", configErr)
		}

		if trainErr := quantizer.Train(ctx, vectors); trainErr == nil {
			return nil // Success
		}

		// Restore original configuration
		config.Bits = originalBits
		quantizer.Configure(config)
	}

	return fmt.Errorf("convergence recovery failed")
}

// recoverFromMemoryExhaustion handles memory exhaustion during training
func (qrm *QuantizationRecoveryManager) recoverFromMemoryExhaustion(
	ctx context.Context,
	quantizer Quantizer,
	vectors [][]float32,
	err *QuantizationError,
) error {
	config := quantizer.Config()
	if config == nil {
		return fmt.Errorf("cannot recover: no configuration available")
	}

	// Strategy 1: Reduce training data size
	if config.TrainRatio > 0.1 {
		originalRatio := config.TrainRatio
		config.TrainRatio = config.TrainRatio / 2

		if configErr := quantizer.Configure(config); configErr != nil {
			return fmt.Errorf("failed to reduce training ratio: %w", configErr)
		}

		if trainErr := quantizer.Train(ctx, vectors); trainErr == nil {
			return nil // Success
		}

		// Restore original configuration
		config.TrainRatio = originalRatio
		quantizer.Configure(config)
	}

	// Strategy 2: Reduce cache size
	if config.CacheSize > 100 {
		originalCacheSize := config.CacheSize
		config.CacheSize = config.CacheSize / 2

		if configErr := quantizer.Configure(config); configErr != nil {
			return fmt.Errorf("failed to reduce cache size: %w", configErr)
		}

		if trainErr := quantizer.Train(ctx, vectors); trainErr == nil {
			return nil // Success
		}

		// Restore original configuration
		config.CacheSize = originalCacheSize
		quantizer.Configure(config)
	}

	return fmt.Errorf("memory exhaustion recovery failed")
}

// reduceQuantizationComplexity reduces the complexity of quantization configuration
func (qrm *QuantizationRecoveryManager) reduceQuantizationComplexity(
	ctx context.Context,
	quantizer Quantizer,
	vectors [][]float32,
) error {
	config := quantizer.Config()
	if config == nil {
		return fmt.Errorf("cannot reduce complexity: no configuration available")
	}

	// Create a simplified configuration
	simplifiedConfig := *config

	switch config.Type {
	case ProductQuantization:
		// Reduce codebooks and bits
		if simplifiedConfig.Codebooks > 4 {
			simplifiedConfig.Codebooks = 4
		}
		if simplifiedConfig.Bits > 4 {
			simplifiedConfig.Bits = 4
		}
	case ScalarQuantization:
		// Reduce bits
		if simplifiedConfig.Bits > 4 {
			simplifiedConfig.Bits = 4
		}
	}

	// Reduce training ratio and cache size
	simplifiedConfig.TrainRatio = 0.05 // Use minimal training data
	simplifiedConfig.CacheSize = 100   // Minimal cache

	if configErr := quantizer.Configure(&simplifiedConfig); configErr != nil {
		return fmt.Errorf("failed to apply simplified configuration: %w", configErr)
	}

	if trainErr := quantizer.Train(ctx, vectors); trainErr != nil {
		return fmt.Errorf("training failed even with simplified configuration: %w", trainErr)
	}

	return nil
}

// retryTraining performs simple retry with backoff
func (qrm *QuantizationRecoveryManager) retryTraining(
	ctx context.Context,
	quantizer Quantizer,
	vectors [][]float32,
	err *QuantizationError,
) error {
	for attempt := 1; attempt <= qrm.maxTrainingRetries; attempt++ {
		// Apply backoff
		if attempt > 1 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(qrm.trainingRetryBackoff * time.Duration(attempt-1)):
			}
		}

		// Retry training
		if trainErr := quantizer.Train(ctx, vectors); trainErr == nil {
			return nil // Success
		}
	}

	return fmt.Errorf("training failed after %d retries", qrm.maxTrainingRetries)
}

// FallbackToUncompressed provides fallback to uncompressed storage
func (qrm *QuantizationRecoveryManager) FallbackToUncompressed() bool {
	return qrm.fallbackToUncompressed
}

// ValidateQuantizationHealth checks the health of a quantizer
func ValidateQuantizationHealth(quantizer Quantizer) error {
	if !quantizer.IsTrained() {
		return NewQuantizationError(
			ErrQuantNotTrained,
			"quantizer",
			"validate",
			"quantizer is not trained",
		).WithRecoverable(true)
	}

	config := quantizer.Config()
	if config == nil {
		return NewQuantizationError(
			ErrQuantConfigInvalid,
			"quantizer",
			"validate",
			"quantizer configuration is nil",
		).WithRecoverable(false)
	}

	if err := config.Validate(); err != nil {
		return NewQuantizationError(
			ErrQuantConfigInvalid,
			"quantizer",
			"validate",
			"quantizer configuration is invalid",
		).WithCause(err).WithRecoverable(true)
	}

	// Check memory usage
	memUsage := quantizer.MemoryUsage()
	if memUsage < 0 {
		return NewQuantizationError(
			ErrQuantMemoryExhausted,
			"quantizer",
			"validate",
			"invalid memory usage reported",
		).WithRecoverable(true)
	}

	return nil
}
