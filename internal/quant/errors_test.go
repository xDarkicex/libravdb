package quant

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestQuantizationError(t *testing.T) {
	t.Run("basic error creation", func(t *testing.T) {
		err := NewQuantizationError(
			ErrQuantTrainingFailed,
			"ProductQuantizer",
			"Train",
			"k-means convergence failed",
		)

		if err.Code != ErrQuantTrainingFailed {
			t.Errorf("expected code %d, got %d", ErrQuantTrainingFailed, err.Code)
		}

		if err.Component != "ProductQuantizer" {
			t.Errorf("expected component 'ProductQuantizer', got '%s'", err.Component)
		}

		if err.Operation != "Train" {
			t.Errorf("expected operation 'Train', got '%s'", err.Operation)
		}

		if err.Message != "k-means convergence failed" {
			t.Errorf("expected message 'k-means convergence failed', got '%s'", err.Message)
		}
	})

	t.Run("error with cause and metadata", func(t *testing.T) {
		cause := errors.New("insufficient memory")
		err := NewQuantizationError(
			ErrQuantMemoryExhausted,
			"ProductQuantizer",
			"Train",
			"training failed due to memory constraints",
		).WithCause(cause).
			WithRetryable(true).
			WithRecoverable(true).
			WithMetadata("memory_required", "2GB").
			WithMetadata("memory_available", "1GB")

		if err.Cause != cause {
			t.Error("expected cause to be set")
		}

		if !err.Retryable {
			t.Error("expected error to be retryable")
		}

		if !err.Recoverable {
			t.Error("expected error to be recoverable")
		}

		if err.Metadata["memory_required"] != "2GB" {
			t.Error("expected memory_required metadata to be set")
		}

		if !errors.Is(err, cause) {
			t.Error("expected error to wrap cause")
		}
	})
}

func TestQuantizationRecoveryManager(t *testing.T) {
	t.Run("recover from insufficient data", func(t *testing.T) {
		qrm := NewQuantizationRecoveryManager(true)

		mockQuantizer := &mockQuantizerForErrors{
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  8,
				Bits:       8,
				TrainRatio: 0.1,
			},
			trainShouldFail: false,
		}

		vectors := generateTestVectors(100, 128)

		err := NewQuantizationError(
			ErrQuantTrainingDataInsufficient,
			"ProductQuantizer",
			"Train",
			"insufficient training data",
		).WithRetryable(true)

		ctx := context.Background()
		if recoveryErr := qrm.RecoverFromTrainingFailure(ctx, mockQuantizer, vectors, err); recoveryErr != nil {
			t.Errorf("expected recovery to succeed, got error: %v", recoveryErr)
		}

		// Check that train ratio was adjusted
		if mockQuantizer.config.TrainRatio != 1.0 {
			t.Errorf("expected train ratio to be adjusted to 1.0, got %f", mockQuantizer.config.TrainRatio)
		}
	})

	t.Run("recover from convergence failure", func(t *testing.T) {
		qrm := NewQuantizationRecoveryManager(true)

		mockQuantizer := &mockQuantizerForErrors{
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  16,
				Bits:       16,
				TrainRatio: 0.1,
			},
			trainShouldFail: false,
		}

		vectors := generateTestVectors(1000, 128)

		err := NewQuantizationError(
			ErrQuantConvergenceFailed,
			"ProductQuantizer",
			"Train",
			"k-means failed to converge",
		).WithRetryable(true)

		ctx := context.Background()
		if recoveryErr := qrm.RecoverFromTrainingFailure(ctx, mockQuantizer, vectors, err); recoveryErr != nil {
			t.Errorf("expected recovery to succeed, got error: %v", recoveryErr)
		}

		// Check that codebooks were reduced
		if mockQuantizer.config.Codebooks >= 16 {
			t.Errorf("expected codebooks to be reduced from 16, got %d", mockQuantizer.config.Codebooks)
		}
	})

	t.Run("recover from memory exhaustion", func(t *testing.T) {
		qrm := NewQuantizationRecoveryManager(true)

		mockQuantizer := &mockQuantizerForErrors{
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  8,
				Bits:       8,
				TrainRatio: 0.5,
				CacheSize:  1000,
			},
			trainShouldFail: false,
		}

		vectors := generateTestVectors(1000, 128)

		err := NewQuantizationError(
			ErrQuantMemoryExhausted,
			"ProductQuantizer",
			"Train",
			"out of memory during training",
		).WithRetryable(true)

		ctx := context.Background()
		if recoveryErr := qrm.RecoverFromTrainingFailure(ctx, mockQuantizer, vectors, err); recoveryErr != nil {
			t.Errorf("expected recovery to succeed, got error: %v", recoveryErr)
		}

		// Check that train ratio was reduced
		if mockQuantizer.config.TrainRatio >= 0.5 {
			t.Errorf("expected train ratio to be reduced from 0.5, got %f", mockQuantizer.config.TrainRatio)
		}
	})

	t.Run("retry training", func(t *testing.T) {
		qrm := NewQuantizationRecoveryManager(true)
		qrm.maxTrainingRetries = 2
		qrm.trainingRetryBackoff = time.Millisecond * 10

		mockQuantizer := &mockQuantizerForErrors{
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  8,
				Bits:       8,
				TrainRatio: 0.1,
			},
			trainFailCount: 1, // Fail once, then succeed
		}

		vectors := generateTestVectors(1000, 128)

		err := NewQuantizationError(
			ErrQuantTrainingFailed,
			"ProductQuantizer",
			"Train",
			"training failed",
		).WithRetryable(true)

		ctx := context.Background()
		if recoveryErr := qrm.RecoverFromTrainingFailure(ctx, mockQuantizer, vectors, err); recoveryErr != nil {
			t.Errorf("expected recovery to succeed after retry, got error: %v", recoveryErr)
		}

		if mockQuantizer.trainCallCount != 2 {
			t.Errorf("expected 2 train calls, got %d", mockQuantizer.trainCallCount)
		}
	})

	t.Run("non-retryable error", func(t *testing.T) {
		qrm := NewQuantizationRecoveryManager(true)

		mockQuantizer := &mockQuantizerForErrors{}
		vectors := generateTestVectors(100, 128)

		err := NewQuantizationError(
			ErrQuantConfigInvalid,
			"ProductQuantizer",
			"Configure",
			"invalid configuration",
		).WithRetryable(false)

		ctx := context.Background()
		if recoveryErr := qrm.RecoverFromTrainingFailure(ctx, mockQuantizer, vectors, err); recoveryErr == nil {
			t.Error("expected recovery to fail for non-retryable error")
		}
	})

	t.Run("context cancellation", func(t *testing.T) {
		qrm := NewQuantizationRecoveryManager(true)
		qrm.trainingRetryBackoff = time.Second

		mockQuantizer := &mockQuantizerForErrors{
			trainShouldFail: true,
		}

		vectors := generateTestVectors(100, 128)

		err := NewQuantizationError(
			ErrQuantTrainingFailed,
			"ProductQuantizer",
			"Train",
			"training failed",
		).WithRetryable(true)

		ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond*100)
		defer cancel()

		recoveryErr := qrm.RecoverFromTrainingFailure(ctx, mockQuantizer, vectors, err)
		if recoveryErr == nil {
			t.Error("expected recovery to fail due to context cancellation")
		}

		if !errors.Is(recoveryErr, context.DeadlineExceeded) {
			t.Error("expected context deadline exceeded error")
		}
	})
}

func TestValidateQuantizationHealth(t *testing.T) {
	t.Run("healthy quantizer", func(t *testing.T) {
		mockQuantizer := &mockQuantizerForErrors{
			trained: true,
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  8,
				Bits:       8,
				TrainRatio: 0.1,
			},
			memoryUsage: 1024,
		}

		if err := ValidateQuantizationHealth(mockQuantizer); err != nil {
			t.Errorf("expected healthy quantizer to pass validation, got error: %v", err)
		}
	})

	t.Run("not trained", func(t *testing.T) {
		mockQuantizer := &mockQuantizerForErrors{
			trained: false,
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  8,
				Bits:       8,
				TrainRatio: 0.1,
			},
		}

		err := ValidateQuantizationHealth(mockQuantizer)
		if err == nil {
			t.Error("expected validation to fail for untrained quantizer")
		}

		qErr, ok := err.(*QuantizationError)
		if !ok {
			t.Error("expected QuantizationError")
		}

		if qErr.Code != ErrQuantNotTrained {
			t.Errorf("expected error code %d, got %d", ErrQuantNotTrained, qErr.Code)
		}

		if !qErr.Recoverable {
			t.Error("expected error to be recoverable")
		}
	})

	t.Run("nil config", func(t *testing.T) {
		mockQuantizer := &mockQuantizerForErrors{
			trained: true,
			config:  nil,
		}

		err := ValidateQuantizationHealth(mockQuantizer)
		if err == nil {
			t.Error("expected validation to fail for nil config")
		}

		qErr, ok := err.(*QuantizationError)
		if !ok {
			t.Error("expected QuantizationError")
		}

		if qErr.Code != ErrQuantConfigInvalid {
			t.Errorf("expected error code %d, got %d", ErrQuantConfigInvalid, qErr.Code)
		}
	})

	t.Run("invalid config", func(t *testing.T) {
		mockQuantizer := &mockQuantizerForErrors{
			trained: true,
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  0, // Invalid
				Bits:       8,
				TrainRatio: 0.1,
			},
		}

		err := ValidateQuantizationHealth(mockQuantizer)
		if err == nil {
			t.Error("expected validation to fail for invalid config")
		}

		qErr, ok := err.(*QuantizationError)
		if !ok {
			t.Error("expected QuantizationError")
		}

		if qErr.Code != ErrQuantConfigInvalid {
			t.Errorf("expected error code %d, got %d", ErrQuantConfigInvalid, qErr.Code)
		}
	})

	t.Run("invalid memory usage", func(t *testing.T) {
		mockQuantizer := &mockQuantizerForErrors{
			trained: true,
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  8,
				Bits:       8,
				TrainRatio: 0.1,
			},
			memoryUsage: -1, // Invalid
		}

		err := ValidateQuantizationHealth(mockQuantizer)
		if err == nil {
			t.Error("expected validation to fail for invalid memory usage")
		}

		qErr, ok := err.(*QuantizationError)
		if !ok {
			t.Error("expected QuantizationError")
		}

		if qErr.Code != ErrQuantMemoryExhausted {
			t.Errorf("expected error code %d, got %d", ErrQuantMemoryExhausted, qErr.Code)
		}
	})
}

// Mock quantizer for testing
type mockQuantizerForErrors struct {
	trained         bool
	config          *QuantizationConfig
	memoryUsage     int64
	trainShouldFail bool
	trainFailCount  int
	trainCallCount  int
}

func (mq *mockQuantizerForErrors) Train(ctx context.Context, vectors [][]float32) error {
	mq.trainCallCount++

	if mq.trainShouldFail {
		return errors.New("training failed")
	}

	if mq.trainFailCount > 0 {
		mq.trainFailCount--
		return errors.New("training failed")
	}

	mq.trained = true
	return nil
}

func (mq *mockQuantizerForErrors) Configure(config *QuantizationConfig) error {
	if config == nil {
		return errors.New("config cannot be nil")
	}
	mq.config = config
	return nil
}

func (mq *mockQuantizerForErrors) Compress(vector []float32) ([]byte, error) {
	return []byte{1, 2, 3, 4}, nil
}

func (mq *mockQuantizerForErrors) Decompress(data []byte) ([]float32, error) {
	return []float32{1.0, 2.0, 3.0, 4.0}, nil
}

func (mq *mockQuantizerForErrors) Distance(compressed1, compressed2 []byte) (float32, error) {
	return 0.5, nil
}

func (mq *mockQuantizerForErrors) DistanceToQuery(compressed []byte, query []float32) (float32, error) {
	return 0.3, nil
}

func (mq *mockQuantizerForErrors) CompressionRatio() float32 {
	return 8.0
}

func (mq *mockQuantizerForErrors) MemoryUsage() int64 {
	return mq.memoryUsage
}

func (mq *mockQuantizerForErrors) IsTrained() bool {
	return mq.trained
}

func (mq *mockQuantizerForErrors) Config() *QuantizationConfig {
	return mq.config
}

// Helper function to generate test vectors
func generateTestVectors(count, dimension int) [][]float32 {
	vectors := make([][]float32, count)
	for i := range vectors {
		vectors[i] = make([]float32, dimension)
		for j := range vectors[i] {
			vectors[i][j] = float32(i*dimension + j)
		}
	}
	return vectors
}
