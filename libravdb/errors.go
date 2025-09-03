package libravdb

import (
	"errors"
	"fmt"
	"time"
)

// Core errors
var (
	ErrDatabaseClosed     = errors.New("database is closed")
	ErrCollectionClosed   = errors.New("collection is closed")
	ErrTooManyCollections = errors.New("maximum number of collections exceeded")
	ErrCollectionNotFound = errors.New("collection not found")
	ErrInvalidDimension   = errors.New("invalid vector dimension")
	ErrInvalidK           = errors.New("k must be positive")
	ErrEmptyIndex         = errors.New("index is empty")
)

// ErrorCode represents structured error codes
type ErrorCode int

const (
	ErrCodeUnknown ErrorCode = iota
	ErrCodeInvalidVector
	ErrCodeIndexCorrupted
	ErrCodeStorageFailure
	ErrCodeMemoryExhausted
	ErrCodeTimeout
	ErrCodeRateLimited
)

// VectorDBError represents a structured error with additional context
type VectorDBError struct {
	Code      ErrorCode   `json:"code"`
	Message   string      `json:"message"`
	Details   interface{} `json:"details,omitempty"`
	Retryable bool        `json:"retryable"`
	Timestamp time.Time   `json:"timestamp"`
}

func (e *VectorDBError) Error() string {
	return fmt.Sprintf("VectorDB Error %d: %s", e.Code, e.Message)
}

// NewVectorDBError creates a new structured error
func NewVectorDBError(code ErrorCode, message string, retryable bool) *VectorDBError {
	return &VectorDBError{
		Code:      code,
		Message:   message,
		Retryable: retryable,
		Timestamp: time.Now(),
	}
}
