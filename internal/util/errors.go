package util

import "errors"

// Index-level sentinels shared across HNSW, IVFPQ, and Flat packages.
var (
	ErrEmptyIndex = errors.New("index is empty")
	ErrInvalidK   = errors.New("k must be positive")
	ErrNotTrained = errors.New("index must be trained before operation")
	ErrDimension  = errors.New("vector dimension does not match index dimension")
	ErrNotFound   = errors.New("not found")
)
