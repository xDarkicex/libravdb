package btree

import "errors"

var (
	errKeyTooLarge  = errors.New("btree: key exceeds maximum size")
	errKeyNotFound  = errors.New("btree: key not found")
	errKeyExists    = errors.New("btree: key already exists")
	errTreeClosed   = errors.New("btree: tree is closed")
	errTreeCorrupt  = errors.New("btree: tree structure is corrupt")
)
