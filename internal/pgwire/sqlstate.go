package pgwire

import (
	"errors"
	"strings"

	"github.com/xDarkicex/libravdb/internal/catalog"
)

// PostgreSQL SQLSTATE error codes used by the pgwire protocol.
// Clients (ORMs, drivers, tools) branch on these codes for error handling.
const (
	// Class 0A — Feature Not Supported
	SQLStateFeatureNotSupported = "0A000"

	// Class 22 — Data Exception
	SQLStateInvalidParameter = "22012"

	// Class 23 — Integrity Constraint Violation
	SQLStateUniqueViolation = "23505"

	// Class 42 — Syntax Error or Access Rule Violation
	SQLStateSyntaxError     = "42601"
	SQLStateUndefinedTable  = "42P01"
	SQLStateUndefinedColumn = "42703"
	SQLStateDuplicateTable  = "42P07"

	// Class 53 — Insufficient Resources
	SQLStateTooManyConnections = "53300"

	// Class 58 — System Error
	SQLStateInternalError = "58000"
)

// errorToSQLState maps a Go error to a PostgreSQL SQLSTATE code.
// It unwraps the error chain looking for known sentinel errors and
// falls back to diagnostic string matching for executor-generated messages.
func errorToSQLState(err error) string {
	if err == nil {
		return SQLStateInternalError
	}

	// Check catalog sentinel errors
	if errors.Is(err, catalog.ErrTableNotFound) {
		return SQLStateUndefinedTable
	}
	if errors.Is(err, catalog.ErrColumnNotFound) {
		return SQLStateUndefinedColumn
	}

	// String matching for executor-generated errors
	msg := err.Error()

	switch {
	case strings.Contains(msg, "parse error"):
		return SQLStateSyntaxError
	case strings.Contains(msg, "table") && strings.Contains(msg, "not found"):
		return SQLStateUndefinedTable
	case strings.Contains(msg, "column") && strings.Contains(msg, "not found"):
		return SQLStateUndefinedColumn
	case strings.Contains(msg, "column") && strings.Contains(msg, "not accepted"):
		return SQLStateInvalidParameter
	case strings.Contains(msg, "identifier") && strings.Contains(msg, "not found"):
		return SQLStateUndefinedColumn
	case strings.Contains(msg, "already exists"):
		return SQLStateUniqueViolation
	case strings.Contains(msg, "duplicate"):
		return SQLStateUniqueViolation
	case strings.Contains(msg, "requires a WHERE"):
		return SQLStateSyntaxError
	case strings.Contains(msg, "requires an 'id'"):
		return SQLStateSyntaxError
	case strings.Contains(msg, "Extended query") && strings.Contains(msg, "not supported"):
		return SQLStateFeatureNotSupported
	case strings.Contains(msg, "catalog not initialized"):
		return SQLStateInternalError
	case strings.Contains(msg, "dimension mismatch") || strings.Contains(msg, "Dimension"):
		return SQLStateInvalidParameter
	default:
		return SQLStateInternalError
	}
}
