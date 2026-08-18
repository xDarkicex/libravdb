package libravdb

import (
	"errors"
	"strings"

	"github.com/xDarkicex/libravdb/internal/catalog"
)

// Stable SQL error codes exposed by the native API and SDK adapters. The
// human-readable message remains available through Error and is intentionally
// kept separate from this machine-readable classification.
const (
	SQLErrorSyntax                 = "sql.syntax_error"
	SQLErrorUnsupported            = "sql.unsupported_feature"
	SQLErrorUnsupportedTemporalAgg = "sql.unsupported_temporal_aggregate"
	SQLErrorUndefinedTable         = "sql.undefined_table"
	SQLErrorUndefinedColumn        = "sql.undefined_column"
	SQLErrorInvalidParameter       = "sql.invalid_parameter"
	SQLErrorStorage                = "sql.storage_error"
	SQLErrorIntegrity              = "sql.integrity_violation"
)

// SQLError is the additive, structured error returned by Database.Query and
// Database.QueryWithParams. Cause remains available to callers using
// errors.Is/errors.As; existing error text is preserved in Message.
type SQLError struct {
	Code     string
	SQLState string
	Message  string
	Cause    error
}

func (e *SQLError) Error() string {
	if e == nil {
		return "<nil>"
	}
	return e.Message
}

func (e *SQLError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

func (e *SQLError) Is(target error) bool {
	other, ok := target.(*SQLError)
	return ok && other != nil && e != nil && e.Code == other.Code
}

func newSQLError(code, state string, cause error) error {
	if cause == nil {
		return nil
	}
	var existing *SQLError
	if errors.As(cause, &existing) {
		return cause
	}
	return &SQLError{Code: code, SQLState: state, Message: cause.Error(), Cause: cause}
}

// normalizeSQLError classifies legacy executor errors at the public SQL
// boundary. New code can return SQLError directly; this compatibility mapper
// keeps older execution paths machine-readable without changing their text.
func normalizeSQLError(err error) error {
	if err == nil {
		return nil
	}
	var structured *SQLError
	if errors.As(err, &structured) {
		return err
	}
	if errors.Is(err, catalog.ErrTableNotFound) || strings.Contains(err.Error(), "table ") && strings.Contains(err.Error(), "not found") {
		return newSQLError(SQLErrorUndefinedTable, "42P01", err)
	}
	if errors.Is(err, catalog.ErrColumnNotFound) || strings.Contains(err.Error(), "column ") && strings.Contains(err.Error(), "not found") || strings.Contains(err.Error(), "identifier ") && strings.Contains(err.Error(), "not found") {
		return newSQLError(SQLErrorUndefinedColumn, "42703", err)
	}
	msg := strings.ToLower(err.Error())
	switch {
	case strings.Contains(msg, "parse error"), strings.Contains(msg, "unexpected token"), strings.Contains(msg, "syntax"):
		return newSQLError(SQLErrorSyntax, "42601", err)
	case strings.Contains(msg, "as of lsn") && strings.Contains(msg, "not supported"), strings.Contains(msg, "temporal execution not supported"):
		return newSQLError(SQLErrorUnsupportedTemporalAgg, "0A000", err)
	case strings.Contains(msg, "parameter") && (strings.Contains(msg, "not bound") || strings.Contains(msg, "must be") || strings.Contains(msg, "invalid")):
		return newSQLError(SQLErrorInvalidParameter, "22023", err)
	case strings.Contains(msg, "duplicate") || strings.Contains(msg, "unique") || strings.Contains(msg, "already exists"):
		return newSQLError(SQLErrorIntegrity, "23505", err)
	case strings.Contains(msg, "unsupported") || strings.Contains(msg, "not implemented"):
		return newSQLError(SQLErrorUnsupported, "0A000", err)
	case strings.Contains(msg, "wal") || strings.Contains(msg, "storage") || strings.Contains(msg, "recovery") || strings.Contains(msg, "btree"):
		return newSQLError(SQLErrorStorage, "58000", err)
	default:
		return newSQLError(SQLErrorStorage, "58000", err)
	}
}

// AsSQLError returns the stable SQL classification for an execution error.
// It is intended for adapters such as pgwire and the SDK FFI boundary; the
// underlying error text and unwrap chain remain unchanged.
func AsSQLError(err error) *SQLError {
	if err == nil {
		return nil
	}
	normalized := normalizeSQLError(err)
	var structured *SQLError
	if errors.As(normalized, &structured) {
		return structured
	}
	return nil
}
