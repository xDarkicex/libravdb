package libravdb

import (
	"bytes"
	"fmt"
	"strings"
	"time"

	"github.com/xDarkicex/lexer/parser"
)

// SessionConfig contains connection-local SQL controls. It is deliberately a
// fixed-shape value, not a map, so changing a setting cannot contend on a
// shared registry or allocate a dynamic value.
type SessionConfig struct {
	StatementTimeout  time.Duration
	MaxRecursionDepth uint32
	EnableSeqScan     bool
	TimeZone          string
	JIT               string
}

const DefaultMaxRecursionDepth uint32 = 10000

func DefaultSessionConfig() SessionConfig {
	return SessionConfig{MaxRecursionDepth: DefaultMaxRecursionDepth, TimeZone: "UTC", JIT: "on"}
}

// ApplySetConfig applies the connection-local subset of PostgreSQL's
// set_config() function. The value is returned by the function and retained
// in the existing session configuration; it is never persisted.
func (c *SessionConfig) ApplySetConfig(name, value string, local bool) error {
	if c == nil {
		return fmt.Errorf("set_config: nil session")
	}
	if local {
		return fmt.Errorf("set_config(..., true) is not supported; use false")
	}
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "timezone", "time zone":
		value = strings.TrimSpace(value)
		if value == "" {
			return fmt.Errorf("set_config: timezone value is empty")
		}
		c.TimeZone = value
		return nil
	case "jit":
		value = strings.ToLower(strings.TrimSpace(value))
		if value != "on" && value != "off" {
			return fmt.Errorf("set_config: jit value must be on or off")
		}
		c.JIT = value
		return nil
	default:
		return fmt.Errorf("set_config: unsupported setting %q", name)
	}
}

// EffectiveTimeout combines the user setting with the server safety ceiling.
// A zero statement_timeout means no user timeout; it never disables the
// server-side resource guard.
func (c SessionConfig) EffectiveTimeout(serverCeiling time.Duration) time.Duration {
	if c.StatementTimeout <= 0 {
		return serverCeiling
	}
	if serverCeiling <= 0 || c.StatementTimeout < serverCeiling {
		return c.StatementTimeout
	}
	return serverCeiling
}

// ApplySessionSetting applies one parsed SET/RESET statement. Session-local
// settings are process state only: they never enter the catalog or WAL.
func (c *SessionConfig) ApplySessionSetting(src []byte, doc *parser.QueryDoc, stmt *parser.SessionSettingStmt) error {
	if c == nil || doc == nil || stmt == nil {
		return fmt.Errorf("session setting: nil argument")
	}
	if stmt.Local {
		return fmt.Errorf("SET LOCAL is not supported; use SET for the connection")
	}
	if stmt.Reset {
		defaults := DefaultSessionConfig()
		switch stmt.Kind {
		case parser.SessionSettingStatementTimeout:
			c.StatementTimeout = defaults.StatementTimeout
		case parser.SessionSettingMaxRecursionDepth:
			c.MaxRecursionDepth = defaults.MaxRecursionDepth
		case parser.SessionSettingEnableSeqScan:
			c.EnableSeqScan = defaults.EnableSeqScan
		default:
			return fmt.Errorf("unsupported session setting kind %d", stmt.Kind)
		}
		return nil
	}

	value, err := sessionSettingValue(src, doc, stmt.Value)
	if err != nil {
		return err
	}
	if bytes.EqualFold(value, []byte("default")) {
		defaults := DefaultSessionConfig()
		switch stmt.Kind {
		case parser.SessionSettingStatementTimeout:
			c.StatementTimeout = defaults.StatementTimeout
		case parser.SessionSettingMaxRecursionDepth:
			c.MaxRecursionDepth = defaults.MaxRecursionDepth
		case parser.SessionSettingEnableSeqScan:
			c.EnableSeqScan = defaults.EnableSeqScan
		default:
			return fmt.Errorf("unsupported session setting kind %d", stmt.Kind)
		}
		return nil
	}
	switch stmt.Kind {
	case parser.SessionSettingStatementTimeout:
		d, err := parseSessionTimeout(value)
		if err != nil {
			return fmt.Errorf("statement_timeout: %w", err)
		}
		c.StatementTimeout = d
	case parser.SessionSettingMaxRecursionDepth:
		depth, err := parseSessionUint(value)
		if err != nil || depth == 0 || depth > uint64(DefaultMaxRecursionDepth) {
			return fmt.Errorf("max_recursion_depth must be between 1 and %d", DefaultMaxRecursionDepth)
		}
		c.MaxRecursionDepth = uint32(depth)
	case parser.SessionSettingEnableSeqScan:
		// The optimizer currently has no sequence-scan/index choice to toggle.
		// Rejecting is intentional; accepting a no-op setting would mislead
		// PostgreSQL clients and make query plans unpredictable.
		return fmt.Errorf("enable_seqscan is not supported by the current planner")
	default:
		return fmt.Errorf("unsupported session setting kind %d", stmt.Kind)
	}
	return nil
}

func sessionSettingValue(src []byte, doc *parser.QueryDoc, ref parser.NodeRef) ([]byte, error) {
	switch ref.Kind {
	case parser.NodeKindNumber:
		if ref.ID < 0 || int(ref.ID) >= len(doc.Numbers) {
			return nil, fmt.Errorf("invalid numeric value")
		}
		n := doc.Numbers[ref.ID]
		return src[n.Start:n.End], nil
	case parser.NodeKindString:
		if ref.ID < 0 || int(ref.ID) >= len(doc.Strings) {
			return nil, fmt.Errorf("invalid string value")
		}
		s := doc.Strings[ref.ID]
		if s.Start >= s.End || int(s.End) > len(src) {
			return nil, fmt.Errorf("invalid string value span")
		}
		value := src[s.Start:s.End]
		if len(value) >= 2 && (value[0] == '\'' || value[0] == '"') {
			return value[1 : len(value)-1], nil
		}
		if len(value) >= 3 && (value[0] == 'e' || value[0] == 'E') && value[1] == '\'' {
			return value[2 : len(value)-1], nil
		}
		return value, nil
	case parser.NodeKindIdentifier:
		if ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
			return nil, fmt.Errorf("invalid identifier value")
		}
		i := doc.Identifiers[ref.ID]
		return src[i.Start:i.End], nil
	default:
		return nil, fmt.Errorf("session setting requires a scalar value")
	}
}

func parseSessionUint(value []byte) (uint64, error) {
	if len(value) == 0 {
		return 0, fmt.Errorf("value is empty")
	}
	var n uint64
	for _, b := range value {
		if b < '0' || b > '9' || n > (^uint64(0)-uint64(b-'0'))/10 {
			return 0, fmt.Errorf("value must be an unsigned integer")
		}
		n = n*10 + uint64(b-'0')
	}
	return n, nil
}

func parseSessionTimeout(value []byte) (time.Duration, error) {
	if n, err := parseSessionUint(value); err == nil {
		// PostgreSQL numeric statement_timeout values are milliseconds.
		if n > uint64((time.Duration(1<<63-1))/time.Millisecond) {
			return 0, fmt.Errorf("duration overflows time.Duration")
		}
		return time.Duration(n) * time.Millisecond, nil
	}
	units := []struct {
		suffix []byte
		mult   time.Duration
	}{
		{[]byte("ms"), time.Millisecond},
		{[]byte("s"), time.Second},
		{[]byte("min"), time.Minute},
		{[]byte("m"), time.Minute},
		{[]byte("h"), time.Hour},
	}
	for _, unit := range units {
		if len(value) > len(unit.suffix) && bytes.EqualFold(value[len(value)-len(unit.suffix):], unit.suffix) {
			n, err := parseSessionUint(value[:len(value)-len(unit.suffix)])
			if err != nil || n > uint64((time.Duration(1<<63-1))/unit.mult) {
				break
			}
			return time.Duration(n) * unit.mult, nil
		}
	}
	return 0, fmt.Errorf("expected milliseconds or an integer duration such as 5s")
}
