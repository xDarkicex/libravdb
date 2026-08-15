package pgwire

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/libravdb"
)

const pgwireSafetyTimeout = 30 * time.Second

func (s *connState) statementContext(base context.Context) (context.Context, context.CancelFunc) {
	if s == nil {
		return context.WithTimeout(base, pgwireSafetyTimeout)
	}
	return context.WithTimeout(base, s.config.EffectiveTimeout(pgwireSafetyTimeout))
}

// applySessionSettingSQL parses and applies one SET/RESET command. The
// parser, rather than string-prefix matching, owns the grammar and value
// spans. It returns handled=false for ordinary SQL.
func applySessionSettingSQL(state *connState, query string) (handled bool, commandTag string, err error) {
	if state == nil {
		return false, "", nil
	}
	// Python PostgreSQL drivers establish a few client-facing compatibility
	// settings before issuing application SQL. These values do not affect the
	// libraVDB execution model, but accepting the standard session commands is
	// required for a real driver handshake.
	if handled, tag := compatibilitySessionSetting(query); handled {
		return true, tag, nil
	}
	doc := &parser.QueryDoc{}
	if err := parser.Parse([]byte(query), doc); err != nil {
		return false, "", nil
	}
	if len(doc.SessionSettingStmts) != 1 || len(doc.Nodes) != 1 || doc.Nodes[0].Kind != parser.NodeKindSessionSettingStmt ||
		len(doc.TransactionStmts) != 0 || len(doc.SelectStmts) != 0 || len(doc.InsertStmts) != 0 ||
		len(doc.UpdateStmts) != 0 || len(doc.DeleteStmts) != 0 {
		return false, "", nil
	}
	stmt := &doc.SessionSettingStmts[0]
	if err := state.config.ApplySessionSetting([]byte(query), doc, stmt); err != nil {
		if stmt.Reset {
			return true, "RESET", err
		}
		return true, "SET", err
	}
	if stmt.Reset {
		return true, "RESET", nil
	}
	return true, "SET", nil
}

func compatibilitySessionSetting(query string) (bool, string) {
	upper := strings.ToUpper(strings.TrimSpace(strings.TrimSuffix(query, ";")))
	if strings.HasPrefix(upper, "SET ") {
		body := strings.TrimSpace(strings.TrimPrefix(upper, "SET "))
		if strings.HasPrefix(body, "SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL") {
			return true, "SET"
		}
		for _, name := range []string{
			"APPLICATION_NAME", "BYTEA_OUTPUT", "CLIENT_ENCODING", "DATESTYLE",
			"EXTRA_FLOAT_DIGITS", "INTERVALSTYLE", "SEARCH_PATH", "STANDARD_CONFORMING_STRINGS",
			"TIME ZONE", "TIMEZONE",
		} {
			if body == name || strings.HasPrefix(body, name+" ") || strings.HasPrefix(body, name+"=") {
				return true, "SET"
			}
		}
	}
	if strings.HasPrefix(upper, "RESET ") {
		name := strings.TrimSpace(strings.TrimPrefix(upper, "RESET "))
		for _, supported := range []string{
			"APPLICATION_NAME", "BYTEA_OUTPUT", "CLIENT_ENCODING", "DATESTYLE",
			"EXTRA_FLOAT_DIGITS", "INTERVALSTYLE", "SEARCH_PATH", "STANDARD_CONFORMING_STRINGS",
			"TIME ZONE", "TIMEZONE",
		} {
			if name == supported {
				return true, "RESET"
			}
		}
	}
	return false, ""
}

// handleSetConfigFunction executes the connection-local subset of
// PostgreSQL's set_config() scalar function. Django uses this during
// connection initialization for TimeZone; it is a result-producing function,
// so it cannot be treated as a command-only SET compatibility case.
func handleSetConfigFunction(query string, config *libravdb.SessionConfig, params *optimizer.ParameterSet) (*libravdb.SearchResults, []ColumnMeta, bool, error) {
	trimmed := strings.TrimSpace(strings.TrimSuffix(query, ";"))
	compact := compactSessionSQL(trimmed)
	if !strings.HasPrefix(compact, "SELECTSET_CONFIG(") || !strings.HasSuffix(trimmed, ")") {
		return nil, nil, false, nil
	}
	args, ok := parseSetConfigCall(trimmed)
	if !ok || len(args) != 3 {
		return nil, nil, true, fmt.Errorf("set_config requires exactly three scalar arguments")
	}
	name, ok := parseSetConfigString(args[0])
	if !ok {
		return nil, nil, true, fmt.Errorf("set_config setting name must be a string literal")
	}
	value, ok := resolveSetConfigValue(args[1], params)
	if !ok {
		return nil, nil, true, fmt.Errorf("set_config setting value must be a string literal or bound text parameter")
	}
	local := strings.EqualFold(strings.TrimSpace(args[2]), "true")
	if !local && !strings.EqualFold(strings.TrimSpace(args[2]), "false") {
		return nil, nil, true, fmt.Errorf("set_config local argument must be boolean")
	}
	if err := config.ApplySetConfig(name, value, local); err != nil {
		return nil, nil, true, err
	}
	columns := []ColumnMeta{{Name: "set_config", TypeOID: OIDText}}
	results := &libravdb.SearchResults{
		Results:     []*libravdb.SearchResult{{ID: value, Score: 1, Metadata: map[string]interface{}{"set_config": value}}},
		Total:       1,
		Columns:     []string{"set_config"},
		ColumnTypes: []uint16{OIDText},
	}
	return results, columns, true, nil
}

func handleAsyncpgJITQuery(query string, config *libravdb.SessionConfig, params *optimizer.ParameterSet) (*libravdb.SearchResults, []ColumnMeta, bool, error) {
	trimmed := strings.TrimSpace(strings.TrimSuffix(query, ";"))
	if !isAsyncpgJITQuery(trimmed) {
		return nil, nil, false, nil
	}
	args, ok := parseSetConfigCall(query)
	if !ok || len(args) != 3 {
		return nil, nil, true, fmt.Errorf("set_config requires exactly three scalar arguments")
	}
	name, ok := parseSetConfigString(args[0])
	if !ok || !strings.EqualFold(name, "jit") {
		return nil, nil, false, nil
	}
	value, ok := resolveSetConfigValue(args[1], params)
	if !ok {
		return nil, nil, true, fmt.Errorf("set_config setting value must be a string literal or bound text parameter")
	}
	local := strings.EqualFold(strings.TrimSpace(args[2]), "true")
	if !local && !strings.EqualFold(strings.TrimSpace(args[2]), "false") {
		return nil, nil, true, fmt.Errorf("set_config local argument must be boolean")
	}
	previous := config.JIT
	if err := config.ApplySetConfig(name, value, local); err != nil {
		return nil, nil, true, err
	}
	columns := []ColumnMeta{{Name: "cur", TypeOID: OIDText}, {Name: "new", TypeOID: OIDText}}
	results := &libravdb.SearchResults{
		Results: []*libravdb.SearchResult{{ID: value, Score: 1, Metadata: map[string]interface{}{"cur": previous, "new": value}}},
		Total:   1, Columns: []string{"cur", "new"}, ColumnTypes: []uint16{OIDText, OIDText},
	}
	return results, columns, true, nil
}

func isAsyncpgJITQuery(query string) bool {
	trimmed := strings.TrimSpace(strings.TrimSuffix(query, ";"))
	compact := compactSessionSQL(trimmed)
	// asyncpg uses two forms while resolving a custom type: one query reads
	// the current setting and turns JIT off, and a second query restores the
	// saved value with a bound parameter. Both must be described as text;
	// otherwise the restore parameter is reported as OID 0 and asyncpg keeps
	// retrying its type lookup.
	if !strings.Contains(compact, "SET_CONFIG(") {
		return false
	}
	args, ok := parseSetConfigCall(trimmed)
	if !ok || len(args) != 3 {
		return false
	}
	name, ok := parseSetConfigString(args[0])
	return ok && strings.EqualFold(name, "jit")
}

func compactSessionSQL(sql string) string {
	return strings.ToUpper(strings.Map(func(r rune) rune {
		switch r {
		case ' ', '\t', '\n', '\r':
			return -1
		default:
			return r
		}
	}, sql))
}

func parseSetConfigCall(query string) ([]string, bool) {
	upper := strings.ToUpper(query)
	start := strings.Index(upper, "SET_CONFIG(")
	if start < 0 {
		return nil, false
	}
	bodyStart := start + len("SET_CONFIG(")
	end := strings.IndexByte(query[bodyStart:], ')')
	if end < 0 {
		return nil, false
	}
	return splitSetConfigArgs(query[bodyStart : bodyStart+end])
}

func resolveSetConfigValue(value string, params *optimizer.ParameterSet) (string, bool) {
	if parsed, ok := parseSetConfigString(value); ok {
		return parsed, true
	}
	value = strings.TrimSpace(value)
	if params == nil || len(value) < 2 || value[0] != '$' {
		return "", false
	}
	for i := 1; i < len(value); i++ {
		if value[i] < '0' || value[i] > '9' {
			return "", false
		}
	}
	lookup, found := params.Lookup([]byte(value), 0, uint32(len(value)))
	if found && !lookup.IsNull() && (lookup.Kind == optimizer.ScalarString || lookup.Kind == optimizer.ScalarBytes) {
		return string(lookup.BytesData), true
	}
	return "", false
}

func splitSetConfigArgs(body string) ([]string, bool) {
	args := make([]string, 0, 3)
	start := 0
	quoted := false
	depth := 0
	for i := 0; i < len(body); i++ {
		switch body[i] {
		case '\'':
			if quoted && i+1 < len(body) && body[i+1] == '\'' {
				i++
				continue
			}
			quoted = !quoted
		case '(':
			if !quoted {
				depth++
			}
		case ')':
			if !quoted {
				depth--
				if depth < 0 {
					return nil, false
				}
			}
		case ',':
			if !quoted && depth == 0 {
				args = append(args, strings.TrimSpace(body[start:i]))
				start = i + 1
			}
		}
	}
	if quoted || depth != 0 {
		return nil, false
	}
	args = append(args, strings.TrimSpace(body[start:]))
	return args, true
}

func parseSetConfigString(value string) (string, bool) {
	value = strings.TrimSpace(value)
	if len(value) < 2 || value[0] != '\'' || value[len(value)-1] != '\'' {
		return "", false
	}
	value = value[1 : len(value)-1]
	return strings.ReplaceAll(value, "''", "'"), true
}
