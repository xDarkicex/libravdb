package pgwire

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/xDarkicex/lexer"
	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/libravdb"
)

// handleSQLPrepareExecute implements SQL-level PREPARE/EXECUTE in the
// connection-local pgwire session. The extended protocol Parse/Bind/Execute
// path remains separate and continues to support native typed parameters.
func handleSQLPrepareExecute(rw interface{ Write([]byte) (int, error) }, db *libravdb.Database, state *connState, query string) (bool, error) {
	if state == nil {
		return false, nil
	}
	src := []byte(query)
	doc := &parser.QueryDoc{}
	if err := parser.Parse(src, doc); err != nil {
		return false, nil // normal query path owns the diagnostic
	}
	if len(doc.PrepareStmts) == 1 {
		stmt := doc.PrepareStmts[0]
		if stmt.NameEnd > uint32(len(src)) || stmt.QueryEnd > uint32(len(src)) {
			return true, sendSimpleError(rw, state, fmt.Errorf("PREPARE source span out of bounds"))
		}
		name := strings.ToLower(string(src[stmt.NameStart:stmt.NameEnd]))
		body := strings.TrimSpace(string(src[stmt.QueryStart:stmt.QueryEnd]))
		if name == "" || body == "" {
			return true, sendSimpleError(rw, state, fmt.Errorf("PREPARE requires a name and query"))
		}
		bodyDoc := &parser.QueryDoc{}
		if err := parser.Parse([]byte(body), bodyDoc); err != nil {
			return true, sendSimpleError(rw, state, fmt.Errorf("PREPARE %q: %w", name, err))
		}
		if state.sqlPrepared == nil {
			state.sqlPrepared = make(map[string]string)
		}
		state.sqlPrepared[name] = body
		if err := sendCommandComplete(rw, "PREPARE"); err != nil {
			return true, err
		}
		return true, sendReadyForQuery(rw, state.readyStatus())
	}
	if len(doc.ExecuteStmts) != 1 {
		return false, nil
	}
	stmt := doc.ExecuteStmts[0]
	if stmt.NameEnd > uint32(len(src)) {
		return true, sendSimpleError(rw, state, fmt.Errorf("EXECUTE source span out of bounds"))
	}
	name := strings.ToLower(string(src[stmt.NameStart:stmt.NameEnd]))
	body, ok := state.sqlPrepared[name]
	if !ok {
		return true, sendSimpleError(rw, state, fmt.Errorf("prepared statement %q does not exist", name))
	}
	params := make(libravdb.QueryParams, stmt.ArgsCount)
	for i := int32(0); i < stmt.ArgsCount; i++ {
		arg := doc.ExecuteArgs[stmt.ArgsStart+i]
		value, err := sqlPrepareArg(doc, src, arg)
		if err != nil {
			return true, sendSimpleError(rw, state, fmt.Errorf("EXECUTE %q argument %d: %w", name, i+1, err))
		}
		params["$"+strconv.Itoa(int(i)+1)] = value
	}
	ctx, cancel := state.statementContext(context.Background())
	defer cancel()
	var results *libravdb.SearchResults
	var err error
	if state.epoch != nil {
		results, err = state.epoch.QueryWithSessionConfig(ctx, body, params, &state.config)
	} else {
		results, err = db.QueryWithSessionConfig(ctx, body, params, &state.config)
	}
	if err != nil {
		return true, sendSimpleError(rw, state, fmt.Errorf("EXECUTE %q: %w", name, err))
	}
	return true, sendQueryResultWithStatus(rw, results, inferColumns(results), state.readyStatus())
}

// handleSQLDeallocate implements the PostgreSQL connection-local prepared
// statement cleanup command used by psycopg/SQLAlchemy when resetting a
// connection. The extended protocol Close message handles wire-level
// statements; this covers SQL-level DEALLOCATE emitted by client pools.
func handleSQLDeallocate(rw interface{ Write([]byte) (int, error) }, state *connState, query string) (bool, error) {
	if state == nil {
		return false, nil
	}
	trimmed := strings.TrimSpace(strings.TrimRight(query, ";"))
	upper := strings.ToUpper(trimmed)
	if !strings.HasPrefix(upper, "DEALLOCATE") {
		return false, nil
	}
	remainder := strings.TrimSpace(trimmed[len("DEALLOCATE"):])
	if strings.HasPrefix(strings.ToUpper(remainder), "PREPARE ") {
		remainder = strings.TrimSpace(remainder[len("PREPARE "):])
	}
	if strings.EqualFold(remainder, "ALL") {
		clear(state.prepared)
		clear(state.sqlPrepared)
		clear(state.portals)
		if err := sendCommandComplete(rw, "DEALLOCATE ALL"); err != nil {
			return true, err
		}
		return true, sendReadyForQuery(rw, state.readyStatus())
	}
	if remainder == "" {
		return true, sendSimpleError(rw, state, fmt.Errorf("DEALLOCATE requires a statement name or ALL"))
	}
	name := strings.ToLower(strings.Trim(remainder, `"`))
	delete(state.prepared, name)
	delete(state.sqlPrepared, name)
	for portalName, portal := range state.portals {
		if portal != nil && portal.Stmt != nil && strings.EqualFold(portal.Stmt.Name, name) {
			delete(state.portals, portalName)
		}
	}
	if err := sendCommandComplete(rw, "DEALLOCATE"); err != nil {
		return true, err
	}
	return true, sendReadyForQuery(rw, state.readyStatus())
}

func sqlPrepareArg(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (interface{}, error) {
	switch ref.Kind {
	case parser.NodeKindString:
		sl := doc.Strings[ref.ID]
		decode := lexer.DecodeStringLiteralInto
		if sl.Escape {
			decode = lexer.DecodeEscapeStringLiteralInto
		}
		if value, ok := decode(src, sl.Start, sl.End, nil); ok {
			return string(value), nil
		}
		return nil, fmt.Errorf("escaped string is too large for SQL EXECUTE")
	case parser.NodeKindNumber:
		n := doc.Numbers[ref.ID]
		raw := string(src[n.Start:n.End])
		if strings.ContainsAny(raw, ".eE") {
			return strconv.ParseFloat(raw, 64)
		}
		return strconv.ParseInt(raw, 10, 64)
	case parser.NodeKindIdentifier:
		id := doc.Identifiers[ref.ID]
		if strings.EqualFold(string(src[id.Start:id.End]), "NULL") {
			return nil, nil
		}
		return string(src[id.Start:id.End]), nil
	default:
		return nil, fmt.Errorf("unsupported argument expression kind %d", ref.Kind)
	}
}
