package pgwire

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/xDarkicex/libravdb/libravdb"
)

// PreparedStmt holds a parsed SQL statement ready for binding and execution.
type PreparedStmt struct {
	Name       string
	Query      string
	ParamCount int // number of $N placeholders (for ParameterDescription)
}

// Portal holds a bound statement with parameter values, ready to execute.
type Portal struct {
	Name      string
	Stmt      *PreparedStmt
	Params    [][]byte // bound parameter values (text format)
	ResultFmt []int16  // 0=text, 1=binary
}

// connState holds the extended query protocol state for a single connection.
type connState struct {
	prepared map[string]*PreparedStmt
	portals  map[string]*Portal
}

func newConnState() *connState {
	return &connState{
		prepared: make(map[string]*PreparedStmt),
		portals:  make(map[string]*Portal),
	}
}

// handleExtendedMessage dispatches an extended query protocol message type.
// Returns true if the connection should continue, false if it should close.
func handleExtendedMessage(rw io.ReadWriter, db *libravdb.Database, state *connState, msgType byte, payload []byte) (bool, error) {
	switch msgType {
	case msgParse:
		return true, handleParse(rw, state, payload)
	case msgBind:
		return true, handleBind(rw, state, payload)
	case msgDescribe:
		return true, handleDescribe(rw, state, payload)
	case msgExecute:
		return true, handleExecute(rw, db, state, payload)
	case msgSync:
		return true, handleSync(rw)
	case msgClose:
		return true, handleClose(state, payload)
	case msgFlush:
		// Flush is a no-op for us — we don't buffer output
		return true, nil
	default:
		return true, nil
	}
}

// handleParse parses a SQL statement and stores it as a prepared statement.
//
// Message format:
//
//	int32 length
//	string stmtName (null-terminated, "" = unnamed)
//	string query   (null-terminated)
//	int16 numParamTypes
//	int32[numParamTypes] paramTypeOIDs
func handleParse(w io.Writer, state *connState, payload []byte) error {
	stmtName, offset := ReadNullTerminated(payload, 0)
	query, offset := ReadNullTerminated(payload, offset)

	// Skip parameter types for now — we don't use them
	// numParamTypes is at payload[offset:offset+2]

	// Trim trailing semicolon and whitespace
	query = strings.TrimSpace(strings.TrimRight(query, ";"))

	// Copy out of the arena: payload memory is reset between messages, but
	// prepared statements must survive across the connection. Retaining
	// arena slices here would corrupt the map keys/query on the next message.
	stmtNameCopy := string(stmtName)
	queryCopy := string(query)

	state.prepared[stmtNameCopy] = &PreparedStmt{
		Name:       stmtNameCopy,
		Query:      queryCopy,
		ParamCount: countParams(queryCopy),
	}

	return WriteMessage(w, msgParseComplete, nil)
}

// countParams returns the highest $N placeholder index in a query
// (0 if none). Used for ParameterDescription.
func countParams(query string) int {
	max := 0
	inStr := false
	for i := 0; i < len(query); i++ {
		c := query[i]
		if c == '\'' {
			inStr = !inStr
			continue
		}
		if !inStr && c == '$' && i+1 < len(query) && query[i+1] >= '0' && query[i+1] <= '9' {
			j := i + 1
			for j < len(query) && query[j] >= '0' && query[j] <= '9' {
				j++
			}
			n := 0
			for k := i + 1; k < j; k++ {
				n = n*10 + int(query[k]-'0')
			}
			if n > max {
				max = n
			}
		}
	}
	return max
}

// handleBind binds parameter values to a prepared statement, creating a portal.
//
// Message format:
//
//	int32 length
//	string portalName   (null-terminated, "" = unnamed)
//	string stmtName     (null-terminated)
//	int16 numParamFormats
//	int16[numParamFormats] paramFormats (0=text, 1=binary)
//	int16 numParams
//	for each param: int32 len + byte[len] value
//	int16 numResultFormats
//	int16[numResultFormats] resultFormats
func handleBind(w io.Writer, state *connState, payload []byte) error {
	portalName, offset := ReadNullTerminated(payload, 0)
	stmtName, offset := ReadNullTerminated(payload, offset)

	stmt, ok := state.prepared[stmtName]
	if !ok {
		return sendError(w, "ERROR", fmt.Errorf("prepared statement %q does not exist", stmtName))
	}

	// Skip parameter formats and values for now — we don't support parameterized queries yet
	// Parse: int16 numParamFormats, int16[numParamFormats], int16 numParams, then per-param values
	if offset+2 > len(payload) {
		return sendError(w, "ERROR", fmt.Errorf("bind payload too short"))
	}
	numParamFormats := int(binary.BigEndian.Uint16(payload[offset : offset+2]))
	offset += 2

	// Skip param format codes
	offset += numParamFormats * 2

	if offset+2 > len(payload) {
		return sendError(w, "ERROR", fmt.Errorf("bind payload too short at params"))
	}
	numParams := int(binary.BigEndian.Uint16(payload[offset : offset+2]))
	offset += 2

	// Parse parameter values and retain them for execution.
	// Values are copied out of the arena payload because the arena is
	// reset between messages — retained slices must own their memory.
	var paramValues [][]byte
	for i := 0; i < numParams; i++ {
		if offset+4 > len(payload) {
			return sendError(w, "ERROR", fmt.Errorf("bind payload too short at param %d", i))
		}
		paramLen := int(binary.BigEndian.Uint32(payload[offset : offset+4]))
		offset += 4
		if paramLen == -1 { // NULL
			paramValues = append(paramValues, nil)
			continue
		}
		if offset+paramLen > len(payload) {
			return sendError(w, "ERROR", fmt.Errorf("bind payload too short at param %d", i))
		}
		val := make([]byte, paramLen)
		copy(val, payload[offset:offset+paramLen])
		paramValues = append(paramValues, val)
		offset += paramLen
	}

	// Parse result formats
	var resultFmts []int16
	if offset+2 <= len(payload) {
		numResultFmts := int(binary.BigEndian.Uint16(payload[offset : offset+2]))
		offset += 2
		for i := 0; i < numResultFmts && offset+2 <= len(payload); i++ {
			resultFmts = append(resultFmts, int16(binary.BigEndian.Uint16(payload[offset:offset+2])))
			offset += 2
		}
	}

	state.portals[portalName] = &Portal{
		Name:      portalName,
		Stmt:      stmt,
		Params:    paramValues,
		ResultFmt: resultFmts,
	}

	return WriteMessage(w, msgBindComplete, nil)
}

// handleDescribe sends ParameterDescription + RowDescription/NoData for a
// prepared statement or portal.
//
// Message format:
//
//	int32 length
//	byte describeType ('S'=statement, 'P'=portal)
//	string name (null-terminated)
func handleDescribe(w io.Writer, state *connState, payload []byte) error {
	if len(payload) < 2 {
		return sendError(w, "ERROR", fmt.Errorf("describe payload too short"))
	}
	describeType := payload[0]
	name, _ := ReadNullTerminated(payload, 1)

	switch describeType {
	case 'S':
		stmt, ok := state.prepared[name]
		if !ok {
			return sendError(w, "ERROR", fmt.Errorf("prepared statement %q does not exist", name))
		}
		// ParameterDescription: clients count parameters from this message.
		// All OIDs 0 (unspecified) → clients use text format for args.
		if err := sendParameterDescription(w, stmt.ParamCount); err != nil {
			return err
		}
		// No result metadata until execution (we don't track columns statically).
		return WriteMessage(w, msgNoData, nil)
	default: // 'P' — portal
		return WriteMessage(w, msgNoData, nil)
	}
}

// sendParameterDescription sends a ParameterDescription ('t') message.
func sendParameterDescription(w io.Writer, count int) error {
	buf := make([]byte, 2+4*count)
	binary.BigEndian.PutUint16(buf[0:2], uint16(count))
	for i := 0; i < count; i++ {
		// 0 = unspecified type OID; clients then use text format for args.
		binary.BigEndian.PutUint32(buf[2+4*i:], 0)
	}
	return WriteMessage(w, msgParameterDescription, buf)
}

// handleExecute runs a portal and sends the results.
//
// Message format:
//
//	int32 length
//	string portalName (null-terminated, "" = unnamed)
//	int32 maxRows (0 = unlimited)
func handleExecute(w io.Writer, db *libravdb.Database, state *connState, payload []byte) error {
	portalName, _ := ReadNullTerminated(payload, 0)

	portal, ok := state.portals[portalName]
	if !ok {
		return sendError(w, "ERROR", fmt.Errorf("portal %q does not exist", portalName))
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Substitute bound parameters into the query ($1, $2, ... → literals),
	// since the engine has no native parameter support yet.
	query := portal.Stmt.Query
	if len(portal.Params) > 0 {
		sub, err := substituteParams(query, portal.Params)
		if err != nil {
			return sendError(w, "ERROR", err)
		}
		query = sub
	}

	// Check for system function interception
	if results, columns, handled := interceptSystemQuery(query, db); handled {
		return sendExtendedQueryResult(w, results, columns)
	}

	results, err := db.Query(ctx, query)
	if err != nil {
		return sendError(w, "ERROR", err)
	}

	return sendExtendedQueryResult(w, results, inferColumns(results))
}

// substituteParams replaces $N placeholders in a query with quoted, escaped
// string literals from the bound parameter values. Single-quoted string
// literals in the original query are skipped so $ inside them stays literal.
func substituteParams(query string, params [][]byte) (string, error) {
	var sb strings.Builder
	sb.Grow(len(query) + 16*len(params))
	inStr := false
	for i := 0; i < len(query); i++ {
		c := query[i]
		if c == '\'' {
			inStr = !inStr
			sb.WriteByte(c)
			continue
		}
		if !inStr && c == '$' && i+1 < len(query) && query[i+1] >= '0' && query[i+1] <= '9' {
			j := i + 1
			for j < len(query) && query[j] >= '0' && query[j] <= '9' {
				j++
			}
			n := 0
			for k := i + 1; k < j; k++ {
				n = n*10 + int(query[k]-'0')
			}
			if n < 1 || n > len(params) {
				return "", fmt.Errorf("parameter $%d out of range (bound %d)", n, len(params))
			}
			sb.WriteByte('\'')
			sb.WriteString(strings.ReplaceAll(string(params[n-1]), "'", "''"))
			sb.WriteByte('\'')
			i = j - 1
			continue
		}
		sb.WriteByte(c)
	}
	return sb.String(), nil
}

// handleSync sends ReadyForQuery, completing a transaction boundary.
func handleSync(w io.Writer) error {
	return WriteMessage(w, msgReadyForQuery, []byte{'I'})
}

// handleClose closes a prepared statement or portal.
//
// Message format:
//
//	int32 length
//	byte closeType ('S'=statement, 'P'=portal)
//	string name (null-terminated)
func handleClose(state *connState, payload []byte) error {
	if len(payload) < 1 {
		return nil
	}
	closeType := payload[0]
	name, _ := ReadNullTerminated(payload, 1)

	switch closeType {
	case 'S':
		delete(state.prepared, name)
	case 'P':
		delete(state.portals, name)
	}
	return nil
}
