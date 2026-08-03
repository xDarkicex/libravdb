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
	Name  string
	Query string
}

// Portal holds a bound statement with parameter values, ready to execute.
type Portal struct {
	Name      string
	Stmt      *PreparedStmt
	ResultFmt []int16 // 0=text, 1=binary
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

	state.prepared[stmtName] = &PreparedStmt{
		Name:  stmtName,
		Query: query,
	}

	return WriteMessage(w, msgParseComplete, nil)
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

	// Skip each parameter value
	for i := 0; i < numParams; i++ {
		if offset+4 > len(payload) {
			return sendError(w, "ERROR", fmt.Errorf("bind payload too short at param %d", i))
		}
		paramLen := int(binary.BigEndian.Uint32(payload[offset : offset+4]))
		offset += 4
		if paramLen == -1 { // NULL
			continue
		}
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
		ResultFmt: resultFmts,
	}

	return WriteMessage(w, msgBindComplete, nil)
}

// handleDescribe sends RowDescription or NoData for a prepared statement or portal.
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
	// describeType := payload[0] // 'S' or 'P' — unused for now
	_, offset := ReadNullTerminated(payload, 1) // skip name

	_ = offset

	// For now, always return NoData — we don't track result column metadata
	// for prepared statements before execution.
	return WriteMessage(w, msgNoData, nil)
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

	// Check for system function interception
	if results, columns, handled := interceptSystemQuery(portal.Stmt.Query, db); handled {
		return sendQueryResult(w, results, columns)
	}

	results, err := db.Query(ctx, portal.Stmt.Query)
	if err != nil {
		return sendError(w, "ERROR", err)
	}

	return sendQueryResult(w, results, inferColumns(results))
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
