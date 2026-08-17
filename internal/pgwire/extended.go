package pgwire

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/xDarkicex/lexer"
	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/libravdb"
)

// PreparedStmt holds a parsed SQL statement ready for binding and execution.
type PreparedStmt struct {
	Name      string
	Query     string
	ParamOIDs []uint32 // parameter type OIDs from Parse message (0 = unspecified)
	// InferredParamOIDs are derived from the parsed query and catalog when the
	// client leaves parameter types unspecified in Parse. They are used for
	// decoding Bind values, not only for Describe output.
	InferredParamOIDs []uint32
	// numPositional is the highest $N seen in the query.
	numPositional int
	// namedOrder holds @name parameters in encounter order.
	namedOrder []string
	// ParamCount returns the total number of expected parameters.
	ParamCount int
	// ServerCursorQuery is populated for psycopg-style DECLARE ... CURSOR
	// statements. Query remains the wire statement for command tags while the
	// underlying SELECT is executed when the portal is run.
	ServerCursorName  string
	ServerCursorQuery string
}

// ParamValue holds a single bound parameter value with its wire-format metadata.
type ParamValue struct {
	Raw    []byte      // raw bytes from Bind message (nil for NULL)
	Format int16       // 0=text, 1=binary
	IsNull bool        // explicit SQL NULL
	OID    uint32      // parameter OID (from Parse or inferred)
	Value  interface{} // decoded Go value (nil when IsNull)
}

// Portal holds a bound statement with typed parameter values, ready to execute.
type Portal struct {
	Name      string
	Stmt      *PreparedStmt
	Params    []ParamValue // bound parameter values
	ResultFmt []int16      // 0=text, 1=binary

	// Execution state is retained so Execute(maxRows) can suspend and resume
	// without rerunning the bound statement.
	Started         bool
	Complete        bool
	Results         *libravdb.SearchResults
	Columns         []ColumnMeta
	RowIndex        int
	CommandTag      string
	CommandOnly     bool
	DescriptionSent bool
}

// transactionState tracks the transaction status of a connection.
type transactionState byte

const (
	transactionIdle       transactionState = 'I'
	transactionInProgress transactionState = 'T'
	transactionFailed     transactionState = 'E'
)

// connState holds the extended query protocol state for a single connection
// and the optional active epoch transaction.
type connState struct {
	prepared                  map[string]*PreparedStmt
	sqlPrepared               map[string]string
	portals                   map[string]*Portal
	serverCursors             map[string]*serverCursor
	maxPreparedStatements     int
	maxPortals                int
	maxPreparedStatementBytes int
	maxPortalBytes            int
	config                    libravdb.SessionConfig
	epoch                     *libravdb.EpochTx
	transactionState          transactionState
	extendedSyncRequired      bool
}

func newConnState() *connState {
	return &connState{
		prepared:                  make(map[string]*PreparedStmt),
		sqlPrepared:               make(map[string]string),
		portals:                   make(map[string]*Portal),
		serverCursors:             make(map[string]*serverCursor),
		maxPreparedStatements:     DefaultMaxPreparedStatements,
		maxPortals:                DefaultMaxPortals,
		maxPreparedStatementBytes: DefaultMaxPreparedStatementBytes,
		maxPortalBytes:            DefaultMaxPortalBytes,
		config:                    libravdb.DefaultSessionConfig(),
	}
}

func (s *connState) rollbackEpoch() {
	if s.epoch != nil {
		_ = s.epoch.Rollback(context.Background())
	}
	s.clearTransaction()
	s.extendedSyncRequired = false
}

func handleExtendedMessage(rw io.ReadWriter, db *libravdb.Database, state *connState, msgType byte, payload []byte) (bool, error) {
	// After an extended-protocol error PostgreSQL ignores the rest of the
	// batch until Sync. ReadyForQuery is emitted by that Sync below.
	if state != nil && state.extendedSyncRequired {
		if msgType == msgSync {
			state.extendedSyncRequired = false
			return true, handleSync(rw, state)
		}
		// Statement-cache cleanup is allowed to close prepared statements and
		// portals while the failed batch is waiting for Sync. PostgreSQL still
		// acknowledges Close with CloseComplete; suppressing it leaves pgx/
		// database/sql pipelines misaligned and the following ReadyForQuery is
		// reported as an unexpected response.
		if msgType == msgClose {
			return true, handleClose(rw, state, payload)
		}
		return true, nil
	}

	switch msgType {
	case msgParse:
		return true, handleParseWithDB(rw, db, state, payload)
	case msgBind:
		return true, handleBind(rw, state, payload)
	case msgDescribe:
		return true, handleDescribe(rw, db, state, payload)
	case msgExecute:
		return true, handleExecute(rw, db, state, payload)
	case msgSync:
		return true, handleSync(rw, state)
	case msgClose:
		return true, handleClose(rw, state, payload)
	case msgFlush:
		return true, nil
	default:
		return true, nil
	}
}

// handleParse parses a SQL statement and stores it as a prepared statement.
// This compatibility wrapper is retained for package-local tests that do not
// have a database handle available for static parameter inference.
func handleParse(w io.Writer, state *connState, payload []byte) error {
	return handleParseWithDB(w, nil, state, payload)
}

// handleParseWithDB is the wire Parse implementation. When a database is
// available it also infers unspecified parameter OIDs from the parsed query;
// this lets Bind decode values natively even when clients send OID zero.
func handleParseWithDB(w io.Writer, db *libravdb.Database, state *connState, payload []byte) error {
	stmtName, offset := ReadNullTerminated(payload, 0)
	query, offset := ReadNullTerminated(payload, offset)
	if len(query) > state.maxPreparedStatementBytes {
		return sendError(w, "ERROR", fmt.Errorf("prepared statement query exceeds %d bytes", state.maxPreparedStatementBytes))
	}
	if _, exists := state.prepared[stmtName]; !exists && len(state.prepared) >= state.maxPreparedStatements {
		return sendError(w, "ERROR", fmt.Errorf("too many prepared statements (limit %d)", state.maxPreparedStatements))
	}

	query = strings.TrimSpace(strings.TrimRight(query, ";"))
	if state.txStatus() == transactionFailed && !transactionQueryAllowedAfterFailure(query) {
		state.extendedSyncRequired = true
		return sendError(w, "ERROR", errCurrentTransactionAborted)
	}

	// Parse parameter type OIDs from the Parse message.
	var paramOIDs []uint32
	if offset+2 <= len(payload) {
		numParamTypes := int(binary.BigEndian.Uint16(payload[offset : offset+2]))
		offset += 2
		for i := 0; i < numParamTypes && offset+4 <= len(payload); i++ {
			oid := binary.BigEndian.Uint32(payload[offset : offset+4])
			paramOIDs = append(paramOIDs, oid)
			offset += 4
		}
	}

	stmtNameCopy := string(stmtName)
	queryCopy := string(query)

	serverCursorName, serverCursorQuery, isServerCursor := parseServerCursorDeclare(query)
	analysisQuery := query
	if isServerCursor {
		analysisQuery = serverCursorQuery
	}
	paramInfo := analyzeParamsBytes([]byte(analysisQuery))
	var inferredParamOIDs []uint32
	if db != nil && paramInfo.total() > 0 {
		// A statement that cannot be described statically is still a valid
		// prepared statement. Defer that error to Bind/Execute and retain any
		// inference that succeeded here.
		describeQuery := queryCopy
		if isServerCursor {
			describeQuery = serverCursorQuery
		}
		if inferred, _, err := describeStatement(db, describeQuery, paramInfo.total()); err == nil {
			inferredParamOIDs = inferred
		}
	}

	state.prepared[stmtNameCopy] = &PreparedStmt{
		Name:              stmtNameCopy,
		Query:             queryCopy,
		ParamOIDs:         paramOIDs,
		InferredParamOIDs: inferredParamOIDs,
		numPositional:     paramInfo.numPositional,
		namedOrder:        paramInfo.namedOrder,
		ParamCount:        paramInfo.total(),
		ServerCursorName:  serverCursorName,
		ServerCursorQuery: serverCursorQuery,
	}

	return WriteMessage(w, msgParseComplete, nil)
}

// paramInfo holds the result of analyzing a query for parameter placeholders.
type paramInfo struct {
	numPositional int
	namedOrder    []string
}

func (pi paramInfo) total() int {
	return pi.numPositional + len(pi.namedOrder)
}

// analyzeParams parses the query and derives parameter metadata from the
// parser's identifier spans. It is retained as a test/publicity wrapper; the
// protocol path calls analyzeParamsBytes directly on the Parse payload.
func analyzeParams(query string) paramInfo {
	return analyzeParamsBytes([]byte(query))
}

// analyzeParamsBytes uses the authoritative lexer/parser AST. It never scans
// SQL text for placeholders, so markers inside quoted strings/comments cannot
// become parameters and aliases are normalized only at parse time.
func analyzeParamsBytes(query []byte) paramInfo {
	var pi paramInfo
	doc := &parser.QueryDoc{}
	if err := parser.Parse(query, doc); err != nil {
		// System-catalog queries emitted by PostgreSQL drivers can contain
		// schema-qualified relations and dialect-specific projection syntax
		// that the SQL parser may reject before execution is intercepted. We
		// still need an AST-safe parameter count for Parse/Bind. The lexer is
		// authoritative about literal/comment boundaries, so fall back to its
		// token stream rather than scanning SQL text.
		return analyzeParamsLexical(query)
	}
	for i := range doc.Identifiers {
		id := &doc.Identifiers[i]
		if id.Start >= uint32(len(query)) || id.End > uint32(len(query)) || id.Start >= id.End {
			continue
		}
		marker := query[id.Start]
		body := query[id.Start+1 : id.End]
		if marker == '$' {
			if ordinal, ok := parseOrdinalBytes(body); ok && ordinal > pi.numPositional {
				pi.numPositional = ordinal
				continue
			}
			// Preserve named $foo parameters as well as the native @foo form.
			// Numeric $N parameters remain positional; malformed markers are
			// ignored rather than being treated as values.
			if len(body) == 0 || !isParamIdentStart(body[0]) {
				continue
			}
		} else if marker != '@' || len(body) == 0 || !isParamIdentStart(body[0]) {
			continue
		}
		name := asciiLowerString(body)
		seen := false
		for _, existing := range pi.namedOrder {
			if bytes.EqualFold([]byte(existing), body) {
				seen = true
				break
			}
		}
		if !seen {
			pi.namedOrder = append(pi.namedOrder, name)
		}
	}
	// Temporal table bounds are stored as spans on TableExpr rather than as
	// expression Identifier nodes. Merge the lexer-authoritative parameter
	// inventory so AS OF TIMESTAMP/LSN parameters participate in Parse/Bind
	// just like parameters in predicates and projections.
	lexical := analyzeParamsLexical(query)
	if lexical.numPositional > pi.numPositional {
		pi.numPositional = lexical.numPositional
	}
	for _, name := range lexical.namedOrder {
		seen := false
		for _, existing := range pi.namedOrder {
			if strings.EqualFold(existing, name) {
				seen = true
				break
			}
		}
		if !seen {
			pi.namedOrder = append(pi.namedOrder, name)
		}
	}
	return pi
}

func analyzeParamsLexical(query []byte) paramInfo {
	var pi paramInfo
	scanner := lexer.New(query)
	for {
		tok, ok := scanner.Next()
		if !ok || tok.Kind == lexer.KindEOF || tok.Kind == lexer.KindError {
			break
		}
		if tok.Kind != lexer.KindParam || tok.End <= tok.Start || tok.End > uint32(len(query)) {
			continue
		}
		raw := query[tok.Start:tok.End]
		if len(raw) < 2 {
			continue
		}
		if raw[0] == '$' {
			if ordinal, ok := parseOrdinalBytes(raw[1:]); ok {
				if ordinal > pi.numPositional {
					pi.numPositional = ordinal
				}
				continue
			}
			if !isParamIdentStart(raw[1]) {
				continue
			}
		} else if raw[0] != '@' || !isParamIdentStart(raw[1]) {
			continue
		}
		name := asciiLowerString(raw[1:])
		seen := false
		for _, existing := range pi.namedOrder {
			if bytes.EqualFold([]byte(existing), raw[1:]) {
				seen = true
				break
			}
		}
		if !seen {
			pi.namedOrder = append(pi.namedOrder, name)
		}
	}
	return pi
}

func parseOrdinalBytes(raw []byte) (int, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	n := 0
	for _, c := range raw {
		if c < '0' || c > '9' {
			return 0, false
		}
		n = n*10 + int(c-'0')
		if n < 0 {
			return 0, false
		}
	}
	return n, true
}

func asciiLowerString(raw []byte) string {
	buf := make([]byte, len(raw))
	for i, c := range raw {
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		buf[i] = c
	}
	return string(buf)
}

func isParamIdentStart(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'
}

func isParamIdentPart(c byte) bool {
	return isParamIdentStart(c) || (c >= '0' && c <= '9')
}

// countParams returns the number of positional parameters required by a query.
func countParams(query string) int {
	pi := analyzeParams(query)
	return pi.total()
}

// handleBind binds parameter values to a prepared statement, creating a portal.
func handleBind(w io.Writer, state *connState, payload []byte) error {
	portalName, offset := ReadNullTerminated(payload, 0)
	stmtName, offset := ReadNullTerminated(payload, offset)
	if _, exists := state.portals[portalName]; !exists && len(state.portals) >= state.maxPortals {
		return sendError(w, "ERROR", fmt.Errorf("too many portals (limit %d)", state.maxPortals))
	}
	if len(payload) > state.maxPortalBytes {
		return sendError(w, "ERROR", fmt.Errorf("portal bind payload exceeds %d bytes", state.maxPortalBytes))
	}

	stmt, ok := state.prepared[stmtName]
	if !ok {
		return sendError(w, "ERROR", fmt.Errorf("prepared statement %q does not exist", stmtName))
	}
	if state.txStatus() == transactionFailed && !transactionQueryAllowedAfterFailure(stmt.Query) {
		state.extendedSyncRequired = true
		return sendError(w, "ERROR", errCurrentTransactionAborted)
	}

	if offset+2 > len(payload) {
		return sendError(w, "ERROR", fmt.Errorf("bind payload too short"))
	}
	numParamFormats := int(binary.BigEndian.Uint16(payload[offset : offset+2]))
	offset += 2
	if numParamFormats != 0 && numParamFormats != 1 && numParamFormats != stmt.ParamCount {
		return sendError(w, "ERROR", fmt.Errorf("bind has %d parameter format codes; expected 0, 1, or %d", numParamFormats, stmt.ParamCount))
	}

	var paramFormats []int16
	for i := 0; i < numParamFormats && offset+2 <= len(payload); i++ {
		pf := int16(binary.BigEndian.Uint16(payload[offset : offset+2]))
		if pf != 0 && pf != 1 {
			return sendError(w, "ERROR", fmt.Errorf("unsupported parameter format %d (expected 0=text or 1=binary)", pf))
		}
		paramFormats = append(paramFormats, pf)
		offset += 2
	}

	if offset+2 > len(payload) {
		return sendError(w, "ERROR", fmt.Errorf("bind payload too short at param count"))
	}
	numParams := int(binary.BigEndian.Uint16(payload[offset : offset+2]))
	offset += 2

	if numParams != stmt.ParamCount {
		return sendError(w, "ERROR", fmt.Errorf("bind supplied %d parameters, statement requires %d", numParams, stmt.ParamCount))
	}

	var paramValues []ParamValue
	for i := 0; i < numParams; i++ {
		if offset+4 > len(payload) {
			return sendError(w, "ERROR", fmt.Errorf("bind payload too short at param %d length", i))
		}
		paramLen := int32(binary.BigEndian.Uint32(payload[offset : offset+4]))
		offset += 4

		pv := ParamValue{}
		if i < len(paramFormats) {
			pv.Format = paramFormats[i]
		} else if len(paramFormats) == 1 {
			pv.Format = paramFormats[0]
		}

		if i < len(stmt.ParamOIDs) && stmt.ParamOIDs[i] != 0 {
			pv.OID = stmt.ParamOIDs[i]
		} else if i < len(stmt.InferredParamOIDs) {
			pv.OID = stmt.InferredParamOIDs[i]
		}
		if paramLen == -1 {
			pv.IsNull = true
		} else {
			if paramLen < 0 || offset+int(paramLen) > len(payload) {
				return sendError(w, "ERROR", fmt.Errorf("bind param %d: length %d exceeds payload", i, paramLen))
			}
			raw := make([]byte, paramLen)
			copy(raw, payload[offset:offset+int(paramLen)])
			pv.Raw = raw
			value, err := decodeParamValue(raw, pv.Format, pv.OID)
			if err != nil {
				return sendError(w, "ERROR", fmt.Errorf("bind parameter %d: %w", i+1, err))
			}
			pv.Value = value
			offset += int(paramLen)
		}
		paramValues = append(paramValues, pv)
	}

	var resultFmts []int16
	if offset+2 <= len(payload) {
		numResultFmts := int(binary.BigEndian.Uint16(payload[offset : offset+2]))
		offset += 2
		for i := 0; i < numResultFmts && offset+2 <= len(payload); i++ {
			format := int16(binary.BigEndian.Uint16(payload[offset : offset+2]))
			if format != 0 && format != 1 {
				return sendError(w, "ERROR", fmt.Errorf("unsupported result format %d (expected 0=text or 1=binary)", format))
			}
			resultFmts = append(resultFmts, format)
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

// handleDescribe sends ParameterDescription + RowDescription/NoData.
func handleDescribe(w io.Writer, db *libravdb.Database, state *connState, payload []byte) error {
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
		if state.txStatus() == transactionFailed && !transactionQueryAllowedAfterFailure(stmt.Query) {
			state.extendedSyncRequired = true
			return sendError(w, "ERROR", errCurrentTransactionAborted)
		}
		describeQuery := stmt.Query
		if stmt.ServerCursorQuery != "" {
			describeQuery = stmt.ServerCursorQuery
		}
		paramOIDs, columns, err := describeStatement(db, describeQuery, stmt.ParamCount)
		if err != nil {
			return sendError(w, "ERROR", err)
		}
		paramOIDs = mergeParamOIDs(paramOIDs, stmt.ParamOIDs)
		if err := sendParameterDescription(w, paramOIDs); err != nil {
			return err
		}
		if len(columns) == 0 {
			return WriteMessage(w, msgNoData, nil)
		}
		return sendRowDescription(w, columns)
	case 'P':
		if cursor, ok := state.serverCursors[name]; ok {
			return sendRowDescription(w, cursor.Columns)
		}
		portal, ok := state.portals[name]
		if !ok {
			return sendError(w, "ERROR", fmt.Errorf("portal %q does not exist", name))
		}
		if state.txStatus() == transactionFailed && !transactionQueryAllowedAfterFailure(portal.Stmt.Query) {
			state.extendedSyncRequired = true
			return sendError(w, "ERROR", errCurrentTransactionAborted)
		}
		_, columns, err := describeStatement(db, portal.Stmt.Query, portal.Stmt.ParamCount)
		if err != nil {
			return sendError(w, "ERROR", err)
		}
		if len(columns) == 0 {
			return WriteMessage(w, msgNoData, nil)
		}
		// A portal Describe has already emitted the RowDescription. PostgreSQL
		// therefore sends only DataRow/CommandComplete when this portal executes;
		// tracking that fact is required by strict extended-protocol clients.
		portal.DescriptionSent = true
		return sendRowDescriptionWithFormats(w, columns, portal.ResultFmt)
	default:
		return sendError(w, "ERROR", fmt.Errorf("unsupported describe type %q", describeType))
	}
}

func mergeParamOIDs(inferred, fromParse []uint32) []uint32 {
	result := make([]uint32, len(inferred))
	copy(result, inferred)
	for i := range result {
		if result[i] == 0 && i < len(fromParse) {
			result[i] = fromParse[i]
		}
	}
	return result
}

// handleExecute runs a portal and sends results through native typed parameters.
func handleExecute(w io.Writer, db *libravdb.Database, state *connState, payload []byte) error {
	portalName, offset := ReadNullTerminated(payload, 0)
	if offset+4 > len(payload) {
		return sendError(w, "ERROR", fmt.Errorf("execute payload too short for maxRows"))
	}
	maxRows := binary.BigEndian.Uint32(payload[offset : offset+4])

	portal, ok := state.portals[portalName]
	if !ok {
		return sendError(w, "ERROR", fmt.Errorf("portal %q does not exist", portalName))
	}
	if state.txStatus() == transactionFailed && !transactionQueryAllowedAfterFailure(portal.Stmt.Query) {
		return sendExtendedExecutionError(w, state, errCurrentTransactionAborted)
	}
	if portal.Complete {
		if portal.CommandOnly {
			return sendCommandComplete(w, portal.CommandTag)
		}
		return sendCommandComplete(w, portal.CommandTag)
	}

	ctx, cancel := state.statementContext(context.Background())
	defer cancel()

	if !portal.Started {
		portal.Started = true
		query := rewritePgCatalogQuery(portal.Stmt.Query)
		boundParams := buildBoundParameterSet(portal)
		if portal.Stmt.ServerCursorQuery != "" {
			cursorQuery := rewritePgCatalogQuery(portal.Stmt.ServerCursorQuery)
			var results *libravdb.SearchResults
			var err error
			if state.epoch != nil {
				results, err = state.epoch.QueryWithBoundParamsAndSessionConfig(ctx, cursorQuery, boundParams, &state.config)
			} else {
				results, err = db.QueryWithBoundParamsAndSessionConfig(ctx, cursorQuery, boundParams, &state.config)
			}
			if err != nil {
				portal.Started = false
				return sendExtendedExecutionError(w, state, err)
			}
			_, columns, describeErr := describeStatement(db, cursorQuery, portal.Stmt.ParamCount)
			if describeErr != nil || len(columns) == 0 {
				columns = inferColumns(results)
			}
			state.serverCursors[portal.Stmt.ServerCursorName] = &serverCursor{
				Name: portal.Stmt.ServerCursorName, Query: cursorQuery, Results: results, Columns: columns,
			}
			portal.Complete = true
			portal.CommandOnly = true
			portal.CommandTag = "DECLARE CURSOR"
			return sendCommandComplete(w, portal.CommandTag)
		}

		if handled, commandTag, settingErr := applySessionSettingSQL(state, query); handled {
			if settingErr != nil {
				portal.Started = false
				return sendExtendedExecutionError(w, state, settingErr)
			}
			portal.Complete = true
			portal.CommandOnly = true
			portal.CommandTag = commandTag
			return sendCommandComplete(w, portal.CommandTag)
		} else if results, columns, handled, settingErr := handleAsyncpgJITQuery(query, &state.config, boundParams); handled {
			if settingErr != nil {
				portal.Started = false
				return sendExtendedExecutionError(w, state, settingErr)
			}
			portal.Results = results
			portal.Columns = columns
		} else if results, columns, handled, settingErr := handleSetConfigFunction(query, &state.config, boundParams); handled {
			if settingErr != nil {
				portal.Started = false
				return sendExtendedExecutionError(w, state, settingErr)
			}
			portal.Results = results
			portal.Columns = columns
		} else if results, columns, handled := interceptSystemQueryWithParams(query, db, boundParams); handled {
			portal.Results = results
			portal.Columns = columns
		} else if stmt, ok, _ := parsePgwireTransactionControl(query); ok {
			before := state.txStatus()
			tag, err := applyTransactionCommand(ctx, db, state, stmt)
			if err != nil {
				portal.Started = false
				if before != transactionIdle {
					state.markTransactionFailed()
				}
				return sendExtendedExecutionError(w, state, err)
			}
			portal.Complete = true
			portal.CommandOnly = true
			portal.CommandTag = tag
			return sendCommandComplete(w, tag)
		} else {
			var err error
			if state.epoch != nil {
				portal.Results, err = state.epoch.QueryWithBoundParamsAndSessionConfig(ctx, query, boundParams, &state.config)
			} else {
				portal.Results, err = db.QueryWithBoundParamsAndSessionConfig(ctx, query, boundParams, &state.config)
			}
			if err != nil {
				portal.Started = false
				return sendExtendedExecutionError(w, state, err)
			}
			// Execute must use the same typed RowDescription that Describe
			// reports. Re-inferring from stored string metadata turns
			// timestamptz columns into text and breaks database/sql scanners
			// such as GORM's *time.Time destinations.
			if _, described, describeErr := describeStatement(db, query, portal.Stmt.ParamCount); describeErr == nil && len(described) > 0 {
				portal.Columns = described
			} else {
				portal.Columns = inferColumns(portal.Results)
			}
		}
	}
	return executePortalRows(w, portal, maxRows)
}

// executePortalRows emits at most maxRows rows. A zero maxRows means no limit.
// A suspended portal emits PortalSuspended instead of CommandComplete and keeps
// its row index for the next Execute message.
func executePortalRows(w io.Writer, portal *Portal, maxRows uint32) error {
	if portal == nil {
		return fmt.Errorf("portal is nil")
	}
	query := ""
	if portal.Stmt != nil {
		query = portal.Stmt.Query
	}
	if portal.Results == nil {
		if isRowProducingSQL(query) {
			if !portal.DescriptionSent {
				if err := sendRowDescriptionWithFormats(w, portal.Columns, portal.ResultFmt); err != nil {
					return err
				}
				portal.DescriptionSent = true
			}
			portal.Complete = true
			portal.CommandTag = commandTagForSQL(query, 0)
			return sendCommandComplete(w, portal.CommandTag)
		}
		portal.Complete = true
		portal.CommandTag = commandTagForSQL(query, 0)
		return WriteMessage(w, msgEmptyQuery, nil)
	}
	// DML without RETURNING produces a command result, not a row stream. The
	// executor preserves the affected-row count in SearchResults.Total; emit a
	// PostgreSQL command tag so database/sql and ORMs populate RowsAffected.
	if isDMLSQL(query) && !hasReturningSQL(query) {
		portal.Complete = true
		portal.CommandTag = commandTagForSQL(query, portal.Results.Total)
		return sendCommandComplete(w, portal.CommandTag)
	}
	if portal.Results.Results == nil {
		if isRowProducingSQL(query) {
			if !portal.DescriptionSent {
				if err := sendRowDescriptionWithFormats(w, portal.Columns, portal.ResultFmt); err != nil {
					return err
				}
				portal.DescriptionSent = true
			}
			portal.Complete = true
			portal.CommandTag = commandTagForSQL(query, 0)
			return sendCommandComplete(w, portal.CommandTag)
		}
		portal.Complete = true
		portal.CommandTag = commandTagForSQL(query, portal.Results.Total)
		return sendCommandComplete(w, portal.CommandTag)
	}
	if !portal.DescriptionSent {
		if err := sendRowDescriptionWithFormats(w, portal.Columns, portal.ResultFmt); err != nil {
			return err
		}
		portal.DescriptionSent = true
	}

	total := len(portal.Results.Results)
	remaining := total - portal.RowIndex
	if remaining < 0 {
		remaining = 0
	}
	count := remaining
	if maxRows > 0 && uint64(count) > uint64(maxRows) {
		count = int(maxRows)
	}
	for i := 0; i < count; i++ {
		if err := sendDataRowWithFormats(w, portal.Results.Results[portal.RowIndex+i], portal.Columns, portal.ResultFmt); err != nil {
			return err
		}
	}
	portal.RowIndex += count
	if portal.RowIndex < total {
		return WriteMessage(w, msgPortalSuspended, nil)
	}
	portal.Complete = true
	portal.CommandTag = commandTagForSQL(query, total)
	return sendCommandComplete(w, portal.CommandTag)
}

func isRowProducingSQL(query string) bool {
	upper := strings.ToUpper(strings.TrimSpace(query))
	return strings.HasPrefix(upper, "SELECT ") || upper == "SELECT" ||
		strings.HasPrefix(upper, "WITH ") || strings.HasPrefix(upper, "COMPUTE ") ||
		strings.HasPrefix(upper, "EXPLAIN ") || upper == "EXPLAIN"
}

func isDMLSQL(query string) bool {
	upper := strings.ToUpper(strings.TrimSpace(query))
	return strings.HasPrefix(upper, "INSERT ") || strings.HasPrefix(upper, "UPDATE ") ||
		strings.HasPrefix(upper, "DELETE ") || upper == "INSERT" || upper == "UPDATE" || upper == "DELETE"
}

func hasReturningSQL(query string) bool {
	upper := strings.ToUpper(query)
	return strings.Contains(upper, " RETURNING ") || strings.HasSuffix(strings.TrimSpace(upper), " RETURNING")
}

func commandTagForSQL(query string, affected int) string {
	upper := strings.ToUpper(strings.TrimSpace(query))
	switch {
	case strings.HasPrefix(upper, "INSERT ") || upper == "INSERT":
		return fmt.Sprintf("INSERT 0 %d", affected)
	case strings.HasPrefix(upper, "UPDATE ") || upper == "UPDATE":
		return fmt.Sprintf("UPDATE %d", affected)
	case strings.HasPrefix(upper, "DELETE ") || upper == "DELETE":
		return fmt.Sprintf("DELETE %d", affected)
	case strings.HasPrefix(upper, "SELECT ") || upper == "SELECT":
		return fmt.Sprintf("SELECT %d", affected)
	case strings.HasPrefix(upper, "CREATE TABLE"):
		return "CREATE TABLE"
	case strings.HasPrefix(upper, "CREATE INDEX"):
		return "CREATE INDEX"
	case strings.HasPrefix(upper, "CREATE COLLECTION"):
		return "CREATE COLLECTION"
	case strings.HasPrefix(upper, "ALTER "):
		return "ALTER TABLE"
	case strings.HasPrefix(upper, "DROP "):
		return "DROP"
	case strings.HasPrefix(upper, "TRUNCATE "):
		return "TRUNCATE TABLE"
	default:
		return "COMMAND"
	}
}

// buildBoundParameterSet converts already-decoded Bind values into the
// optimizer's native ordinal/name representation. No SQL text is inspected or
// rewritten here; PreparedStmt's AST-derived parameter metadata supplies the
// slots.
func buildBoundParameterSet(portal *Portal) *optimizer.ParameterSet {
	if portal == nil || len(portal.Params) == 0 {
		return nil
	}
	stmt := portal.Stmt
	params := &optimizer.ParameterSet{}
	if stmt.numPositional > 0 {
		params.Positional = make([]optimizer.ScalarValue, stmt.numPositional)
	}
	for i, pv := range portal.Params {
		value := optimizer.ScalarFromInterface(paramGoValue(pv))
		if i < stmt.numPositional {
			params.Positional[i] = value
			continue
		}
		namedIdx := i - stmt.numPositional
		if namedIdx >= 0 && namedIdx < len(stmt.namedOrder) {
			params.Named = append(params.Named, optimizer.NamedScalar{
				Name:  []byte(stmt.namedOrder[namedIdx]),
				Value: value,
			})
		}
	}
	return params
}

// buildQueryParams converts bound portal parameters into QueryParams.
func buildQueryParams(portal *Portal) libravdb.QueryParams {
	if len(portal.Params) == 0 {
		return nil
	}
	params := make(libravdb.QueryParams, len(portal.Params))
	stmt := portal.Stmt

	for i, pv := range portal.Params {
		pos := i + 1
		if pos > stmt.numPositional && i >= stmt.numPositional {
			namedIdx := i - stmt.numPositional
			if namedIdx < len(stmt.namedOrder) {
				// QueryParams uses canonical unprefixed names. The optimizer
				// accepts both the public form and wire-prefixed aliases.
				key := stmt.namedOrder[namedIdx]
				params[key] = paramGoValue(pv)
			}
			continue
		}
		// Positional parameters are canonicalized to their decimal ordinal;
		// the optimizer resolves both "$1" and "1" forms.
		key := strconv.Itoa(pos)
		params[key] = paramGoValue(pv)
	}

	return params
}

func paramGoValue(pv ParamValue) interface{} {
	if pv.IsNull {
		return nil
	}
	return pv.Value
}

// decodeParamValue decodes raw parameter bytes into a typed Go value.
// PostgreSQL binary values are big-endian. Vector parameters use the
// PostgreSQL one-dimensional float array representation on the wire.
func decodeParamValue(raw []byte, format int16, oid uint32) (interface{}, error) {
	if raw == nil {
		return nil, nil
	}
	if format == 1 {
		return decodeBinaryParam(raw, oid)
	}
	switch oid {
	case OIDBool:
		value, err := parseBoolBytes(raw)
		return value, err
	case OIDInt2, OIDInt4:
		bits := 32
		if oid == OIDInt2 {
			bits = 16
		}
		n, err := parseIntBytes(raw, bits)
		if err != nil {
			return nil, fmt.Errorf("invalid integer: %w", err)
		}
		if oid == OIDInt2 {
			return int16(n), nil
		}
		return int32(n), nil
	case OIDInt8:
		n, err := parseIntBytes(raw, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid bigint: %w", err)
		}
		return n, nil
	case OIDFloat4:
		f, err := parseFloatBytes(raw, 32)
		if err != nil {
			return nil, fmt.Errorf("invalid real: %w", err)
		}
		return float32(f), nil
	case OIDFloat8:
		f, err := parseFloatBytes(raw, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid double precision: %w", err)
		}
		return f, nil
	case OIDFloat4Array, OIDFloat8Array:
		if vec := parseVectorParamBytes(raw); vec != nil {
			return vec, nil
		}
		return nil, fmt.Errorf("invalid vector/float-array value")
	case OIDVector:
		// pgvector parameters sent in text format (the normal asyncpg path)
		// use the extension type OID, not the float4[] compatibility OID. Decode
		// them to the same native []float32 value used by binary vector binds so
		// the optimizer can execute vector operators instead of falling through
		// to the virtual relational evaluator.
		if vec := parseVectorParamBytes(raw); vec != nil {
			return vec, nil
		}
		return nil, fmt.Errorf("invalid vector value")
	default:
		// Preserve unknown text types as bytes. The typed execution boundary
		// can compare them without manufacturing a SQL string.
		return append([]byte(nil), raw...), nil
	}
}

func decodeBinaryParam(raw []byte, oid uint32) (interface{}, error) {
	switch oid {
	case OIDBool:
		if len(raw) != 1 || (raw[0] != 0 && raw[0] != 1) {
			return nil, fmt.Errorf("invalid binary boolean")
		}
		return raw[0] == 1, nil
	case OIDInt2:
		if len(raw) != 2 {
			return nil, fmt.Errorf("binary int2 requires 2 bytes")
		}
		return int16(binary.BigEndian.Uint16(raw)), nil
	case OIDInt4:
		if len(raw) != 4 {
			return nil, fmt.Errorf("binary int4 requires 4 bytes")
		}
		return int32(binary.BigEndian.Uint32(raw)), nil
	case OIDInt8:
		if len(raw) != 8 {
			return nil, fmt.Errorf("binary int8 requires 8 bytes")
		}
		return int64(binary.BigEndian.Uint64(raw)), nil
	case OIDFloat4:
		if len(raw) != 4 {
			return nil, fmt.Errorf("binary float4 requires 4 bytes")
		}
		return math.Float32frombits(binary.BigEndian.Uint32(raw)), nil
	case OIDFloat8:
		if len(raw) != 8 {
			return nil, fmt.Errorf("binary float8 requires 8 bytes")
		}
		return math.Float64frombits(binary.BigEndian.Uint64(raw)), nil
	case OIDFloat4Array, OIDFloat8Array:
		return decodeBinaryFloatArray(raw, oid)
	case OIDOIDArray:
		return decodeBinaryOIDArray(raw)
	case OIDVector:
		return decodeBinaryVector(raw)
	case OIDTimestamp, OIDTimestamptz:
		if len(raw) != 8 {
			return nil, fmt.Errorf("binary timestamp requires 8 bytes")
		}
		micros := int64(binary.BigEndian.Uint64(raw))
		const postgresEpochOffset = int64(946684800) // 2000-01-01 from Unix epoch
		seconds, remainder := micros/1_000_000, micros%1_000_000
		return time.Unix(postgresEpochOffset+seconds, remainder*1000).UTC().Format(time.RFC3339Nano), nil
	case OIDDate:
		if len(raw) != 4 {
			return nil, fmt.Errorf("binary date requires 4 bytes")
		}
		days := int32(binary.BigEndian.Uint32(raw))
		const postgresEpochDays = int64(10957) // 2000-01-01 from Unix epoch
		return time.Unix((postgresEpochDays+int64(days))*86400, 0).UTC().Format("2006-01-02"), nil
	case OIDJSON, OIDJSONB:
		// PostgreSQL's binary jsonb representation starts with a one-byte
		// version marker. The SQL executor consumes the JSON document itself.
		if oid == OIDJSONB && len(raw) > 0 && raw[0] == 1 {
			raw = raw[1:]
		}
		return append([]byte(nil), raw...), nil
	default:
		// Unknown binary types are preserved as bytes rather than being
		// silently coerced into SQL text.
		return append([]byte(nil), raw...), nil
	}
}

// decodeBinaryVector decodes pgvector's native binary representation:
// int16 dimension, int16 reserved, then big-endian float4 elements.
func decodeBinaryVector(raw []byte) ([]float32, error) {
	if len(raw) < 4 {
		return nil, fmt.Errorf("binary vector header is truncated")
	}
	dimension := int(binary.BigEndian.Uint16(raw[:2]))
	want := 4 + dimension*4
	if len(raw) != want {
		return nil, fmt.Errorf("binary vector length %d does not match dimension %d", len(raw), dimension)
	}
	values := make([]float32, dimension)
	for i := range values {
		values[i] = math.Float32frombits(binary.BigEndian.Uint32(raw[4+i*4 : 8+i*4]))
	}
	return values, nil
}

func decodeBinaryFloatArray(raw []byte, oid uint32) ([]float32, error) {
	if len(raw) < 12 {
		return nil, fmt.Errorf("binary array header is truncated")
	}
	ndim := int32(binary.BigEndian.Uint32(raw[0:4]))
	hasNull := int32(binary.BigEndian.Uint32(raw[4:8]))
	elemOID := binary.BigEndian.Uint32(raw[8:12])
	if ndim == 0 {
		return []float32{}, nil
	}
	if ndim != 1 {
		return nil, fmt.Errorf("vector parameter must be one-dimensional, got %d dimensions", ndim)
	}
	wantElemOID := uint32(OIDFloat4)
	if oid == OIDFloat8Array {
		wantElemOID = OIDFloat8
	}
	if elemOID != wantElemOID {
		return nil, fmt.Errorf("binary array element OID %d does not match %d", elemOID, wantElemOID)
	}
	off := 12
	if off+8 > len(raw) {
		return nil, fmt.Errorf("binary array dimensions are truncated")
	}
	length := int32(binary.BigEndian.Uint32(raw[off : off+4]))
	off += 8 // length plus lower bound
	if length < 0 {
		return nil, fmt.Errorf("negative binary array length")
	}
	values := make([]float32, 0, length)
	for i := int32(0); i < length; i++ {
		if off+4 > len(raw) {
			return nil, fmt.Errorf("binary array element %d is truncated", i)
		}
		n := int32(binary.BigEndian.Uint32(raw[off : off+4]))
		off += 4
		if n == -1 {
			if hasNull != 1 {
				return nil, fmt.Errorf("binary array contains an unexpected NULL element")
			}
			return nil, fmt.Errorf("vector parameter cannot contain NULL elements")
		}
		if n < 0 || off+int(n) > len(raw) {
			return nil, fmt.Errorf("binary array element %d length is invalid", i)
		}
		if oid == OIDFloat4Array && n != 4 || oid == OIDFloat8Array && n != 8 {
			return nil, fmt.Errorf("binary array element %d has invalid width %d", i, n)
		}
		if oid == OIDFloat4Array {
			values = append(values, math.Float32frombits(binary.BigEndian.Uint32(raw[off:off+4])))
		} else {
			values = append(values, float32(math.Float64frombits(binary.BigEndian.Uint64(raw[off:off+8]))))
		}
		off += int(n)
	}
	return values, nil
}

// decodeBinaryOIDArray decodes PostgreSQL's one-dimensional binary oid[]
// representation. asyncpg uses this type for its recursive type-catalog
// lookup parameter. Returning PostgreSQL text-array syntax keeps the value
// usable by the catalog projection without losing the original OIDs.
func decodeBinaryOIDArray(raw []byte) (string, error) {
	if len(raw) < 12 {
		return "", fmt.Errorf("binary oid array header is truncated")
	}
	ndim := int32(binary.BigEndian.Uint32(raw[0:4]))
	hasNull := int32(binary.BigEndian.Uint32(raw[4:8]))
	elemOID := binary.BigEndian.Uint32(raw[8:12])
	if ndim == 0 {
		if len(raw) != 12 {
			return "", fmt.Errorf("binary oid array has trailing data")
		}
		return "{}", nil
	}
	if ndim != 1 {
		return "", fmt.Errorf("oid parameter must be one-dimensional, got %d dimensions", ndim)
	}
	if elemOID != uint32(OIDOID) {
		return "", fmt.Errorf("binary oid array element OID %d does not match %d", elemOID, OIDOID)
	}
	off := 12
	if off+8 > len(raw) {
		return "", fmt.Errorf("binary oid array dimensions are truncated")
	}
	length := int32(binary.BigEndian.Uint32(raw[off : off+4]))
	off += 8 // length plus lower bound
	if length < 0 {
		return "", fmt.Errorf("negative binary oid array length")
	}
	values := make([]byte, 0, 2+int(length)*11)
	values = append(values, '{')
	for i := int32(0); i < length; i++ {
		if off+4 > len(raw) {
			return "", fmt.Errorf("binary oid array element %d is truncated", i)
		}
		n := int32(binary.BigEndian.Uint32(raw[off : off+4]))
		off += 4
		if n == -1 {
			if hasNull != 1 {
				return "", fmt.Errorf("binary oid array contains an unexpected NULL element")
			}
			return "", fmt.Errorf("oid parameter cannot contain NULL elements")
		}
		if n != 4 || off+int(n) > len(raw) {
			return "", fmt.Errorf("binary oid array element %d has invalid width %d", i, n)
		}
		if i > 0 {
			values = append(values, ',')
		}
		values = strconv.AppendUint(values, uint64(binary.BigEndian.Uint32(raw[off:off+4])), 10)
		off += int(n)
	}
	if off != len(raw) {
		return "", fmt.Errorf("binary oid array has trailing data")
	}
	values = append(values, '}')
	return string(values), nil
}

func parseVectorParam(s string) []float32 {
	return parseVectorParamBytes([]byte(s))
}

func parseVectorParamBytes(raw []byte) []float32 {
	raw = trimASCIIWhitespace(raw)
	if len(raw) == 0 {
		return nil
	}
	if (raw[0] == '[' && raw[len(raw)-1] == ']') || (raw[0] == '{' && raw[len(raw)-1] == '}') {
		raw = trimASCIIWhitespace(raw[1 : len(raw)-1])
	}
	if len(raw) == 0 {
		return nil
	}
	vec := make([]float32, 0, 8)
	start := 0
	for start < len(raw) {
		for start < len(raw) && (raw[start] == ',' || isASCIIWhitespace(raw[start])) {
			start++
		}
		if start == len(raw) {
			break
		}
		end := start
		for end < len(raw) && raw[end] != ',' && !isASCIIWhitespace(raw[end]) {
			end++
		}
		f, err := parseFloatBytes(raw[start:end], 32)
		if err != nil {
			return nil
		}
		vec = append(vec, float32(f))
		start = end
	}
	if len(vec) == 0 {
		return nil
	}
	return vec
}

func parseBoolStrict(s string) (bool, error) {
	return parseBoolBytes([]byte(s))
}

func parseBoolBytes(raw []byte) (bool, error) {
	raw = trimASCIIWhitespace(raw)
	switch {
	case asciiEqualFoldBytes(raw, []byte("true")), asciiEqualFoldBytes(raw, []byte("t")),
		asciiEqualFoldBytes(raw, []byte("yes")), asciiEqualFoldBytes(raw, []byte("y")),
		asciiEqualFoldBytes(raw, []byte("on")), asciiEqualFoldBytes(raw, []byte("1")):
		return true, nil
	case asciiEqualFoldBytes(raw, []byte("false")), asciiEqualFoldBytes(raw, []byte("f")),
		asciiEqualFoldBytes(raw, []byte("no")), asciiEqualFoldBytes(raw, []byte("n")),
		asciiEqualFoldBytes(raw, []byte("off")), asciiEqualFoldBytes(raw, []byte("0")):
		return false, nil
	default:
		return false, fmt.Errorf("invalid boolean")
	}
}

func parseIntBytes(raw []byte, bits int) (int64, error) {
	raw = trimASCIIWhitespace(raw)
	if len(raw) == 0 {
		return 0, fmt.Errorf("empty integer")
	}
	neg := false
	start := 0
	if raw[0] == '+' || raw[0] == '-' {
		neg = raw[0] == '-'
		start = 1
	}
	if start == len(raw) {
		return 0, fmt.Errorf("invalid integer")
	}
	var value uint64
	for _, c := range raw[start:] {
		if c < '0' || c > '9' {
			return 0, fmt.Errorf("invalid integer")
		}
		value = value*10 + uint64(c-'0')
	}
	max := uint64(1)<<(bits-1) - 1
	if neg {
		if value > max+1 {
			return 0, fmt.Errorf("integer overflow")
		}
		if value == max+1 {
			return -int64(max) - 1, nil
		}
		return -int64(value), nil
	}
	if value > max {
		return 0, fmt.Errorf("integer overflow")
	}
	return int64(value), nil
}

func parseFloatBytes(raw []byte, bits int) (float64, error) {
	raw = trimASCIIWhitespace(raw)
	if len(raw) == 0 {
		return 0, fmt.Errorf("empty float")
	}
	if asciiEqualFoldBytes(raw, []byte("nan")) {
		return math.NaN(), nil
	}
	neg := false
	start := 0
	if raw[0] == '+' || raw[0] == '-' {
		neg = raw[0] == '-'
		start = 1
	}
	if start >= len(raw) {
		return 0, fmt.Errorf("invalid float")
	}
	if asciiEqualFoldBytes(raw[start:], []byte("inf")) || asciiEqualFoldBytes(raw[start:], []byte("infinity")) {
		if neg {
			return math.Inf(-1), nil
		}
		return math.Inf(1), nil
	}
	var value float64
	digits := 0
	for start < len(raw) && raw[start] >= '0' && raw[start] <= '9' {
		value = value*10 + float64(raw[start]-'0')
		start++
		digits++
	}
	if start < len(raw) && raw[start] == '.' {
		start++
		place := 0.1
		for start < len(raw) && raw[start] >= '0' && raw[start] <= '9' {
			value += float64(raw[start]-'0') * place
			place *= 0.1
			start++
			digits++
		}
	}
	if digits == 0 {
		return 0, fmt.Errorf("invalid float")
	}
	exponent := 0
	expNeg := false
	if start < len(raw) && (raw[start] == 'e' || raw[start] == 'E') {
		start++
		if start < len(raw) && (raw[start] == '+' || raw[start] == '-') {
			expNeg = raw[start] == '-'
			start++
		}
		if start == len(raw) {
			return 0, fmt.Errorf("invalid exponent")
		}
		for start < len(raw) && raw[start] >= '0' && raw[start] <= '9' {
			exponent = exponent*10 + int(raw[start]-'0')
			start++
		}
	}
	if start != len(raw) {
		return 0, fmt.Errorf("invalid float")
	}
	if expNeg {
		exponent = -exponent
	}
	value *= math.Pow10(exponent)
	if neg {
		value = -value
	}
	if bits == 32 {
		value = float64(float32(value))
	}
	return value, nil
}

func trimASCIIWhitespace(raw []byte) []byte {
	start, end := 0, len(raw)
	for start < end && isASCIIWhitespace(raw[start]) {
		start++
	}
	for end > start && isASCIIWhitespace(raw[end-1]) {
		end--
	}
	return raw[start:end]
}

func isASCIIWhitespace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f'
}

func asciiEqualFoldBytes(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		x, y := a[i], b[i]
		if x >= 'A' && x <= 'Z' {
			x += 'a' - 'A'
		}
		if y >= 'A' && y <= 'Z' {
			y += 'a' - 'A'
		}
		if x != y {
			return false
		}
	}
	return true
}

func handleSync(w io.Writer, state *connState) error {
	return WriteMessage(w, msgReadyForQuery, []byte{state.readyStatus()})
}

func handleClose(w io.Writer, state *connState, payload []byte) error {
	if len(payload) < 1 {
		return WriteMessage(w, msgCloseComplete, nil)
	}
	closeType := payload[0]
	name, _ := ReadNullTerminated(payload, 1)

	switch closeType {
	case 'S':
		delete(state.prepared, name)
	case 'P':
		delete(state.portals, name)
	}
	return WriteMessage(w, msgCloseComplete, nil)
}
