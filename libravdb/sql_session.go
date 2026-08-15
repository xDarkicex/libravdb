package libravdb

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"

	"github.com/xDarkicex/lexer"
	"github.com/xDarkicex/lexer/parser"
)

var ErrSessionClosed = fmt.Errorf("session is closed")
var ErrEpochAlreadyActive = fmt.Errorf("epoch transaction already active")
var ErrNoActiveEpoch = fmt.Errorf("no active epoch transaction")

type SQLSession struct {
	db       *Database
	mu       sync.Mutex
	epoch    *EpochTx
	prepared map[string]string
	config   SessionConfig
	closed   bool
}

func (db *Database) NewSQLSession(ctx context.Context) (*SQLSession, error) {
	if db == nil {
		return nil, fmt.Errorf("database is nil")
	}
	return &SQLSession{db: db, prepared: make(map[string]string), config: DefaultSessionConfig()}, nil
}

// SessionConfig returns a copy of the connection-local settings.
func (s *SQLSession) SessionConfig() SessionConfig {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.config
}

func (s *SQLSession) Exec(sql string) error {
	return s.ExecWithParams(sql, nil)
}

func (s *SQLSession) ExecWithParams(sql string, params QueryParams) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrSessionClosed
	}

	doc, err := parseSQL(sql)
	if err != nil {
		return err
	}
	if len(doc.TransactionStmts) > 0 {
		return s.handleTransactionStmts(doc.TransactionStmts, params)
	}
	if len(doc.PrepareStmts) > 0 {
		return s.handlePrepare(doc, []byte(sql))
	}
	if len(doc.ExecuteStmts) > 0 {
		_, err := s.executePrepared(doc, sql)
		return err
	}
	if len(doc.SessionSettingStmts) > 0 {
		if len(doc.SessionSettingStmts) != 1 {
			return fmt.Errorf("multiple session settings are not supported in one call")
		}
		return s.config.ApplySessionSetting([]byte(sql), doc, &doc.SessionSettingStmts[0])
	}
	return s.executeStatements(sql, params)
}

func (s *SQLSession) Query(sql string) (*SearchResults, error) {
	return s.QueryWithParams(sql, nil)
}

func (s *SQLSession) QueryWithParams(sql string, params QueryParams) (*SearchResults, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, ErrSessionClosed
	}

	doc, err := parseSQL(sql)
	if err != nil {
		return nil, err
	}
	if len(doc.TransactionStmts) > 0 {
		if err := s.handleTransactionStmts(doc.TransactionStmts, params); err != nil {
			return nil, err
		}
		return &SearchResults{}, nil
	}
	if len(doc.PrepareStmts) > 0 {
		if err := s.handlePrepare(doc, []byte(sql)); err != nil {
			return nil, err
		}
		return &SearchResults{}, nil
	}
	if len(doc.ExecuteStmts) > 0 {
		return s.executePrepared(doc, sql)
	}
	if len(doc.SessionSettingStmts) > 0 {
		if len(doc.SessionSettingStmts) != 1 {
			return nil, fmt.Errorf("multiple session settings are not supported in one call")
		}
		if err := s.config.ApplySessionSetting([]byte(sql), doc, &doc.SessionSettingStmts[0]); err != nil {
			return nil, err
		}
		return &SearchResults{}, nil
	}
	if s.epoch != nil {
		ctx := s.epoch.Context(context.Background())
		ctx, cancel := s.withStatementTimeout(ctx)
		defer cancel()
		return s.db.queryWithSessionConfig(ctx, sql, params, &s.config)
	}
	ctx, cancel := s.withStatementTimeout(context.Background())
	defer cancel()
	return s.db.queryWithSessionConfig(ctx, sql, params, &s.config)
}

func (s *SQLSession) withStatementTimeout(base context.Context) (context.Context, context.CancelFunc) {
	timeout := s.config.EffectiveTimeout(0)
	if timeout <= 0 {
		return base, func() {}
	}
	return context.WithTimeout(base, timeout)
}

func (s *SQLSession) handlePrepare(doc *parser.QueryDoc, src []byte) error {
	if doc == nil || len(doc.PrepareStmts) != 1 {
		return fmt.Errorf("PREPARE requires exactly one statement")
	}
	stmt := doc.PrepareStmts[0]
	if stmt.NameEnd > uint32(len(src)) || stmt.QueryEnd > uint32(len(src)) {
		return fmt.Errorf("PREPARE source span out of bounds")
	}
	name := strings.ToLower(string(src[stmt.NameStart:stmt.NameEnd]))
	query := string(src[stmt.QueryStart:stmt.QueryEnd])
	if name == "" || strings.TrimSpace(query) == "" {
		return fmt.Errorf("PREPARE requires a name and query")
	}
	// The parser deliberately stores the body span rather than reparsing it
	// into the outer document. Validate the body now with the authoritative
	// parser before publishing it in the session-local prepared map.
	body := &parser.QueryDoc{}
	if err := parser.Parse([]byte(query), body); err != nil {
		return fmt.Errorf("PREPARE %q: %w", name, err)
	}
	s.prepared[name] = query
	return nil
}

func (s *SQLSession) executePrepared(doc *parser.QueryDoc, src string) (*SearchResults, error) {
	if doc == nil || len(doc.ExecuteStmts) != 1 {
		return nil, fmt.Errorf("EXECUTE requires exactly one statement")
	}
	stmt := doc.ExecuteStmts[0]
	name := strings.ToLower(string([]byte(src)[stmt.NameStart:stmt.NameEnd]))
	query, ok := s.prepared[name]
	if !ok {
		return nil, fmt.Errorf("prepared statement %q does not exist", name)
	}
	params := make(QueryParams, stmt.ArgsCount)
	for i := int32(0); i < stmt.ArgsCount; i++ {
		arg := doc.ExecuteArgs[stmt.ArgsStart+i]
		value, err := executeArgValue(doc, []byte(src), arg)
		if err != nil {
			return nil, fmt.Errorf("EXECUTE %q argument %d: %w", name, i+1, err)
		}
		params["$"+strconv.Itoa(int(i)+1)] = value
	}
	if s.epoch != nil {
		ctx, cancel := s.withStatementTimeout(s.epoch.Context(context.Background()))
		defer cancel()
		return s.db.queryWithSessionConfig(ctx, query, params, &s.config)
	}
	ctx, cancel := s.withStatementTimeout(context.Background())
	defer cancel()
	return s.db.queryWithSessionConfig(ctx, query, params, &s.config)
}

func executeArgValue(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (interface{}, error) {
	switch ref.Kind {
	case parser.NodeKindString:
		if ref.ID < 0 || int(ref.ID) >= len(doc.Strings) {
			return nil, fmt.Errorf("invalid string argument")
		}
		sl := doc.Strings[ref.ID]
		decode := lexer.DecodeStringLiteralInto
		if sl.Escape {
			decode = lexer.DecodeEscapeStringLiteralInto
		}
		if value, ok := decode(src, sl.Start, sl.End, nil); ok {
			return string(value), nil
		}
		return nil, fmt.Errorf("string argument requires a caller scratch buffer")
	case parser.NodeKindNumber:
		if ref.ID < 0 || int(ref.ID) >= len(doc.Numbers) {
			return nil, fmt.Errorf("invalid numeric argument")
		}
		n := doc.Numbers[ref.ID]
		raw := string(src[n.Start:n.End])
		if strings.ContainsAny(raw, ".eE") {
			value, err := strconv.ParseFloat(raw, 64)
			return value, err
		}
		value, err := strconv.ParseInt(raw, 10, 64)
		return value, err
	case parser.NodeKindIdentifier:
		id := doc.Identifiers[ref.ID]
		raw := src[id.Start:id.End]
		if strings.EqualFold(string(raw), "NULL") {
			return nil, nil
		}
		return string(raw), nil
	default:
		return nil, fmt.Errorf("unsupported argument expression kind %d", ref.Kind)
	}
}

func (s *SQLSession) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	if s.epoch != nil {
		_ = s.epoch.Rollback(context.Background())
		s.epoch = nil
	}
	return nil
}

func (s *SQLSession) handleTransactionStmts(stmts []parser.TransactionStmt, params QueryParams) error {
	for _, stmt := range stmts {
		switch stmt.Kind {
		case parser.TransactionBeginEpoch:
			if s.epoch != nil {
				return ErrEpochAlreadyActive
			}
			epoch, err := s.db.BeginEpochTx(context.Background())
			if err != nil {
				return fmt.Errorf("BEGIN EPOCH: %w", err)
			}
			s.epoch = epoch

		case parser.TransactionCommit:
			if s.epoch == nil {
				return ErrNoActiveEpoch
			}
			if err := s.epoch.Commit(context.Background()); err != nil {
				return fmt.Errorf("COMMIT: %w", err)
			}
			s.epoch = nil

		case parser.TransactionRollback:
			if s.epoch == nil {
				return ErrNoActiveEpoch
			}
			if err := s.epoch.Rollback(context.Background()); err != nil {
				return fmt.Errorf("ROLLBACK: %w", err)
			}
			s.epoch = nil

		case parser.TransactionSavepoint:
			if s.epoch == nil {
				return ErrSavepointOutsideEpoch
			}
			if err := s.epoch.Savepoint(stmt.SavepointName); err != nil {
				return err
			}

		case parser.TransactionRollbackToSavepoint:
			if s.epoch == nil {
				return ErrSavepointOutsideEpoch
			}
			if err := s.epoch.RollbackTo(stmt.SavepointName); err != nil {
				return err
			}

		case parser.TransactionReleaseSavepoint:
			if s.epoch == nil {
				return ErrSavepointOutsideEpoch
			}
			if err := s.epoch.ReleaseSavepoint(stmt.SavepointName); err != nil {
				return err
			}

		default:
			return fmt.Errorf("unknown transaction statement kind: %d", stmt.Kind)
		}
	}
	return nil
}

func (s *SQLSession) executeStatements(sql string, params QueryParams) error {
	if s.epoch != nil {
		ctx := s.epoch.Context(context.Background())
		ctx, cancel := s.withStatementTimeout(ctx)
		defer cancel()
		_, err := s.db.queryWithSessionConfig(ctx, sql, params, &s.config)
		return err
	}
	ctx, cancel := s.withStatementTimeout(context.Background())
	defer cancel()
	_, err := s.db.queryWithSessionConfig(ctx, sql, params, &s.config)
	return err
}

func parseSQL(sql string) (*parser.QueryDoc, error) {
	src := []byte(sql)
	doc := &parser.QueryDoc{}
	if err := parser.Parse(src, doc); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}
	// Count standalone statements. ComputeLeidenStmts referenced by a
	// CTE (SelectStmt.CTEsCount > 0) are not standalone — they are part
	// of the enclosing SELECT and should not trigger the multi-statement
	// guard.
	standaloneLeidenCount := len(doc.ComputeLeidenStmts)
	for i := range doc.SelectStmts {
		if doc.SelectStmts[i].CTEsCount > 0 {
			standaloneLeidenCount--
		}
	}

	// Nested SELECTs in generic CTEs, subqueries, and UNION branches are part
	// of one root statement. Count only root nodes so SQLSession does not
	// misclassify a valid query composition as multi-statement input.
	rootSelectCount := 0
	for _, node := range doc.Nodes {
		if node.Kind == parser.NodeKindSelectStmt {
			rootSelectCount++
		}
	}
	if rootSelectCount == 0 && len(doc.SelectStmts) > 0 {
		rootSelectCount = 1
	}

	stmtCount := rootSelectCount + len(doc.InsertStmts) +
		len(doc.InsertGraphEdgeStmts) + len(doc.UpdateStmts) +
		len(doc.DeleteStmts) + len(doc.CreateTableStmts) + len(doc.CreateEdgeTypeStmts) +
		len(doc.DropTableStmts) + len(doc.CreateIndexStmts) +
		len(doc.DropIndexStmts) + len(doc.AlterTableStmts) +
		len(doc.TransactionStmts) + len(doc.PrepareStmts) + len(doc.ExecuteStmts) + len(doc.SessionSettingStmts) + standaloneLeidenCount
	if stmtCount > 1 {
		return nil, fmt.Errorf("multi-statement input is not supported; execute one statement per call")
	}
	return doc, nil
}
