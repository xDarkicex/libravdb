package libravdb

import (
	"context"
	"fmt"
	"sync"

	"github.com/xDarkicex/lexer/parser"
)

var ErrSessionClosed = fmt.Errorf("session is closed")
var ErrEpochAlreadyActive = fmt.Errorf("epoch transaction already active")
var ErrNoActiveEpoch = fmt.Errorf("no active epoch transaction")

type SQLSession struct {
	db     *Database
	mu     sync.Mutex
	epoch  *EpochTx
	closed bool
}

func (db *Database) NewSQLSession(ctx context.Context) (*SQLSession, error) {
	if db == nil {
		return nil, fmt.Errorf("database is nil")
	}
	return &SQLSession{db: db}, nil
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
	if s.epoch != nil {
		ctx := s.epoch.Context(context.Background())
		return s.db.queryWithContext(ctx, sql, params)
	}
	return s.db.QueryWithParams(context.Background(), sql, params)
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
		_, err := s.db.queryWithContext(ctx, sql, params)
		return err
	}
	_, err := s.db.queryWithContext(context.Background(), sql, params)
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

	stmtCount := len(doc.SelectStmts) + len(doc.InsertStmts) +
		len(doc.InsertGraphEdgeStmts) + len(doc.UpdateStmts) +
		len(doc.DeleteStmts) + len(doc.CreateTableStmts) +
		len(doc.DropTableStmts) + len(doc.CreateIndexStmts) +
		len(doc.DropIndexStmts) + len(doc.AlterTableStmts) +
		len(doc.TransactionStmts) + standaloneLeidenCount
	if stmtCount > 1 {
		return nil, fmt.Errorf("multi-statement input is not supported; execute one statement per call")
	}
	return doc, nil
}
