package libravdb

import (
	"context"
	"fmt"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// Query parses, binds, optimizes, and executes a SQL/PGQ query.
// It is the primary embedded entrypoint for analytical query execution.
func (db *Database) Query(ctx context.Context, sql string) (*SearchResults, error) {
	src := []byte(sql)

	// 1 & 2. Lex & Parse
	doc := &parser.QueryDoc{}
	if err := parser.Parse(src, doc); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	// 3. Bind OIDs (Modifies doc in place)
	db.mu.RLock()
	cat := db.catalog
	db.mu.RUnlock()

	if cat == nil {
		return nil, fmt.Errorf("catalog not initialized")
	}

	// Empty statement list (e.g. comment-only SQL like pgx's "-- ping")
	// is a valid no-op, mirroring Postgres' EmptyQueryResponse.
	if len(doc.SelectStmts) == 0 && len(doc.InsertStmts) == 0 &&
		len(doc.UpdateStmts) == 0 && len(doc.DeleteStmts) == 0 &&
		len(doc.CreateTableStmts) == 0 && len(doc.DropTableStmts) == 0 &&
		len(doc.CreateIndexStmts) == 0 && len(doc.DropIndexStmts) == 0 &&
		len(doc.AlterTableStmts) == 0 {
		return &SearchResults{}, nil
	}

	binder := catalog.NewBinder(cat, src)
	if err := binder.Bind(doc); err != nil {
		return nil, fmt.Errorf("bind error: %w", err)
	}

	// 4. Optimize (AST -> Physical Plan)
	opt := optimizer.NewOptimizer(cat)
	plan, err := opt.Optimize(doc, src)
	if err != nil {
		return nil, fmt.Errorf("optimize error: %w", err)
	}

	// 5. Execute Physical Plan
	return newExecutor(db).Execute(ctx, plan)
}
