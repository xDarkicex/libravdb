package libravdb

import (
	"context"
	"fmt"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// QueryParams holds typed query parameter values.
type QueryParams map[string]interface{}

// Query parses, binds, optimizes, and executes a SQL/PGQ query.
func (db *Database) Query(ctx context.Context, sql string) (*SearchResults, error) {
	return db.QueryWithParams(ctx, sql, nil)
}

// QueryWithParams executes SQL with named parameters ($param or @param).
func (db *Database) QueryWithParams(ctx context.Context, sql string, params QueryParams) (*SearchResults, error) {
	return db.queryWithContext(ctx, sql, params)
}

func (db *Database) queryWithContext(ctx context.Context, sql string, params QueryParams) (*SearchResults, error) {
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

	// COMPUTE LEIDEN is a standalone statement that bypasses the catalog/optimizer.
	// It requires an active epoch transaction. Skip if the COMPUTE LEIDEN is
	// contained within a CTE (WITH ... AS).
	standaloneLeiden := false
	for i := range doc.ComputeLeidenStmts {
		cteReferenced := false
		for _, cte := range doc.CTEs {
			if cte.Body.Kind == parser.NodeKindComputeLeidenStmt && cte.Body.ID == int32(i) {
				cteReferenced = true
				break
			}
		}
		if !cteReferenced {
			standaloneLeiden = true
			break
		}
	}
	if standaloneLeiden && len(doc.ComputeLeidenStmts) > 0 {
		return db.executeComputeLeiden(ctx, src, doc)
	}

	// CTE SELECT: WITH local_clusters AS (COMPUTE LEIDEN ...) SELECT ... JOIN ...
	if len(doc.SelectStmts) > 0 && doc.SelectStmts[0].CTEsCount > 0 {
		return db.executeLeidenCTE(ctx, src, doc, params)
	}

	// Empty statement list is a valid no-op.
	if len(doc.SelectStmts) == 0 && len(doc.InsertStmts) == 0 &&
		len(doc.InsertGraphEdgeStmts) == 0 &&
		len(doc.UpdateStmts) == 0 && len(doc.DeleteStmts) == 0 &&
		len(doc.CreateTableStmts) == 0 && len(doc.DropTableStmts) == 0 &&
		len(doc.CreateIndexStmts) == 0 && len(doc.DropIndexStmts) == 0 &&
		len(doc.AlterTableStmts) == 0 {
		if len(doc.TransactionStmts) > 0 {
			if epoch := epochFromContext(ctx); epoch != nil {
				return db.handleEpochTransactionStmts(ctx, epoch, doc.TransactionStmts)
			}
		}
		return &SearchResults{}, nil
	}

	binder := catalog.NewBinder(cat, src)
	if err := binder.Bind(doc); err != nil {
		return nil, fmt.Errorf("bind error: %w", err)
	}

	// 4. Optimize (AST -> Physical Plan)
	opt := optimizer.NewOptimizer(cat)
	plan, err := opt.OptimizeWithParams(doc, src, params)
	if err != nil {
		return nil, fmt.Errorf("optimize error: %w", err)
	}

	// 5. Execute Physical Plan
	return newExecutor(db).Execute(ctx, plan)
}

// executeComputeLeiden runs the full COMPUTE LEIDEN pipeline:
// parse → lower → resolve collection → bind → execute → materialize → SearchResults.
func (db *Database) executeComputeLeiden(ctx context.Context, src []byte, doc *parser.QueryDoc) (*SearchResults, error) {
	epoch := epochFromContext(ctx)
	if epoch == nil {
		return nil, fmt.Errorf("COMPUTE LEIDEN requires an active epoch transaction")
	}

	if len(doc.ComputeLeidenStmts) != 1 {
		return nil, fmt.Errorf("COMPUTE LEIDEN: expected 1 statement, got %d", len(doc.ComputeLeidenStmts))
	}

	// Lower AST → logical plan.
	plan, err := LowerComputeLeidenPlan(src, doc, 0)
	if err != nil {
		return nil, fmt.Errorf("lower COMPUTE LEIDEN plan: %w", err)
	}

	// Resolve collection: find the collection whose graph has labeled nodes
	// matching the seed label. For the standalone COMPUTE LEIDEN grammar,
	// the collection is not specified in the SQL text.
	if plan.Collection == "" {
		coll, err := db.resolveLeidenCollection(ctx, plan.SeedLabel)
		if err != nil {
			return nil, fmt.Errorf("resolve COMPUTE LEIDEN collection: %w", err)
		}
		plan.Collection = coll
	}

	// Bind against the active epoch.
	bound, err := epoch.BindLeidenMatchPlan(ctx, plan, "")
	if err != nil {
		return nil, fmt.Errorf("bind COMPUTE LEIDEN plan: %w", err)
	}

	// Execute.
	result, err := epoch.ExecuteBoundLeidenMatchPlan(ctx, bound)
	if err != nil {
		return nil, fmt.Errorf("execute COMPUTE LEIDEN: %w", err)
	}

	// Convert LeidenRelation rows to SearchResults.
	return leidenRelationToSearchResults(result.Relation), nil
}

// leidenRelationToSearchResults converts a LeidenRelation into the standard
// SearchResults shape. Each community row becomes one result with node_id and
// community_id in metadata, plus the propagated Leiden diagnostics.
func leidenRelationToSearchResults(rel *LeidenRelation) *SearchResults {
	if rel == nil {
		return &SearchResults{}
	}
	results := &SearchResults{
		Results: make([]*SearchResult, len(rel.Rows)),
		Total:   len(rel.Rows),
		// COMPUTE LEIDEN result columns in exact order:
		//   node_id, community_id, collection, record_id, truncated, scope, modularity
		Columns: []string{
			"node_id", "community_id", "collection", "record_id",
			"truncated", "scope", "modularity",
		},
	}
	for i, row := range rel.Rows {
		results.Results[i] = &SearchResult{
			ID:    fmt.Sprintf("%d", row.NodeID),
			Score: 1.0,
			Metadata: map[string]interface{}{
				"node_id":      row.NodeID,
				"community_id": row.CommunityID,
				"collection":   row.Collection,
				"record_id":    row.RecordID,
				"truncated":    rel.Truncated,
				"scope":        string(rel.Scope),
				"modularity":   rel.Modularity,
			},
		}
	}
	return results
}

// executeLeidenCTE executes a CTE SELECT: binds the CTE, executes the Leiden
// plan, builds the local_clusters virtual relation, and performs the JOIN
// against the outer FROM table.
func (db *Database) executeLeidenCTE(ctx context.Context, src []byte, doc *parser.QueryDoc, params QueryParams) (*SearchResults, error) {
	epoch := epochFromContext(ctx)
	if epoch == nil {
		return nil, fmt.Errorf("COMPUTE LEIDEN CTE requires an active epoch transaction")
	}

	// Bind the CTE.
	bound, err := epoch.BindLeidenCTE(ctx, src, doc, 0)
	if err != nil {
		return nil, fmt.Errorf("bind CTE: %w", err)
	}

	// Execute the full pipeline: Leiden → JOIN → SearchResults.
	return epoch.ExecuteLeidenCTE(ctx, src, doc, bound, 0, params)
}

// handleEpochTransactionStmts processes transaction statements inside an epoch.
func (db *Database) handleEpochTransactionStmts(ctx context.Context, epoch *EpochTx, stmts []parser.TransactionStmt) (*SearchResults, error) {
	for _, stmt := range stmts {
		switch stmt.Kind {
		case parser.TransactionBeginEpoch:
		case parser.TransactionCommit:
			if err := epoch.Commit(ctx); err != nil {
				return nil, fmt.Errorf("COMMIT: %w", err)
			}
		case parser.TransactionRollback:
			if err := epoch.Rollback(ctx); err != nil {
				return nil, fmt.Errorf("ROLLBACK: %w", err)
			}
		}
	}
	return &SearchResults{}, nil
}
