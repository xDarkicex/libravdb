package libravdb

import (
	"bytes"
	"context"
	"strconv"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/catalog"
)

const latestCommitLSNFunction = "LIBRAVDB_LATEST_COMMIT_LSN"
const latestCommitLSNColumn = "libravdb_latest_commit_lsn"

// executeLatestCommitLSNQuery handles the zero-argument SQL form of the
// existing LatestCommitLSN API. It is deliberately a virtual scalar query:
// reading the token must not require a user relation, catalog row, or WAL
// mutation.
func (db *Database) executeLatestCommitLSNQuery(ctx context.Context, src []byte, doc *parser.QueryDoc) (*SearchResults, bool, error) {
	if doc == nil || len(doc.SelectStmts) != 1 {
		return nil, false, nil
	}
	stmt := &doc.SelectStmts[0]
	if stmt.FromTable.Kind != parser.NodeKindUnknown ||
		stmt.CTEsCount != 0 || len(stmt.Joins) != 0 ||
		stmt.WhereExpr.Kind != parser.NodeKindUnknown ||
		stmt.HavingExpr.Kind != parser.NodeKindUnknown ||
		len(stmt.GroupBy) != 0 || len(stmt.OrderTerms) != 0 ||
		stmt.OrderBy.Kind != parser.NodeKindUnknown ||
		stmt.Limit != -1 || stmt.Offset != -1 ||
		stmt.LimitExpr.Kind != parser.NodeKindUnknown ||
		stmt.OffsetExpr.Kind != parser.NodeKindUnknown ||
		stmt.ProjectionsCount != 1 {
		return nil, false, nil
	}

	projection := doc.Projections[stmt.ProjectionsStart]
	if projection.Expr.Kind != parser.NodeKindFunctionExpr ||
		projection.Expr.ID < 0 || int(projection.Expr.ID) >= len(doc.FunctionExprs) {
		return nil, false, nil
	}
	fn := doc.FunctionExprs[projection.Expr.ID]
	if fn.ArgsCount != 0 || fn.HasWindow || fn.NameStart >= fn.NameEnd || fn.NameEnd > uint32(len(src)) ||
		!bytes.EqualFold(src[fn.NameStart:fn.NameEnd], []byte(latestCommitLSNFunction)) {
		return nil, false, nil
	}

	lsn, err := db.LatestCommitLSN(ctx)
	if err != nil {
		return nil, true, err
	}
	column := latestCommitLSNColumn
	if projection.Alias != 0 && projection.Alias < projection.AliasEnd && projection.AliasEnd <= uint32(len(src)) {
		column = string(src[projection.Alias:projection.AliasEnd])
	}
	return &SearchResults{
		Results: []*SearchResult{{
			ID:       strconv.FormatUint(lsn, 10),
			Score:    1,
			Metadata: map[string]interface{}{column: lsn},
		}},
		Total:       1,
		Columns:     []string{column},
		ColumnTypes: []uint16{catalog.TypeBigInt},
	}, true, nil
}
