package libravdb

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/optimizer"
)

func virtualSelectHasWindow(doc *parser.QueryDoc, stmt *parser.SelectStmt) bool {
	if doc == nil || stmt == nil {
		return false
	}
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		projectionIndex := stmt.ProjectionsStart + i
		if projectionIndex < 0 || int(projectionIndex) >= len(doc.Projections) {
			continue
		}
		projection := doc.Projections[projectionIndex]
		if projection.Expr.Kind != parser.NodeKindFunctionExpr || projection.Expr.ID < 0 || int(projection.Expr.ID) >= len(doc.FunctionExprs) {
			if projection.Expr.Kind == parser.NodeKindAggregateExpr && projection.Expr.ID >= 0 && int(projection.Expr.ID) < len(doc.AggregateExprs) && doc.AggregateExprs[projection.Expr.ID].HasWindow {
				return true
			}
			continue
		}
		if doc.FunctionExprs[projection.Expr.ID].HasWindow {
			return true
		}
	}
	return false
}

func (db *Database) projectVirtualWindowRows(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) ([]virtualSQLRow, []string, error) {
	columns := make([]string, 0, stmt.ProjectionsCount)
	windowValues := make([][]interface{}, stmt.ProjectionsCount)
	projectionNames := make([]string, stmt.ProjectionsCount)
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		projection := &doc.Projections[stmt.ProjectionsStart+i]
		if projection.Star {
			if len(rows) > 0 {
				starColumns := rows[0].visibleVirtualKeys()
				sort.Strings(starColumns)
				columns = append(columns, starColumns...)
			}
			continue
		}
		name, err := virtualWindowProjectionName(src, doc, projection)
		if err != nil {
			return nil, nil, err
		}
		columns = append(columns, name)
		projectionNames[i] = name
		switch projection.Expr.Kind {
		case parser.NodeKindFunctionExpr:
			fn := &doc.FunctionExprs[projection.Expr.ID]
			if !fn.HasWindow {
				return nil, nil, fmt.Errorf("function %q is not a window function", sourceSpan(src, fn.NameStart, fn.NameEnd))
			}
			values, err := db.evaluateWindowFunction(ctx, src, doc, fn, rows, params, legacy)
			if err != nil {
				return nil, nil, err
			}
			windowValues[i] = values
		case parser.NodeKindAggregateExpr:
			ae := &doc.AggregateExprs[projection.Expr.ID]
			if !ae.HasWindow {
				continue
			}
			values, err := db.evaluateWindowAggregate(ctx, src, doc, ae, rows, params, legacy)
			if err != nil {
				return nil, nil, err
			}
			windowValues[i] = values
		case parser.NodeKindIdentifier:
			// Ordinary projections are copied from the input row below.
		default:
			return nil, nil, fmt.Errorf("unsupported window projection expression")
		}
	}

	out := make([]virtualSQLRow, 0, len(rows))
	for rowIndex, row := range rows {
		values := make(map[string]interface{}, len(columns))
		for i := int32(0); i < stmt.ProjectionsCount; i++ {
			projection := &doc.Projections[stmt.ProjectionsStart+i]
			if projection.Star {
				row.forEachVisibleVirtualValue(func(key string, value interface{}) {
					values[key] = value
				})
				continue
			}
			name := projectionNames[i]
			var value interface{}
			var ok bool
			switch projection.Expr.Kind {
			case parser.NodeKindIdentifier:
				value, ok = virtualIdentifierValue(src, &doc.Identifiers[projection.Expr.ID], row)
			case parser.NodeKindFunctionExpr:
				value = windowValues[i][rowIndex]
				ok = value != nil
			case parser.NodeKindAggregateExpr:
				ae := &doc.AggregateExprs[projection.Expr.ID]
				if ae.HasWindow {
					value = windowValues[i][rowIndex]
					ok = value != nil
				} else {
					value, ok = row.lookup("", name)
					if !ok {
						value, ok, _ = db.virtualExprValue(ctx, src, doc, projection.Expr, row, params, legacy)
					}
				}
			default:
				return nil, nil, fmt.Errorf("window projection supports identifiers, aggregates, and window functions")
			}
			if ok {
				values[name] = value
			} else {
				values[name] = nil
			}
		}
		out = append(out, virtualSQLRow{ID: row.ID, Values: values})
	}
	return out, columns, nil
}

func virtualWindowProjectionName(src []byte, doc *parser.QueryDoc, projection *parser.Projection) (string, error) {
	if projection == nil {
		return "", fmt.Errorf("nil window projection")
	}
	name := ""
	switch projection.Expr.Kind {
	case parser.NodeKindIdentifier:
		id := &doc.Identifiers[projection.Expr.ID]
		name = sourceSpan(src, id.Start, id.End)
	case parser.NodeKindFunctionExpr:
		if projection.Expr.ID < 0 || int(projection.Expr.ID) >= len(doc.FunctionExprs) {
			return "", fmt.Errorf("invalid window function reference")
		}
		fn := &doc.FunctionExprs[projection.Expr.ID]
		name = sourceSpan(src, fn.NameStart, fn.NameEnd)
	case parser.NodeKindAggregateExpr:
		if projection.Expr.ID < 0 || int(projection.Expr.ID) >= len(doc.AggregateExprs) {
			return "", fmt.Errorf("invalid aggregate projection reference")
		}
		name = aggregateColumnName(uint8(doc.AggregateExprs[projection.Expr.ID].Func))
	default:
		return "", fmt.Errorf("window projection supports identifiers, aggregates, and window functions")
	}
	if projection.AliasEnd > projection.Alias {
		name = sourceSpan(src, projection.Alias, projection.AliasEnd)
	}
	return name, nil
}

type windowRow struct {
	index      int
	orders     []interface{}
	orderValid []bool
	partition  string
}

func (db *Database) prepareWindowRows(ctx context.Context, src []byte, doc *parser.QueryDoc, window *parser.WindowSpec, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) ([]windowRow, []string, map[string][]int, int, error) {
	orderCount := window.OrderCount
	if orderCount == 0 && window.OrderBy.Kind != parser.NodeKindUnknown {
		orderCount = 1 // compatibility with the original single-order AST view
	}
	ordered := make([]windowRow, len(rows))
	for i := range rows {
		item := windowRow{index: i}
		for j := int32(0); j < window.PartitionCount; j++ {
			if window.PartitionStart+j < 0 || int(window.PartitionStart+j) >= len(doc.Nodes) {
				return nil, nil, nil, 0, fmt.Errorf("invalid window partition expression")
			}
			ref := doc.Nodes[window.PartitionStart+j]
			value, ok, err := db.virtualExprValue(ctx, src, doc, ref, rows[i], params, legacy)
			if err != nil {
				return nil, nil, nil, 0, err
			}
			if !ok || value == nil {
				item.partition += "<NULL>\x00"
			} else {
				item.partition += aggregateValueKey(value) + "\x00"
			}
		}
		if orderCount > 0 {
			item.orders = make([]interface{}, orderCount)
			item.orderValid = make([]bool, orderCount)
			for orderIndex := int32(0); orderIndex < orderCount; orderIndex++ {
				order := windowOrderAt(doc, window, orderIndex)
				if order.Expr.Kind == parser.NodeKindUnknown {
					return nil, nil, nil, 0, fmt.Errorf("invalid window ORDER BY expression")
				}
				value, ok, err := db.virtualExprValue(ctx, src, doc, order.Expr, rows[i], params, legacy)
				if err != nil {
					return nil, nil, nil, 0, err
				}
				item.orders[orderIndex], item.orderValid[orderIndex] = value, ok && value != nil
			}
		}
		ordered[i] = item
	}

	partitions := make(map[string][]int)
	partitionOrder := make([]string, 0)
	for i := range ordered {
		key := ordered[i].partition
		if _, exists := partitions[key]; !exists {
			partitionOrder = append(partitionOrder, key)
		}
		partitions[key] = append(partitions[key], i)
	}
	for _, partitionKey := range partitionOrder {
		indices := partitions[partitionKey]
		sort.SliceStable(indices, func(i, j int) bool {
			return compareWindowOrderWithSpec(ordered[indices[i]], ordered[indices[j]], doc, window) < 0
		})
	}
	return ordered, partitionOrder, partitions, int(orderCount), nil
}

func (db *Database) evaluateWindowFunction(ctx context.Context, src []byte, doc *parser.QueryDoc, fn *parser.FunctionExpr, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) ([]interface{}, error) {
	if fn == nil || !fn.HasWindow || fn.WindowID < 0 || int(fn.WindowID) >= len(doc.WindowSpecs) {
		return nil, fmt.Errorf("invalid window function specification")
	}
	window := &doc.WindowSpecs[fn.WindowID]
	name := strings.ToLower(sourceSpan(src, fn.NameStart, fn.NameEnd))
	switch name {
	case "row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile", "lag", "lead":
	default:
		return nil, fmt.Errorf("unsupported window function %q", sourceSpan(src, fn.NameStart, fn.NameEnd))
	}
	if (name == "row_number" || name == "rank" || name == "dense_rank" || name == "percent_rank" || name == "cume_dist") && fn.ArgsCount != 0 {
		return nil, fmt.Errorf("window function %s does not accept arguments", name)
	}
	if name == "ntile" && fn.ArgsCount != 1 {
		return nil, fmt.Errorf("window function NTILE requires one argument")
	}
	if (name == "lag" || name == "lead") && (fn.ArgsCount < 1 || fn.ArgsCount > 3) {
		return nil, fmt.Errorf("window function %s requires one to three arguments", name)
	}
	if fn.ArgsStart < 0 || fn.ArgsStart+fn.ArgsCount > int32(len(doc.FunctionArgs)) {
		return nil, fmt.Errorf("invalid %s argument arena", name)
	}

	ordered, partitionOrder, partitions, orderCount, err := db.prepareWindowRows(ctx, src, doc, window, rows, params, legacy)
	if err != nil {
		return nil, err
	}
	values := make([]interface{}, len(rows))
	for _, partitionKey := range partitionOrder {
		indices := partitions[partitionKey]
		for position, orderedIndex := range indices {
			switch name {
			case "row_number":
				values[ordered[orderedIndex].index] = int64(position + 1)
			case "rank":
				rank := position + 1
				if orderCount == 0 {
					// Without ORDER BY every row in the partition is a peer.
					rank = 1
				} else if position > 0 && compareWindowOrderWithSpec(ordered[indices[position-1]], ordered[orderedIndex], doc, window) == 0 {
					rank = int(values[ordered[indices[position-1]].index].(int64))
				}
				values[ordered[orderedIndex].index] = int64(rank)
			case "dense_rank":
				rank := 1
				if position > 0 && orderCount > 0 && compareWindowOrderWithSpec(ordered[indices[position-1]], ordered[orderedIndex], doc, window) != 0 {
					rank = int(values[ordered[indices[position-1]].index].(int64)) + 1
				} else if position > 0 {
					rank = int(values[ordered[indices[position-1]].index].(int64))
				}
				values[ordered[orderedIndex].index] = int64(rank)
			case "percent_rank":
				if len(indices) <= 1 {
					values[ordered[orderedIndex].index] = float64(0)
					continue
				}
				rank := position + 1
				if orderCount > 0 && position > 0 && compareWindowOrderWithSpec(ordered[indices[position-1]], ordered[orderedIndex], doc, window) == 0 {
					firstPeer := position - 1
					for firstPeer > 0 && compareWindowOrderWithSpec(ordered[indices[firstPeer-1]], ordered[orderedIndex], doc, window) == 0 {
						firstPeer--
					}
					rank = firstPeer + 1
				}
				values[ordered[orderedIndex].index] = float64(rank-1) / float64(len(indices)-1)
			case "cume_dist":
				end := position
				if orderCount == 0 {
					end = len(indices) - 1
				} else {
					for end+1 < len(indices) && compareWindowOrderWithSpec(ordered[indices[end+1]], ordered[orderedIndex], doc, window) == 0 {
						end++
					}
				}
				values[ordered[orderedIndex].index] = float64(end+1) / float64(len(indices))
			case "ntile":
				bucketValue, ok, err := db.windowArgumentValue(ctx, src, doc, doc.FunctionArgs[fn.ArgsStart], rows[ordered[orderedIndex].index], params, legacy)
				if err != nil {
					return nil, err
				}
				bucketsFloat, okFloat := toFloat(bucketValue)
				if !ok || !okFloat || bucketsFloat < 1 || bucketsFloat != float64(int(bucketsFloat)) {
					return nil, fmt.Errorf("window function NTILE requires a positive integer bucket count")
				}
				buckets := int(bucketsFloat)
				base, remainder := len(indices)/buckets, len(indices)%buckets
				var tile int
				if base == 0 || position < remainder*(base+1) {
					tile = position/(base+1) + 1
				} else {
					tile = remainder + (position-remainder*(base+1))/base + 1
				}
				values[ordered[orderedIndex].index] = int64(tile)
			case "lag", "lead":
				offset, err := db.windowOffset(ctx, src, doc, fn, rows[ordered[orderedIndex].index], params, legacy)
				if err != nil {
					return nil, err
				}
				target := position - offset
				if name == "lead" {
					target = position + offset
				}
				if target < 0 || target >= len(indices) {
					if fn.ArgsCount == 3 {
						defaultValue, ok, err := db.windowArgumentValue(ctx, src, doc, doc.FunctionArgs[fn.ArgsStart+2], rows[ordered[orderedIndex].index], params, legacy)
						if err != nil {
							return nil, err
						}
						if ok {
							values[ordered[orderedIndex].index] = defaultValue
						}
					}
					continue
				}
				value, ok, err := db.windowArgumentValue(ctx, src, doc, doc.FunctionArgs[fn.ArgsStart], rows[ordered[indices[target]].index], params, legacy)
				if err != nil {
					return nil, err
				}
				if ok {
					values[ordered[orderedIndex].index] = value
				}
			}
		}
	}
	return values, nil
}

func (db *Database) evaluateWindowAggregate(ctx context.Context, src []byte, doc *parser.QueryDoc, ae *parser.AggregateExpr, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) ([]interface{}, error) {
	if ae == nil || !ae.HasWindow || ae.WindowID < 0 || int(ae.WindowID) >= len(doc.WindowSpecs) {
		return nil, fmt.Errorf("invalid aggregate window specification")
	}
	window := &doc.WindowSpecs[ae.WindowID]
	ordered, partitionOrder, partitions, _, err := db.prepareWindowRows(ctx, src, doc, window, rows, params, legacy)
	if err != nil {
		return nil, err
	}
	values := make([]interface{}, len(rows))
	for _, partitionKey := range partitionOrder {
		indices := partitions[partitionKey]
		for position, orderedIndex := range indices {
			start, end, err := db.windowFrameBounds(ctx, src, doc, window, ordered, indices, position, rows, params, legacy)
			if err != nil {
				return nil, err
			}
			frameRows := make([]virtualSQLRow, 0, end-start+1)
			for framePosition := start; framePosition <= end; framePosition++ {
				frameRows = append(frameRows, rows[ordered[indices[framePosition]].index])
			}
			value, err := db.evaluateVirtualAggregate(ctx, src, doc, ae, frameRows, params, legacy)
			if err != nil {
				return nil, err
			}
			values[ordered[orderedIndex].index] = value
		}
	}
	return values, nil
}

func (db *Database) windowFrameBounds(ctx context.Context, src []byte, doc *parser.QueryDoc, window *parser.WindowSpec, ordered []windowRow, indices []int, position int, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) (int, int, error) {
	last := len(indices) - 1
	if last < 0 {
		return 0, -1, nil
	}
	if !window.Frame.HasFrame {
		if len(ordered[indices[position]].orders) == 0 {
			return 0, last, nil
		}
		// SQL's default ordered aggregate frame is RANGE UNBOUNDED
		// PRECEDING through CURRENT ROW, including current-row peers.
		start := 0
		end := position
		for end+1 <= last && compareWindowOrderWithSpec(ordered[indices[end]], ordered[indices[end+1]], doc, window) == 0 {
			end++
		}
		return start, end, nil
	}
	start, err := db.windowFrameBoundIndex(ctx, src, doc, window, ordered, indices, position, window.Frame.Start, true, rows, params, legacy)
	if err != nil {
		return 0, 0, err
	}
	end, err := db.windowFrameBoundIndex(ctx, src, doc, window, ordered, indices, position, window.Frame.End, false, rows, params, legacy)
	if err != nil {
		return 0, 0, err
	}
	if start > end {
		return 0, -1, nil
	}
	return start, end, nil
}

func (db *Database) windowFrameBoundIndex(ctx context.Context, src []byte, doc *parser.QueryDoc, window *parser.WindowSpec, ordered []windowRow, indices []int, position int, bound parser.WindowFrameBound, lower bool, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) (int, error) {
	last := len(indices) - 1
	switch bound.Kind {
	case parser.WindowFrameUnboundedPreceding:
		return 0, nil
	case parser.WindowFrameUnboundedFollowing:
		return last, nil
	case parser.WindowFrameCurrentRow:
		if !window.Frame.IsRange || len(ordered[indices[position]].orders) == 0 {
			return position, nil
		}
		if lower {
			start := position
			for start > 0 && compareWindowOrderWithSpec(ordered[indices[start-1]], ordered[indices[position]], doc, window) == 0 {
				start--
			}
			return start, nil
		}
		end := position
		for end < last && compareWindowOrderWithSpec(ordered[indices[end+1]], ordered[indices[position]], doc, window) == 0 {
			end++
		}
		return end, nil
	case parser.WindowFramePreceding, parser.WindowFrameFollowing:
		if window.Frame.IsRange {
			return db.windowRangeFrameBoundIndex(ctx, src, doc, window, ordered, indices, position, bound, lower, rows, params, legacy)
		}
		value, ok, err := db.virtualExprValue(ctx, src, doc, bound.Expr, rows[ordered[indices[position]].index], params, legacy)
		if err != nil {
			return 0, err
		}
		if !ok || value == nil {
			return 0, fmt.Errorf("window frame offset is NULL")
		}
		offset, ok := toFloat(value)
		if !ok || offset < 0 || offset != float64(int(offset)) {
			return 0, fmt.Errorf("window frame offset must be a non-negative integer")
		}
		n := int(offset)
		if bound.Kind == parser.WindowFramePreceding {
			if lower {
				if n > position {
					return 0, nil
				}
				return position - n, nil
			}
			return position, nil
		}
		if lower {
			return position, nil
		}
		if n > last-position {
			return last, nil
		}
		return position + n, nil
	default:
		return 0, fmt.Errorf("invalid window frame bound")
	}
}

// windowRangeFrameBoundIndex resolves a numeric RANGE offset against the
// single ORDER BY key. RANGE offsets are defined in the ordering coordinate:
// descending order reverses that coordinate while preserving PRECEDING and
// FOLLOWING semantics. Numeric offsets intentionally require one numeric key;
// SQL's multi-key RANGE offset rules are not portable and are rejected rather
// than approximated.
func (db *Database) windowRangeFrameBoundIndex(ctx context.Context, src []byte, doc *parser.QueryDoc, window *parser.WindowSpec, ordered []windowRow, indices []int, position int, bound parser.WindowFrameBound, lower bool, rows []virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) (int, error) {
	orderCount := window.OrderCount
	if orderCount == 0 && window.OrderBy.Kind != parser.NodeKindUnknown {
		orderCount = 1
	}
	if orderCount != 1 {
		return 0, fmt.Errorf("RANGE offset frames require exactly one numeric ORDER BY expression")
	}
	order := windowOrderAt(doc, window, 0)
	current := ordered[indices[position]]
	if len(current.orders) == 0 || !current.orderValid[0] {
		return 0, fmt.Errorf("RANGE offset frame cannot use a NULL ORDER BY value")
	}
	currentValue, ok := toFloat(current.orders[0])
	if !ok {
		return 0, fmt.Errorf("RANGE offset frames require a numeric ORDER BY expression")
	}
	offsetValue, offsetOK, err := db.virtualExprValue(ctx, src, doc, bound.Expr, rows[current.index], params, legacy)
	if err != nil {
		return 0, err
	}
	if !offsetOK || offsetValue == nil {
		return 0, fmt.Errorf("window frame offset is NULL")
	}
	offset, ok := toFloat(offsetValue)
	if !ok || offset < 0 {
		return 0, fmt.Errorf("RANGE frame offset must be a non-negative number")
	}
	coordinate := currentValue
	if order.IsDesc {
		coordinate = -coordinate
	}
	if bound.Kind == parser.WindowFramePreceding {
		coordinate -= offset
	} else {
		coordinate += offset
	}
	last := len(indices) - 1
	if lower {
		for i, index := range indices {
			item := ordered[index]
			if len(item.orders) == 0 || !item.orderValid[0] {
				continue
			}
			itemValue, itemOK := toFloat(item.orders[0])
			if !itemOK {
				return 0, fmt.Errorf("RANGE offset frames require numeric ORDER BY values")
			}
			if order.IsDesc {
				itemValue = -itemValue
			}
			if itemValue >= coordinate {
				return i, nil
			}
		}
		return last + 1, nil
	}
	result := -1
	for i, index := range indices {
		item := ordered[index]
		if len(item.orders) == 0 || !item.orderValid[0] {
			continue
		}
		itemValue, itemOK := toFloat(item.orders[0])
		if !itemOK {
			return 0, fmt.Errorf("RANGE offset frames require numeric ORDER BY values")
		}
		if order.IsDesc {
			itemValue = -itemValue
		}
		if itemValue <= coordinate {
			result = i
			continue
		}
		break
	}
	return result, nil
}

func windowOrderAt(doc *parser.QueryDoc, window *parser.WindowSpec, index int32) parser.WindowOrder {
	if window == nil || index < 0 {
		return parser.WindowOrder{}
	}
	if window.OrderCount > 0 {
		position := window.OrderStart + index
		if position >= 0 && int(position) < len(doc.WindowOrders) {
			return doc.WindowOrders[position]
		}
		return parser.WindowOrder{}
	}
	if index == 0 {
		return parser.WindowOrder{Expr: window.OrderBy, IsDesc: window.IsDesc}
	}
	return parser.WindowOrder{}
}

func compareWindowOrderWithSpec(left, right windowRow, doc *parser.QueryDoc, window *parser.WindowSpec) int {
	count := len(left.orders)
	if len(right.orders) < count {
		count = len(right.orders)
	}
	for i := 0; i < count; i++ {
		leftValid, rightValid := left.orderValid[i], right.orderValid[i]
		if !leftValid || !rightValid {
			if leftValid == rightValid {
				continue
			}
			order := windowOrderAt(doc, window, int32(i))
			nullsFirst := order.NullsOrder == parser.WindowNullsFirst ||
				(order.NullsOrder == parser.WindowNullsDefault && order.IsDesc)
			leftIsNull := !leftValid
			if leftIsNull == nullsFirst {
				return -1
			}
			return 1
		}
		cmp := compareVirtualValues(left.orders[i], right.orders[i])
		if cmp == 0 {
			continue
		}
		if windowOrderAt(doc, window, int32(i)).IsDesc {
			return -cmp
		}
		return cmp
	}
	return 0
}

func (db *Database) windowOffset(ctx context.Context, src []byte, doc *parser.QueryDoc, fn *parser.FunctionExpr, row virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) (int, error) {
	if fn.ArgsCount < 2 {
		return 1, nil
	}
	if fn.ArgsStart < 0 || fn.ArgsStart+1 >= int32(len(doc.FunctionArgs)) {
		return 0, fmt.Errorf("invalid window offset expression")
	}
	value, ok, err := db.virtualExprValue(ctx, src, doc, doc.FunctionArgs[fn.ArgsStart+1], row, params, legacy)
	if err != nil {
		return 0, err
	}
	if !ok || value == nil {
		return 1, nil
	}
	n, ok := toFloat(value)
	if !ok || n < 0 || n != float64(int(n)) {
		return 0, fmt.Errorf("window offset must be a non-negative integer")
	}
	return int(n), nil
}

func (db *Database) windowArgumentValue(ctx context.Context, src []byte, doc *parser.QueryDoc, ref parser.NodeRef, row virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) (interface{}, bool, error) {
	value, ok, err := db.virtualExprValue(ctx, src, doc, ref, row, params, legacy)
	if err != nil || !ok || ref.Kind != parser.NodeKindNumber {
		return value, ok, err
	}
	if ref.ID < 0 || int(ref.ID) >= len(doc.Numbers) {
		return value, ok, nil
	}
	literal := sourceSpan(src, doc.Numbers[ref.ID].Start, doc.Numbers[ref.ID].End)
	if integer, parseErr := strconv.ParseInt(literal, 10, 64); parseErr == nil {
		return integer, true, nil
	}
	if decimal, parseErr := strconv.ParseFloat(literal, 64); parseErr == nil {
		return decimal, true, nil
	}
	return value, ok, nil
}
