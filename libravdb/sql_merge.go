package libravdb

import (
	"context"
	"fmt"
	"strings"

	apexjson "github.com/xDarkicex/apexJSON/v2"
	"github.com/xDarkicex/lexer"
	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// mergeVertexState is the query-local identity and mutation image for one
// vertex in a MERGE pattern. The record transaction receives the final image;
// graph endpoints are resolved through EpochTx so new vertices and edges share
// one atomic record/graph commit.
type mergeVertexState struct {
	alias    string
	label    string
	id       string
	metadata map[string]interface{}
	created  bool
	changed  bool
	delta    map[string]interface{}
}

func (db *Database) executeSQLMerge(ctx context.Context, src []byte, doc *parser.QueryDoc, params *optimizer.ParameterSet, legacy QueryParams) (*SearchResults, error) {
	if len(doc.MergeStmts) != 1 {
		return nil, fmt.Errorf("MERGE currently requires exactly one graph pattern")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if epoch := epochFromContext(ctx); epoch != nil {
		return db.executeMergeInEpoch(ctx, src, doc, &doc.MergeStmts[0], epoch, params, legacy)
	}
	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		return nil, fmt.Errorf("MERGE begin transaction: %w", err)
	}
	mergeCtx := epoch.Context(ctx)
	result, err := db.executeMergeInEpoch(mergeCtx, src, doc, &doc.MergeStmts[0], epoch, params, legacy)
	if err != nil {
		_ = epoch.Rollback(ctx)
		return nil, err
	}
	if err := epoch.Commit(ctx); err != nil {
		return nil, fmt.Errorf("MERGE commit: %w", err)
	}
	return result, nil
}

func (db *Database) executeMergeInEpoch(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.MergeStmt, epoch *EpochTx, params *optimizer.ParameterSet, legacy QueryParams) (*SearchResults, error) {
	if stmt == nil || stmt.MatchPath.Kind != parser.NodeKindMatchPath || stmt.MatchPath.ID < 0 || int(stmt.MatchPath.ID) >= len(doc.MatchPaths) {
		return nil, fmt.Errorf("MERGE requires a valid graph pattern")
	}
	path := &doc.MatchPaths[stmt.MatchPath.ID]
	collection, err := db.mergeGraphCollection(doc, src, path)
	if collection == nil {
		return nil, fmt.Errorf("MERGE requires a graph-backed collection")
	}
	vertices, edges := mergePathElements(doc, path)
	if len(vertices) == 0 || len(edges) != len(vertices)-1 {
		return nil, fmt.Errorf("MERGE requires a connected graph pattern")
	}

	records, err := epoch.ListRecords(ctx, collection.name)
	if err != nil {
		return nil, fmt.Errorf("MERGE read %s: %w", collection.name, err)
	}
	states := make([]*mergeVertexState, len(vertices))
	byID := make(map[string]*mergeVertexState, len(vertices))
	byAlias := make(map[string]*mergeVertexState, len(vertices))
	for i, ref := range vertices {
		vertex := &doc.Vertexes[ref.ID]
		alias := sourceSpan(src, vertex.Alias, vertex.AliasEnd)
		if alias == "" {
			return nil, fmt.Errorf("MERGE vertices must have aliases")
		}
		properties, err := db.mergePropertyMap(ctx, src, doc, vertex.Predicate, params, legacy)
		if err != nil {
			return nil, fmt.Errorf("MERGE vertex %s: %w", alias, err)
		}
		identity, ok := mergeIdentity(properties)
		if !ok {
			return nil, fmt.Errorf("MERGE vertex %s requires an id or uuid property", alias)
		}
		if existing := byID[identity]; existing != nil {
			states[i] = existing
			byAlias[strings.ToLower(alias)] = existing
			continue
		}
		var found *Record
		for j := range records {
			if mergeRecordMatches(&records[j], properties) {
				found = &records[j]
				break
			}
		}
		state := &mergeVertexState{
			alias:    alias,
			label:    sourceSpan(src, vertex.LabelStart, vertex.LabelEnd),
			id:       identity,
			metadata: make(map[string]interface{}, len(properties)),
			delta:    make(map[string]interface{}),
		}
		if found != nil {
			state.id = found.ID
			state.metadata = cloneMetadata(found.Metadata)
			if state.metadata == nil {
				state.metadata = make(map[string]interface{})
			}
			for key, value := range properties {
				if key != "id" && !sqlValueEqual(state.metadata[key], value) {
					return nil, fmt.Errorf("MERGE identity mismatch for existing record %q", found.ID)
				}
			}
		} else {
			state.created = true
			for key, value := range properties {
				if key != "id" {
					state.metadata[key] = value
				}
			}
		}
		states[i] = state
		byID[identity] = state
		byAlias[strings.ToLower(alias)] = state
	}

	assignments := stmt.OnMatchStart
	count := stmt.OnMatchCount
	for _, state := range states {
		if state.created {
			assignments = stmt.OnCreateStart
			count = stmt.OnCreateCount
		}
		for i := int32(0); i < count; i++ {
			assignment := doc.MergeAssignments[assignments+i]
			if assignment.Column.Kind != parser.NodeKindIdentifier || assignment.Column.ID < 0 || int(assignment.Column.ID) >= len(doc.Identifiers) {
				return nil, fmt.Errorf("MERGE assignment target is not an identifier")
			}
			column := &doc.Identifiers[assignment.Column.ID]
			field := sourceSpan(src, column.Start, column.End)
			alias := sourceSpan(src, column.QualStart, column.QualEnd)
			target := state
			if alias != "" {
				target = byAlias[strings.ToLower(alias)]
			}
			if target == nil {
				return nil, fmt.Errorf("MERGE assignment references unknown vertex alias %q", alias)
			}
			if strings.EqualFold(field, "id") {
				return nil, fmt.Errorf("MERGE cannot update the graph record id")
			}
			row := mergeStateRow(target)
			value, ok, err := db.virtualExprValue(ctx, src, doc, assignment.Value, row, params, legacy)
			if err != nil {
				return nil, fmt.Errorf("MERGE assignment %s.%s: %w", alias, field, err)
			}
			if !ok {
				return nil, fmt.Errorf("MERGE assignment %s.%s could not be evaluated", alias, field)
			}
			value = materializeSQLJSONValue(value)
			target.metadata[field] = value
			target.delta[field] = value
			target.changed = true
		}
	}

	for _, state := range uniqueMergeStates(states) {
		if state.created {
			vector := mergeZeroVector(collection.Dimension())
			if err := epoch.Insert(ctx, collection.name, state.id, vector, state.metadata); err != nil {
				return nil, fmt.Errorf("MERGE create vertex %s: %w", state.id, err)
			}
		} else if state.changed {
			if err := epoch.Update(ctx, collection.name, state.id, nil, state.delta); err != nil {
				return nil, fmt.Errorf("MERGE update vertex %s: %w", state.id, err)
			}
		}
		if state.label != "" {
			if err := epoch.RegisterVertexLabel(collection.name, state.id, state.label); err != nil {
				return nil, err
			}
		}
	}

	for i, edgeRef := range edges {
		edge := &doc.Edges[edgeRef.ID]
		kind := uint8(0)
		if edge.TypeStart < edge.TypeEnd {
			kind = ResolveEdgeKind(sourceSpan(src, edge.TypeStart, edge.TypeEnd))
		}
		if kind == 0 {
			return nil, fmt.Errorf("MERGE edge requires a registered edge type")
		}
		sourceState, targetState := states[i], states[i+1]
		if edge.Direction < 0 {
			sourceState, targetState = targetState, sourceState
		}
		from, err := epoch.LookupNodeID(ctx, collection.name, sourceState.id)
		if err != nil {
			return nil, err
		}
		to, err := epoch.LookupNodeID(ctx, collection.name, targetState.id)
		if err != nil {
			return nil, err
		}
		properties, weight, err := db.mergeEdgeProperties(ctx, src, doc, edge.Predicate, params, legacy)
		if err != nil {
			return nil, fmt.Errorf("MERGE edge: %w", err)
		}
		encoded := []byte(nil)
		if len(properties) > 0 {
			encoded, err = apexjson.Marshal(properties)
			if err != nil {
				return nil, fmt.Errorf("MERGE edge properties: %w", err)
			}
		}
		gtx, err := epoch.GraphTxn(collection.name)
		if err != nil {
			return nil, fmt.Errorf("MERGE graph transaction: %w", err)
		}
		existing, err := gtx.NeighborsOverlay(from)
		if err != nil {
			return nil, fmt.Errorf("MERGE inspect edge: %w", err)
		}
		alreadyExists := false
		for _, candidate := range existing {
			if candidate.Target == to && candidate.GetKind() == kind {
				alreadyExists = true
				break
			}
		}
		if alreadyExists {
			continue
		}
		if err := epoch.AddGraphEdgeWithPropertiesJSON(collection.name, from, to, weight, kind, encoded); err != nil {
			return nil, fmt.Errorf("MERGE edge %s: %w", sourceSpan(src, edge.TypeStart, edge.TypeEnd), err)
		}
	}

	return mergeResults(states, stmt, doc, src), nil
}

// mergeGraphCollection resolves the graph relation for a MERGE pattern. The
// Cypher surface does not carry a SQL table name, so selecting an arbitrary
// graph collection would silently mutate the wrong graph as soon as a database
// owns more than one graph-backed relation. Prefer the unique graph collection
// whose declared metadata schema contains every property named by the vertex
// maps; retain the single-graph fallback for existing Graphiti deployments.
func (db *Database) mergeGraphCollection(doc *parser.QueryDoc, src []byte, path *parser.MatchPath) (*Collection, error) {
	keys := mergePathPropertyKeys(doc, src, path)
	names := db.graphCollectionNames("")
	candidates := make([]*Collection, 0, len(names))
	for _, name := range names {
		collection, err := db.GetCollection(name)
		if err != nil || collection.GetGraph() == nil {
			continue
		}
		if len(keys) == 0 || mergeSchemaContainsKeys(collection.Config().MetadataSchema, keys) {
			candidates = append(candidates, collection)
		}
	}
	if len(candidates) == 1 {
		return candidates[0], nil
	}
	if len(candidates) == 0 && len(names) == 1 {
		return db.GetCollection(names[0])
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf("MERGE graph pattern is ambiguous: no graph collection declares its vertex properties")
	}
	return nil, fmt.Errorf("MERGE graph pattern is ambiguous across %d graph collections", len(candidates))
}

func mergePathPropertyKeys(doc *parser.QueryDoc, src []byte, path *parser.MatchPath) map[string]struct{} {
	keys := make(map[string]struct{})
	if path == nil {
		return keys
	}
	for i := int32(0); i < path.PathNodesCount; i++ {
		ref := doc.Nodes[path.PathNodesStart+i]
		if ref.Kind != parser.NodeKindVertex || ref.ID < 0 || int(ref.ID) >= len(doc.Vertexes) {
			continue
		}
		var walk func(parser.NodeRef)
		walk = func(node parser.NodeRef) {
			if node.Kind != parser.NodeKindBinaryExpr || node.ID < 0 || int(node.ID) >= len(doc.BinaryExprs) {
				return
			}
			be := &doc.BinaryExprs[node.ID]
			if be.Operator == uint8(lexer.KindAnd) {
				walk(be.Left)
				walk(be.Right)
				return
			}
			if be.Left.Kind == parser.NodeKindIdentifier && be.Left.ID >= 0 && int(be.Left.ID) < len(doc.Identifiers) {
				identifier := &doc.Identifiers[be.Left.ID]
				keys[strings.ToLower(sourceSpan(src, identifier.Start, identifier.End))] = struct{}{}
			}
		}
		walk(doc.Vertexes[ref.ID].Predicate)
	}
	return keys
}

func mergeSchemaContainsKeys(schema MetadataSchema, keys map[string]struct{}) bool {
	for key := range keys {
		if key == "id" {
			continue
		}
		found := false
		for declared := range schema {
			if strings.EqualFold(declared, key) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func mergePathElements(doc *parser.QueryDoc, path *parser.MatchPath) ([]parser.NodeRef, []parser.NodeRef) {
	vertices := make([]parser.NodeRef, 0, path.PathNodesCount/2+1)
	edges := make([]parser.NodeRef, 0, path.PathNodesCount/2)
	for i := int32(0); i < path.PathNodesCount; i++ {
		ref := doc.Nodes[path.PathNodesStart+i]
		switch ref.Kind {
		case parser.NodeKindVertex:
			vertices = append(vertices, ref)
		case parser.NodeKindEdge:
			edges = append(edges, ref)
		}
	}
	return vertices, edges
}

func (db *Database) mergePropertyMap(ctx context.Context, src []byte, doc *parser.QueryDoc, ref parser.NodeRef, params *optimizer.ParameterSet, legacy QueryParams) (map[string]interface{}, error) {
	properties := make(map[string]interface{})
	var walk func(parser.NodeRef) error
	walk = func(node parser.NodeRef) error {
		if node.Kind != parser.NodeKindBinaryExpr || node.ID < 0 || int(node.ID) >= len(doc.BinaryExprs) {
			return fmt.Errorf("property map must contain comparisons")
		}
		be := &doc.BinaryExprs[node.ID]
		if be.Operator == uint8(lexer.KindAnd) {
			if err := walk(be.Left); err != nil {
				return err
			}
			return walk(be.Right)
		}
		if be.Operator != uint8(lexer.KindEquals) || be.Left.Kind != parser.NodeKindIdentifier {
			return fmt.Errorf("identity/property maps support equality comparisons only")
		}
		id := &doc.Identifiers[be.Left.ID]
		field := sourceSpan(src, id.Start, id.End)
		value, ok, err := db.virtualExprValue(ctx, src, doc, be.Right, virtualSQLRow{}, params, legacy)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("property %s could not be evaluated", field)
		}
		properties[field] = materializeSQLJSONValue(value)
		return nil
	}
	if ref.Kind == parser.NodeKindUnknown {
		return properties, nil
	}
	if err := walk(ref); err != nil {
		return nil, err
	}
	return properties, nil
}

func (db *Database) mergeEdgeProperties(ctx context.Context, src []byte, doc *parser.QueryDoc, ref parser.NodeRef, params *optimizer.ParameterSet, legacy QueryParams) (map[string]interface{}, float32, error) {
	properties, err := db.mergePropertyMap(ctx, src, doc, ref, params, legacy)
	if err != nil {
		return nil, 1, err
	}
	weight := float32(1)
	if value, ok := properties["weight"]; ok {
		switch number := value.(type) {
		case float64:
			weight = float32(number)
		case float32:
			weight = number
		case int:
			weight = float32(number)
		case int64:
			weight = float32(number)
		default:
			return nil, 0, fmt.Errorf("edge weight must be numeric")
		}
	}
	return properties, weight, nil
}

func mergeIdentity(properties map[string]interface{}) (string, bool) {
	for _, key := range []string{"id", "uuid"} {
		if value, ok := properties[key]; ok && value != nil {
			id := recordMetaToString(value)
			if id != "" && id != "null" {
				return id, true
			}
		}
	}
	return "", false
}

func mergeRecordMatches(record *Record, properties map[string]interface{}) bool {
	if record == nil {
		return false
	}
	for key, expected := range properties {
		actual := interface{}(nil)
		if key == "id" {
			actual = record.ID
		} else if record.Metadata != nil {
			actual = record.Metadata[key]
		}
		if !sqlValueEqual(actual, expected) {
			return false
		}
	}
	return true
}

func mergeStateRow(state *mergeVertexState) virtualSQLRow {
	values := cloneMetadata(state.metadata)
	if values == nil {
		values = make(map[string]interface{})
	}
	values["id"] = state.id
	return virtualSQLRow{ID: state.id, Values: values, Scopes: []virtualSQLScope{{Alias: state.alias, Values: values}}}
}

func uniqueMergeStates(states []*mergeVertexState) []*mergeVertexState {
	seen := make(map[*mergeVertexState]struct{}, len(states))
	result := make([]*mergeVertexState, 0, len(states))
	for _, state := range states {
		if state == nil {
			continue
		}
		if _, ok := seen[state]; ok {
			continue
		}
		seen[state] = struct{}{}
		result = append(result, state)
	}
	return result
}

func mergeZeroVector(dimension int) []float32 {
	if dimension <= 0 {
		return nil
	}
	return make([]float32, dimension)
}

func mergeResults(states []*mergeVertexState, stmt *parser.MergeStmt, doc *parser.QueryDoc, src []byte) *SearchResults {
	unique := uniqueMergeStates(states)
	result := &SearchResults{Results: make([]*SearchResult, 0, len(unique))}
	if stmt.ReturningStar || len(stmt.Returning) == 0 {
		result.Columns = []string{"id"}
	}
	if len(stmt.Returning) > 0 && !stmt.ReturningStar {
		result.Columns = make([]string, 0, len(stmt.Returning))
	}
	for _, state := range unique {
		metadata := make(map[string]interface{})
		if stmt.ReturningStar || len(stmt.Returning) == 0 {
			metadata["id"] = state.id
		} else {
			for _, ref := range stmt.Returning {
				if ref.Kind != parser.NodeKindIdentifier || ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
					continue
				}
				identifier := &doc.Identifiers[ref.ID]
				field := sourceSpan(src, identifier.Start, identifier.End)
				if field == "" {
					continue
				}
				result.Columns = append(result.Columns, field)
				if strings.EqualFold(field, "id") {
					metadata[field] = state.id
				} else {
					metadata[field] = state.metadata[field]
				}
			}
		}
		result.Results = append(result.Results, &SearchResult{ID: state.id, Score: 1, Metadata: metadata})
	}
	result.Total = len(result.Results)
	return result
}
