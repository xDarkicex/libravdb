package libravdb

import (
	"context"
	"fmt"
	"strings"

	apexjson "github.com/xDarkicex/apexJSON/v2"
	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/graph"
	"github.com/xDarkicex/libravdb/internal/optimizer"
)

type cypherEdgeBinding struct {
	from       uint64
	target     uint64
	kind       uint8
	weight     float32
	properties map[string]interface{}
}

type cypherMatchBinding struct {
	base     virtualSQLRow
	vertices map[string]Record
	nodes    map[string]uint64
	edges    map[string]cypherEdgeBinding
}

func (db *Database) executeSQLCypherDelete(ctx context.Context, src []byte, doc *parser.QueryDoc, params *optimizer.ParameterSet, legacy QueryParams) (*SearchResults, error) {
	if len(doc.DeleteStmts) != 1 || !doc.DeleteStmts[0].Cypher {
		return nil, fmt.Errorf("invalid Cypher DELETE")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if epoch := epochFromContext(ctx); epoch != nil {
		return db.executeCypherDeleteInEpoch(epoch.Context(ctx), src, doc, &doc.DeleteStmts[0], epoch, params, legacy)
	}
	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		return nil, fmt.Errorf("DELETE begin transaction: %w", err)
	}
	result, err := db.executeCypherDeleteInEpoch(epoch.Context(ctx), src, doc, &doc.DeleteStmts[0], epoch, params, legacy)
	if err != nil {
		_ = epoch.Rollback(ctx)
		return nil, err
	}
	if err := epoch.Commit(ctx); err != nil {
		return nil, fmt.Errorf("DELETE commit: %w", err)
	}
	return result, nil
}

func (db *Database) executeCypherDeleteInEpoch(ctx context.Context, src []byte, doc *parser.QueryDoc, stmt *parser.DeleteStmt, epoch *EpochTx, params *optimizer.ParameterSet, legacy QueryParams) (*SearchResults, error) {
	if stmt == nil || stmt.MatchPath.Kind != parser.NodeKindMatchPath || stmt.MatchPath.ID < 0 || int(stmt.MatchPath.ID) >= len(doc.MatchPaths) {
		return nil, fmt.Errorf("DELETE requires a valid graph pattern")
	}
	path := &doc.MatchPaths[stmt.MatchPath.ID]
	collection, err := db.cypherMatchGraphCollection(doc, src, path)
	if err != nil {
		return nil, err
	}
	if collection == nil || collection.GetGraph() == nil {
		return nil, fmt.Errorf("DELETE requires a graph-backed collection")
	}
	bindings, err := db.collectCypherMatchBindings(ctx, src, doc, path, collection, stmt.WhereExpr, epoch, params, legacy)
	if err != nil {
		return nil, err
	}
	requested := make(map[string]struct{}, len(stmt.Targets))
	for _, target := range stmt.Targets {
		if target.Kind != parser.NodeKindIdentifier || target.ID < 0 || int(target.ID) >= len(doc.Identifiers) {
			return nil, fmt.Errorf("DELETE target is not a graph alias")
		}
		id := &doc.Identifiers[target.ID]
		requested[strings.ToLower(sourceSpan(src, id.Start, id.End))] = struct{}{}
	}

	type vertexDelete struct {
		id   string
		node uint64
	}
	vertices := make(map[string]vertexDelete)
	type edgeDelete struct {
		from, target uint64
		kind         uint8
	}
	edges := make(map[string]edgeDelete)
	for _, binding := range bindings {
		for alias := range requested {
			if edge, ok := binding.edges[alias]; ok {
				key := fmt.Sprintf("%d/%d/%d", edge.from, edge.target, edge.kind)
				edges[key] = edgeDelete{from: edge.from, target: edge.target, kind: edge.kind}
				continue
			}
			if record, ok := binding.vertices[alias]; ok {
				node := binding.nodes[alias]
				vertices[collection.name+"\x00"+record.ID] = vertexDelete{id: record.ID, node: node}
			}
		}
	}

	// MATCH is evaluated against the collection's current graph view. This is
	// important for graph transactions created through the public Graph API:
	// those commits are durable/live graph state even when they were not part
	// of the record epoch that opened this write transaction.
	if !stmt.Detach {
		for _, vertex := range vertices {
			outbound, outErr := collection.GetGraph().Neighbors(vertex.node)
			if outErr != nil {
				return nil, outErr
			}
			inbound, inErr := collection.GetGraph().InboundNeighbors(vertex.node)
			if inErr != nil {
				return nil, inErr
			}
			if len(outbound) > 0 || len(inbound) > 0 {
				return nil, fmt.Errorf("cannot delete graph vertex %q with incident edges; use DETACH DELETE", vertex.id)
			}
		}
	}
	for _, edge := range edges {
		if err := epoch.RemoveGraphEdge(collection.name, edge.from, edge.target, edge.kind); err != nil {
			return nil, err
		}
	}
	for _, vertex := range vertices {
		if stmt.Detach {
			if err := epoch.DropGraphNodeEdges(collection.name, vertex.node); err != nil {
				return nil, err
			}
		}
		if err := epoch.Delete(ctx, collection.name, vertex.id); err != nil {
			return nil, err
		}
	}
	return &SearchResults{Results: []*SearchResult{}, Total: 0}, nil
}

// cypherMatchGraphCollection applies the same deterministic implicit graph
// relation selection used by top-level MATCH. Native Cypher has no FROM table,
// so a database with several graph collections must narrow by labels, edge
// kinds, and property-map schema before falling back to stable collection
// order. MERGE keeps its stricter ambiguity checks because writes must never
// silently choose a destination relation.
func (db *Database) cypherMatchGraphCollection(doc *parser.QueryDoc, src []byte, path *parser.MatchPath) (*Collection, error) {
	names := db.graphCollectionNames("")
	if len(names) == 0 {
		return nil, fmt.Errorf("no graph-enabled collection found for implicit MATCH source")
	}
	vertices, edges := mergePathElements(doc, path)
	keys := mergePathPropertyKeys(doc, src, path)
	candidates := make([]*Collection, 0, len(names))
	for _, name := range names {
		collection, getErr := db.GetCollection(name)
		if getErr != nil || collection.GetGraph() == nil {
			continue
		}
		if len(keys) > 0 && !mergeSchemaContainsKeys(collection.Config().MetadataSchema, keys) {
			continue
		}
		g := collection.GetGraph()
		matched := true
		for _, ref := range vertices {
			if ref.ID < 0 || int(ref.ID) >= len(doc.Vertexes) {
				continue
			}
			vertex := &doc.Vertexes[ref.ID]
			labels := make([]string, 0, 1+vertex.LabelsCount)
			if vertex.LabelStart < vertex.LabelEnd {
				labels = append(labels, sourceSpan(src, vertex.LabelStart, vertex.LabelEnd))
			}
			for i := int32(0); i < vertex.LabelsCount; i++ {
				index := vertex.LabelsStart + i
				if index >= 0 && int(index) < len(doc.VertexLabels) {
					label := doc.VertexLabels[index]
					labels = append(labels, sourceSpan(src, label.Start, label.End))
				}
			}
			if len(labels) > 0 && graphHasVertexLabels(g) && !graphCollectionMatchesLabelsForDatabase(db, collection, labels) {
				matched = false
				break
			}
		}
		if !matched {
			continue
		}
		for _, ref := range edges {
			if ref.ID < 0 || int(ref.ID) >= len(doc.Edges) {
				continue
			}
			edge := &doc.Edges[ref.ID]
			if edge.TypeStart >= edge.TypeEnd {
				continue
			}
			kind := ResolveEdgeKind(sourceSpan(src, edge.TypeStart, edge.TypeEnd))
			if kind == 0 {
				continue
			}
			if !graphCollectionHasEdgeOriginForDatabase(db, collection, kind) {
				matched = false
				break
			}
		}
		if matched {
			candidates = append(candidates, collection)
		}
	}
	if len(candidates) > 0 {
		// A graph without explicit vertex labels can still satisfy a label as
		// a schema-side hint. Prefer a candidate with durable labels when one
		// exists; otherwise an earlier unlabeled relation (for example a
		// bootstrap documents table) can shadow the actual labeled graph table.
		for _, candidate := range candidates {
			if graphHasVertexLabels(candidate.GetGraph()) {
				return candidate, nil
			}
		}
		return candidates[0], nil
	}
	// Preserve the established implicit MATCH behavior for an empty graph or
	// a pattern whose labels/edge kinds have not been registered yet.
	return db.GetCollection(names[0])
}

// collectCypherMatchBindings materializes the alias-to-record/node/edge
// bindings needed by write clauses. It intentionally shares the same path
// lowering and traversal rules as the graph SELECT executor.
func (db *Database) collectCypherMatchBindings(ctx context.Context, src []byte, doc *parser.QueryDoc, path *parser.MatchPath, collection *Collection, where parser.NodeRef, epoch *EpochTx, params *optimizer.ParameterSet, legacy QueryParams) ([]cypherMatchBinding, error) {
	return db.collectCypherMatchBindingsFromRows(ctx, src, doc, path, collection, where, epoch, params, legacy, nil)
}

// collectCypherMatchBindingsFromRows evaluates a MATCH against the rows
// emitted by a preceding WITH boundary. If the first vertex alias is still
// bound to a node in the input scope, the scan is restricted to that node;
// otherwise the path's vertex/property predicates are evaluated against the
// full graph relation while retaining the input scope for downstream clauses.
func (db *Database) collectCypherMatchBindingsFromRows(ctx context.Context, src []byte, doc *parser.QueryDoc, path *parser.MatchPath, collection *Collection, where parser.NodeRef, epoch *EpochTx, params *optimizer.ParameterSet, legacy QueryParams, inputRows []virtualSQLRow) ([]cypherMatchBinding, error) {
	vertices, edgeRefs := mergePathElements(doc, path)
	if len(vertices) == 0 {
		return nil, fmt.Errorf("MATCH pattern has no vertices")
	}
	records, err := epoch.ListRecords(ctx, collection.name)
	if err != nil {
		return nil, err
	}
	db.mu.RLock()
	cat := db.catalog
	db.mu.RUnlock()
	optimizerEdges, maxHops, err := optimizer.NewOptimizer(cat).ExtractMatchPath(doc, src, path, params)
	if err != nil {
		return nil, err
	}
	edges := make([]EdgePlan, len(optimizerEdges))
	for i, edge := range optimizerEdges {
		edges[i] = graphEdgePlanForTraversal(edge)
	}
	recordsByNode := make(map[uint64]Record, len(records))
	for i := range records {
		nodeID, lookupErr := epoch.LookupNodeID(ctx, collection.name, records[i].ID)
		if lookupErr == nil {
			recordsByNode[nodeID] = records[i]
		}
	}
	result := make([]cypherMatchBinding, 0)
	inputs := inputRows
	if len(inputs) == 0 {
		inputs = []virtualSQLRow{{}}
	}
	for _, input := range inputs {
		candidates := records
		firstAlias := strings.ToLower(sourceSpan(src, doc.Vertexes[vertices[0].ID].Alias, doc.Vertexes[vertices[0].ID].AliasEnd))
		if inputID, ok := cypherInputRecordID(input, firstAlias); ok {
			candidates = nil
			for i := range records {
				if records[i].ID == inputID {
					candidates = records[i : i+1]
					break
				}
			}
		}
		for i := range candidates {
			source := candidates[i]
			first := &doc.Vertexes[vertices[0].ID]
			sourceNode, lookupErr := epoch.LookupNodeID(ctx, collection.name, source.ID)
			if lookupErr != nil || !db.cypherVertexMatchesInRow(ctx, src, doc, first, collection.GetGraph(), source, sourceNode, input, params, legacy) {
				continue
			}
			if len(edges) == 0 {
				binding := cypherMatchBinding{base: input, vertices: map[string]Record{}, nodes: map[string]uint64{}, edges: map[string]cypherEdgeBinding{}}
				db.addCypherVertexBinding(src, doc, &binding, &doc.Vertexes[vertices[0].ID], source, sourceNode)
				if db.cypherWhereMatches(ctx, src, doc, binding, where, params, legacy) {
					result = append(result, binding)
				}
				continue
			}
			states, traverseErr := collectGraphJoinPaths(sourceNode, edges, maxHops, func(nodeID uint64, direction int8) ([]graph.EdgeView, error) {
				if direction > 0 {
					return collection.GetGraph().NeighborsWithProperties(nodeID)
				}
				if direction < 0 {
					return collection.GetGraph().InboundNeighborsWithProperties(nodeID)
				}
				return graphPatternNeighbors(collection.GetGraph(), nodeID, direction)
			})
			if traverseErr != nil {
				return nil, traverseErr
			}
			for _, state := range states {
				binding := cypherMatchBinding{base: input, vertices: map[string]Record{}, nodes: map[string]uint64{}, edges: map[string]cypherEdgeBinding{}}
				db.addCypherVertexBinding(src, doc, &binding, first, source, sourceNode)
				for vertexIndex := 1; vertexIndex < len(vertices); vertexIndex++ {
					if vertexIndex >= len(state.nodes) {
						break
					}
					record, recordOK := recordsByNode[state.nodes[vertexIndex]]
					if !recordOK {
						binding = cypherMatchBinding{}
						break
					}
					nodeID := state.nodes[vertexIndex]
					if !db.cypherVertexMatchesInRow(ctx, src, doc, &doc.Vertexes[vertices[vertexIndex].ID], collection.GetGraph(), record, nodeID, input, params, legacy) {
						binding = cypherMatchBinding{}
						break
					}
					db.addCypherVertexBinding(src, doc, &binding, &doc.Vertexes[vertices[vertexIndex].ID], record, state.nodes[vertexIndex])
				}
				if len(binding.vertices) == 0 || !db.cypherWhereMatches(ctx, src, doc, binding, where, params, legacy) {
					continue
				}
				for edgeIndex, edgeRef := range edgeRefs {
					if edgeIndex >= len(state.edges) || edgeRef.ID < 0 || int(edgeRef.ID) >= len(doc.Edges) {
						continue
					}
					edge := &doc.Edges[edgeRef.ID]
					from, target := state.nodes[edgeIndex], state.nodes[edgeIndex+1]
					if edge.Direction < 0 {
						from, target = target, from
					}
					properties, _ := graph.EdgePropertyJSON(state.edges[edgeIndex].Properties)
					var values map[string]interface{}
					if len(properties) > 0 {
						_ = apexjson.Unmarshal(properties, &values)
					}
					alias := strings.ToLower(sourceSpan(src, edge.Alias, edge.AliasEnd))
					if alias != "" {
						binding.edges[alias] = cypherEdgeBinding{from: from, target: target, kind: state.edges[edgeIndex].Edge.GetKind(), weight: state.edges[edgeIndex].Edge.Weight, properties: values}
					}
				}
				result = append(result, binding)
			}
		}
	}
	return result, nil
}

func cypherInputRecordID(row virtualSQLRow, alias string) (string, bool) {
	if alias == "" {
		return "", false
	}
	for _, scope := range virtualRowScopes(row) {
		if !strings.EqualFold(scope.Alias, alias) {
			continue
		}
		value, ok := scope.Values["id"]
		if !ok {
			return "", false
		}
		id, ok := value.(string)
		return id, ok && id != ""
	}
	return "", false
}

func (db *Database) cypherVertexMatches(ctx context.Context, src []byte, doc *parser.QueryDoc, vertex *parser.Vertex, g Graph, record Record, nodeID uint64, params *optimizer.ParameterSet, legacy QueryParams) bool {
	return db.cypherVertexMatchesInRow(ctx, src, doc, vertex, g, record, nodeID, virtualSQLRow{}, params, legacy)
}

func (db *Database) cypherVertexMatchesInRow(ctx context.Context, src []byte, doc *parser.QueryDoc, vertex *parser.Vertex, g Graph, record Record, nodeID uint64, row virtualSQLRow, params *optimizer.ParameterSet, legacy QueryParams) bool {
	if vertex == nil {
		return false
	}
	labels := make([]string, 0, vertex.LabelsCount)
	if vertex.LabelStart < vertex.LabelEnd {
		labels = append(labels, sourceSpan(src, vertex.LabelStart, vertex.LabelEnd))
	}
	for i := int32(0); i < vertex.LabelsCount; i++ {
		index := vertex.LabelsStart + i
		if index >= 0 && int(index) < len(doc.VertexLabels) {
			label := doc.VertexLabels[index]
			labels = append(labels, sourceSpan(src, label.Start, label.End))
		}
	}
	if !graphLabelsMatch(g, nodeID, labels) {
		return false
	}
	properties, err := db.mergePropertyMapInRow(ctx, src, doc, vertex.Predicate, row, params, legacy)
	return err == nil && mergeRecordMatches(&record, properties)
}

func (db *Database) addCypherVertexBinding(src []byte, doc *parser.QueryDoc, binding *cypherMatchBinding, vertex *parser.Vertex, record Record, nodeID uint64) {
	alias := strings.ToLower(sourceSpan(src, vertex.Alias, vertex.AliasEnd))
	if alias == "" {
		return
	}
	binding.vertices[alias] = record
	binding.nodes[alias] = nodeID
}

func (db *Database) cypherWhereMatches(ctx context.Context, src []byte, doc *parser.QueryDoc, binding cypherMatchBinding, where parser.NodeRef, params *optimizer.ParameterSet, legacy QueryParams) bool {
	if where.Kind == parser.NodeKindUnknown {
		return true
	}
	row := binding.base
	row.Scopes = append([]virtualSQLScope(nil), virtualRowScopes(binding.base)...)
	for alias, record := range binding.vertices {
		values := cloneMetadata(record.Metadata)
		if values == nil {
			values = make(map[string]interface{})
		}
		values["id"] = record.ID
		row.Scopes = append(row.Scopes, virtualSQLScope{Alias: alias, Values: values})
	}
	value, ok, err := db.virtualExprValue(ctx, src, doc, where, row, params, legacy)
	return err == nil && ok && isVirtualTrue(value)
}
