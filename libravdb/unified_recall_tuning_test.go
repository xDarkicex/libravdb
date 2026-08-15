package libravdb

// This is an opt-in production-path tuning loop. Run with:
//   LIBRAVDB_TUNE=1 go test ./libravdb -run TestUnifiedRecallTuning -v -count=1
// It intentionally does not run in the normal suite because it builds several
// multi-thousand-vector HNSW indexes.

import (
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

type unifiedRecallProfile struct {
	name string
	m    int
	efc  int
	efs  int
}

func TestUnifiedRecallTuning(t *testing.T) {
	if os.Getenv("LIBRAVDB_TUNE") != "1" {
		t.Skip("set LIBRAVDB_TUNE=1 to run the production recall sweep")
	}
	ctx := context.Background()
	profiles := []unifiedRecallProfile{
		{name: "fast-m16-50", m: 16, efc: 200, efs: 50},
		{name: "balanced-m16-200", m: 16, efc: 200, efs: 200},
		{name: "production-m32-200", m: 32, efc: 200, efs: 200},
		{name: "high-recall-m36-200", m: 36, efc: 200, efs: 200},
	}

	for _, profile := range profiles {
		profile := profile
		t.Run(profile.name, func(t *testing.T) {
			const n, dim, k, reachable = 5000, 512, 10, 1200
			db, err := Open(WithStoragePath(":memory:unified_tune_"+profile.name), WithMetrics(false))
			if err != nil {
				t.Fatal(err)
			}
			defer db.Close()
			graph, err := NewGraph(GraphConfig{})
			if err != nil {
				t.Fatal(err)
			}
			defer graph.Close()
			col, err := db.CreateCollection(ctx, "items", WithDimension(dim), WithMetric(CosineDistance),
				WithHNSW(profile.m, profile.efc, profile.efs), WithGraph(graph),
				WithMetadataSchema(MetadataSchema{"cat": IntField}), WithIndexedFields("cat"))
			if err != nil {
				t.Fatal(err)
			}

			vectors := make([][]float32, n)
			for i := 0; i < n; i++ {
				vectors[i] = tuningVector(i, dim)
				cat := int64(0)
				if i < reachable {
					cat = 1
				}
				if err := col.Insert(ctx, fmt.Sprintf("v-%05d", i), vectors[i], map[string]interface{}{"cat": cat}); err != nil {
					t.Fatal(err)
				}
			}
			seed, err := db.GetNodeID(ctx, "items", "v-00000")
			if err != nil {
				t.Fatal(err)
			}
			for i := 1; i < reachable; i++ {
				target, err := db.GetNodeID(ctx, "items", fmt.Sprintf("v-%05d", i))
				if err != nil {
					t.Fatal(err)
				}
				txn := graph.BeginTxn()
				if err := graph.AddEdge(txn, seed, target, 1, 0); err != nil {
					t.Fatal(err)
				}
				if err := txn.Commit(ctx); err != nil {
					t.Fatal(err)
				}
			}
			graph.RegisterVertexLabel(seed, "Service")

			plan := &optimizer.PhysicalPlan{CollectionName: "items", Kind: optimizer.QueryKindKNN,
				HasVectorSearch: true, QueryVector: tuningVector(7777, dim), Limit: k,
				HasGraphTraversal: true, HasExplicitSeed: true, ExplicitSeedID: seed,
				GraphEdges: []optimizer.GraphEdgePlan{{Direction: 1, QuantMin: 1, QuantMax: 1}}, MaxHops: 1,
				HasRelationalQuery: true, Predicates: []optimizer.RelationalPredicate{{Column: "cat", Operator: 12, Value: []byte("1")}},
				RecallContract: optimizer.RecallBounded}
			exec := newExecutor(db)
			metrics := &QueryMetrics{}
			constraints, err := exec.prepareHybridConstraints(ctx, plan, metrics)
			if err != nil {
				t.Fatal(err)
			}
			all, err := col.ListAll(ctx)
			if err != nil {
				t.Fatal(err)
			}
			records := make([]Record, 0, len(constraints.graphRecordIDs))
			for _, record := range all {
				if constraints.graphAllows(record.ID) && recordMatchesPredicates(record, plan.Predicates) {
					records = append(records, record)
				}
			}
			exact := scoreAndSelectTopK(col, records, plan.QueryVector, k)
			want := make(map[string]struct{}, len(exact.Results))
			for _, r := range exact.Results {
				want[r.ID] = struct{}{}
			}
			// Force the production filtered operator once as a quality probe. The
			// normal Execute call below remains the authoritative dispatcher result.
			filteredMetrics := &QueryMetrics{EstConjunctionCandidates: len(records)}
			filtered, err := exec.executeFilteredANN(ctx, plan, filteredMetrics, constraints)
			if err != nil {
				t.Fatal(err)
			}
			filteredHits := 0
			for _, r := range filtered.Results {
				if _, ok := want[r.ID]; ok {
					filteredHits++
				}
			}
			filteredRecall := float64(filteredHits) / float64(len(want))
			got, err := exec.Execute(ctx, plan)
			if err != nil {
				t.Fatal(err)
			}
			hits := 0
			for _, r := range got.Results {
				if _, ok := want[r.ID]; ok {
					hits++
				}
			}
			recall := float64(hits) / float64(len(want))
			t.Logf("profile=%s dispatch=%s candidates=%d filteredRecall=%.3f dispatcherRecall=%.3f", profile.name, metrics.PlanChosen, len(records), filteredRecall, recall)
			if filteredRecall < 0.97 {
				t.Errorf("production unified filtered recall %.3f below 0.970", filteredRecall)
			}

			// Finally exercise the literal SQL parser/optimizer path. The query
			// shape is GRAPH_TABLE + MATCH + scalar predicate + vector function.
			sql := fmt.Sprintf("SELECT id FROM GRAPH_TABLE(items MATCH (s:Service)-[e]->(x)) WHERE SIMILARITY(vector, '%s') > 0 AND cat = 1 LIMIT %d", tuningVectorLiteral(plan.QueryVector), k)
			sqlResults, err := db.Query(ctx, sql)
			if err != nil {
				t.Fatalf("literal unified SQL failed: %v", err)
			}
			sqlHits := 0
			for _, r := range sqlResults.Results {
				if _, ok := want[r.ID]; ok {
					sqlHits++
				}
			}
			sqlRecall := float64(sqlHits) / float64(len(want))
			t.Logf("profile=%s literalSQL rows=%d recall=%.3f", profile.name, len(sqlResults.Results), sqlRecall)
			if sqlRecall < 0.97 {
				t.Errorf("literal unified SQL recall %.3f below 0.970", sqlRecall)
			}
		})
	}
}

func tuningVectorLiteral(v []float32) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = fmt.Sprintf("%.7g", x)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

func tuningVector(seed, dim int) []float32 {
	v := make([]float32, dim)
	for j := range v {
		x := float64((seed+1)*(j+3)%997) / 997.0
		v[j] = float32(x - 0.5)
	}
	norm := float32(0)
	for _, x := range v {
		norm += x * x
	}
	norm = float32(math.Sqrt(float64(norm)))
	for j := range v {
		v[j] /= norm
	}
	return v
}
