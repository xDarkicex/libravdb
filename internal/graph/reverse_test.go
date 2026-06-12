package graph

import (
	"reflect"
	"testing"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

type edgeOp struct {
	Add    bool
	Src    uint64
	Tgt    uint64
	Kind   uint8
	Weight float32
}

func genEdgeOp() gopter.Gen {
	return gen.Struct(reflect.TypeOf(edgeOp{}), map[string]gopter.Gen{
		"Add":    gen.Bool(),
		"Src":    gen.UInt64Range(1, 100),
		"Tgt":    gen.UInt64Range(1, 100),
		"Kind":   gen.UInt8(),
		"Weight": gen.Float32(),
	})
}

func TestReverseIndex_Symmetry(t *testing.T) {
	// Property 13: Forward-Reverse Index Symmetry
	properties := gopter.NewProperties(nil)

	properties.Property("Every forward edge has a corresponding reverse edge", prop.ForAll(
		func(ops []edgeOp) bool {
			cfg := testConfig()
			store, err := NewGraph(cfg)
			if err != nil {
				return false
			}
			defer store.Close()

			txn := &Txn{ID: 1}
			expected := make(map[uint64]map[uint64]map[uint8]bool)
			
			for _, op := range ops {
				if op.Add {
					_ = store.AddEdge(txn, op.Src, op.Tgt, op.Weight, op.Kind)
					if expected[op.Src] == nil {
						expected[op.Src] = make(map[uint64]map[uint8]bool)
					}
					if expected[op.Src][op.Tgt] == nil {
						expected[op.Src][op.Tgt] = make(map[uint8]bool)
					}
					expected[op.Src][op.Tgt][op.Kind] = true
				} else {
					_ = store.RemoveEdge(txn, op.Src, op.Tgt, op.Kind)
					if expected[op.Src] != nil && expected[op.Src][op.Tgt] != nil {
						delete(expected[op.Src][op.Tgt], op.Kind)
					}
				}
			}

			gs := store.(*graphStore)
			
			for src, tgts := range expected {
				for tgt, kinds := range tgts {
					for kind, exists := range kinds {
						if exists {
							revEdges, _ := gs.neighborsFromTable(tgt, gs.reverse.locator, gs.reverse.pool, gs.cfg.PageShards)
							found := false
							for _, re := range revEdges {
								if re.Target == src && re.Kind == kind {
									found = true
									break
								}
							}
							if !found {
								return false
							}
						}
					}
				}
			}
			
			return true
		},
		gen.SliceOfN(100, genEdgeOp()),
	))

	properties.TestingRun(t)
}

func TestDropNodeEdges_Completeness(t *testing.T) {
	// Property 7: DropNodeEdges Completeness
	properties := gopter.NewProperties(nil)

	properties.Property("DropNodeEdges removes all inbound and outbound edges", prop.ForAll(
		func(ops []edgeOp, dropNode uint64) bool {
			cfg := testConfig()
			store, err := NewGraph(cfg)
			if err != nil {
				return false
			}
			defer store.Close()

			txn := &Txn{ID: 1}
			
			for _, op := range ops {
				store.AddEdge(txn, op.Src, op.Tgt, op.Weight, op.Kind)
			}
			
			store.AddEdge(txn, dropNode, 101, 1.0, 1)
			store.AddEdge(txn, 102, dropNode, 1.0, 1)
			store.AddEdge(txn, dropNode, dropNode, 1.0, 1) // test self loop
			
			err = store.DropNodeEdges(txn, dropNode)
			if err != nil {
				return false
			}
			
			deg, _ := store.Degree(dropNode)
			if deg != 0 {
				return false
			}
			
			neighbors, _ := store.Neighbors(dropNode)
			if len(neighbors) != 0 {
				return false
			}
			
			gs := store.(*graphStore)
			revEdges, _ := gs.neighborsFromTable(dropNode, gs.reverse.locator, gs.reverse.pool, gs.cfg.PageShards)
			if len(revEdges) != 0 {
				return false
			}
			
			nodes := make(map[uint64]bool)
			for _, op := range ops {
				nodes[op.Src] = true
				nodes[op.Tgt] = true
			}
			nodes[101] = true
			nodes[102] = true
			
			for n := range nodes {
				if n == dropNode {
					continue
				}
				nEdges, _ := store.Neighbors(n)
				for _, e := range nEdges {
					if e.Target == dropNode {
						return false
					}
				}
			}
			
			return true
		},
		gen.SliceOfN(50, genEdgeOp()), gen.UInt64Range(1, 100),
	))

	properties.TestingRun(t)
}
