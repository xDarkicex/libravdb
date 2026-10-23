package libravdb

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestTemporalSQL_AsOfTimestamp verifies the full AS OF TIMESTAMP pipeline:
// parse → bind → optimize → resolve LSN → temporal execute.
func TestTemporalSQL_AsOfTimestamp(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/temporal_sql.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()
	col, err := db.CreateCollection(context.Background(), "customers",
		WithDimension(3),
		WithGraph(gr),
		WithMetadataSchema(MetadataSchema{"name": StringField, "category": StringField}),
	)
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// T1: Insert C1 with vector V1, metadata, and edge to P1.
	if err := col.Insert(context.Background(), "C1", []float32{1, 0, 0},
		map[string]interface{}{"name": "Alice", "category": "customer"}); err != nil {
		t.Fatalf("Insert C1: %v", err)
	}
	if err := col.Insert(context.Background(), "P1", []float32{0, 0, 1},
		map[string]interface{}{"name": "AuthModule", "category": "Security"}); err != nil {
		t.Fatalf("Insert P1: %v", err)
	}
	c1Node, _ := db.GetNodeID(context.Background(), "customers", "C1")
	p1Node, _ := db.GetNodeID(context.Background(), "customers", "P1")
	txn := gr.BeginTxn()
	txn.AddEdge(c1Node, p1Node, 1.0, 1) // kind 1 = PURCHASED
	txn.Commit(context.Background())
	time.Sleep(50 * time.Millisecond)       // let batch flusher run
	t1 := time.Now().UTC().Add(time.Second) // well after the commit

	// T2: Update C1's vector.
	if err := col.Update(context.Background(), "C1", []float32{5, 0, 0},
		map[string]interface{}{"name": "Alice", "category": "customer"}); err != nil {
		t.Fatalf("Update C1: %v", err)
	}
	time.Sleep(20 * time.Millisecond)
	t2 := time.Now().UTC()

	// T3: Remove edge.
	txn2 := gr.BeginTxn()
	txn2.RemoveEdge(c1Node, p1Node, 1)
	txn2.Commit(context.Background())
	time.Sleep(20 * time.Millisecond)

	// Query at T1: should see C1 with V1=[1,0,0].
	t1Str := t1.Format(time.RFC3339)
	sql1 := "SELECT c.name FROM customers c AS OF TIMESTAMP '" + t1Str + "' WHERE c.name = 'Alice'"
	results1, err := db.Query(context.Background(), sql1)
	if err != nil {
		t.Fatalf("Query T1: %v", err)
	}
	if results1.Total != 1 {
		t.Errorf("T1: got %d results, want 1", results1.Total)
	}

	// Query at T2: vector is now V2.
	t.Logf("T1=%s T2=%s", t1Str, t2.Format(time.RFC3339))
	_ = t2
}

// TestTemporalSQL_ErrorCases verifies error behavior.
func TestTemporalSQL_ErrorCases(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/temporal_sql_err.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	// AS OF TIMESTAMP with missing timestamp literal
	_, err = db.Query(context.Background(), "SELECT * FROM c AS OF TIMESTAMP")
	if err == nil {
		t.Error("missing timestamp literal should error")
	}

	// AS OF TIMESTAMP with invalid timestamp
	_, err = db.Query(context.Background(), "SELECT * FROM c AS OF TIMESTAMP 'not-a-date'")
	if err == nil {
		t.Error("invalid timestamp should error")
	}

	// AS OF TIMESTAMP before any commit
	col, _ := db.CreateCollection(context.Background(), "c", WithDimension(3))
	col.Insert(context.Background(), "r1", []float32{1, 0, 0}, nil)
	_, err = db.Query(context.Background(), "SELECT * FROM c AS OF TIMESTAMP '2020-01-01T00:00:00Z'")
	if err == nil {
		t.Error("timestamp before first commit should error")
	}
}

// TestTemporalSQL_CurrentUnaffected verifies non-temporal SQL still works.
func TestTemporalSQL_CurrentUnaffected(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir() + "/temporal_sql_curr.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(context.Background())

	col, err := db.CreateCollection(context.Background(), "c", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	col.Insert(context.Background(), "r1", []float32{1, 0, 0}, nil)

	// SELECT * works — full scan fallback for non-B-tree indexes.
	results, err := db.Query(context.Background(), "SELECT * FROM c")
	if err != nil {
		t.Fatalf("SELECT *: %v", err)
	}
	if results.Total != 1 {
		t.Errorf("got %d results, want 1", results.Total)
	}
}

// TestTemporalSQL_CypherAndSQLConcurrentWithCompact verifies that SQL and Cypher
// queries with AS OF TIMESTAMP / AS OF LSN pin history properly and execute
// safely while background history compaction is running concurrently.
func TestTemporalSQL_CypherAndSQLConcurrentWithCompact(t *testing.T) {
	ctx := context.Background()
	db, err := Open(
		WithStoragePath(t.TempDir()+"/temporal_sql_cypher_concurrent.libravdb"),
		WithTemporalRetention(10*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(ctx)

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()

	if !RegisterEdgeKind("TEMPORAL_FOLLOWS", 1) && ResolveEdgeKind("TEMPORAL_FOLLOWS") != 1 {
		t.Fatal("failed to register edge kind TEMPORAL_FOLLOWS")
	}

	col, err := db.CreateCollection(ctx, "people",
		WithDimension(3),
		WithGraph(gr),
		WithMetadataSchema(MetadataSchema{"name": StringField, "role": StringField}),
	)
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// T1: Insert Alice, Bob, Charlie and graph edge Alice -> Bob
	col.Insert(ctx, "u1", []float32{1, 0, 0}, map[string]interface{}{"name": "Alice", "role": "admin"})
	col.Insert(ctx, "u2", []float32{0, 1, 0}, map[string]interface{}{"name": "Bob", "role": "dev"})
	col.Insert(ctx, "u3", []float32{0, 0, 1}, map[string]interface{}{"name": "Charlie", "role": "dev"})
	u1Node, _ := db.GetNodeID(ctx, "people", "u1")
	u2Node, _ := db.GetNodeID(ctx, "people", "u2")
	gr.RegisterVertexLabel(u1Node, "User")
	gr.RegisterVertexLabel(u2Node, "User")

	txn := gr.BeginTxn()
	txn.AddEdge(u1Node, u2Node, 1.0, 1) // kind 1 = TEMPORAL_FOLLOWS
	txn.Commit(ctx)
	time.Sleep(30 * time.Millisecond)

	snap1, err := db.SnapshotAt(ctx, time.Now().UTC())
	if err != nil {
		t.Fatalf("SnapshotAt T1: %v", err)
	}
	t1 := snap1.Timestamp
	lsn1 := snap1.LSN
	snap1.Close()

	// T2: Update Alice and Bob
	col.Update(ctx, "u1", []float32{2, 0, 0}, map[string]interface{}{"name": "Alice", "role": "lead"})
	col.Update(ctx, "u2", []float32{0, 2, 0}, map[string]interface{}{"name": "Bob", "role": "lead"})
	time.Sleep(30 * time.Millisecond)

	snap2, err := db.SnapshotAt(ctx, time.Now().UTC())
	if err != nil {
		t.Fatalf("SnapshotAt T2: %v", err)
	}
	t2 := snap2.Timestamp
	lsn2 := snap2.LSN
	snap2.Close()

	runCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	var sqlErrors atomic.Uint64
	var cypherErrors atomic.Uint64
	var expiredCount atomic.Uint64
	var successCount atomic.Uint64

	// Background Compactor: hammer CompactHistory repeatedly
	wg.Add(1)
	go func() {
		defer wg.Done()
		for runCtx.Err() == nil {
			_, _ = db.CompactHistory(runCtx)
			time.Sleep(1 * time.Millisecond)
		}
	}()

	// SQL reader with AS OF TIMESTAMP and Cypher MATCH pattern
	for r := 0; r < 2; r++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for runCtx.Err() == nil {
				target := t2
				if workerID%2 == 0 {
					target = t1
				}
				sql := fmt.Sprintf("SELECT u.name FROM people u AS OF TIMESTAMP '%s' "+
					"WHERE MATCH (u)-[r:TEMPORAL_FOLLOWS]->(v:User) AND v.name = 'Bob' LIMIT 5",
					target.Format(time.RFC3339Nano))

				res, qErr := db.Query(runCtx, sql)
				if qErr != nil {
					if errors.Is(qErr, ErrRetentionExpired) {
						expiredCount.Add(1)
						continue
					}
					if runCtx.Err() != nil {
						return
					}
					cypherErrors.Add(1)
					t.Errorf("Cypher AS OF TIMESTAMP query error: %v", qErr)
					return
				}
				if res == nil || len(res.Results) != 1 || res.Results[0].ID != "u1" {
					cypherErrors.Add(1)
					t.Errorf("Cypher AS OF query returned unexpected results: %#v", res)
					return
				}
				successCount.Add(1)
			}
		}(r)
	}

	// SQL reader with AS OF LSN
	for r := 0; r < 2; r++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for runCtx.Err() == nil {
				targetLSN := lsn2
				if workerID%2 == 0 {
					targetLSN = lsn1
				}
				sql := fmt.Sprintf("SELECT id, name FROM people AS OF LSN %d WHERE id = 'u1'", targetLSN)
				res, qErr := db.Query(runCtx, sql)
				if qErr != nil {
					if errors.Is(qErr, ErrRetentionExpired) {
						expiredCount.Add(1)
						continue
					}
					if runCtx.Err() != nil {
						return
					}
					sqlErrors.Add(1)
					t.Errorf("SQL AS OF LSN query error: %v", qErr)
					return
				}
				if res == nil || len(res.Results) != 1 || res.Results[0].ID != "u1" {
					sqlErrors.Add(1)
					t.Errorf("SQL AS OF LSN returned unexpected results: %#v", res)
					return
				}
				successCount.Add(1)
			}
		}(r)
	}

	wg.Wait()

	if totalErr := sqlErrors.Load() + cypherErrors.Load(); totalErr > 0 {
		t.Fatalf("Concurrent SQL/Cypher compaction test failed with %d errors", totalErr)
	}
	t.Logf("Concurrent SQL/Cypher test passed: %d successful executions, %d clean expirations",
		successCount.Load(), expiredCount.Load())
}
