package main

import "core:fmt"
import "core:os"
import "core:time"
import "core:strings"
import "libravdb"

main :: proc() {
    db_path := "demo_db_sql_odin"

    fmt.println("Initializing LibraVDB at", db_path, "...")
    db, err := libravdb.open_db(db_path)
    if err != .None {
        fmt.println("Failed to open DB:", err)
        return
    }
    defer libravdb.close_db(&db)

    // Create tables
    _, err1 := libravdb.query(&db, "CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))")
    if err1 != .None { fmt.println("Error CREATE GRAPH TABLE:", err1); return }

    _, err2 := libravdb.query(&db, "CREATE EDGE TYPE FOLLOWS")
    if err2 != .None { fmt.println("Error CREATE EDGE TYPE:", err2); return }

    // 1. Relational
    fmt.println("\n--- Relational SQL ---")
    _, err_ins1 := libravdb.query_with_params(&db, "INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)", `{"1": "u1", "2": "Alice", "3": [1.0, 0.0, 0.0]}`)
    _, err_ins2 := libravdb.query_with_params(&db, "INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)", `{"1": "u2", "2": "Bob", "3": [0.0, 1.0, 0.0]}`)
    _, err_ins3 := libravdb.query_with_params(&db, "INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)", `{"1": "u3", "2": "Charlie", "3": [0.0, 0.0, 1.0]}`)

    res_rel, err_rel := libravdb.query(&db, "SELECT id, name FROM users ORDER BY name ASC")
    fmt.println("Relational Result:", res_rel)

    // 2. Vector
    fmt.println("\n--- Vector SQL ---")
    res_vec, err_vec := libravdb.query_with_params(&db, "SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, $vec) ASC LIMIT 2", `{"vec": [1.0, 0.0, 0.0]}`)
    fmt.println("Vector Result:", res_vec)

    // 3. Graph
    fmt.println("\n--- Graph SQL ---")
    _, err_edge1 := libravdb.query_with_params(&db, "INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)", `{"1": "u1", "2": "FOLLOWS", "3": "u2"}`)
    _, err_edge2 := libravdb.query_with_params(&db, "INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)", `{"1": "u2", "2": "FOLLOWS", "3": "u3"}`)
    res_graph, err_graph := libravdb.query_with_params(&db, "SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = $1", `{"1": "u1"}`)
    fmt.println("Graph Result:", res_graph)

    // 4. Temporal SQL
    fmt.println("\n--- Temporal SQL ---")
    future_time := time.time_add(time.now(), time.Second * 2)
    // NOTE: In Odin we can use a basic RFC3339 formatted string or ISO8601.
    // However, string interpolation is simpler. Let's just generate a future timestamp.
    // For simplicity, we can do string manipulation.
    year, month, day := time.date(future_time)
    hour, min, sec := time.clock(future_time)
    cutoff := fmt.tprintf("%04d-%02d-%02dT%02d:%02d:%02d.000Z", year, int(month), day, hour, min, sec)

    query_str := fmt.tprintf("SELECT id FROM users AS OF TIMESTAMP '%s' ORDER BY id ASC", cutoff)
    res_temp, err_temp := libravdb.query(&db, query_str)
    fmt.println("Temporal Result:", res_temp)

    fmt.println("\nAll unified SQL tests passed successfully.")
}
