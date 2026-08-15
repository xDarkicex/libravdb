use libravdb::LibraVDB;
use serde_json::json;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use chrono::{DateTime, Utc};

#[test]
fn test_sql() {
    let db_path = "demo_db_sql_rust";
    if Path::new(db_path).exists() {
        let _ = fs::remove_dir_all(db_path);
        let _ = fs::remove_file(db_path);
    }

    println!("Initializing LibraVDB at {}...", db_path);
    let db = LibraVDB::new(db_path).expect("Failed to init DB");

    // Create tables
    db.query("CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))")
        .expect("CREATE GRAPH TABLE failed");
    db.query("CREATE EDGE TYPE FOLLOWS")
        .expect("CREATE EDGE TYPE failed");

    // 1. Relational
    println!("\n--- Relational SQL ---");
    db.query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)", Some(json!({"1": "u1", "2": "Alice", "3": [1.0, 0.0, 0.0]}))).unwrap();
    db.query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)", Some(json!({"1": "u2", "2": "Bob", "3": [0.0, 1.0, 0.0]}))).unwrap();
    db.query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)", Some(json!({"1": "u3", "2": "Charlie", "3": [0.0, 0.0, 1.0]}))).unwrap();

    let res = db.query("SELECT id, name FROM users ORDER BY name ASC").unwrap();
    println!("Relational Result: {}", res);

    // 2. Vector
    println!("\n--- Vector SQL ---");
    let res = db.query_with_params("SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, $vec) ASC LIMIT 2", Some(json!({"vec": [1.0, 0.0, 0.0]}))).unwrap();
    println!("Vector Result: {}", res);

    // 3. Graph
    println!("\n--- Graph SQL ---");
    db.query_with_params("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)", Some(json!({"1": "u1", "2": "FOLLOWS", "3": "u2"}))).unwrap();
    db.query_with_params("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)", Some(json!({"1": "u2", "2": "FOLLOWS", "3": "u3"}))).unwrap();
    let res = db.query_with_params("SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = $1", Some(json!({"1": "u1"}))).unwrap();
    println!("Graph Result: {}", res);

    // 4. Temporal SQL
    println!("\n--- Temporal SQL ---");
    let future_time = SystemTime::now() + Duration::from_secs(2);
    let datetime: DateTime<Utc> = future_time.into();
    let cutoff = datetime.to_rfc3339();

    let res = db.query(&format!("SELECT id FROM users AS OF TIMESTAMP '{}' ORDER BY id ASC", cutoff)).unwrap();
    println!("Temporal Result: {}", res);

    println!("\nAll unified SQL tests passed successfully.");
}
