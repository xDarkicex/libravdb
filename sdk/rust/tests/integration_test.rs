use libravdb::{LibraVDB, Filter};
use serde_json::json;
use std::fs;

#[test]
fn test_libravdb_integration() {
    let db_path = "./demo_db_rust";
    if fs::metadata(db_path).is_ok() {
        fs::remove_dir_all(db_path).unwrap();
    }

    println!("Initializing LibraVDB at {}...", db_path);
    let db = LibraVDB::new(db_path).expect("Failed to open DB");

    println!("Database Ping OK: {:?}", db.ping());

    println!("Setting global memory limit...");
    db.set_memory_limit(10 * 1024 * 1024).unwrap();

    println!("Creating collection 'docs' (dim: 3)...");
    let col = db.create_collection("docs", 3).unwrap();

    let collections = db.list_collections().unwrap();
    println!("List collections: {:?}", collections);
    assert_eq!(collections, vec!["docs"]);

    println!("Testing InsertBatch with 1000 vectors...");
    let mut ids = Vec::new();
    let mut vectors = Vec::new();
    let mut metadata = Vec::new();

    for i in 0..1000 {
        ids.push(format!("vec_{}", i));
        vectors.push(vec![0.1_f32, 0.2_f32, 0.3_f32]);
        metadata.push(json!({
            "source": "rust_test",
            "index": i,
            "active": i % 2 == 0
        }));
    }

    col.insert_batch(&ids, &vectors, Some(&metadata)).unwrap();
    println!("Batch Insert Complete!");

    println!("Testing Update...");
    col.update("vec_0", &[1.0, 1.0, 1.0], Some(json!({ "updated": true }))).unwrap();

    println!("Testing Get...");
    let rec = col.get("vec_0").unwrap();
    println!("Got record: {}", rec);
    assert_eq!(rec["id"], "vec_0");

    println!("Testing Search with Filters (active = true)...");
    let filter = Filter::eq("active", true);
    let results = col.search(&[1.0, 1.0, 1.0], 5, Some(&filter)).unwrap();
    println!("Search Results: {:?}", results);
    assert!(results.as_array().unwrap().len() > 0);

    println!("Testing Scan (Offset 0, Limit 2)...");
    let scanned = col.scan(0, 2).unwrap();
    println!("Scanned: {}", scanned);
    assert_eq!(scanned.as_array().unwrap().len(), 2);

    println!("Testing DeleteBatch (Deleting 500 vectors)...");
    col.delete_batch(&ids[0..500]).unwrap();

    let count = col.count().unwrap();
    println!("Collection Count: {}", count);
    assert_eq!(count, 500);

    println!("Testing Vacuum...");
    db.vacuum().unwrap();

    println!("Dropping database...");
    db.drop_db().unwrap();

    assert!(fs::metadata(db_path).is_err());
    println!("Database successfully dropped!");
}
