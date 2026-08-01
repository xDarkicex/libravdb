package main

import "core:fmt"
import "core:os"
import "core:strings"
import "libravdb"

main :: proc() {
    db_path := "./test_db_odin"


    // Open DB
    db, err := libravdb.open_db(db_path)
    if err != .None {
        fmt.println("Failed to open DB")
        os.exit(1)
    }
    defer libravdb.close_db(&db)
    
    // Create Collection
    col, col_err := libravdb.create_collection(&db, "test_col", 3)
    if col_err != .None {
        fmt.println("Failed to create collection")
        os.exit(1)
    }

    // Insert
    vec1 := []f32{1.0, 2.0, 3.0}
    meta1 := `{"category": "A"}`
    insert_err := libravdb.insert(&col, "1", vec1, meta1)
    if insert_err != .None {
        fmt.println("Failed to insert")
        os.exit(1)
    }

    // Search
    filter := libravdb.eq_str("category", "A")
    filter_json := libravdb.to_json(filter)
    
    search_res, search_err := libravdb.search(&col, vec1, 10, filter_json)
    if search_err != .None {
        fmt.println("Failed to search")
        os.exit(1)
    }
    
    if !strings.contains(search_res, `"id":"1"`) {
        fmt.println("Search results missing inserted vector")
        os.exit(1)
    }

    // Batch Insert
    ids := []string{"2", "3"}
    vecs := [][]f32{
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
    }
    metas := []string{`{"category": "B"}`, `{"category": "C"}`}
    
    batch_err := libravdb.insert_batch(&col, ids, vecs, metas)
    if batch_err != .None {
        fmt.println("Failed batch insert")
        os.exit(1)
    }

    // Scan
    scan_res, _ := libravdb.scan(&col)
    if !strings.contains(scan_res, `"id":"2"`) || !strings.contains(scan_res, `"id":"3"`) {
        fmt.println("Scan results missing batch inserted vectors")
        os.exit(1)
    }

    // Delete
    libravdb.delete_vector(&col, "2")
    scan_res2, _ := libravdb.scan(&col)
    if strings.contains(scan_res2, `"id":"2"`) {
        fmt.println("Delete failed")
        os.exit(1)
    }
    
    fmt.println("All Odin integration tests passed!")
}
