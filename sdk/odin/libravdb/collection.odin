package libravdb

import "core:strings"

Collection :: struct {
    db_id:     i32,
    col_id:    i32,
    name:      string,
    dimension: int,
}

// insert inserts a single vector.
insert :: proc(col: ^Collection, id: string, vector: []f32, metadata: string = "{}") -> LibraError {
    if len(vector) != col.dimension {
        return .NativeError
    }
    
    c_id := strings.clone_to_cstring(id)
    defer delete(c_id)
    
    c_meta := strings.clone_to_cstring(metadata)
    defer delete(c_meta)
    
    vec_ptr := raw_data(vector)
    
    err_ptr := InsertVector(col.col_id, c_id, vec_ptr, i32(col.dimension), c_meta)
    return check_error(err_ptr)
}

// insert_batch inserts multiple vectors efficiently via zero-copy batching.
insert_batch :: proc(col: ^Collection, ids: []string, vectors: [][]f32, metadata: []string = nil) -> LibraError {
    count := len(ids)
    if len(vectors) != count {
        return .NativeError
    }
    
    // Flatten vectors
    flat_vectors := make([]f32, count * col.dimension)
    defer delete(flat_vectors)
    
    for i in 0..<count {
        if len(vectors[i]) != col.dimension {
            return .NativeError
        }
        for j in 0..<col.dimension {
            flat_vectors[i * col.dimension + j] = vectors[i][j]
        }
    }
    
    // Convert strings to cstrings
    c_ids := make([]cstring, count)
    defer {
        for cstr in c_ids { delete(cstr) }
        delete(c_ids)
    }
    for i in 0..<count {
        c_ids[i] = strings.clone_to_cstring(ids[i])
    }
    
    c_metas := make([]cstring, count)
    defer {
        for cstr in c_metas { delete(cstr) }
        delete(c_metas)
    }
    for i in 0..<count {
        meta_str := "{}"
        if metadata != nil && i < len(metadata) {
            meta_str = metadata[i]
        }
        c_metas[i] = strings.clone_to_cstring(meta_str)
    }
    
    err_ptr := InsertBatch(
        col.col_id, 
        raw_data(c_ids), 
        raw_data(flat_vectors), 
        i32(count), 
        i32(col.dimension), 
        raw_data(c_metas)
    )
    
    return check_error(err_ptr)
}

// search queries the collection using a vector and an optional JSON AST filter.
search :: proc(col: ^Collection, vector: []f32, k: int, filter: string = "{}") -> (string, LibraError) {
    if len(vector) != col.dimension {
        return "", .NativeError
    }
    
    c_filter := strings.clone_to_cstring(filter)
    defer delete(c_filter)
    
    res_ptr := QueryVector(col.col_id, raw_data(vector), i32(col.dimension), i32(k), c_filter)
    return extract_string(res_ptr)
}

// scan fetches all documents with pagination.
scan :: proc(col: ^Collection, offset: int = 0, limit: int = 100) -> (string, LibraError) {
    res_ptr := ScanCollection(col.col_id, i32(offset), i32(limit))
    return extract_string(res_ptr)
}

// delete_vector deletes a single vector by ID.
delete_vector :: proc(col: ^Collection, id: string) -> LibraError {
    c_id := strings.clone_to_cstring(id)
    defer delete(c_id)
    
    err_ptr := DeleteVector(col.col_id, c_id)
    return check_error(err_ptr)
}
