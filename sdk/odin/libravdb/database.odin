package libravdb

import "core:strings"

Database :: struct {
    id: i32,
}

// Error definitions
LibraError :: enum {
    None,
    InitFailed,
    NativeError,
}

// OpenDB opens a database at the given path
open_db :: proc(path: string) -> (Database, LibraError) {
    c_path := strings.clone_to_cstring(path)
    defer delete(c_path)

    db_id := OpenDB(c_path)
    if db_id < 0 {
        return Database{id = -1}, .InitFailed
    }

    return Database{id = db_id}, .None
}

// CloseDB closes the database connection
close_db :: proc(db: ^Database) {
    if db.id >= 0 {
        CloseDB(db.id)
        db.id = -1
    }
}

// CreateCollection creates a new collection
create_collection :: proc(db: ^Database, name: string, dimension: int) -> (Collection, LibraError) {
    c_name := strings.clone_to_cstring(name)
    defer delete(c_name)

    col_id := CreateCollection(db.id, c_name, i32(dimension))
    if col_id < 0 {
        return Collection{}, .NativeError
    }

    return Collection{
        db_id = db.id,
        col_id = col_id,
        name = name,
        dimension = dimension,
    }, .None
}

// GetCollection retrieves an existing collection
get_collection :: proc(db: ^Database, name: string, dimension: int) -> (Collection, LibraError) {
    c_name := strings.clone_to_cstring(name)
    defer delete(c_name)

    col_id := GetCollection(db.id, c_name)
    if col_id < 0 {
        return Collection{}, .NativeError
    }

    return Collection{
        db_id = db.id,
        col_id = col_id,
        name = name,
        dimension = dimension,
    }, .None
}

// Vacuum triggers database compaction
vacuum :: proc(db: ^Database) -> LibraError {
    err_ptr := Vacuum(db.id)
    return check_error(err_ptr)
}

// DropDatabase drops the database completely
drop_database :: proc(db: ^Database) -> LibraError {
    err_ptr := DropDatabase(db.id)
    return check_error(err_ptr)
}

// Helper to check for native string errors from Go
@(private="package")
check_error :: proc(err_ptr: cstring) -> LibraError {
    if err_ptr == nil {
        return .None
    }

    msg := string(err_ptr)
    defer FreeString(err_ptr)

    if msg == "OK" {
        return .None
    }
    if strings.has_prefix(msg, "ERROR:") || strings.has_prefix(msg, "error") {
        return .NativeError
    }
    return .None
}

@(private="package")
extract_string :: proc(ptr: cstring) -> (string, LibraError) {
    if ptr == nil {
        return "", .NativeError
    }

    msg := string(ptr)
    // We clone the string so we can safely free the Go allocated memory
    result := strings.clone(msg)
    FreeString(ptr)

    if strings.has_prefix(result, "ERROR:") || strings.has_prefix(result, "error") {
        delete(result)
        return "", .NativeError
    }

    return result, .None
}

// Extract query result from C-string and catch {"error": ...} JSON envelopes
@(private="package")
extract_query_result :: proc(ptr: cstring) -> (string, LibraError) {
    if ptr == nil {
        return "", .NativeError
    }

    msg := string(ptr)
    result := strings.clone(msg)
    FreeString(ptr)

    if strings.has_prefix(result, "{\"error\"") {
        delete(result)
        return "", .NativeError
    }

    return result, .None
}

// Execute a bare SQL query and return the JSON string payload
query :: proc(db: ^Database, sql: string) -> (string, LibraError) {
    c_sql := strings.clone_to_cstring(sql)
    defer delete(c_sql)

    res_ptr := DatabaseQuery(db.id, c_sql)
    return extract_query_result(res_ptr)
}

// Execute a SQL query with parameters and return the JSON string payload
query_with_params :: proc(db: ^Database, sql: string, params: string) -> (string, LibraError) {
    c_sql := strings.clone_to_cstring(sql)
    c_params := strings.clone_to_cstring(params)
    defer delete(c_sql)
    defer delete(c_params)

    res_ptr := DatabaseQueryWithParams(db.id, c_sql, c_params)
    return extract_query_result(res_ptr)
}
