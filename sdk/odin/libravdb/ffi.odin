package libravdb

when ODIN_OS == .Darwin {
    foreign import libravdb "../../cgo/libravdb.dylib"
} else when ODIN_OS == .Linux {
    foreign import libravdb "../../cgo/libravdb.so"
} else when ODIN_OS == .Windows {
    foreign import libravdb "../../cgo/libravdb.dll"
}

@(default_calling_convention="c")
foreign libravdb {
    OpenDB :: proc(path: cstring) -> i32 ---
    CloseDB :: proc(dbID: i32) ---
    CreateCollection :: proc(dbID: i32, name: cstring, dimension: i32) -> i32 ---
    GetCollection :: proc(dbID: i32, name: cstring) -> i32 ---
    InsertVector :: proc(colID: i32, id: cstring, vector: ^f32, dimension: i32, metadata: cstring) -> cstring ---
    QueryVector :: proc(colID: i32, vector: ^f32, dimension: i32, k: i32, filter: cstring) -> cstring ---
    ScanCollection :: proc(colID: i32, offset: i32, limit: i32) -> cstring ---
    Vacuum :: proc(dbID: i32) -> cstring ---
    DropDatabase :: proc(dbID: i32) -> cstring ---
    InsertBatch :: proc(colID: i32, ids: ^cstring, vectors: ^f32, count: i32, dimension: i32, metadata: ^cstring) -> cstring ---
    DeleteVector :: proc(colID: i32, id: cstring) -> cstring ---
    DeleteBatch :: proc(colID: i32, ids: ^cstring, count: i32) -> cstring ---
    FreeString :: proc(ptr: cstring) ---
}
