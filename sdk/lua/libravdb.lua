local ffi = require("ffi")
local json = require("json")

local is_windows = package.config:sub(1,1) == "\\"
local ext = is_windows and "dll" or (jit.os == "OSX" and "dylib" or "so")
local lib_path = string.format("../cgo/libravdb.%s", ext)
local lib = ffi.load(lib_path)

ffi.cdef[[
    int OpenDB(const char* path);
    int CloseDB(int dbID);
    void FreeString(char* ptr);
    char* Ping(int dbID);
    char* SetGlobalMemoryLimit(int dbID, long long limit);
    char* Vacuum(int dbID);
    char* DropDatabase(int dbID);
    char* ListCollections(int dbID);

    int CreateCollection(int dbID, const char* name, int dim);
    int GetCollection(int dbID, const char* name);
    long long GetCollectionCount(int colID);

    char* DatabaseQuery(int dbID, const char* sql);
    char* DatabaseQueryWithParams(int dbID, const char* sql, const char* paramsStr);

    char* InsertVector(int colID, const char* id, float* vector, int dim, const char* metadataJSON);
    char* UpdateVector(int colID, const char* id, float* vector, int dim, const char* metadataJSON);
    char* UpdateVectorIfVersion(int colID, const char* id, float* vector, int dim, const char* metadataJSON, long long expectedVersion);
    char* DeleteVector(int colID, const char* id);
    char* GetVector(int colID, const char* id);
    char* QueryVector(int colID, float* vector, int dim, int limit, const char* filterJSON);
    char* ScanCollection(int colID, int offset, int limit);

    char* InsertBatch(int colID, const char** ids, float* vectors, int count, int dim, const char** metadataJSON);
    char* DeleteBatch(int colID, const char** ids, int count);
]]

local function check_error(res_ptr, context)
    if res_ptr ~= nil then
        local msg = ffi.string(res_ptr)
        lib.FreeString(res_ptr)
        if msg ~= "" then
            error(string.format("%s failed: %s", context, msg))
        end
    end
end

local function parse_query_result(res_ptr, context)
    if res_ptr == nil then
        error(string.format("%s failed: null pointer returned", context))
    end
    local msg = ffi.string(res_ptr)
    lib.FreeString(res_ptr)

    if msg:sub(1, 8) == '{"error"' then
        error(string.format("%s failed: %s", context, msg))
    end

    if msg == "" then return nil end
    local s, res = pcall(json.decode, msg)
    if s then return res else return msg end
end

-- ==========================================
-- Collection Class
-- ==========================================
local Collection = {}
Collection.__index = Collection

function Collection.new(col_handle, dimension)
    local self = setmetatable({}, Collection)
    self.handle = col_handle
    self.dimension = dimension
    return self
end

function Collection:count()
    local cnt = lib.GetCollectionCount(self.handle)
    return tonumber(cnt)
end

function Collection:insert(id, vector, metadata)
    if #vector ~= self.dimension then
        error("Vector dimension mismatch")
    end
    local vec_c = ffi.new("float[?]", self.dimension, vector)
    local meta_str = metadata and json.encode(metadata) or ""

    local res_ptr = lib.InsertVector(self.handle, id, vec_c, self.dimension, meta_str)
    check_error(res_ptr, "InsertVector")
end

function Collection:insert_batch(ids, vectors, metadatas)
    local count = #ids
    local id_arr = ffi.new("const char*[?]", count)
    local meta_arr = ffi.new("const char*[?]", count)
    local vec_arr = ffi.new("float[?]", count * self.dimension)

    for i=1, count do
        id_arr[i-1] = ids[i]
        meta_arr[i-1] = metadatas and metadatas[i] and json.encode(metadatas[i]) or ""
        for j=1, self.dimension do
            vec_arr[(i-1)*self.dimension + (j-1)] = vectors[i][j]
        end
    end

    local res_ptr = lib.InsertBatch(self.handle, id_arr, vec_arr, count, self.dimension, meta_arr)
    check_error(res_ptr, "InsertBatch")
end

function Collection:update(id, vector, metadata)
    local vec_c = ffi.new("float[?]", self.dimension, vector)
    local meta_str = metadata and json.encode(metadata) or ""

    local res_ptr = lib.UpdateVector(self.handle, id, vec_c, self.dimension, meta_str)
    check_error(res_ptr, "UpdateVector")
end

function Collection:update_if_version(id, vector, metadata, expected_version)
    local vec_c = ffi.new("float[?]", self.dimension, vector)
    local meta_str = metadata and json.encode(metadata) or ""

    local res_ptr = lib.UpdateVectorIfVersion(self.handle, id, vec_c, self.dimension, meta_str, expected_version)
    check_error(res_ptr, "UpdateVectorIfVersion")
end

function Collection:delete(id)
    local res_ptr = lib.DeleteVector(self.handle, id)
    check_error(res_ptr, "DeleteVector")
end

function Collection:delete_batch(ids)
    local count = #ids
    local id_arr = ffi.new("const char*[?]", count)
    for i=1, count do
        id_arr[i-1] = ids[i]
    end
    local res_ptr = lib.DeleteBatch(self.handle, id_arr, count)
    check_error(res_ptr, "DeleteBatch")
end

function Collection:get(id)
    local res_ptr = lib.GetVector(self.handle, id)
    return parse_query_result(res_ptr, "GetVector")
end

function Collection:search(vector, limit, filter)
    local vec_c = ffi.new("float[?]", self.dimension, vector)
    local filter_str = filter and json.encode(filter) or ""
    local res_ptr = lib.QueryVector(self.handle, vec_c, self.dimension, limit, filter_str)
    return parse_query_result(res_ptr, "QueryVector")
end

function Collection:scan(offset, limit)
    local res_ptr = lib.ScanCollection(self.handle, offset, limit)
    return parse_query_result(res_ptr, "ScanCollection")
end


-- ==========================================
-- LibraVDB Class
-- ==========================================
local LibraVDB = {}
LibraVDB.__index = LibraVDB

function LibraVDB.new(path)
    local self = setmetatable({}, LibraVDB)
    self.handle = lib.OpenDB(path)
    if self.handle < 0 then
        error("Failed to open database at " .. path)
    end
    return self
end

function LibraVDB:ping()
    check_error(lib.Ping(self.handle), "Ping")
end

function LibraVDB:set_memory_limit(limit)
    check_error(lib.SetGlobalMemoryLimit(self.handle, limit), "SetMemoryLimit")
end

function LibraVDB:vacuum()
    check_error(lib.Vacuum(self.handle), "Vacuum")
end

function LibraVDB:drop_database()
    check_error(lib.DropDatabase(self.handle), "DropDatabase")
end

function LibraVDB:list_collections()
    local res_ptr = lib.ListCollections(self.handle)
    return parse_query_result(res_ptr, "ListCollections") or {}
end

function LibraVDB:create_collection(name, dimension)
    local col_handle = lib.CreateCollection(self.handle, name, dimension)
    if col_handle < 0 then
        error("Failed to create collection: " .. name)
    end
    return Collection.new(col_handle, dimension)
end

function LibraVDB:get_collection(name, dimension)
    local col_handle = lib.GetCollection(self.handle, name)
    if col_handle < 0 then
        error("Failed to get collection: " .. name)
    end
    return Collection.new(col_handle, dimension)
end

function LibraVDB:query(sql)
    local res_ptr = lib.DatabaseQuery(self.handle, sql)
    return parse_query_result(res_ptr, "Query")
end

function LibraVDB:query_with_params(sql, params)
    local params_str = params and json.encode(params) or ""
    local res_ptr = lib.DatabaseQueryWithParams(self.handle, sql, params_str)
    return parse_query_result(res_ptr, "QueryWithParams")
end

function LibraVDB:close()
    if self.handle >= 0 then
        lib.CloseDB(self.handle)
        self.handle = -1
    end
end

return {
    LibraVDB = LibraVDB,
    json = json
}
