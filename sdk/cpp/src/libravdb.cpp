#include "libravdb.hpp"
#include <iostream>

namespace libravdb {

// Helper to safely convert C string to C++ string and free the C string
static std::optional<std::string> from_c_string(char* ptr) {
    if (!ptr) return std::nullopt;
    std::string str(ptr);
    FreeString(ptr);
    return str;
}

// ==========================================
// Filter
// ==========================================

Filter Filter::eq(const std::string& field, const json& value) {
    return Filter("eq", field, value);
}

Filter Filter::neq(const std::string& field, const json& value) {
    return Filter("neq", field, value);
}

Filter Filter::gt(const std::string& field, const json& value) {
    return Filter("gt", field, value);
}

Filter Filter::AND(const std::vector<Filter>& filters) {
    json arr = json::array();
    for (const auto& f : filters) arr.push_back(f.to_json());
    return Filter("and", std::nullopt, arr);
}

Filter Filter::OR(const std::vector<Filter>& filters) {
    json arr = json::array();
    for (const auto& f : filters) arr.push_back(f.to_json());
    return Filter("or", std::nullopt, arr);
}

json Filter::to_json() const {
    json j;
    j["type"] = type;
    if (field) j["field"] = *field;
    if (value) j["value"] = *value;
    return j;
}

// ==========================================
// Collection
// ==========================================

Collection::Collection(int handle, int dim) : handle(handle), dim(dim) {}

void Collection::check_error(char* err_ptr, const std::string& op_name) const {
    auto err_msg = from_c_string(err_ptr);
    if (err_msg) {
        if (err_msg->rfind("error: ", 0) == 0) {
            throw LibraException(op_name + " failed: " + err_msg->substr(7));
        }
        throw LibraException(op_name + " failed: " + *err_msg);
    }
}

json Collection::parse_result(char* res_ptr, const std::string& op_name) const {
    auto msg = from_c_string(res_ptr);
    if (!msg || msg->empty()) {
        return json(nullptr);
    }

    if (msg->rfind("{\"error\"", 0) == 0) {
        try {
            auto j = json::parse(*msg);
            throw LibraException(op_name + " failed: " + j.value("error", "Unknown error"));
        } catch (const json::parse_error&) {
            throw LibraException(op_name + " failed: " + *msg);
        }
    }

    try {
        return json::parse(*msg);
    } catch (const json::parse_error& e) {
        throw LibraException(std::string("JSON Parse error: ") + e.what());
    }
}

void Collection::insert(const std::string& id, const std::vector<float>& vector, const std::optional<json>& metadata) const {
    if (vector.size() != static_cast<size_t>(dim)) {
        throw LibraException("Vector dimension mismatch");
    }

    std::string meta_str = metadata ? metadata->dump() : "";
    char* err_ptr = InsertVector(
        handle,
        const_cast<char*>(id.c_str()),
        const_cast<float*>(vector.data()),
        dim,
        const_cast<char*>(meta_str.c_str())
    );
    check_error(err_ptr, "Insert");
}

void Collection::update(const std::string& id, const std::vector<float>& vector, const std::optional<json>& metadata) const {
    std::string meta_str = metadata ? metadata->dump() : "";
    char* err_ptr = UpdateVector(
        handle,
        const_cast<char*>(id.c_str()),
        const_cast<float*>(vector.data()),
        dim,
        const_cast<char*>(meta_str.c_str())
    );
    check_error(err_ptr, "Update");
}

void Collection::update_if_version(const std::string& id, const std::vector<float>& vector, uint64_t expected_version, const std::optional<json>& metadata) const {
    std::string meta_str = metadata ? metadata->dump() : "";
    char* err_ptr = UpdateVectorIfVersion(
        handle,
        const_cast<char*>(id.c_str()),
        const_cast<float*>(vector.data()),
        dim,
        const_cast<char*>(meta_str.c_str()),
        expected_version
    );
    check_error(err_ptr, "UpdateIfVersion");
}

json Collection::get(const std::string& id) const {
    char* res_ptr = GetVector(handle, const_cast<char*>(id.c_str()));
    return parse_result(res_ptr, "Get");
}

json Collection::search(const std::vector<float>& vector, int k, const std::optional<Filter>& filter) const {
    std::string filter_str = filter ? filter->to_json().dump() : "";
    char* res_ptr = QueryVector(
        handle,
        const_cast<float*>(vector.data()),
        dim,
        k,
        const_cast<char*>(filter_str.c_str())
    );
    return parse_result(res_ptr, "Search");
}

json Collection::scan(int offset, int limit) const {
    char* res_ptr = ScanCollection(handle, offset, limit);
    return parse_result(res_ptr, "Scan");
}

void Collection::insert_batch(const std::vector<std::string>& ids, const std::vector<std::vector<float>>& vectors, const std::optional<std::vector<json>>& metadata) const {
    if (ids.size() != vectors.size()) {
        throw LibraException("ids and vectors must have the same length");
    }

    std::vector<float> flat_vectors;
    flat_vectors.reserve(ids.size() * dim);
    for (const auto& vec : vectors) {
        if (vec.size() != static_cast<size_t>(dim)) {
            throw LibraException("Vector dimension mismatch");
        }
        flat_vectors.insert(flat_vectors.end(), vec.begin(), vec.end());
    }

    std::vector<char*> c_ids;
    c_ids.reserve(ids.size());
    for (const auto& id : ids) c_ids.push_back(const_cast<char*>(id.c_str()));

    std::vector<std::string> meta_strings;
    std::vector<char*> c_metas;
    char** meta_ptr = nullptr;

    if (metadata) {
        if (metadata->size() != ids.size()) throw LibraException("ids and metadata length mismatch");
        meta_strings.reserve(metadata->size());
        c_metas.reserve(metadata->size());
        for (const auto& m : *metadata) {
            meta_strings.push_back(m.dump());
        }
        for (const auto& ms : meta_strings) {
            c_metas.push_back(const_cast<char*>(ms.c_str()));
        }
        meta_ptr = c_metas.data();
    }

    char* err_ptr = InsertBatch(
        handle,
        c_ids.data(),
        flat_vectors.data(),
        static_cast<int>(ids.size()),
        dim,
        meta_ptr
    );
    check_error(err_ptr, "InsertBatch");
}

void Collection::delete_batch(const std::vector<std::string>& ids) const {
    std::vector<char*> c_ids;
    c_ids.reserve(ids.size());
    for (const auto& id : ids) c_ids.push_back(const_cast<char*>(id.c_str()));

    char* err_ptr = DeleteBatch(handle, c_ids.data(), static_cast<int>(ids.size()));
    check_error(err_ptr, "DeleteBatch");
}

int64_t Collection::count() const {
    long long c = GetCollectionCount(handle);
    if (c < 0) throw LibraException("Failed to get collection count");
    return c;
}

void Collection::enable_memory_mapping(const std::string& path) const {
    char* err_ptr = EnableMemoryMapping(handle, const_cast<char*>(path.c_str()));
    check_error(err_ptr, "EnableMemoryMapping");
}


// ==========================================
// LibraVDB
// ==========================================

LibraVDB::LibraVDB(const std::string& path) {
    handle = OpenDB(const_cast<char*>(path.c_str()));
    if (handle < 0) {
        throw LibraException("Failed to open database at " + path);
    }
}

LibraVDB::~LibraVDB() {
    if (handle >= 0) {
        CloseDB(handle);
    }
}

LibraVDB::LibraVDB(LibraVDB&& other) noexcept : handle(other.handle) {
    other.handle = -1;
}

LibraVDB& LibraVDB::operator=(LibraVDB&& other) noexcept {
    if (this != &other) {
        if (handle >= 0) CloseDB(handle);
        handle = other.handle;
        other.handle = -1;
    }
    return *this;
}

void LibraVDB::ping() const {
    char* err_ptr = Ping(handle);
    if (from_c_string(err_ptr)) {
        throw LibraException("Ping failed");
    }
}

void LibraVDB::set_memory_limit(int64_t limit) const {
    char* err_ptr = SetGlobalMemoryLimit(handle, limit);
    if (from_c_string(err_ptr)) {
        throw LibraException("Set memory limit failed");
    }
}

void LibraVDB::vacuum() const {
    char* err_ptr = Vacuum(handle);
    if (from_c_string(err_ptr)) {
        throw LibraException("Vacuum failed");
    }
}

void LibraVDB::drop_db() const {
    char* err_ptr = DropDatabase(handle);
    if (from_c_string(err_ptr)) {
        throw LibraException("Drop database failed");
    }
}

std::vector<std::string> LibraVDB::list_collections() const {
    char* res_ptr = ListCollections(handle);
    auto msg = from_c_string(res_ptr);
    if (msg) {
        try {
            auto j = json::parse(*msg);
            return j.get<std::vector<std::string>>();
        } catch (...) {
            return {};
        }
    }
    return {};
}

Collection LibraVDB::create_collection(const std::string& name, int dimension) const {
    int col_handle = CreateCollection(handle, const_cast<char*>(name.c_str()), dimension);
    if (col_handle < 0) {
        throw LibraException("Failed to create collection " + name);
    }
    return Collection(col_handle, dimension);
}

Collection LibraVDB::get_collection(const std::string& name, int dimension) const {
    int col_handle = GetCollection(handle, const_cast<char*>(name.c_str()));
    if (col_handle < 0) {
        throw LibraException("Failed to get collection " + name);
    }
    return Collection(col_handle, dimension);
}

json LibraVDB::parse_query_result(char* res_ptr, const std::string& op_name) const {
    auto msg = from_c_string(res_ptr);
    if (!msg || msg->empty()) {
        return json(nullptr);
    }

    if (msg->rfind("{\"error\"", 0) == 0) {
        try {
            auto j = json::parse(*msg);
            throw LibraException(op_name + " failed: " + j.value("error", "Unknown error"));
        } catch (const json::parse_error&) {
            throw LibraException(op_name + " failed: " + *msg);
        }
    }

    try {
        return json::parse(*msg);
    } catch (const json::parse_error& e) {
        throw LibraException("JSON Parse error: " + std::string(e.what()));
    }
}

json LibraVDB::query(const std::string& sql) const {
    char* res_ptr = DatabaseQuery(handle, const_cast<char*>(sql.c_str()));
    return parse_query_result(res_ptr, "Query");
}

json LibraVDB::query_with_params(const std::string& sql, const std::optional<json>& params) const {
    std::string params_str = params ? params->dump() : "";
    char* res_ptr = DatabaseQueryWithParams(
        handle,
        const_cast<char*>(sql.c_str()),
        const_cast<char*>(params_str.c_str())
    );
    return parse_query_result(res_ptr, "QueryWithParams");
}

uint64_t LibraVDB::latest_commit_lsn() const {
    char* res_ptr = DatabaseLatestCommitLSN(handle);
    json result = parse_query_result(res_ptr, "LatestCommitLSN");
    if (!result.contains("lsn")) {
        throw LibraException("LatestCommitLSN failed: response did not contain lsn");
    }
    if (result.at("lsn").is_string()) {
        return std::stoull(result.at("lsn").get<std::string>());
    }
    return result.at("lsn").get<uint64_t>();
}

} // namespace libravdb
