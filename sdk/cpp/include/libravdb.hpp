#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <optional>
#include <nlohmann/json.hpp>
#include "libravdb.h"

namespace libravdb {

using json = nlohmann::json;

class LibraException : public std::runtime_error {
public:
    explicit LibraException(const std::string& message) : std::runtime_error(message) {}
};

class Filter {
public:
    std::string type;
    std::optional<std::string> field;
    std::optional<json> value;

    Filter(std::string t, std::optional<std::string> f, std::optional<json> v)
        : type(std::move(t)), field(std::move(f)), value(std::move(v)) {}

    static Filter eq(const std::string& field, const json& value);
    static Filter neq(const std::string& field, const json& value);
    static Filter gt(const std::string& field, const json& value);
    static Filter AND(const std::vector<Filter>& filters);
    static Filter OR(const std::vector<Filter>& filters);

    json to_json() const;
};

class Collection {
private:
    int handle;
    int dim;

    void check_error(char* err_ptr, const std::string& op_name) const;
    json parse_result(char* res_ptr, const std::string& op_name) const;

public:
    Collection(int handle, int dim);

    void insert(const std::string& id, const std::vector<float>& vector, const std::optional<json>& metadata = std::nullopt) const;
    void update(const std::string& id, const std::vector<float>& vector, const std::optional<json>& metadata = std::nullopt) const;
    void update_if_version(const std::string& id, const std::vector<float>& vector, uint64_t expected_version, const std::optional<json>& metadata = std::nullopt) const;
    
    json get(const std::string& id) const;
    json search(const std::vector<float>& vector, int k, const std::optional<Filter>& filter = std::nullopt) const;
    json scan(int offset, int limit) const;
    
    void insert_batch(const std::vector<std::string>& ids, const std::vector<std::vector<float>>& vectors, const std::optional<std::vector<json>>& metadata = std::nullopt) const;
    void delete_batch(const std::vector<std::string>& ids) const;
    
    int64_t count() const;
    void enable_memory_mapping(const std::string& path) const;
};

class LibraVDB {
private:
    int handle;

public:
    explicit LibraVDB(const std::string& path);
    ~LibraVDB();

    // Prevent copying because of the C-handle
    LibraVDB(const LibraVDB&) = delete;
    LibraVDB& operator=(const LibraVDB&) = delete;
    
    LibraVDB(LibraVDB&& other) noexcept;
    LibraVDB& operator=(LibraVDB&& other) noexcept;

    void ping() const;
    void set_memory_limit(int64_t limit) const;
    void vacuum() const;
    void drop_db() const;
    
    std::vector<std::string> list_collections() const;
    Collection create_collection(const std::string& name, int dimension) const;
    Collection get_collection(const std::string& name, int dimension) const;
};

} // namespace libravdb
