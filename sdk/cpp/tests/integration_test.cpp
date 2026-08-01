#include "libravdb.hpp"
#include <iostream>
#include <cassert>
#include <filesystem>

using namespace libravdb;
namespace fs = std::filesystem;

int main() {
    std::string db_path = "./demo_db_cpp";
    if (fs::exists(db_path)) {
        fs::remove_all(db_path);
    }

    try {
        std::cout << "Initializing LibraVDB at " << db_path << "..." << std::endl;
        LibraVDB db(db_path);

        std::cout << "Database Ping OK." << std::endl;
        db.ping();

        std::cout << "Setting global memory limit..." << std::endl;
        db.set_memory_limit(10 * 1024 * 1024);

        std::cout << "Creating collection 'docs' (dim: 3)..." << std::endl;
        Collection col = db.create_collection("docs", 3);

        auto collections = db.list_collections();
        std::cout << "List collections: [" << (collections.empty() ? "" : collections[0]) << "]" << std::endl;
        assert(collections.size() == 1 && collections[0] == "docs");

        std::cout << "Testing InsertBatch with 1000 vectors..." << std::endl;
        std::vector<std::string> ids;
        std::vector<std::vector<float>> vectors;
        std::vector<json> metadata;

        for (int i = 0; i < 1000; ++i) {
            ids.push_back("vec_" + std::to_string(i));
            vectors.push_back({0.1f, 0.2f, 0.3f});
            metadata.push_back({
                {"source", "cpp_test"},
                {"index", i},
                {"active", (i % 2 == 0)}
            });
        }

        col.insert_batch(ids, vectors, metadata);
        std::cout << "Batch Insert Complete!" << std::endl;

        std::cout << "Testing Update..." << std::endl;
        col.update("vec_0", {1.0f, 1.0f, 1.0f}, json{{"updated", true}});

        std::cout << "Testing Get..." << std::endl;
        json rec = col.get("vec_0");
        std::cout << "Got record: " << rec.dump() << std::endl;
        assert(rec["id"] == "vec_0");

        std::cout << "Testing Search with Filters (active = true)..." << std::endl;
        Filter filter = Filter::eq("active", true);
        json results = col.search({1.0f, 1.0f, 1.0f}, 5, filter);
        std::cout << "Search Results: " << results.dump() << std::endl;
        assert(results.is_array() && !results.empty());

        std::cout << "Testing Scan (Offset 0, Limit 2)..." << std::endl;
        json scanned = col.scan(0, 2);
        std::cout << "Scanned: " << scanned.dump() << std::endl;
        assert(scanned.is_array() && scanned.size() == 2);

        std::cout << "Testing DeleteBatch (Deleting 500 vectors)..." << std::endl;
        std::vector<std::string> del_ids(ids.begin(), ids.begin() + 500);
        col.delete_batch(del_ids);

        int64_t count = col.count();
        std::cout << "Collection Count: " << count << std::endl;
        assert(count == 500);

        std::cout << "Testing Vacuum..." << std::endl;
        db.vacuum();

        std::cout << "Dropping database..." << std::endl;
        db.drop_db();
        
        assert(!fs::exists(db_path));
        std::cout << "Database successfully dropped!" << std::endl;

    } catch (const LibraException& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
