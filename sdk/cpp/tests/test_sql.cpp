#include <iostream>
#include <chrono>
#include <filesystem>
#include "LibraVDB.hpp"

using namespace libravdb;

void cleanup(const std::string& path) {
    if (std::filesystem::exists(path)) {
        std::filesystem::remove_all(path);
    }
}

int main() {
    std::string db_path = "demo_db_sql_cpp";
    cleanup(db_path);

    std::cout << "Initializing LibraVDB at " << db_path << "...\n";
    LibraVDB db(db_path);

    // Create tables
    db.query("CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))");
    db.query("CREATE EDGE TYPE FOLLOWS");

    // 1. Relational
    std::cout << "\n--- Relational SQL ---\n";
    db.query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
        json{{"1", "u1"}, {"2", "Alice"}, {"3", {1.0, 0.0, 0.0}}});
    db.query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
        json{{"1", "u2"}, {"2", "Bob"}, {"3", {0.0, 1.0, 0.0}}});
    db.query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
        json{{"1", "u3"}, {"2", "Charlie"}, {"3", {0.0, 0.0, 1.0}}});

    auto res = db.query("SELECT id, name FROM users ORDER BY name ASC");
    std::cout << "Relational Result: " << res.dump() << "\n";

    // 2. Vector
    std::cout << "\n--- Vector SQL ---\n";
    res = db.query_with_params("SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, $vec) ASC LIMIT 2",
        json{{"vec", {1.0, 0.0, 0.0}}});
    std::cout << "Vector Result: " << res.dump() << "\n";

    // 3. Graph
    std::cout << "\n--- Graph SQL ---\n";
    db.query_with_params("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
        json{{"1", "u1"}, {"2", "FOLLOWS"}, {"3", "u2"}});
    db.query_with_params("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
        json{{"1", "u2"}, {"2", "FOLLOWS"}, {"3", "u3"}});
    res = db.query_with_params("SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = $1",
        json{{"1", "u1"}});
    std::cout << "Graph Result: " << res.dump() << "\n";

    // 4. Temporal SQL
    std::cout << "\n--- Temporal SQL ---\n";
    // Get future timestamp 2s from now
    auto future_time = std::chrono::system_clock::now() + std::chrono::seconds(2);
    std::time_t future_tt = std::chrono::system_clock::to_time_t(future_time);
    std::tm* gmtime = std::gmtime(&future_tt);
    char buf[128];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S.000Z", gmtime);
    std::string cutoff(buf);

    std::string query_str = "SELECT id FROM users AS OF TIMESTAMP '" + cutoff + "' ORDER BY id ASC";
    res = db.query(query_str);
    std::cout << "Temporal Result: " << res.dump() << "\n";

    std::cout << "\nAll unified SQL tests passed successfully.\n";
    return 0;
}
