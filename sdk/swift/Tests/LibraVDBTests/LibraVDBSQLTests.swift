import XCTest
@testable import LibraVDB
import Foundation

final class LibraVDBSQLTests: XCTestCase {

    func cleanup(path: String) {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: path) {
            try? fileManager.removeItem(atPath: path)
        }
    }

    func testUnifiedSQL() throws {
        let dbPath = "demo_db_sql_swift"
        cleanup(path: dbPath)

        print("Initializing LibraVDB at \(dbPath)...")
        let db = try Database(path: dbPath)
        defer {
            try? db.dropDatabase()
        }

        // Create tables
        _ = try db.query(sql: "CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))")
        _ = try db.query(sql: "CREATE EDGE TYPE FOLLOWS")

        // 1. Relational
        print("\n--- Relational SQL ---")
        _ = try db.queryWithParams(sql: "INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
            params: "{\"1\": \"u1\", \"2\": \"Alice\", \"3\": [1.0, 0.0, 0.0]}")
        _ = try db.queryWithParams(sql: "INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
            params: "{\"1\": \"u2\", \"2\": \"Bob\", \"3\": [0.0, 1.0, 0.0]}")
        _ = try db.queryWithParams(sql: "INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
            params: "{\"1\": \"u3\", \"2\": \"Charlie\", \"3\": [0.0, 0.0, 1.0]}")

        let resRel = try db.query(sql: "SELECT id, name FROM users ORDER BY name ASC")
        print("Relational Result: \(resRel)")

        // 2. Vector
        print("\n--- Vector SQL ---")
        let resVec = try db.queryWithParams(sql: "SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, $vec) ASC LIMIT 2",
            params: "{\"vec\": [1.0, 0.0, 0.0]}")
        print("Vector Result: \(resVec)")

        // 3. Graph
        print("\n--- Graph SQL ---")
        _ = try db.queryWithParams(sql: "INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
            params: "{\"1\": \"u1\", \"2\": \"FOLLOWS\", \"3\": \"u2\"}")
        _ = try db.queryWithParams(sql: "INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
            params: "{\"1\": \"u2\", \"2\": \"FOLLOWS\", \"3\": \"u3\"}")
        let resGraph = try db.queryWithParams(sql: "SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = $1",
            params: "{\"1\": \"u1\"}")
        print("Graph Result: \(resGraph)")

        // 4. Temporal SQL
        print("\n--- Temporal SQL ---")
        let futureDate = Date().addingTimeInterval(2)
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let cutoff = formatter.string(from: futureDate)

        let resTemp = try db.query(sql: "SELECT id FROM users AS OF TIMESTAMP '\(cutoff)' ORDER BY id ASC")
        print("Temporal Result: \(resTemp)")

        print("\nAll unified SQL tests passed successfully.")
    }
}
