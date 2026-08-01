import XCTest
@testable import LibraVDB

final class LibraVDBTests: XCTestCase {
    let dbPath = "./test_db_\(UUID().uuidString)"

    override func setUpWithError() throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: dbPath) {
            try fm.removeItem(atPath: dbPath)
        }
    }

    override func tearDownWithError() throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: dbPath) {
            try fm.removeItem(atPath: dbPath)
        }
    }

    func testEndToEnd() throws {
        let db = try Database(path: dbPath)
        let col = try db.createCollection(name: "test_col", dimension: 3)

        // Insert
        try col.insert(id: "1", vector: [1.0, 2.0, 3.0], metadata: ["category": "A"])

        // Search
        let filter = Filter.eq("category", "A")
        let searchResults = try col.search(vector: [1.0, 2.0, 3.0], k: 10, filter: filter)
        XCTAssertTrue(searchResults.contains("\"id\":\"1\""), "Search results should contain the inserted vector")

        // Batch Insert
        try col.insertBatch(
            ids: ["2", "3"],
            vectors: [
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]
            ],
            metadata: [
                ["category": "B"],
                ["category": "C"]
            ]
        )

        // Scan
        var scanRes = try col.scan()
        XCTAssertTrue(scanRes.contains("\"id\":\"1\""))
        XCTAssertTrue(scanRes.contains("\"id\":\"2\""))
        XCTAssertTrue(scanRes.contains("\"id\":\"3\""))

        // Delete
        try col.delete(id: "2")
        scanRes = try col.scan()
        XCTAssertFalse(scanRes.contains("\"id\":\"2\""))

        // Vacuum
        try db.vacuum()
    }
}
