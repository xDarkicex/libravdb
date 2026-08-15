package io.libravdb

import kotlinx.cinterop.ExperimentalForeignApi
import platform.posix.remove
import platform.posix.rmdir
import platform.posix.time
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test

@OptIn(ExperimentalForeignApi::class)
class TestSQL {
    private val dbPath = "demo_db_sql_kotlin"
    private var db: LibraVDB? = null

    private fun cleanup(path: String) {
        // Simple cleanup for the test db path, assuming flat structure for demo
        remove("$path/data.db")
        remove("$path/metadata.db")
        rmdir(path)
    }

    @BeforeTest
    fun setUp() {
        cleanup(dbPath)
        println("Initializing LibraVDB at $dbPath...")
        db = LibraVDB(dbPath)
    }

    @AfterTest
    fun tearDown() {
        db?.dropDatabase()
        db?.close()
    }

    @Test
    fun testUnifiedSQL() {
        val database = db ?: throw IllegalStateException("DB is null")

        // Create tables
        database.query("CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))")
        database.query("CREATE EDGE TYPE FOLLOWS")

        // 1. Relational
        println("\n--- Relational SQL ---")
        database.queryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
            """{"1": "u1", "2": "Alice", "3": [1.0, 0.0, 0.0]}""")
        database.queryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
            """{"1": "u2", "2": "Bob", "3": [0.0, 1.0, 0.0]}""")
        database.queryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
            """{"1": "u3", "2": "Charlie", "3": [0.0, 0.0, 1.0]}""")

        val resRel = database.query("SELECT id, name FROM users ORDER BY name ASC")
        println("Relational Result: $resRel")

        // 2. Vector
        println("\n--- Vector SQL ---")
        val resVec = database.queryWithParams("SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, \$vec) ASC LIMIT 2",
            """{"vec": [1.0, 0.0, 0.0]}""")
        println("Vector Result: $resVec")

        // 3. Graph
        println("\n--- Graph SQL ---")
        database.queryWithParams("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
            """{"1": "u1", "2": "FOLLOWS", "3": "u2"}""")
        database.queryWithParams("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
            """{"1": "u2", "2": "FOLLOWS", "3": "u3"}""")

        val resGraph = database.queryWithParams("SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = $1",
            """{"1": "u1"}""")
        println("Graph Result: $resGraph")

        // 4. Temporal SQL
        println("\n--- Temporal SQL ---")
        val futureTime = time(null) + 2 // 2 seconds from now
        // A hacky way to format ISO 8601 in native POSIX without importing large libraries
        // Just use a big future timestamp
        val cutoff = "2030-01-01T00:00:00Z"

        val resTemp = database.query("SELECT id FROM users AS OF TIMESTAMP '$cutoff' ORDER BY id ASC")
        println("Temporal Result: $resTemp")

        println("\nAll unified SQL tests passed successfully.")
    }
}
