package io.libravdb

import kotlinx.serialization.json.*
import kotlin.test.*

class IntegrationTest {

    private val dbPath = "./demo_db_kotlin"

    @BeforeTest
    fun setUp() {
        // Clean up before test if needed. In a real shell we'd rm -rf.
    }

    @AfterTest
    fun tearDown() {
        // Clean up after test
    }

    @Test
    fun testFullIntegration() {
        println("Initializing LibraVDB Native...")
        val db = LibraVDB(dbPath)

        println("Testing Ping...")
        db.ping()

        println("Setting Memory Limit...")
        db.setMemoryLimit(10 * 1024 * 1024)

        println("Creating Collection docs...")
        val col = db.createCollection("docs", 3)

        val collections = db.listCollections()
        assertTrue(collections.contains("docs"))

        println("Testing InsertBatch with 1000 vectors...")
        val ids = mutableListOf<String>()
        val vectors = mutableListOf<FloatArray>()
        val metadata = mutableListOf<JsonObject>()

        for (i in 0 until 1000) {
            ids.add("vec_$i")
            vectors.add(floatArrayOf(0.1f, 0.2f, 0.3f))
            metadata.add(buildJsonObject {
                put("source", "kotlin_test")
                put("index", i)
                put("active", i % 2 == 0)
            })
        }

        col.insertBatch(ids, vectors, metadata)

        println("Testing Update...")
        col.update("vec_0", floatArrayOf(1.0f, 1.0f, 1.0f), buildJsonObject { put("updated", true) })

        println("Testing Get...")
        val rec = col.get("vec_0")
        assertNotNull(rec)
        assertTrue(rec is JsonObject)
        assertEquals("vec_0", rec["id"]?.jsonPrimitive?.content)

        println("Testing Search...")
        val filter = Filter.eq("active", true)
        val results = col.search(floatArrayOf(1.0f, 1.0f, 1.0f), 5, filter)
        assertNotNull(results)
        assertTrue(results is JsonArray)
        assertTrue(results.size > 0)

        println("Testing Scan...")
        val scanned = col.scan(0, 2)
        assertNotNull(scanned)
        assertTrue(scanned is JsonArray)
        assertEquals(2, scanned.size)

        println("Testing DeleteBatch...")
        col.deleteBatch(ids.subList(0, 500))

        val count = col.count()
        assertEquals(500, count)

        println("Testing Vacuum...")
        db.vacuum()

        println("Testing DropDatabase...")
        db.dropDatabase()
        
        db.close()

        println("Integration Test Passed!")
    }
}
