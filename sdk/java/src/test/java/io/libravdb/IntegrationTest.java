package io.libravdb;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

public class IntegrationTest {
    private static final String DB_PATH = "./demo_db_java";
    private static LibraVDB db;
    private static final ObjectMapper mapper = new ObjectMapper();

    @BeforeAll
    public static void setup() {
        File dir = new File(DB_PATH);
        if (dir.exists()) {
            deleteDirectory(dir);
        }
        db = new LibraVDB(DB_PATH);
    }

    @AfterAll
    public static void teardown() {
        if (db != null) {
            db.dropDatabase();
            db.close();
        }
    }

    private static void deleteDirectory(File dir) {
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                deleteDirectory(file);
            }
        }
        dir.delete();
    }

    @Test
    public void testFullIntegration() throws Exception {
        System.out.println("Testing Ping...");
        db.ping();

        System.out.println("Setting Memory Limit...");
        db.setMemoryLimit(10 * 1024 * 1024);

        System.out.println("Creating Collection 'docs'...");
        Collection col = db.createCollection("docs", 3);

        List<String> collections = db.listCollections();
        assertTrue(collections.contains("docs"));

        System.out.println("Testing InsertBatch with 1000 vectors...");
        List<String> ids = new ArrayList<>();
        List<float[]> vectors = new ArrayList<>();
        List<JsonNode> metadata = new ArrayList<>();

        for (int i = 0; i < 1000; i++) {
            ids.add("vec_" + i);
            vectors.add(new float[]{0.1f, 0.2f, 0.3f});
            metadata.add(mapper.readTree("{\"source\":\"java_test\",\"index\":" + i + ",\"active\":" + (i % 2 == 0) + "}"));
        }

        col.insertBatch(ids, vectors, Optional.of(metadata));

        System.out.println("Testing Update...");
        col.update("vec_0", new float[]{1.0f, 1.0f, 1.0f}, Optional.of(mapper.readTree("{\"updated\":true}")));

        System.out.println("Testing Get...");
        JsonNode rec = col.get("vec_0");
        assertNotNull(rec);
        assertEquals("vec_0", rec.get("id").asText());

        System.out.println("Testing Search...");
        Filter filter = Filter.eq("active", true);
        JsonNode results = col.search(new float[]{1.0f, 1.0f, 1.0f}, 5, Optional.of(filter));
        assertTrue(results.isArray());
        assertTrue(results.size() > 0);

        System.out.println("Testing Scan...");
        JsonNode scanned = col.scan(0, 2);
        assertTrue(scanned.isArray());
        assertEquals(2, scanned.size());

        System.out.println("Testing DeleteBatch...");
        col.deleteBatch(ids.subList(0, 500));

        long count = col.count();
        assertEquals(500, count);

        System.out.println("Testing Vacuum...");
        db.vacuum();
        
        System.out.println("Integration Test Passed!");
    }
}
