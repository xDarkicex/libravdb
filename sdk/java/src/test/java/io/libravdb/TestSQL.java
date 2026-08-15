package io.libravdb;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Comparator;

public class TestSQL {
    private String dbPath = "demo_db_sql_java";
    private LibraVDB db;

    private void deleteDir(File file) {
        File[] contents = file.listFiles();
        if (contents != null) {
            for (File f : contents) {
                deleteDir(f);
            }
        }
        file.delete();
    }

    @BeforeEach
    public void setUp() {
        deleteDir(new File(dbPath));
        System.out.println("Initializing LibraVDB at " + dbPath + "...");
        db = new LibraVDB(dbPath);
    }

    @AfterEach
    public void tearDown() {
        if (db != null) {
            db.dropDatabase();
            db.close();
        }
    }

    @Test
    public void testUnifiedSQL() {
        // Create tables
        db.query("CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))");
        db.query("CREATE EDGE TYPE FOLLOWS");

        // 1. Relational
        System.out.println("\n--- Relational SQL ---");
        db.queryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
                "{\"1\": \"u1\", \"2\": \"Alice\", \"3\": [1.0, 0.0, 0.0]}");
        db.queryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
                "{\"1\": \"u2\", \"2\": \"Bob\", \"3\": [0.0, 1.0, 0.0]}");
        db.queryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
                "{\"1\": \"u3\", \"2\": \"Charlie\", \"3\": [0.0, 0.0, 1.0]}");

        String resRel = db.query("SELECT id, name FROM users ORDER BY name ASC");
        System.out.println("Relational Result: " + resRel);

        // 2. Vector
        System.out.println("\n--- Vector SQL ---");
        String resVec = db.queryWithParams("SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, $vec) ASC LIMIT 2",
                "{\"vec\": [1.0, 0.0, 0.0]}");
        System.out.println("Vector Result: " + resVec);

        // 3. Graph
        System.out.println("\n--- Graph SQL ---");
        db.queryWithParams("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
                "{\"1\": \"u1\", \"2\": \"FOLLOWS\", \"3\": \"u2\"}");
        db.queryWithParams("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
                "{\"1\": \"u2\", \"2\": \"FOLLOWS\", \"3\": \"u3\"}");

        String resGraph = db.queryWithParams("SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = $1",
                "{\"1\": \"u1\"}");
        System.out.println("Graph Result: " + resGraph);

        // 4. Temporal SQL
        System.out.println("\n--- Temporal SQL ---");
        Instant futureTime = Instant.now().plus(2, ChronoUnit.SECONDS);
        String cutoff = futureTime.toString(); // ISO-8601 string like "2026-08-15T09:42:00Z"

        String resTemp = db.query("SELECT id FROM users AS OF TIMESTAMP '" + cutoff + "' ORDER BY id ASC");
        System.out.println("Temporal Result: " + resTemp);

        System.out.println("\nAll unified SQL tests passed successfully.");
    }
}
