using System;
using System.IO;
using Xunit;

namespace LibraVDB.Tests
{
    public class TestSQL : IDisposable
    {
        private string _dbPath = "demo_db_sql_csharp";
        private Database _db;

        public TestSQL()
        {
            if (Directory.Exists(_dbPath))
            {
                Directory.Delete(_dbPath, true);
            }
            Console.WriteLine($"Initializing LibraVDB at {_dbPath}...");
            _db = new Database(_dbPath);
        }

        public void Dispose()
        {
            _db?.DropDatabase();
            _db?.Dispose();
        }

        [Fact]
        public void TestUnifiedSQL()
        {
            // Create tables
            _db.Query("CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))");
            _db.Query("CREATE EDGE TYPE FOLLOWS");

            // 1. Relational
            Console.WriteLine("\n--- Relational SQL ---");
            _db.QueryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
                "{\"1\": \"u1\", \"2\": \"Alice\", \"3\": [1.0, 0.0, 0.0]}");
            _db.QueryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
                "{\"1\": \"u2\", \"2\": \"Bob\", \"3\": [0.0, 1.0, 0.0]}");
            _db.QueryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
                "{\"1\": \"u3\", \"2\": \"Charlie\", \"3\": [0.0, 0.0, 1.0]}");

            string resRel = _db.Query("SELECT id, name FROM users ORDER BY name ASC");
            Console.WriteLine("Relational Result: " + resRel);

            // 2. Vector
            Console.WriteLine("\n--- Vector SQL ---");
            string resVec = _db.QueryWithParams("SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, $vec) ASC LIMIT 2",
                "{\"vec\": [1.0, 0.0, 0.0]}");
            Console.WriteLine("Vector Result: " + resVec);

            // 3. Graph
            Console.WriteLine("\n--- Graph SQL ---");
            _db.QueryWithParams("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
                "{\"1\": \"u1\", \"2\": \"FOLLOWS\", \"3\": \"u2\"}");
            _db.QueryWithParams("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
                "{\"1\": \"u2\", \"2\": \"FOLLOWS\", \"3\": \"u3\"}");

            string resGraph = _db.QueryWithParams("SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = $1",
                "{\"1\": \"u1\"}");
            Console.WriteLine("Graph Result: " + resGraph);

            // 4. Temporal SQL
            Console.WriteLine("\n--- Temporal SQL ---");
            DateTime futureTime = DateTime.UtcNow.AddSeconds(2);
            string cutoff = futureTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ");

            string resTemp = _db.Query($"SELECT id FROM users AS OF TIMESTAMP '{cutoff}' ORDER BY id ASC");
            Console.WriteLine("Temporal Result: " + resTemp);

            Console.WriteLine("\nAll unified SQL tests passed successfully.");
        }
    }
}
