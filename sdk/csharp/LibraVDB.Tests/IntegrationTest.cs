using System;
using System.IO;
using System.Text.Json;
using Xunit;
using LibraVDB;

namespace LibraVDB.Tests
{
    public class IntegrationTest : IDisposable
    {
        private readonly string _dbPath = $"./test_db_{Guid.NewGuid()}";

        public IntegrationTest()
        {
            if (Directory.Exists(_dbPath))
            {
                Directory.Delete(_dbPath, true);
            }
        }

        public void Dispose()
        {
            if (Directory.Exists(_dbPath))
            {
                Directory.Delete(_dbPath, true);
            }
        }

        [Fact]
        public void TestEndToEnd()
        {
            using (var db = new Database(_dbPath))
            {
                var col = db.CreateCollection("test_col", 3);

                // Insert
                col.Insert("1", new float[] { 1.0f, 2.0f, 3.0f }, new { category = "A" });

                // Query
                string results = col.Search(new float[] { 1.0f, 2.0f, 3.0f }, 10, Filter.Eq("category", "A"));
                Assert.Contains("\"id\":\"1\"", results);

                // Insert Batch
                string[] batchIds = new string[] { "2", "3" };
                float[][] batchVectors = new float[][] {
                    new float[] { 4.0f, 5.0f, 6.0f },
                    new float[] { 7.0f, 8.0f, 9.0f }
                };
                object[] batchMeta = new object[] {
                    new { category = "B" },
                    new { category = "C" }
                };

                col.InsertBatch(batchIds, batchVectors, batchMeta);

                // Scan
                string scanRes = col.Scan();
                Assert.Contains("\"id\":\"1\"", scanRes);
                Assert.Contains("\"id\":\"2\"", scanRes);
                Assert.Contains("\"id\":\"3\"", scanRes);

                // Delete
                col.Delete("2");
                scanRes = col.Scan();
                Assert.DoesNotContain("\"id\":\"2\"", scanRes);

                // Vacuum
                db.Vacuum();
            }
        }
    }
}
