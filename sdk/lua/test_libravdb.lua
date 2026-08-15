local libra = require("libravdb")
local json = libra.json
local LibraVDB = libra.LibraVDB

os.execute("rm -rf demo_db_lua")
local db = LibraVDB.new("demo_db_lua")

print("Initializing LibraVDB at demo_db_lua...")

db:ping()
print("Ping successful")

-- Test SQL features directly
db:query("CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))")
db:query("CREATE EDGE TYPE FOLLOWS")

print("\n--- Relational SQL ---")
db:query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
    {["1"] = "u1", ["2"] = "Alice", ["3"] = {1.0, 0.0, 0.0}})
db:query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
    {["1"] = "u2", ["2"] = "Bob", ["3"] = {0.0, 1.0, 0.0}})
db:query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
    {["1"] = "u3", ["2"] = "Charlie", ["3"] = {0.0, 0.0, 1.0}})

local res_rel = db:query("SELECT id, name FROM users ORDER BY name ASC")
print("Relational Result: " .. json.encode(res_rel))

print("\n--- Vector SQL ---")
local res_vec = db:query_with_params("SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, $vec) ASC LIMIT 2",
    {vec = {1.0, 0.0, 0.0}})
print("Vector Result: " .. json.encode(res_vec))

print("\n--- Graph SQL ---")
db:query_with_params("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
    {["1"] = "u1", ["2"] = "FOLLOWS", ["3"] = "u2"})
db:query_with_params("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
    {["1"] = "u2", ["2"] = "FOLLOWS", ["3"] = "u3"})

local res_graph = db:query_with_params("SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = $1",
    {["1"] = "u1"})
print("Graph Result: " .. json.encode(res_graph))

print("\n--- Temporal SQL ---")
-- Format ISO 8601 for a future time
local res_temp = db:query("SELECT id FROM users AS OF TIMESTAMP '2030-01-01T00:00:00Z' ORDER BY id ASC")
print("Temporal Result: " .. json.encode(res_temp))

print("\nAll unified SQL tests passed successfully.")

db:drop_database()
db:close()
