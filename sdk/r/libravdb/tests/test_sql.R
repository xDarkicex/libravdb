# Wait to ensure we are testing the compiled C-shared FFI
library(jsonlite)
library(libravdb)

db_path <- "demo_db_r"
unlink(db_path, recursive = TRUE)

cat("Initializing LibraVDB at", db_path, "...\n")
db <- LibraVDB$new(db_path)

# Test unified SQL
db$query("CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))")
db$query("CREATE EDGE TYPE FOLLOWS")

cat("\n--- Relational SQL ---\n")
db$query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
    list(`1`="u1", `2`="Alice", `3`=c(1.0, 0.0, 0.0)))
db$query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
    list(`1`="u2", `2`="Bob", `3`=c(0.0, 1.0, 0.0)))
db$query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
    list(`1`="u3", `2`="Charlie", `3`=c(0.0, 0.0, 1.0)))

res_rel <- db$query("SELECT id, name FROM users ORDER BY name ASC")
cat("Relational Result:\n")
print(res_rel)

cat("\n--- Vector SQL ---\n")
res_vec <- db$query_with_params("SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, $vec) ASC LIMIT 2",
    list(vec=c(1.0, 0.0, 0.0)))
cat("Vector Result:\n")
print(res_vec)

cat("\n--- Graph SQL ---\n")
db$query_with_params("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
    list(`1`="u1", `2`="FOLLOWS", `3`="u2"))
db$query_with_params("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
    list(`1`="u2", `2`="FOLLOWS", `3`="u3"))

res_graph <- db$query_with_params("SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = $1",
    list(`1`="u1"))
cat("Graph Result:\n")
print(res_graph)

cat("\n--- Temporal SQL ---\n")
# Future timestamp
cutoff <- format(Sys.time() + 2, "%Y-%m-%dT%H:%M:%S.000Z", tz="UTC")
query_str <- sprintf("SELECT id FROM users AS OF TIMESTAMP '%s' ORDER BY id ASC", cutoff)
res_temp <- db$query(query_str)
cat("Temporal Result:\n")
print(res_temp)

cat("\nAll unified SQL tests passed successfully.\n")

db$drop_database()
db$close()
