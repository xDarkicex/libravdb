require_relative '../lib/libravdb'
require 'json'
require 'fileutils'
require 'time'

def cleanup(path)
  FileUtils.rm_rf(path) if Dir.exist?(path) || File.exist?(path)
end

db_path = "demo_db_sql_ruby"
cleanup(db_path)

puts "Initializing LibraVDB at #{db_path}..."
db = LibraVDB::Client.new(db_path)

begin
  # Create tables
  db.query("CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))")
  db.query("CREATE EDGE TYPE FOLLOWS")

  # 1. Relational
  puts "\n--- Relational SQL ---"
  db.query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
                       {"1" => "u1", "2" => "Alice", "3" => [1.0, 0.0, 0.0]}.to_json)
  db.query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
                       {"1" => "u2", "2" => "Bob", "3" => [0.0, 1.0, 0.0]}.to_json)
  db.query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
                       {"1" => "u3", "2" => "Charlie", "3" => [0.0, 0.0, 1.0]}.to_json)

  res_rel = db.query("SELECT id, name FROM users ORDER BY name ASC")
  puts "Relational Result: #{res_rel}"

  # 2. Vector
  puts "\n--- Vector SQL ---"
  res_vec = db.query_with_params("SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, $vec) ASC LIMIT 2",
                                 {"vec" => [1.0, 0.0, 0.0]}.to_json)
  puts "Vector Result: #{res_vec}"

  # 3. Graph
  puts "\n--- Graph SQL ---"
  db.query_with_params("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
                       {"1" => "u1", "2" => "FOLLOWS", "3" => "u2"}.to_json)
  db.query_with_params("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
                       {"1" => "u2", "2" => "FOLLOWS", "3" => "u3"}.to_json)

  res_graph = db.query_with_params("SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = $1",
                                   {"1" => "u1"}.to_json)
  puts "Graph Result: #{res_graph}"

  # 4. Temporal SQL
  puts "\n--- Temporal SQL ---"
  future_time = Time.now.utc + 2
  cutoff = future_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')

  res_temp = db.query("SELECT id FROM users AS OF TIMESTAMP '#{cutoff}' ORDER BY id ASC")
  puts "Temporal Result: #{res_temp}"

  puts "\nAll unified SQL tests passed successfully."
ensure
  db.drop
end
