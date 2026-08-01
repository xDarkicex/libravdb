require 'fileutils'
require_relative '../lib/libravdb'

db_path = "./demo_db_ruby"

FileUtils.rm_rf(db_path) if Dir.exist?(db_path)

puts "Initializing LibraVDB at #{db_path}..."
db = LibraVDB::Client.new(db_path)

puts "Database Ping OK: #{db.ping.nil?}"

puts "Setting global memory limit..."
db.set_memory_limit(10 * 1024 * 1024)

puts "Creating collection 'docs' (dim: 3)..."
col = db.create_collection("docs", 3)

puts "List collections: #{db.list_collections}"

puts "Testing InsertBatch with 1000 vectors..."
ids = []
vectors = []
metadata = []

1000.times do |i|
  ids << "vec_#{i}"
  vectors << [rand, rand, rand]
  metadata << { source: "ruby_test", index: i, active: i.even? }
end

col.insert_batch(ids, vectors, metadata)
puts "Batch Insert Complete!"

puts "Testing Update..."
col.update("vec_0", [1.0, 1.0, 1.0], { updated: true })

puts "Testing Get..."
rec = col.get("vec_0")
puts "Got record: #{rec}"

puts "Testing Search with Filters (active = true)..."
filter = LibraVDB::Filter.eq("active", true)
results = col.search([1.0, 1.0, 1.0], 5, filter)
puts "Search Results: #{results.empty? ? 'No matches' : 'Found matches!'}"

puts "Testing Scan (Offset 0, Limit 2)..."
scanned = col.scan(0, 2)
puts "Scanned: #{scanned.map { |s| s['id'] }}"

puts "Testing DeleteBatch (Deleting 500 vectors)..."
col.delete_batch(ids[0...500])

puts "Collection Count: #{col.count}"

puts "Testing Vacuum..."
db.vacuum

puts "Database Health: #{db.health['status']}"

puts "Dropping database..."
db.drop

if !Dir.exist?(db_path)
  puts "Database successfully dropped!"
end

db.close
