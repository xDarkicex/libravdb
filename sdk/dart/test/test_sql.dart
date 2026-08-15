import 'dart:convert';
import 'dart:io';
import 'package:libravdb/libravdb.dart';

void cleanup(String path) {
  final dir = Directory(path);
  if (dir.existsSync()) {
    dir.deleteSync(recursive: true);
  }
}

void main() {
  final dbPath = 'demo_db_sql_dart';
  cleanup(dbPath);

  print('Initializing LibraVDB at $dbPath...');
  final db = LibraVDB(dbPath);

  try {
    // Create tables
    db.query("CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))");
    db.query("CREATE EDGE TYPE FOLLOWS");

    // 1. Relational
    print("\n--- Relational SQL ---");
    db.queryWithParams("INSERT INTO users (id, name, embedding) VALUES (\$1, \$2, \$3)",
        jsonEncode({"1": "u1", "2": "Alice", "3": [1.0, 0.0, 0.0]}));
    db.queryWithParams("INSERT INTO users (id, name, embedding) VALUES (\$1, \$2, \$3)",
        jsonEncode({"1": "u2", "2": "Bob", "3": [0.0, 1.0, 0.0]}));
    db.queryWithParams("INSERT INTO users (id, name, embedding) VALUES (\$1, \$2, \$3)",
        jsonEncode({"1": "u3", "2": "Charlie", "3": [0.0, 0.0, 1.0]}));

    final resRel = db.query("SELECT id, name FROM users ORDER BY name ASC");
    print("Relational Result: $resRel");

    // 2. Vector
    print("\n--- Vector SQL ---");
    final resVec = db.queryWithParams("SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, \$vec) ASC LIMIT 2",
        jsonEncode({"vec": [1.0, 0.0, 0.0]}));
    print("Vector Result: $resVec");

    // 3. Graph
    print("\n--- Graph SQL ---");
    db.queryWithParams("INSERT INTO GRAPH_EDGES VALUES (\$1, \$2, \$3)",
        jsonEncode({"1": "u1", "2": "FOLLOWS", "3": "u2"}));
    db.queryWithParams("INSERT INTO GRAPH_EDGES VALUES (\$1, \$2, \$3)",
        jsonEncode({"1": "u2", "2": "FOLLOWS", "3": "u3"}));

    final resGraph = db.queryWithParams("SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = \$1",
        jsonEncode({"1": "u1"}));
    print("Graph Result: $resGraph");

    // 4. Temporal SQL
    print("\n--- Temporal SQL ---");
    final futureTime = DateTime.now().toUtc().add(Duration(seconds: 2));
    final cutoff = futureTime.toIso8601String();

    final resTemp = db.query("SELECT id FROM users AS OF TIMESTAMP '$cutoff' ORDER BY id ASC");
    print("Temporal Result: $resTemp");

    print("\nAll unified SQL tests passed successfully.");
  } finally {
    db.dropDatabase();
    db.close();
  }
}
