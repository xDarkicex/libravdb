<?php

require_once __DIR__ . '/../src/LibraVDB.php';
require_once __DIR__ . '/../src/Database.php';

use LibraVDB\Database;

function cleanup($path) {
    if (file_exists($path)) {
        if (is_dir($path)) {
            $files = array_diff(scandir($path), array('.', '..'));
            foreach ($files as $file) {
                unlink("$path/$file");
            }
            rmdir($path);
        } else {
            unlink($path);
        }
    }
}

$dbPath = "demo_db_sql_php";
cleanup($dbPath);

echo "Initializing LibraVDB at $dbPath...\n";
$db = new Database($dbPath);

try {
    // Create tables
    $db->query("CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))");
    $db->query("CREATE EDGE TYPE FOLLOWS");

    // 1. Relational
    echo "\n--- Relational SQL ---\n";
    $db->queryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
        json_encode(["1" => "u1", "2" => "Alice", "3" => [1.0, 0.0, 0.0]]));
    $db->queryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
        json_encode(["1" => "u2", "2" => "Bob", "3" => [0.0, 1.0, 0.0]]));
    $db->queryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)",
        json_encode(["1" => "u3", "2" => "Charlie", "3" => [0.0, 0.0, 1.0]]));

    $resRel = $db->query("SELECT id, name FROM users ORDER BY name ASC");
    echo "Relational Result: " . $resRel . "\n";

    // 2. Vector
    echo "\n--- Vector SQL ---\n";
    $resVec = $db->queryWithParams("SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, \$vec) ASC LIMIT 2",
        json_encode(["vec" => [1.0, 0.0, 0.0]]));
    echo "Vector Result: " . $resVec . "\n";

    // 3. Graph
    echo "\n--- Graph SQL ---\n";
    $db->queryWithParams("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
        json_encode(["1" => "u1", "2" => "FOLLOWS", "3" => "u2"]));
    $db->queryWithParams("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)",
        json_encode(["1" => "u2", "2" => "FOLLOWS", "3" => "u3"]));

    $resGraph = $db->queryWithParams("SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = $1",
        json_encode(["1" => "u1"]));
    echo "Graph Result: " . $resGraph . "\n";

    // 4. Temporal SQL
    echo "\n--- Temporal SQL ---\n";
    $futureTime = time() + 2;
    $cutoff = gmdate('Y-m-d\TH:i:s.000\Z', $futureTime);

    $resTemp = $db->query("SELECT id FROM users AS OF TIMESTAMP '$cutoff' ORDER BY id ASC");
    echo "Temporal Result: " . $resTemp . "\n";

    echo "\nAll unified SQL tests passed successfully.\n";
} finally {
    $db->dropDatabase();
}
