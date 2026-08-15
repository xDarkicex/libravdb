import * as fs from 'fs';
import * as path from 'path';
import { LibraVDB } from '../src';

function main() {
    const dbPath = path.resolve(__dirname, 'demo_db_sql_ts');

    if (fs.existsSync(dbPath)) {
        if (fs.statSync(dbPath).isDirectory()) {
            fs.rmSync(dbPath, { recursive: true, force: true });
        } else {
            fs.unlinkSync(dbPath);
        }
    }

    console.log(`Initializing LibraVDB at ${dbPath}...`);
    const db = new LibraVDB(dbPath);

    // Create tables
    db.query("CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))");
    db.query("CREATE EDGE TYPE FOLLOWS");

    // 1. Relational
    console.log("\n--- Relational SQL ---");
    db.queryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)", {"1": "u1", "2": "Alice", "3": [1.0, 0.0, 0.0]});
    db.queryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)", {"1": "u2", "2": "Bob", "3": [0.0, 1.0, 0.0]});
    db.queryWithParams("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)", {"1": "u3", "2": "Charlie", "3": [0.0, 0.0, 1.0]});

    let res = db.query("SELECT id, name FROM users ORDER BY name ASC");
    console.log(`Relational Result: ${JSON.stringify(res)}`);

    // 2. Vector
    console.log("\n--- Vector SQL ---");
    res = db.queryWithParams("SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, $vec) ASC LIMIT 2", {"vec": [1.0, 0.0, 0.0]});
    console.log(`Vector Result: ${JSON.stringify(res)}`);

    // 3. Graph
    console.log("\n--- Graph SQL ---");
    db.queryWithParams("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)", {"1": "u1", "2": "FOLLOWS", "3": "u2"});
    db.queryWithParams("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)", {"1": "u2", "2": "FOLLOWS", "3": "u3"});
    res = db.queryWithParams("SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = $1", {"1": "u1"});
    console.log(`Graph Result: ${JSON.stringify(res)}`);

    // 4. Temporal SQL
    console.log("\n--- Temporal SQL ---");
    const cutoff = new Date(Date.now() + 2000).toISOString();
    res = db.query(`SELECT id FROM users AS OF TIMESTAMP '${cutoff}' ORDER BY id ASC`);
    console.log(`Temporal Result: ${JSON.stringify(res)}`);

    db.close();
    console.log("\nAll unified SQL tests passed successfully.");
}

main();
