import * as fs from 'fs';
import { LibraVDB, Filter } from '../src';

const dbPath = "./demo_db_ts";

if (fs.existsSync(dbPath)) {
    fs.rmSync(dbPath, { recursive: true, force: true });
}

console.log(`Initializing LibraVDB at ${dbPath}...`);
const db = new LibraVDB(dbPath);

console.log("Database Ping OK:", (() => { db.ping(); return true; })());

console.log("Setting global memory limit...");
db.setMemoryLimit(10 * 1024 * 1024);

console.log("Creating collection 'docs' (dim: 3)...");
const col = db.createCollection("docs", 3);

console.log("List collections:", db.listCollections());

console.log("Testing InsertBatch with 1000 vectors...");
const ids: string[] = [];
const vectors: number[][] = [];
const metadata: any[] = [];
for (let i = 0; i < 1000; i++) {
    ids.push(`vec_${i}`);
    vectors.push([Math.random(), Math.random(), Math.random()]);
    metadata.push({ source: "ts_test", index: i, active: i % 2 === 0 });
}

col.insertBatch(ids, vectors, metadata);
console.log("Batch Insert Complete!");

console.log("Testing Update...");
col.update("vec_0", [1.0, 1.0, 1.0], { updated: true });

console.log("Testing Get...");
const rec = col.get("vec_0");
console.log("Got record:", rec);

console.log("Testing Search with Filters (active = true)...");
const filter = Filter.eq("active", true);
const results = col.search([1.0, 1.0, 1.0], 5, filter);
console.log("Search Results:", results.length > 0 ? "Found matches!" : "No matches");

console.log("Testing Scan (Offset 0, Limit 2)...");
const scanned = col.scan(0, 2);
console.log("Scanned:", scanned.map(s => s.id));

console.log("Testing DeleteBatch (Deleting 500 vectors)...");
col.deleteBatch(ids.slice(0, 500));

console.log(`Collection Count: ${col.count()}`);

console.log("Testing Vacuum...");
db.vacuum();

console.log("Database Health:", db.health().status);

console.log("Dropping database...");
db.drop();

if (!fs.existsSync(dbPath)) {
    console.log("Database successfully dropped!");
}

db.close();
