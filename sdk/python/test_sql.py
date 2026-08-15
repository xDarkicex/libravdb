import os
import shutil
from datetime import datetime, timedelta, timezone
from libravdb import LibraVDB


def row_ids(result):
    return [row.get("metadata", {}).get("id", row.get("id")) for row in result.get("results", [])]

def main():
    db_path = "./demo_db_sql"

    if os.path.exists(db_path):
        if os.path.isdir(db_path):
            shutil.rmtree(db_path)
        else:
            os.remove(db_path)

    print(f"Initializing LibraVDB at {db_path}...")
    db = LibraVDB(db_path)

    # One SQL DDL path creates the relational/vector collection and attaches
    # the real graph layer. Records become GRAPH_NODES as they are inserted.
    db.query("CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))")
    db.query("CREATE EDGE TYPE FOLLOWS")
    db.query("CREATE EDGE TYPE KNOWS UNDIRECTED")

    # 1. Relational
    print("\n--- Relational SQL ---")
    db.query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)", {"1": "u1", "2": "Alice", "3": [1.0, 0.0, 0.0]})
    db.query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)", {"1": "u2", "2": "Bob", "3": [0.0, 1.0, 0.0]})
    db.query_with_params("INSERT INTO users (id, name, embedding) VALUES ($1, $2, $3)", {"1": "u3", "2": "Charlie", "3": [0.0, 0.0, 1.0]})

    res = db.query("SELECT id, name FROM users ORDER BY name ASC")
    assert len(res["results"]) == 3
    print(f"Relational Result: {res}")

    # 2. Vector
    print("\n--- Vector SQL ---")
    res = db.query_with_params("SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, $vec) ASC LIMIT 2", {"vec": [1.0, 0.0, 0.0]})
    print(f"Vector Result: {res}")

    # 2b. Parameterized JSONB upsert. Structured SDK values travel through
    # the same shared DatabaseQueryWithParams path as vectors and graph IDs.
    print("\n--- JSONB upsert SQL ---")
    db.query("CREATE TABLE people (id STRING PRIMARY KEY, metadata JSONB, vector VECTOR(3))")
    profile = {
        "name": "Ada Lovelace",
        "roles": ["admin", "owner"],
        "settings": {"alerts": True},
    }
    db.query_with_params(
        """INSERT INTO people (id, metadata, vector)
           VALUES ($1, $2::jsonb, $3)
           ON CONFLICT (id) DO UPDATE SET
             metadata = EXCLUDED.metadata,
             vector = EXCLUDED.vector""",
        {"1": "p1", "2": profile, "3": [0.0, 1.0, 0.0]},
    )
    updated = dict(profile)
    updated["name"] = "Ada Lovelace Byron"
    db.query_with_params(
        """INSERT INTO people (id, metadata, vector)
           VALUES ($1, $2::jsonb, $3)
           ON CONFLICT (id) DO UPDATE SET metadata = EXCLUDED.metadata""",
        {"1": "p1", "2": updated, "3": [1.0, 0.0, 0.0]},
    )
    res = db.query("SELECT metadata->>'name' AS name FROM people WHERE id = 'p1'")
    assert res["results"][0]["metadata"]["name"] == "Ada Lovelace Byron", res

    # 3. Graph
    print("\n--- Graph SQL ---")
    db.query_with_params("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)", {"1": "u1", "2": "FOLLOWS", "3": "u2"})
    db.query_with_params("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)", {"1": "u2", "2": "FOLLOWS", "3": "u3"})
    res = db.query_with_params("SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = $1", {"1": "u1"})
    assert len(res["results"]) == 1, res
    print(f"Graph Result: {res}")
    db.query_with_params("INSERT INTO GRAPH_EDGES VALUES ($1, $2, $3)", {"1": "u1", "2": "KNOWS", "3": "u2"})
    res = db.query_with_params("SELECT tgt.id FROM users src JOIN MATCH (src)-[:KNOWS]->(tgt) WHERE src.id = $1", {"1": "u2"})
    assert len(res["results"]) == 1, res
    print(f"Undirected graph Result: {res}")

    # 4. Temporal SQL
    print("\n--- Temporal SQL ---")
    cutoff = (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()
    res = db.query(f"SELECT id FROM users AS OF TIMESTAMP '{cutoff}' ORDER BY id ASC")
    assert len(res["results"]) == 3
    print(f"Temporal Result: {res}")

    print("\nAll unified SQL tests passed successfully.")

    db.close()

if __name__ == "__main__":
    main()
