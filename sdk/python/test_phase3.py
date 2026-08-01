import os
import shutil
import random
from libravdb import LibraVDB

def main():
    db_path = "./demo_db_phase3"
    backup_path = "./demo_db_phase3_backup"
    
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(backup_path):
        os.remove(backup_path)
        
    print(f"Initializing LibraVDB at {db_path}...")
    db = LibraVDB(db_path)
    
    db.ping()
    print("Database Ping OK")
    
    print("Setting memory limit to 10MB...")
    db.set_memory_limit(10 * 1024 * 1024)
    
    print("Creating collection 'logs' with dimension 3...")
    col = db.create_collection("logs", 3)
    
    collections = db.list_collections()
    print(f"Collections: {collections}")
    
    print("Testing InsertBatch with 1000 vectors...")
    ids = [f"vec_{i}" for i in range(1000)]
    vectors = [[random.random(), random.random(), random.random()] for _ in range(1000)]
    metadata = [{"source": "test", "index": i} for i in range(1000)]
    
    col.insert_batch(ids, vectors, metadata)
    print("Batch Insert Complete!")
    
    print("Testing Update...")
    col.update("vec_0", [1.0, 1.0, 1.0], {"updated": True})
    
    print("Testing Scan (Offset 0, Limit 5)...")
    scanned = col.scan(0, 5)
    for rec in scanned:
        print(f"  - {rec['id']}: {rec['metadata']}")
        
    print("Testing DeleteBatch (Deleting 500 vectors)...")
    col.delete_batch(ids[:500])
    
    print("Testing Vacuum...")
    db.vacuum()
    
    print("Testing Backup...")
    db.backup(backup_path)
    if os.path.exists(backup_path):
        print(f"Backup successfully created at {backup_path}")
        
    health = db.health()
    print(f"Database Health: {health['status']}")
    
    print("Dropping database...")
    db.drop()
    
    if not os.path.exists(db_path):
        print("Database successfully dropped!")
        
    db.close()
    
if __name__ == "__main__":
    main()
