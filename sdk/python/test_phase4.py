import os
import shutil
from libravdb import LibraVDB

def main():
    db_path = "./demo_db_phase4"
    index_path = "./demo_index_phase4.bin"
    
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(index_path):
        os.remove(index_path)
        
    print(f"Initializing LibraVDB at {db_path}...")
    db = LibraVDB(db_path)
    
    print("Creating collection 'docs' with dimension 4...")
    col = db.create_collection("docs", 4)
    
    print("Setting Collection Memory Limit to 5MB...")
    col.set_memory_limit(5 * 1024 * 1024)
    
    print("Enabling Memory Mapping...")
    try:
        col.enable_memory_mapping(db_path + "/docs_mmap.bin")
    except Exception as e:
        print(f"EnableMemoryMapping intentionally caught: {e}")
    
    print("Inserting vector...")
    col.insert("doc1", [1.0, 2.0, 3.0, 4.0], {"title": "Hello"})
    
    print("Fetching vector using Get...")
    rec = col.get("doc1")
    print(f"Got record: {rec}")
    
    print(f"Collection Count: {col.count()}")
    
    print("Testing UpdateIfVersion...")
    try:
        # Assuming version 1 since it was just inserted
        col.update_if_version("doc1", [4.0, 3.0, 2.0, 1.0], expected_version=1, metadata={"title": "Updated"})
        print("UpdateIfVersion succeeded!")
    except Exception as e:
        print(f"UpdateIfVersion failed: {e}")
        
    print("Testing DeleteIfVersion with wrong version...")
    try:
        col.delete_if_version("doc1", expected_version=1)
        print("DeleteIfVersion succeeded (Unexpected!)")
    except Exception as e:
        print(f"DeleteIfVersion correctly failed: {e}")
        
    print(f"Collection Count: {col.count()}")
    
    usage = col.memory_usage()
    print(f"Collection Memory Usage: {usage.get('total')} bytes")
    
    print("Triggering Collection GC...")
    col.trigger_gc()
    
    print("Saving Index...")
    col.save_index(index_path)
    print("Loading Index...")
    col.load_index(index_path)
    
    print("Disabling Memory Mapping...")
    try:
        col.disable_memory_mapping()
    except Exception as e:
        print(f"DisableMemoryMapping intentionally caught: {e}")
    
    db.close()
    
if __name__ == "__main__":
    main()
