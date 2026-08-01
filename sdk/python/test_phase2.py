import os
import shutil
from libravdb import LibraVDB, Eq, Gt, And

def main():
    db_path = "./demo_db_phase2"
    
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    print(f"Initializing LibraVDB at {db_path}...")
    db = LibraVDB(db_path)
    
    print("Creating collection 'products' with dimension 2...")
    collection = db.create_collection("products", 2)
    
    print("Inserting vectors...")
    collection.insert(id="item1", vector=[1.0, 0.0], metadata={"category": "A", "price": 100.0})
    collection.insert(id="item2", vector=[0.0, 1.0], metadata={"category": "B", "price": 200.0})
    collection.insert(id="item3", vector=[0.5, 0.5], metadata={"category": "A", "price": 300.0})
    
    print("Upserting item1...")
    collection.upsert(id="item1", vector=[1.0, 0.0], metadata={"category": "A", "price": 150.0})
    
    print("Deleting item2...")
    collection.delete(id="item2")
    
    print("Optimizing collection...")
    db.optimize_collection("products")
    
    print("Fetching stats...")
    stats = collection.stats()
    print(f"Stats: {stats}")
    
    print("Searching with Filter: category == 'A' AND price > 100.0 ...")
    f = And(Eq("category", "A"), Gt("price", 100.0))
    
    results = collection.search([1.0, 0.0], k=5, filter=f)
    
    print("\nFiltered Results:")
    for i, res in enumerate(results):
        print(f"{i+1}. ID: {res['id']}, Score: {res['score']:.4f}, Meta: {res['metadata']}")
        
    db.close()
    
if __name__ == "__main__":
    main()
