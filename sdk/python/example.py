import os
import shutil
from libravdb import LibraVDB

def main():
    db_path = "./demo_db"
    
    # Cleanup old demo db if exists
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    print(f"Initializing LibraVDB at {db_path}...")
    db = LibraVDB(db_path)
    
    print("Creating collection 'products' with dimension 3...")
    collection = db.create_collection("products", 3)
    
    print("Inserting vectors...")
    collection.insert(
        id="phone1",
        vector=[1.0, 0.0, 0.0],
        metadata={"category": "electronics", "price": 699.99}
    )
    
    collection.insert(
        id="laptop1",
        vector=[0.0, 1.0, 0.0],
        metadata={"category": "electronics", "price": 1299.99}
    )
    
    collection.insert(
        id="book1",
        vector=[0.0, 0.0, 1.0],
        metadata={"category": "books", "price": 24.99}
    )
    
    print("Searching for similar items to [0.9, 0.1, 0.0]...")
    results = collection.search([0.9, 0.1, 0.0], k=2)
    
    print("\nResults:")
    for i, res in enumerate(results):
        print(f"{i+1}. ID: {res['id']}, Score: {res['score']:.4f}, Meta: {res['metadata']}")
        
    print("\nTesting complete! Cleaning up...")
    db.close()
    
if __name__ == "__main__":
    main()
