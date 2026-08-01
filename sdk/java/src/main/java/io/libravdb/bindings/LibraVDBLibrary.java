package io.libravdb.bindings;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.StringArray;

public interface LibraVDBLibrary extends Library {
    LibraVDBLibrary INSTANCE = Native.load("libravdb", LibraVDBLibrary.class);

    int OpenDB(String path);
    int CloseDB(int dbID);
    
    Pointer Ping(int dbID);
    Pointer Vacuum(int dbID);
    Pointer DropDatabase(int dbID);
    Pointer SetGlobalMemoryLimit(int dbID, long limit);
    Pointer ListCollections(int dbID);
    
    int CreateCollection(int dbID, String name, int dim);
    int GetCollection(int dbID, String name);
    
    Pointer InsertVector(int colID, String id, float[] vec, int dim, String metadataJSON);
    Pointer UpdateVector(int colID, String id, float[] vec, int dim, String metadataJSON);
    Pointer UpdateVectorIfVersion(int colID, String id, float[] vec, int dim, String metadataJSON, long expectedVersion);
    
    Pointer GetVector(int colID, String id);
    Pointer QueryVector(int colID, float[] vec, int dim, int limit, String filterJSON);
    Pointer ScanCollection(int colID, int offset, int limit);
    
    Pointer InsertBatch(int colID, StringArray ids, float[] vecs, int count, int dim, StringArray metas);
    Pointer DeleteBatch(int colID, StringArray ids, int count);
    
    long GetCollectionCount(int colID);
    Pointer EnableMemoryMapping(int colID, String path);
    
    void FreeString(Pointer str);
}
