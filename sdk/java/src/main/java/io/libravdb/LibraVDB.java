package io.libravdb;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.jna.Pointer;
import io.libravdb.bindings.LibraVDBLibrary;

import java.util.ArrayList;
import java.util.List;

public class LibraVDB implements AutoCloseable {
    private int handle;
    private static final ObjectMapper mapper = new ObjectMapper();

    public LibraVDB(String path) {
        this.handle = LibraVDBLibrary.INSTANCE.OpenDB(path);
        if (this.handle < 0) {
            throw new LibraException("Failed to open database at " + path);
        }
    }

    private void checkError(Pointer errPtr, String opName) {
        if (errPtr != null) {
            String msg = errPtr.getString(0);
            LibraVDBLibrary.INSTANCE.FreeString(errPtr);
            throw new LibraException(opName + " failed: " + msg);
        }
    }

    public void ping() {
        Pointer errPtr = LibraVDBLibrary.INSTANCE.Ping(handle);
        checkError(errPtr, "Ping");
    }

    public void setMemoryLimit(long limit) {
        Pointer errPtr = LibraVDBLibrary.INSTANCE.SetGlobalMemoryLimit(handle, limit);
        checkError(errPtr, "SetMemoryLimit");
    }

    public void vacuum() {
        Pointer errPtr = LibraVDBLibrary.INSTANCE.Vacuum(handle);
        checkError(errPtr, "Vacuum");
    }

    public void dropDatabase() {
        Pointer errPtr = LibraVDBLibrary.INSTANCE.DropDatabase(handle);
        checkError(errPtr, "DropDatabase");
    }

    public List<String> listCollections() {
        Pointer resPtr = LibraVDBLibrary.INSTANCE.ListCollections(handle);
        if (resPtr == null) return new ArrayList<>();
        
        String msg = resPtr.getString(0);
        LibraVDBLibrary.INSTANCE.FreeString(resPtr);

        try {
            JsonNode node = mapper.readTree(msg);
            List<String> collections = new ArrayList<>();
            if (node.isArray()) {
                for (JsonNode elem : node) {
                    collections.add(elem.asText());
                }
            }
            return collections;
        } catch (Exception e) {
            return new ArrayList<>();
        }
    }

    public Collection createCollection(String name, int dimension) {
        int colHandle = LibraVDBLibrary.INSTANCE.CreateCollection(handle, name, dimension);
        if (colHandle < 0) {
            throw new LibraException("Failed to create collection " + name);
        }
        return new Collection(colHandle, dimension);
    }

    public Collection getCollection(String name, int dimension) {
        int colHandle = LibraVDBLibrary.INSTANCE.GetCollection(handle, name);
        if (colHandle < 0) {
            throw new LibraException("Failed to get collection " + name);
        }
        return new Collection(colHandle, dimension);
    }

    @Override
    public void close() {
        if (handle >= 0) {
            LibraVDBLibrary.INSTANCE.CloseDB(handle);
            handle = -1;
        }
    }
}
