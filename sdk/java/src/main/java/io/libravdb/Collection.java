package io.libravdb;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.jna.Pointer;
import com.sun.jna.StringArray;
import io.libravdb.bindings.LibraVDBLibrary;

import java.util.List;
import java.util.Optional;

public class Collection {
    private final int handle;
    private final int dim;
    private static final ObjectMapper mapper = new ObjectMapper();

    Collection(int handle, int dim) {
        this.handle = handle;
        this.dim = dim;
    }

    private void checkError(Pointer errPtr, String opName) {
        if (errPtr != null) {
            String msg = errPtr.getString(0);
            LibraVDBLibrary.INSTANCE.FreeString(errPtr);
            if (msg.startsWith("error: ")) {
                throw new LibraException(opName + " failed: " + msg.substring(7));
            }
            throw new LibraException(opName + " failed: " + msg);
        }
    }

    private JsonNode parseResult(Pointer resPtr, String opName) {
        if (resPtr == null) {
            return null;
        }
        String msg = resPtr.getString(0);
        LibraVDBLibrary.INSTANCE.FreeString(resPtr);

        if (msg.isEmpty()) {
            return null;
        }

        try {
            JsonNode node = mapper.readTree(msg);
            if (node.has("error")) {
                throw new LibraException(opName + " failed: " + node.get("error").asText());
            }
            return node;
        } catch (Exception e) {
            if (e instanceof LibraException) throw (LibraException) e;
            throw new LibraException("JSON Parse error: " + e.getMessage());
        }
    }

    public void insert(String id, float[] vector, Optional<JsonNode> metadata) {
        if (vector.length != dim) throw new LibraException("Vector dimension mismatch");
        String metaStr = metadata.map(JsonNode::toString).orElse("");
        Pointer errPtr = LibraVDBLibrary.INSTANCE.InsertVector(handle, id, vector, dim, metaStr);
        checkError(errPtr, "Insert");
    }

    public void update(String id, float[] vector, Optional<JsonNode> metadata) {
        if (vector.length != dim) throw new LibraException("Vector dimension mismatch");
        String metaStr = metadata.map(JsonNode::toString).orElse("");
        Pointer errPtr = LibraVDBLibrary.INSTANCE.UpdateVector(handle, id, vector, dim, metaStr);
        checkError(errPtr, "Update");
    }

    public void updateIfVersion(String id, float[] vector, long expectedVersion, Optional<JsonNode> metadata) {
        if (vector.length != dim) throw new LibraException("Vector dimension mismatch");
        String metaStr = metadata.map(JsonNode::toString).orElse("");
        Pointer errPtr = LibraVDBLibrary.INSTANCE.UpdateVectorIfVersion(handle, id, vector, dim, metaStr, expectedVersion);
        checkError(errPtr, "UpdateIfVersion");
    }

    public JsonNode get(String id) {
        Pointer resPtr = LibraVDBLibrary.INSTANCE.GetVector(handle, id);
        return parseResult(resPtr, "Get");
    }

    public JsonNode search(float[] vector, int k, Optional<Filter> filter) {
        if (vector.length != dim) throw new LibraException("Vector dimension mismatch");
        String filterStr = filter.map(Filter::toJsonString).orElse("");
        Pointer resPtr = LibraVDBLibrary.INSTANCE.QueryVector(handle, vector, dim, k, filterStr);
        return parseResult(resPtr, "Search");
    }

    public JsonNode scan(int offset, int limit) {
        Pointer resPtr = LibraVDBLibrary.INSTANCE.ScanCollection(handle, offset, limit);
        return parseResult(resPtr, "Scan");
    }

    public void insertBatch(List<String> ids, List<float[]> vectors, Optional<List<JsonNode>> metadata) {
        int count = ids.size();
        if (vectors.size() != count) throw new LibraException("ids and vectors size mismatch");

        float[] flatVectors = new float[count * dim];
        for (int i = 0; i < count; i++) {
            float[] vec = vectors.get(i);
            if (vec.length != dim) throw new LibraException("Vector dimension mismatch at index " + i);
            System.arraycopy(vec, 0, flatVectors, i * dim, dim);
        }

        StringArray cIds = new StringArray(ids.toArray(new String[0]));
        StringArray cMetas = null;

        if (metadata.isPresent()) {
            List<JsonNode> metaList = metadata.get();
            if (metaList.size() != count) throw new LibraException("ids and metadata size mismatch");
            String[] metaStrings = new String[count];
            for (int i = 0; i < count; i++) {
                metaStrings[i] = metaList.get(i).toString();
            }
            cMetas = new StringArray(metaStrings);
        }

        Pointer errPtr = LibraVDBLibrary.INSTANCE.InsertBatch(handle, cIds, flatVectors, count, dim, cMetas);
        checkError(errPtr, "InsertBatch");
    }

    public void deleteBatch(List<String> ids) {
        StringArray cIds = new StringArray(ids.toArray(new String[0]));
        Pointer errPtr = LibraVDBLibrary.INSTANCE.DeleteBatch(handle, cIds, ids.size());
        checkError(errPtr, "DeleteBatch");
    }

    public long count() {
        long c = LibraVDBLibrary.INSTANCE.GetCollectionCount(handle);
        if (c < 0) throw new LibraException("Failed to get collection count");
        return c;
    }

    public void enableMemoryMapping(String path) {
        Pointer errPtr = LibraVDBLibrary.INSTANCE.EnableMemoryMapping(handle, path);
        checkError(errPtr, "EnableMemoryMapping");
    }
}
