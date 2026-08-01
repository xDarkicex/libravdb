@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
package io.libravdb

import io.libravdb.cgo.*
import kotlinx.cinterop.*
import kotlinx.serialization.json.*

class Collection internal constructor(private val handle: Int, private val dim: Int) {

    private fun checkError(errPtr: CPointer<ByteVar>?, opName: String) {
        if (errPtr != null) {
            val msg = errPtr.toKString()
            FreeString(errPtr)
            if (msg.startsWith("error: ")) {
                throw LibraException("$opName failed: ${msg.substring(7)}")
            }
            throw LibraException("$opName failed: $msg")
        }
    }

    private fun parseResult(resPtr: CPointer<ByteVar>?, opName: String): JsonElement? {
        if (resPtr == null) return null
        
        val msg = resPtr.toKString()
        FreeString(resPtr)
        
        if (msg.isEmpty()) return null

        try {
            val element = Json.parseToJsonElement(msg)
            if (element is JsonObject && element.containsKey("error")) {
                throw LibraException("$opName failed: ${element["error"]}")
            }
            return element
        } catch (e: Exception) {
            if (e is LibraException) throw e
            throw LibraException("JSON Parse error: ${e.message}")
        }
    }

    fun insert(id: String, vector: FloatArray, metadata: JsonObject? = null) {
        if (vector.size != dim) throw LibraException("Vector dimension mismatch")
        memScoped {
            val metaStr = metadata?.toString() ?: ""
            val vecPtr = allocArray<FloatVar>(dim)
            for (i in 0 until dim) vecPtr[i] = vector[i]

            val errPtr = InsertVector(handle, id.cstr.ptr, vecPtr, dim, metaStr.cstr.ptr)
            checkError(errPtr, "Insert")
        }
    }

    fun update(id: String, vector: FloatArray, metadata: JsonObject? = null) {
        if (vector.size != dim) throw LibraException("Vector dimension mismatch")
        memScoped {
            val metaStr = metadata?.toString() ?: ""
            val vecPtr = allocArray<FloatVar>(dim)
            for (i in 0 until dim) vecPtr[i] = vector[i]

            val errPtr = UpdateVector(handle, id.cstr.ptr, vecPtr, dim, metaStr.cstr.ptr)
            checkError(errPtr, "Update")
        }
    }

    fun updateIfVersion(id: String, vector: FloatArray, expectedVersion: Long, metadata: JsonObject? = null) {
        if (vector.size != dim) throw LibraException("Vector dimension mismatch")
        memScoped {
            val metaStr = metadata?.toString() ?: ""
            val vecPtr = allocArray<FloatVar>(dim)
            for (i in 0 until dim) vecPtr[i] = vector[i]

            val errPtr = UpdateVectorIfVersion(handle, id.cstr.ptr, vecPtr, dim, metaStr.cstr.ptr, expectedVersion.toULong())
            checkError(errPtr, "UpdateIfVersion")
        }
    }

    fun get(id: String): JsonElement? {
        return memScoped {
            val resPtr = GetVector(handle, id.cstr.ptr)
            parseResult(resPtr, "Get")
        }
    }

    fun search(vector: FloatArray, k: Int, filter: Filter? = null): JsonElement? {
        if (vector.size != dim) throw LibraException("Vector dimension mismatch")
        return memScoped {
            val filterStr = filter?.toJsonString() ?: ""
            val vecPtr = allocArray<FloatVar>(dim)
            for (i in 0 until dim) vecPtr[i] = vector[i]

            val resPtr = QueryVector(handle, vecPtr, dim, k, filterStr.cstr.ptr)
            parseResult(resPtr, "Search")
        }
    }

    fun scan(offset: Int, limit: Int): JsonElement? {
        val resPtr = ScanCollection(handle, offset, limit)
        return parseResult(resPtr, "Scan")
    }

    fun insertBatch(ids: List<String>, vectors: List<FloatArray>, metadata: List<JsonObject>? = null) {
        val count = ids.size
        if (vectors.size != count) throw LibraException("ids and vectors size mismatch")

        memScoped {
            val idsPtr = allocArray<CPointerVar<ByteVar>>(count)
            for (i in 0 until count) {
                idsPtr[i] = ids[i].cstr.ptr
            }

            val vecsPtr = allocArray<FloatVar>(count * dim)
            for (i in 0 until count) {
                if (vectors[i].size != dim) throw LibraException("Vector dimension mismatch at index $i")
                for (j in 0 until dim) {
                    vecsPtr[i * dim + j] = vectors[i][j]
                }
            }

            var metasPtr: CArrayPointer<CPointerVar<ByteVar>>? = null
            if (metadata != null) {
                if (metadata.size != count) throw LibraException("ids and metadata size mismatch")
                metasPtr = allocArray<CPointerVar<ByteVar>>(count)
                for (i in 0 until count) {
                    metasPtr[i] = metadata[i].toString().cstr.ptr
                }
            }

            val errPtr = InsertBatch(handle, idsPtr, vecsPtr, count, dim, metasPtr)
            checkError(errPtr, "InsertBatch")
        }
    }

    fun deleteBatch(ids: List<String>) {
        val count = ids.size
        memScoped {
            val idsPtr = allocArray<CPointerVar<ByteVar>>(count)
            for (i in 0 until count) {
                idsPtr[i] = ids[i].cstr.ptr
            }

            val errPtr = DeleteBatch(handle, idsPtr, count)
            checkError(errPtr, "DeleteBatch")
        }
    }

    fun count(): Long {
        val c = GetCollectionCount(handle)
        if (c < 0) throw LibraException("Failed to get collection count")
        return c
    }

    fun enableMemoryMapping(path: String) {
        memScoped {
            val errPtr = EnableMemoryMapping(handle, path.cstr.ptr)
            checkError(errPtr, "EnableMemoryMapping")
        }
    }
}
