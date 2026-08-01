@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
package io.libravdb

import io.libravdb.cgo.*
import kotlinx.cinterop.*
import kotlinx.serialization.json.*

class LibraException(message: String) : Exception(message)

class LibraVDB(path: String) {
    private var handle: Int = -1

    init {
        memScoped {
            handle = OpenDB(path.cstr.ptr)
            if (handle < 0) {
                throw LibraException("Failed to open database at $path")
            }
        }
    }

    private fun checkError(errPtr: CPointer<ByteVar>?, opName: String) {
        if (errPtr != null) {
            val msg = errPtr.toKString()
            FreeString(errPtr)
            throw LibraException("$opName failed: $msg")
        }
    }

    fun ping() {
        val errPtr = Ping(handle)
        checkError(errPtr, "Ping")
    }

    fun setMemoryLimit(limit: Long) {
        val errPtr = SetGlobalMemoryLimit(handle, limit)
        checkError(errPtr, "SetMemoryLimit")
    }

    fun vacuum() {
        val errPtr = Vacuum(handle)
        checkError(errPtr, "Vacuum")
    }

    fun dropDatabase() {
        val errPtr = DropDatabase(handle)
        checkError(errPtr, "DropDatabase")
    }

    fun listCollections(): List<String> {
        val resPtr = ListCollections(handle) ?: return emptyList()
        val msg = resPtr.toKString()
        FreeString(resPtr)

        if (msg.isEmpty()) return emptyList()

        return try {
            val element = Json.parseToJsonElement(msg)
            if (element is JsonArray) {
                element.map { it.jsonPrimitive.content }
            } else emptyList()
        } catch (e: Exception) {
            emptyList()
        }
    }

    fun createCollection(name: String, dimension: Int): Collection {
        return memScoped {
            val colHandle = CreateCollection(handle, name.cstr.ptr, dimension)
            if (colHandle < 0) {
                throw LibraException("Failed to create collection $name")
            }
            Collection(colHandle, dimension)
        }
    }

    fun getCollection(name: String, dimension: Int): Collection {
        return memScoped {
            val colHandle = GetCollection(handle, name.cstr.ptr)
            if (colHandle < 0) {
                throw LibraException("Failed to get collection $name")
            }
            Collection(colHandle, dimension)
        }
    }

    fun close() {
        if (handle >= 0) {
            CloseDB(handle)
            handle = -1
        }
    }
}
