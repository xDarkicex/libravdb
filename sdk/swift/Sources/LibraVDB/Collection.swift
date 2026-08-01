import Foundation
import CLibraVDB

public class Collection {
    private let dbID: Int32
    private let colID: Int32
    public let name: String
    public let dimension: Int

    internal init(dbID: Int32, colID: Int32, name: String, dimension: Int) {
        self.dbID = dbID
        self.colID = colID
        self.name = name
        self.dimension = dimension
    }

    public func insert(id: String, vector: [Float], metadata: [String: Any]? = nil) throws {
        guard vector.count == dimension else {
            throw LibraVDBError.nativeError("Vector dimension mismatch. Expected \(dimension), got \(vector.count)")
        }

        let metaStr = metadataToJson(metadata)
        
        try id.withCString { cId in
            try metaStr.withCString { cMeta in
                var mutableVec = vector
                try mutableVec.withUnsafeMutableBufferPointer { vecBuffer in
                    if let ptr = vecBuffer.baseAddress {
                        let errPtr = InsertVector(colID, UnsafeMutablePointer(mutating: cId), ptr, Int32(dimension), UnsafeMutablePointer(mutating: cMeta))
                        try checkError(errPtr)
                    }
                }
            }
        }
    }

    public func insertBatch(ids: [String], vectors: [[Float]], metadata: [[String: Any]?]? = nil) throws {
        let count = ids.count
        guard vectors.count == count else {
            throw LibraVDBError.nativeError("Length of ids and vectors must match")
        }
        if let metas = metadata, metas.count != count {
            throw LibraVDBError.nativeError("Length of ids and metadata must match")
        }

        var flatVectors = [Float]()
        flatVectors.reserveCapacity(count * dimension)
        
        var cIds = [UnsafeMutablePointer<CChar>?]()
        var cMetas = [UnsafeMutablePointer<CChar>?]()
        
        for i in 0..<count {
            guard vectors[i].count == dimension else {
                throw LibraVDBError.nativeError("Vector at index \(i) has dimension \(vectors[i].count), expected \(dimension)")
            }
            flatVectors.append(contentsOf: vectors[i])
            cIds.append(strdup(ids[i]))
            
            let metaStr = metadataToJson(metadata?[i])
            cMetas.append(strdup(metaStr))
        }

        defer {
            for ptr in cIds { free(ptr) }
            for ptr in cMetas { free(ptr) }
        }

        try flatVectors.withUnsafeMutableBufferPointer { vecBuffer in
            if let vecPtr = vecBuffer.baseAddress {
                try cIds.withUnsafeMutableBufferPointer { idsBuffer in
                    try cMetas.withUnsafeMutableBufferPointer { metasBuffer in
                        if let cIdsPtr = idsBuffer.baseAddress, let cMetasPtr = metasBuffer.baseAddress {
                            let errPtr = InsertBatch(colID, cIdsPtr, vecPtr, Int32(count), Int32(dimension), cMetasPtr)
                            try checkError(errPtr)
                        }
                    }
                }
            }
        }
    }

    public func search(vector: [Float], k: Int, filter: Filter? = nil) throws -> String {
        guard vector.count == dimension else {
            throw LibraVDBError.nativeError("Vector dimension mismatch")
        }

        let filterJson = filter?.toJson() ?? "{}"

        return try filterJson.withCString { cFilter in
            var mutableVec = vector
            return try mutableVec.withUnsafeMutableBufferPointer { vecBuffer in
                if let ptr = vecBuffer.baseAddress {
                    let resPtr = QueryVector(colID, ptr, Int32(dimension), Int32(k), UnsafeMutablePointer(mutating: cFilter))
                    return try extractString(resPtr) ?? "{}"
                }
                return "{}"
            }
        }
    }

    public func scan(offset: Int = 0, limit: Int = 100) throws -> String {
        let resPtr = ScanCollection(colID, Int32(offset), Int32(limit))
        return try extractString(resPtr) ?? "{}"
    }

    public func delete(id: String) throws {
        try id.withCString { cId in
            let errPtr = DeleteVector(colID, UnsafeMutablePointer(mutating: cId))
            try checkError(errPtr)
        }
    }

    private func metadataToJson(_ metadata: [String: Any]?) -> String {
        guard let metadata = metadata else { return "{}" }
        guard let data = try? JSONSerialization.data(withJSONObject: metadata, options: []) else { return "{}" }
        return String(data: data, encoding: .utf8) ?? "{}"
    }
}
