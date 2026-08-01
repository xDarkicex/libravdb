import Foundation
import CLibraVDB

public enum LibraVDBError: Error {
    case initializationFailed(String)
    case nativeError(String)
}

public class Database {
    private var dbID: Int32 = -1

    public init(path: String) throws {
        self.dbID = path.withCString { cPath in
            OpenDB(UnsafeMutablePointer(mutating: cPath))
        }
        
        if self.dbID < 0 {
            throw LibraVDBError.initializationFailed("Failed to open database at path: \(path)")
        }
    }

    public func createCollection(name: String, dimension: Int) throws -> Collection {
        let colID = name.withCString { cName in
            CreateCollection(dbID, UnsafeMutablePointer(mutating: cName), Int32(dimension))
        }
        
        if colID < 0 {
            throw LibraVDBError.nativeError("Failed to create collection: \(name)")
        }
        
        return Collection(dbID: dbID, colID: colID, name: name, dimension: dimension)
    }

    public func getCollection(name: String, dimension: Int) throws -> Collection {
        let colID = name.withCString { cName in
            GetCollection(dbID, UnsafeMutablePointer(mutating: cName))
        }
        
        if colID < 0 {
            throw LibraVDBError.nativeError("Failed to get collection: \(name)")
        }
        
        return Collection(dbID: dbID, colID: colID, name: name, dimension: dimension)
    }

    public func vacuum() throws {
        if let errPtr = Vacuum(dbID) {
            try checkError(errPtr)
        }
    }

    public func dropDatabase() throws {
        if let errPtr = DropDatabase(dbID) {
            try checkError(errPtr)
        }
    }

    deinit {
        if dbID >= 0 {
            CloseDB(dbID)
        }
    }
}

internal func checkError(_ ptr: UnsafeMutablePointer<CChar>?) throws {
    guard let ptr = ptr else { return }
    let msg = String(cString: ptr)
    FreeString(ptr)
    
    if msg != "OK" && !msg.hasPrefix("{") && !msg.hasPrefix("[") && msg.hasPrefix("ERROR:") {
        throw LibraVDBError.nativeError(msg)
    } else if msg.hasPrefix("error ") || msg.hasPrefix("error:") {
        throw LibraVDBError.nativeError(msg)
    }
}

internal func extractString(_ ptr: UnsafeMutablePointer<CChar>?) throws -> String? {
    guard let ptr = ptr else { return nil }
    let result = String(cString: ptr)
    FreeString(ptr)
    
    if result.hasPrefix("ERROR:") || result.hasPrefix("error ") || result.hasPrefix("error:") {
        throw LibraVDBError.nativeError(result)
    }
    
    return result
}
