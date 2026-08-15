import { _lib, _from_c_string } from './core';
import { Filter } from './filters';

export class Collection {
    private handle: number;
    private dim: number;

    constructor(handle: number, dim: number) {
        this.handle = handle;
        this.dim = dim;
    }

    private checkError(resPtr: any, opName: string): any {
        const jsonStr = _from_c_string(resPtr);
        if (jsonStr) {
            if (jsonStr.startsWith('{"error"')) {
                const err = JSON.parse(jsonStr);
                throw new Error(`${opName} failed: ${err.error}`);
            }
            if (jsonStr.startsWith("error: ")) {
                throw new Error(`${opName} failed: ${jsonStr.substring(7)}`);
            }
        }
        return jsonStr;
    }

    insert(id: string, vector: number[], metadata: Record<string, any> = {}): void {
        if (vector.length !== this.dim) throw new Error(`Vector dimension must be ${this.dim}`);
        const floatArray = new Float32Array(vector);
        const metaStr = JSON.stringify(metadata);
        
        const errPtr = _lib.InsertVector(this.handle, id, floatArray, this.dim, metaStr);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`Insert failed: ${errMsg}`);
        }
    }

    upsert(id: string, vector: number[], metadata: Record<string, any> = {}): void {
        if (vector.length !== this.dim) throw new Error(`Vector dimension must be ${this.dim}`);
        const floatArray = new Float32Array(vector);
        const metaStr = JSON.stringify(metadata);
        
        const errPtr = _lib.UpsertVector(this.handle, id, floatArray, this.dim, metaStr);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`Upsert failed: ${errMsg}`);
        }
    }

    update(id: string, vector: number[], metadata: Record<string, any> = {}): void {
        if (vector.length !== this.dim) throw new Error(`Vector dimension must be ${this.dim}`);
        const floatArray = new Float32Array(vector);
        const metaStr = JSON.stringify(metadata);
        
        const errPtr = _lib.UpdateVector(this.handle, id, floatArray, this.dim, metaStr);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`Update failed: ${errMsg}`);
        }
    }

    delete(id: string): void {
        const errPtr = _lib.DeleteVector(this.handle, id);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`Delete failed: ${errMsg}`);
        }
    }

    stats(): any {
        const resPtr = _lib.GetCollectionStats(this.handle);
        const jsonStr = this.checkError(resPtr, "Stats");
        return jsonStr ? JSON.parse(jsonStr) : {};
    }

    search(vector: number[], k: number = 10, filter?: Filter): any[] {
        if (vector.length !== this.dim) throw new Error(`Vector dimension must be ${this.dim}`);
        const floatArray = new Float32Array(vector);
        const filterStr = filter ? JSON.stringify(filter.toJSON()) : "";
        
        const resPtr = _lib.QueryVector(this.handle, floatArray, this.dim, k, filterStr);
        const jsonStr = this.checkError(resPtr, "Search");
        return jsonStr ? JSON.parse(jsonStr) : [];
    }

    scan(offset: number = 0, limit: number = 100): any[] {
        const resPtr = _lib.ScanCollection(this.handle, offset, limit);
        const jsonStr = this.checkError(resPtr, "Scan");
        return jsonStr ? JSON.parse(jsonStr) : [];
    }

    insertBatch(ids: string[], vectors: number[][], metadata?: Record<string, any>[]): void {
        const count = ids.length;
        if (vectors.length !== count) throw new Error("ids and vectors must have same length");
        if (metadata && metadata.length !== count) throw new Error("ids and metadata must have same length");

        const flatVectors = new Float32Array(count * this.dim);
        for (let i = 0; i < count; i++) {
            if (vectors[i].length !== this.dim) throw new Error(`Vector dimension must be ${this.dim}`);
            flatVectors.set(vectors[i], i * this.dim);
        }

        const metasArray = metadata ? metadata.map(m => JSON.stringify(m)) : null;

        const errPtr = _lib.InsertBatch(this.handle, ids, flatVectors, count, this.dim, metasArray);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`InsertBatch failed: ${errMsg}`);
        }
    }

    deleteBatch(ids: string[]): void {
        const errPtr = _lib.DeleteBatch(this.handle, ids, ids.length);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`DeleteBatch failed: ${errMsg}`);
        }
    }

    get(id: string): any {
        const resPtr = _lib.GetVector(this.handle, id);
        const jsonStr = this.checkError(resPtr, "Get");
        return jsonStr ? JSON.parse(jsonStr) : null;
    }

    count(): number {
        const c = _lib.GetCollectionCount(this.handle);
        if (c < 0) throw new Error("Failed to get collection count");
        return Number(c);
    }

    updateIfVersion(id: string, vector: number[], expectedVersion: number, metadata: Record<string, any> = {}): void {
        if (vector.length !== this.dim) throw new Error(`Vector dimension must be ${this.dim}`);
        const floatArray = new Float32Array(vector);
        const metaStr = JSON.stringify(metadata);

        const errPtr = _lib.UpdateVectorIfVersion(this.handle, id, floatArray, this.dim, metaStr, expectedVersion);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`UpdateIfVersion failed: ${errMsg}`);
        }
    }

    deleteIfVersion(id: string, expectedVersion: number): void {
        const errPtr = _lib.DeleteVectorIfVersion(this.handle, id, expectedVersion);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`DeleteIfVersion failed: ${errMsg}`);
        }
    }

    setMemoryLimit(limit: number): void {
        const errPtr = _lib.SetCollectionMemoryLimit(this.handle, limit);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`SetMemoryLimit failed: ${errMsg}`);
        }
    }

    memoryUsage(): any {
        const resPtr = _lib.GetCollectionMemoryUsage(this.handle);
        const jsonStr = this.checkError(resPtr, "Memory usage");
        return jsonStr ? JSON.parse(jsonStr) : {};
    }

    triggerGC(): void {
        const errPtr = _lib.TriggerCollectionGC(this.handle);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`TriggerGC failed: ${errMsg}`);
        }
    }

    enableMemoryMapping(path: string): void {
        const errPtr = _lib.EnableMemoryMapping(this.handle, path);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`EnableMemoryMapping failed: ${errMsg}`);
        }
    }

    disableMemoryMapping(): void {
        const errPtr = _lib.DisableMemoryMapping(this.handle);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`DisableMemoryMapping failed: ${errMsg}`);
        }
    }

    saveIndex(path: string): void {
        const errPtr = _lib.SaveIndex(this.handle, path);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`SaveIndex failed: ${errMsg}`);
        }
    }

    loadIndex(path: string): void {
        const errPtr = _lib.LoadIndex(this.handle, path);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`LoadIndex failed: ${errMsg}`);
        }
    }
}

export class LibraVDB {
    private handle: number;

    constructor(path: string) {
        this.handle = _lib.OpenDB(path);
        if (this.handle < 0) {
            throw new Error(`Failed to open database at ${path}`);
        }
    }

    private checkError(resPtr: any, opName: string): any {
        const jsonStr = _from_c_string(resPtr);
        if (jsonStr) {
            if (jsonStr.startsWith('{"error"')) {
                const err = JSON.parse(jsonStr);
                throw new Error(`${opName} failed: ${err.error}`);
            }
            if (jsonStr.startsWith("error: ")) {
                throw new Error(`${opName} failed: ${jsonStr.substring(7)}`);
            }
        }
        return jsonStr;
    }

    close(): void {
        if (this.handle >= 0) {
            _lib.CloseDB(this.handle);
            this.handle = -1;
        }
    }

    createCollection(name: string, dimension: number): Collection {
        const colHandle = _lib.CreateCollection(this.handle, name, dimension);
        if (colHandle < 0) throw new Error(`Failed to create collection ${name}`);
        return new Collection(colHandle, dimension);
    }

    getCollection(name: string, dimension: number): Collection {
        const colHandle = _lib.GetCollection(this.handle, name);
        if (colHandle < 0) throw new Error(`Failed to get collection ${name}`);
        return new Collection(colHandle, dimension);
    }

    listCollections(): string[] {
        const resPtr = _lib.ListCollections(this.handle);
        const jsonStr = this.checkError(resPtr, "List collections");
        return jsonStr ? JSON.parse(jsonStr) : [];
    }

    deleteCollection(name: string): void {
        const errPtr = _lib.DeleteCollection(this.handle, name);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`Delete collection failed: ${errMsg}`);
        }
    }

    optimizeCollection(name: string): void {
        const errPtr = _lib.OptimizeCollection(this.handle, name);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`Optimize failed: ${errMsg}`);
        }
    }

    vacuum(): void {
        const errPtr = _lib.Vacuum(this.handle);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`Vacuum failed: ${errMsg}`);
        }
    }

    backup(dest: string): void {
        const errPtr = _lib.Backup(this.handle, dest);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`Backup failed: ${errMsg}`);
        }
    }

    drop(): void {
        const errPtr = _lib.DropDatabase(this.handle);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`Drop database failed: ${errMsg}`);
        }
    }

    setMemoryLimit(limit: number): void {
        const errPtr = _lib.SetGlobalMemoryLimit(this.handle, limit);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`Set memory limit failed: ${errMsg}`);
        }
    }

    memoryUsage(): any {
        const resPtr = _lib.GetGlobalMemoryUsage(this.handle);
        const jsonStr = this.checkError(resPtr, "Memory usage");
        return jsonStr ? JSON.parse(jsonStr) : {};
    }

    triggerGC(): void {
        const errPtr = _lib.TriggerGlobalGC(this.handle);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`Trigger GC failed: ${errMsg}`);
        }
    }

    ping(): void {
        const errPtr = _lib.Ping(this.handle);
        if (errPtr) {
            const errMsg = _from_c_string(errPtr);
            if (errMsg) throw new Error(`Ping failed: ${errMsg}`);
        }
    }

    health(): any {
        const resPtr = _lib.GetDatabaseHealth(this.handle);
        const jsonStr = this.checkError(resPtr, "Health");
        return jsonStr ? JSON.parse(jsonStr) : {};
    }

    stats(): any {
        const resPtr = _lib.GetDatabaseStats(this.handle);
        const jsonStr = this.checkError(resPtr, "Database Stats");
        return jsonStr ? JSON.parse(jsonStr) : {};
    }

    query(sql: string): any {
        const resPtr = _lib.DatabaseQuery(this.handle, sql);
        const jsonStr = this.checkError(resPtr, "Query");
        return jsonStr ? JSON.parse(jsonStr) : {};
    }

    queryWithParams(sql: string, params: Record<string, any>): any {
        const paramsStr = params ? JSON.stringify(params) : "";
        const resPtr = _lib.DatabaseQueryWithParams(this.handle, sql, paramsStr);
        const jsonStr = this.checkError(resPtr, "QueryWithParams");
        return jsonStr ? JSON.parse(jsonStr) : {};
    }
}
