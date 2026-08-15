package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"unsafe"

	"github.com/xDarkicex/libravdb/libravdb"
)

var (
	mu          sync.RWMutex
	dbs         = make(map[int]*libravdb.Database)
	collections = make(map[int]*libravdb.Collection)
	nextDBID    = 1
	nextColID   = 1
)

//export OpenDB
func OpenDB(path *C.char) C.int {
	goPath := C.GoString(path)

	db, err := libravdb.Open(libravdb.WithStoragePath(goPath))
	if err != nil {
		fmt.Printf("Error opening DB: %v\n", err)
		return -1
	}

	mu.Lock()
	defer mu.Unlock()
	id := nextDBID
	nextDBID++
	dbs[id] = db

	return C.int(id)
}

//export CloseDB
func CloseDB(dbID C.int) C.int {
	mu.Lock()
	defer mu.Unlock()
	
	id := int(dbID)
	db, ok := dbs[id]
	if !ok {
		return -1
	}
	
	if err := db.Close(); err != nil {
		return -1
	}
	
	delete(dbs, id)
	return 0
}

//export CreateCollection
func CreateCollection(dbID C.int, name *C.char, dim C.int) C.int {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return -1
	}

	goName := C.GoString(name)
	ctx := context.Background()

	// Using cosine distance as default for the MVP
	col, err := db.CreateCollection(ctx, goName, 
		libravdb.WithDimension(int(dim)),
		libravdb.WithMetric(libravdb.CosineDistance),
	)
	if err != nil {
		fmt.Printf("Error creating collection: %v\n", err)
		return -1
	}

	mu.Lock()
	defer mu.Unlock()
	id := nextColID
	nextColID++
	collections[id] = col

	return C.int(id)
}

//export GetCollection
func GetCollection(dbID C.int, name *C.char) C.int {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return -1
	}

	goName := C.GoString(name)
	col, err := db.GetCollection(goName)
	if err != nil {
		return -1
	}

	mu.Lock()
	defer mu.Unlock()
	id := nextColID
	nextColID++
	collections[id] = col

	return C.int(id)
}

//export InsertVector
func InsertVector(colID C.int, id *C.char, vec *C.float, dim C.int, metadataJSON *C.char) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: collection not found")
	}

	goID := C.GoString(id)
	goDim := int(dim)
	
	// Convert C float array to Go slice safely
	slice := unsafe.Slice(vec, goDim)
	goVec := make([]float32, goDim)
	for i := 0; i < goDim; i++ {
		goVec[i] = float32(slice[i])
	}

	var metadata map[string]interface{}
	if metadataJSON != nil {
		goMetaJSON := C.GoString(metadataJSON)
		if goMetaJSON != "" {
			if err := json.Unmarshal([]byte(goMetaJSON), &metadata); err != nil {
				return C.CString(fmt.Sprintf("error parsing metadata: %v", err))
			}
		}
	}

	ctx := context.Background()
	err := col.Insert(ctx, goID, goVec, metadata)
	if err != nil {
		return C.CString(fmt.Sprintf("error inserting: %v", err))
	}

	return nil // Success
}

//export DeleteVector
func DeleteVector(colID C.int, id *C.char) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: collection not found")
	}

	goID := C.GoString(id)
	ctx := context.Background()
	err := col.Delete(ctx, goID)
	if err != nil {
		return C.CString(fmt.Sprintf("error deleting: %v", err))
	}
	return nil
}

//export UpsertVector
func UpsertVector(colID C.int, id *C.char, vec *C.float, dim C.int, metadataJSON *C.char) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: collection not found")
	}

	goID := C.GoString(id)
	goDim := int(dim)
	
	slice := unsafe.Slice(vec, goDim)
	goVec := make([]float32, goDim)
	for i := 0; i < goDim; i++ {
		goVec[i] = float32(slice[i])
	}

	var metadata map[string]interface{}
	if metadataJSON != nil {
		goMetaJSON := C.GoString(metadataJSON)
		if goMetaJSON != "" {
			if err := json.Unmarshal([]byte(goMetaJSON), &metadata); err != nil {
				return C.CString(fmt.Sprintf("error parsing metadata: %v", err))
			}
		}
	}

	ctx := context.Background()
	err := col.Upsert(ctx, goID, goVec, metadata)
	if err != nil {
		return C.CString(fmt.Sprintf("error upserting: %v", err))
	}

	return nil
}

//export OptimizeCollection
func OptimizeCollection(dbID C.int, name *C.char) *C.char {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: database not found")
	}

	goName := C.GoString(name)
	ctx := context.Background()
	err := db.OptimizeCollection(ctx, goName, nil)
	if err != nil {
		return C.CString(fmt.Sprintf("error optimizing: %v", err))
	}
	return nil
}

//export GetCollectionStats
func GetCollectionStats(colID C.int) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString(`{"error": "collection not found"}`)
	}

	ctx := context.Background()
	stats := col.Stats(ctx)
	
	bytes, err := json.Marshal(stats)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "failed to marshal stats: %v"}`, err))
	}

	return C.CString(string(bytes))
}

//export QueryVector
func QueryVector(colID C.int, vec *C.float, dim C.int, limit C.int, filterJSON *C.char) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString(`{"error": "collection not found"}`)
	}

	goDim := int(dim)
	slice := unsafe.Slice(vec, goDim)
	goVec := make([]float32, goDim)
	for i := 0; i < goDim; i++ {
		goVec[i] = float32(slice[i])
	}

	ctx := context.Background()
	q := col.Query(ctx).WithVector(goVec).Limit(int(limit))
	
	if filterJSON != nil {
		goFilterJSON := C.GoString(filterJSON)
		if goFilterJSON != "" {
			f, err := parseFilterJSON(goFilterJSON)
			if err != nil {
				return C.CString(fmt.Sprintf(`{"error": "failed to parse filter: %v"}`, err))
			}
			if f != nil {
				q = q.WithFilter(f)
			}
		}
	}

	results, err := q.Execute()
		
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	// For MVP, we serialize the Results directly to JSON
	type simpleResult struct {
		ID       string                 `json:"id"`
		Score    float32                `json:"score"`
		Metadata map[string]interface{} `json:"metadata"`
	}

	var formattedResults []simpleResult
	for _, r := range results.Results {
		formattedResults = append(formattedResults, simpleResult{
			ID:       r.ID,
			Score:    r.Score,
			Metadata: r.Metadata,
		})
	}

	bytes, err := json.Marshal(formattedResults)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "failed to marshal results: %v"}`, err))
	}

	return C.CString(string(bytes))
}

//export ScanCollection
func ScanCollection(colID C.int, offset C.int, limit C.int) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString(`{"error": "collection not found"}`)
	}

	type simpleResult struct {
		ID       string                 `json:"id"`
		Metadata map[string]interface{} `json:"metadata"`
	}
	var results []simpleResult

	goOffset := int(offset)
	goLimit := int(limit)
	count := 0

	ctx := context.Background()
	_ = col.Iterate(ctx, func(record libravdb.Record) error {
		if count >= goOffset && count < goOffset+goLimit {
			results = append(results, simpleResult{
				ID:       record.ID,
				Metadata: record.Metadata,
			})
		}
		count++
		if count >= goOffset+goLimit {
			return fmt.Errorf("limit_reached")
		}
		return nil
	})

	bytes, err := json.Marshal(results)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "failed to marshal results: %v"}`, err))
	}
	return C.CString(string(bytes))
}

//export UpdateVector
func UpdateVector(colID C.int, id *C.char, vec *C.float, dim C.int, metadataJSON *C.char) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: collection not found")
	}

	goID := C.GoString(id)
	goDim := int(dim)
	slice := unsafe.Slice(vec, goDim)
	goVec := make([]float32, goDim)
	for i := 0; i < goDim; i++ {
		goVec[i] = float32(slice[i])
	}

	var metadata map[string]interface{}
	if metadataJSON != nil {
		goMetaJSON := C.GoString(metadataJSON)
		if goMetaJSON != "" {
			_ = json.Unmarshal([]byte(goMetaJSON), &metadata)
		}
	}

	ctx := context.Background()
	err := col.Update(ctx, goID, goVec, metadata)
	if err != nil {
		return C.CString(fmt.Sprintf("error updating: %v", err))
	}
	return nil
}

//export InsertBatch
func InsertBatch(colID C.int, ids **C.char, vecs *C.float, count C.int, dim C.int, metas **C.char) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: collection not found")
	}

	goCount := int(count)
	goDim := int(dim)

	idsSlice := unsafe.Slice(ids, goCount)
	vecsSlice := unsafe.Slice(vecs, goCount*goDim)
	
	var metasSlice []*C.char
	if metas != nil {
		metasSlice = unsafe.Slice(metas, goCount)
	}

	entries := make([]libravdb.VectorEntry, goCount)
	for i := 0; i < goCount; i++ {
		goID := C.GoString(idsSlice[i])

		v := make([]float32, goDim)
		for j := 0; j < goDim; j++ {
			v[j] = float32(vecsSlice[i*goDim+j])
		}

		var metadata map[string]interface{}
		if metasSlice != nil && metasSlice[i] != nil {
			mStr := C.GoString(metasSlice[i])
			if mStr != "" {
				_ = json.Unmarshal([]byte(mStr), &metadata)
			}
		}

		entries[i] = libravdb.VectorEntry{
			ID:       goID,
			Vector:   v,
			Metadata: metadata,
		}
	}

	ctx := context.Background()
	err := col.InsertBatch(ctx, entries)
	if err != nil {
		return C.CString(fmt.Sprintf("error insert batch: %v", err))
	}
	return nil
}

//export DeleteBatch
func DeleteBatch(colID C.int, ids **C.char, count C.int) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: collection not found")
	}

	goCount := int(count)
	idsSlice := unsafe.Slice(ids, goCount)
	
	goIDs := make([]string, goCount)
	for i := 0; i < goCount; i++ {
		goIDs[i] = C.GoString(idsSlice[i])
	}

	ctx := context.Background()
	err := col.DeleteBatch(ctx, goIDs)
	if err != nil {
		return C.CString(fmt.Sprintf("error delete batch: %v", err))
	}
	return nil
}

//export ListCollections
func ListCollections(dbID C.int) *C.char {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return C.CString(`{"error": "database not found"}`)
	}

	names := db.ListCollections()
	bytes, err := json.Marshal(names)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}
	return C.CString(string(bytes))
}

//export DeleteCollection
func DeleteCollection(dbID C.int, name *C.char) *C.char {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: database not found")
	}
	
	goName := C.GoString(name)
	err := db.DeleteCollection(context.Background(), goName)
	if err != nil {
		return C.CString(fmt.Sprintf("error deleting collection: %v", err))
	}
	return nil
}

//export Vacuum
func Vacuum(dbID C.int) *C.char {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: database not found")
	}
	err := db.Vacuum(context.Background())
	if err != nil {
		return C.CString(fmt.Sprintf("error vacuuming: %v", err))
	}
	return nil
}

//export Backup
func Backup(dbID C.int, dest *C.char) *C.char {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: database not found")
	}
	goDest := C.GoString(dest)
	err := db.Backup(context.Background(), goDest)
	if err != nil {
		return C.CString(fmt.Sprintf("error backing up: %v", err))
	}
	return nil
}

//export DropDatabase
func DropDatabase(dbID C.int) *C.char {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: database not found")
	}
	err := db.Drop(context.Background())
	if err != nil {
		return C.CString(fmt.Sprintf("error dropping database: %v", err))
	}
	return nil
}

//export SetGlobalMemoryLimit
func SetGlobalMemoryLimit(dbID C.int, limit C.longlong) *C.char {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: database not found")
	}
	err := db.SetGlobalMemoryLimit(int64(limit))
	if err != nil {
		return C.CString(fmt.Sprintf("error setting memory limit: %v", err))
	}
	return nil
}

//export GetGlobalMemoryUsage
func GetGlobalMemoryUsage(dbID C.int) *C.char {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return C.CString(`{"error": "database not found"}`)
	}
	usage, err := db.GetGlobalMemoryUsage(context.Background())
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}
	bytes, _ := json.Marshal(usage)
	return C.CString(string(bytes))
}

//export TriggerGlobalGC
func TriggerGlobalGC(dbID C.int) *C.char {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: database not found")
	}
	err := db.TriggerGlobalGC()
	if err != nil {
		return C.CString(fmt.Sprintf("error triggering GC: %v", err))
	}
	return nil
}

//export Ping
func Ping(dbID C.int) *C.char {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: database not found")
	}
	err := db.Ping(context.Background())
	if err != nil {
		return C.CString(fmt.Sprintf("error pinging: %v", err))
	}
	return nil
}

//export GetDatabaseHealth
func GetDatabaseHealth(dbID C.int) *C.char {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return C.CString(`{"error": "database not found"}`)
	}
	health, err := db.Health(context.Background())
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}
	bytes, _ := json.Marshal(health)
	return C.CString(string(bytes))
}

//export GetDatabaseStats
func GetDatabaseStats(dbID C.int) *C.char {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return C.CString(`{"error": "database not found"}`)
	}
	stats := db.Stats(context.Background())
	bytes, _ := json.Marshal(stats)
	return C.CString(string(bytes))
}

//export GetVector
func GetVector(colID C.int, id *C.char) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString(`{"error": "collection not found"}`)
	}

	goID := C.GoString(id)
	record, err := col.Get(context.Background(), goID)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	type simpleRecord struct {
		ID       string                 `json:"id"`
		Metadata map[string]interface{} `json:"metadata"`
	}

	res := simpleRecord{
		ID:       record.ID,
		Metadata: record.Metadata,
	}

	bytes, _ := json.Marshal(res)
	return C.CString(string(bytes))
}

//export GetCollectionCount
func GetCollectionCount(colID C.int) C.longlong {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return -1
	}

	count, err := col.Count(context.Background())
	if err != nil {
		return -1
	}
	return C.longlong(count)
}

//export UpdateVectorIfVersion
func UpdateVectorIfVersion(colID C.int, id *C.char, vec *C.float, dim C.int, metadataJSON *C.char, expectedVersion C.ulonglong) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: collection not found")
	}

	goID := C.GoString(id)
	goDim := int(dim)
	slice := unsafe.Slice(vec, goDim)
	goVec := make([]float32, goDim)
	for i := 0; i < goDim; i++ {
		goVec[i] = float32(slice[i])
	}

	var metadata map[string]interface{}
	if metadataJSON != nil {
		goMetaJSON := C.GoString(metadataJSON)
		if goMetaJSON != "" {
			_ = json.Unmarshal([]byte(goMetaJSON), &metadata)
		}
	}

	ctx := context.Background()
	err := col.UpdateIfVersion(ctx, goID, goVec, metadata, uint64(expectedVersion))
	if err != nil {
		return C.CString(fmt.Sprintf("error: %v", err))
	}
	return nil
}

//export DeleteVectorIfVersion
func DeleteVectorIfVersion(colID C.int, id *C.char, expectedVersion C.ulonglong) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: collection not found")
	}

	goID := C.GoString(id)
	ctx := context.Background()
	err := col.DeleteIfVersion(ctx, goID, uint64(expectedVersion))
	if err != nil {
		return C.CString(fmt.Sprintf("error: %v", err))
	}
	return nil
}

//export SetCollectionMemoryLimit
func SetCollectionMemoryLimit(colID C.int, limit C.longlong) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: collection not found")
	}

	err := col.SetMemoryLimit(int64(limit))
	if err != nil {
		return C.CString(fmt.Sprintf("error: %v", err))
	}
	return nil
}

//export GetCollectionMemoryUsage
func GetCollectionMemoryUsage(colID C.int) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString(`{"error": "collection not found"}`)
	}

	usage, err := col.GetMemoryUsage(context.Background())
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}
	bytes, _ := json.Marshal(usage)
	return C.CString(string(bytes))
}

//export TriggerCollectionGC
func TriggerCollectionGC(colID C.int) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: collection not found")
	}

	err := col.TriggerGC()
	if err != nil {
		return C.CString(fmt.Sprintf("error: %v", err))
	}
	return nil
}

//export EnableMemoryMapping
func EnableMemoryMapping(colID C.int, path *C.char) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: collection not found")
	}

	goPath := C.GoString(path)
	err := col.EnableMemoryMapping(goPath)
	if err != nil {
		return C.CString(fmt.Sprintf("error: %v", err))
	}
	return nil
}

//export DisableMemoryMapping
func DisableMemoryMapping(colID C.int) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: collection not found")
	}

	err := col.DisableMemoryMapping()
	if err != nil {
		return C.CString(fmt.Sprintf("error: %v", err))
	}
	return nil
}

//export SaveIndex
func SaveIndex(colID C.int, path *C.char) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: collection not found")
	}

	goPath := C.GoString(path)
	err := col.SaveIndex(context.Background(), goPath)
	if err != nil {
		return C.CString(fmt.Sprintf("error: %v", err))
	}
	return nil
}

//export LoadIndex
func LoadIndex(colID C.int, path *C.char) *C.char {
	mu.RLock()
	col, ok := collections[int(colID)]
	mu.RUnlock()

	if !ok {
		return C.CString("error: collection not found")
	}

	goPath := C.GoString(path)
	err := col.LoadIndex(context.Background(), goPath)
	if err != nil {
		return C.CString(fmt.Sprintf("error: %v", err))
	}
	return nil
}

//export FreeString
func FreeString(str *C.char) {
	if str != nil {
		C.free(unsafe.Pointer(str))
	}
}

//export DatabaseQuery
func DatabaseQuery(dbID C.int, sql *C.char) *C.char {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return C.CString(`{"error": "database not found"}`)
	}

	goSQL := C.GoString(sql)

	results, err := db.Query(context.Background(), goSQL)
	if err != nil {
		errBytes, _ := json.Marshal(map[string]string{"error": err.Error()})
		return C.CString(string(errBytes))
	}

	bytes, err := json.Marshal(results)
	if err != nil {
		errBytes, _ := json.Marshal(map[string]string{"error": fmt.Sprintf("failed to marshal results: %v", err)})
		return C.CString(string(errBytes))
	}
	return C.CString(string(bytes))
}

//export DatabaseQueryWithParams
func DatabaseQueryWithParams(dbID C.int, sql *C.char, paramsJSON *C.char) *C.char {
	mu.RLock()
	db, ok := dbs[int(dbID)]
	mu.RUnlock()

	if !ok {
		return C.CString(`{"error": "database not found"}`)
	}

	goSQL := C.GoString(sql)

	var params libravdb.QueryParams
	if paramsJSON != nil {
		goParamsJSON := C.GoString(paramsJSON)
		if goParamsJSON != "" {
			if err := json.Unmarshal([]byte(goParamsJSON), &params); err != nil {
				errBytes, _ := json.Marshal(map[string]string{"error": fmt.Sprintf("failed to parse params: %v", err)})
				return C.CString(string(errBytes))
			}
			// Normalize []interface{} to []float32 for vectors
			for k, v := range params {
				if slice, ok := v.([]interface{}); ok {
					vec := make([]float32, len(slice))
					isVec := true
					for i, val := range slice {
						if f, ok := val.(float64); ok {
							vec[i] = float32(f)
						} else {
							isVec = false
							break
						}
					}
					if isVec {
						params[k] = vec
					}
				}
			}
		}
	}

	results, err := db.QueryWithParams(context.Background(), goSQL, params)
	if err != nil {
		errBytes, _ := json.Marshal(map[string]string{"error": err.Error()})
		return C.CString(string(errBytes))
	}

	bytes, err := json.Marshal(results)
	if err != nil {
		errBytes, _ := json.Marshal(map[string]string{"error": fmt.Sprintf("failed to marshal results: %v", err)})
		return C.CString(string(errBytes))
	}
	return C.CString(string(bytes))
}

func main() {
	// Required for c-shared build mode, but ignored
}
