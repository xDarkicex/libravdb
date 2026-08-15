<?php
namespace LibraVDB;

require_once __DIR__ . '/LibraVDB.php';
require_once __DIR__ . '/Collection.php';

class Database {
    private int $dbID;
    private string $path;

    public function __construct(string $path) {
        $ffi = LibraVDB::getFFI();
        $this->dbID = $ffi->OpenDB($path);
        
        if ($this->dbID < 0) {
            throw new \Exception("Failed to open database at path: $path");
        }
        $this->path = $path;
    }

    public function createCollection(string $name, int $dimension): Collection {
        $ffi = LibraVDB::getFFI();
        $colID = $ffi->CreateCollection($this->dbID, $name, $dimension);
        
        if ($colID < 0) {
            throw new \Exception("Failed to create collection: $name");
        }
        
        return new Collection($this->dbID, $colID, $name, $dimension);
    }

    public function getCollection(string $name, int $dimension): Collection {
        $ffi = LibraVDB::getFFI();
        $colID = $ffi->GetCollection($this->dbID, $name);
        
        if ($colID < 0) {
            throw new \Exception("Failed to get collection: $name");
        }
        
        return new Collection($this->dbID, $colID, $name, $dimension);
    }

    public function vacuum(): void {
        $ffi = LibraVDB::getFFI();
        $errPtr = $ffi->Vacuum($this->dbID);
        LibraVDB::checkError($errPtr);
    }

    public function dropDatabase(): void {
        $ffi = LibraVDB::getFFI();
        $errPtr = $ffi->DropDatabase($this->dbID);
        LibraVDB::checkError($errPtr);
    }

    public function query(string $sql): string {
        $ffi = LibraVDB::getFFI();
        $resPtr = $ffi->DatabaseQuery($this->dbID, $sql);
        return LibraVDB::extractQueryResult($resPtr);
    }

    public function queryWithParams(string $sql, ?string $params = null): string {
        $ffi = LibraVDB::getFFI();
        $paramsStr = $params ?? "";
        $resPtr = $ffi->DatabaseQueryWithParams($this->dbID, $sql, $paramsStr);
        return LibraVDB::extractQueryResult($resPtr);
    }

    public function close(): void {
        if ($this->dbID >= 0) {
            LibraVDB::getFFI()->CloseDB($this->dbID);
            $this->dbID = -1;
        }
    }

    public function __destruct() {
        $this->close();
    }
}
