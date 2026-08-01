<?php
namespace LibraVDB;

require_once __DIR__ . '/LibraVDB.php';

class Collection {
    private int $dbID;
    private int $colID;
    private string $name;
    private int $dimension;

    public function __construct(int $dbID, int $colID, string $name, int $dimension) {
        $this->dbID = $dbID;
        $this->colID = $colID;
        $this->name = $name;
        $this->dimension = $dimension;
    }

    public function insert(string $id, array $vector, string $metadata = "{}"): void {
        if (count($vector) !== $this->dimension) {
            throw new \Exception("Vector dimension mismatch");
        }

        $ffi = LibraVDB::getFFI();
        $c_vector = $ffi->new("float[" . $this->dimension . "]");
        for ($i = 0; $i < $this->dimension; $i++) {
            $c_vector[$i] = $vector[$i];
        }

        $errPtr = $ffi->InsertVector($this->colID, $id, $ffi->cast("float*", $c_vector), $this->dimension, $metadata);
        LibraVDB::checkError($errPtr);
    }

    public function insertBatch(array $ids, array $vectors, ?array $metadata = null): void {
        $count = count($ids);
        if (count($vectors) !== $count) {
            throw new \Exception("Length of ids and vectors must match");
        }

        $ffi = LibraVDB::getFFI();

        // 1. Flatten vectors
        $c_vectors = $ffi->new("float[" . ($count * $this->dimension) . "]");
        for ($i = 0; $i < $count; $i++) {
            if (count($vectors[$i]) !== $this->dimension) {
                throw new \Exception("Vector dimension mismatch at index $i");
            }
            for ($j = 0; $j < $this->dimension; $j++) {
                $c_vectors[$i * $this->dimension + $j] = $vectors[$i][$j];
            }
        }

        // 2. Prepare string arrays
        $c_ids = $ffi->new("const char*[$count]");
        $c_metas = $ffi->new("const char*[$count]");
        
        // We must keep references to FFI strings so they aren't garbage collected
        $strRefs = [];

        for ($i = 0; $i < $count; $i++) {
            // Memory must be allocated or C string conversion handled
            // PHP FFI automatically converts strings when passing them as parameters,
            // but for arrays of strings, we need to manually allocate them.
            // Using FFI::new("char[]") for each string
            
            $idStr = $ids[$i];
            $idLen = strlen($idStr) + 1;
            $cId = $ffi->new("char[$idLen]");
            \FFI::memcpy($cId, $idStr, strlen($idStr));
            $c_ids[$i] = $ffi->cast("const char*", $cId);
            $strRefs[] = $cId;

            $metaStr = $metadata !== null ? $metadata[$i] : "{}";
            $metaLen = strlen($metaStr) + 1;
            $cMeta = $ffi->new("char[$metaLen]");
            \FFI::memcpy($cMeta, $metaStr, strlen($metaStr));
            $c_metas[$i] = $ffi->cast("const char*", $cMeta);
            $strRefs[] = $cMeta;
        }

        $errPtr = $ffi->InsertBatch(
            $this->colID,
            $c_ids,
            $ffi->cast("float*", $c_vectors),
            $count,
            $this->dimension,
            $c_metas
        );

        LibraVDB::checkError($errPtr);
    }

    public function search(array $vector, int $k, string $filter = "{}"): string {
        if (count($vector) !== $this->dimension) {
            throw new \Exception("Vector dimension mismatch");
        }

        $ffi = LibraVDB::getFFI();
        $c_vector = $ffi->new("float[" . $this->dimension . "]");
        for ($i = 0; $i < $this->dimension; $i++) {
            $c_vector[$i] = $vector[$i];
        }

        $resPtr = $ffi->QueryVector($this->colID, $ffi->cast("float*", $c_vector), $this->dimension, $k, $filter);
        return LibraVDB::extractString($resPtr);
    }

    public function scan(int $offset = 0, int $limit = 100): string {
        $ffi = LibraVDB::getFFI();
        $resPtr = $ffi->ScanCollection($this->colID, $offset, $limit);
        return LibraVDB::extractString($resPtr);
    }

    public function delete(string $id): void {
        $ffi = LibraVDB::getFFI();
        $errPtr = $ffi->DeleteVector($this->colID, $id);
        LibraVDB::checkError($errPtr);
    }
}
