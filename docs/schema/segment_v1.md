# Segment Format Specification v1

This document defines the binary format specification for LibraVDB segment files (SSTables) version 1.

## Overview

Segment files are immutable, sorted collections of vector entries stored on disk. They form the foundation of LibraVDB's LSM-tree storage architecture.

## File Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                         File Header                             │  (32 bytes)
├─────────────────────────────────────────────────────────────────┤
│                         Data Blocks                             │  (variable)
├─────────────────────────────────────────────────────────────────┤
│                         Index Block                             │  (variable)
├─────────────────────────────────────────────────────────────────┤
│                        Bloom Filter                             │  (variable)
├─────────────────────────────────────────────────────────────────┤
│                         Metadata                                │  (variable)
├─────────────────────────────────────────────────────────────────┤
│                          Footer                                 │  (32 bytes)
└─────────────────────────────────────────────────────────────────┘
```

## File Header (32 bytes)

The file header appears at the beginning of every segment file:

```
Offset | Size | Field           | Description
-------|------|-----------------|----------------------------------
0      | 4    | Magic           | Magic number: 0x4C565342 ("LVSB")
4      | 2    | Version         | Format version (1)
6      | 2    | Flags           | Feature flags (see below)
8      | 4    | Compression     | Compression algorithm ID
12     | 4    | BlockSize       | Data block size in bytes
16     | 4    | EntryCount      | Total number of entries
20     | 4    | Dimension       | Vector dimension
24     | 4    | HeaderChecksum  | CRC32 of header bytes 0-23
28     | 4    | Reserved        | Reserved for future use
```

### Magic Number
- **Value**: `0x4C565342` (ASCII: "LVSB" - LibraVDB Segment Binary)
- **Purpose**: File format identification and corruption detection

### Version
- **Value**: `1` for this specification
- **Purpose**: Format version for backward compatibility

### Flags (16 bits)
```
Bit | Description
----|------------------------------------------
0   | Has Quantization (1 if vectors are quantized)
1   | Has Metadata (1 if entries contain metadata)
2   | Has Bloom Filter (1 if bloom filter is present)
3   | Compressed Data Blocks (1 if data blocks are compressed)
4   | Compressed Index (1 if index block is compressed)
5-15| Reserved (must be 0)
```

### Compression Algorithm IDs
```
ID | Algorithm
---|----------
0  | None
1  | LZ4
2  | Zstd
3  | Snappy
4  | Reserved
```

## Data Blocks

Data blocks contain the actual vector entries, organized for efficient access:

### Data Block Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      Block Header                               │  (16 bytes)
├─────────────────────────────────────────────────────────────────┤
│                      Entry Data                                 │  (variable)
├─────────────────────────────────────────────────────────────────┤
│                      Block Footer                               │  (8 bytes)
└─────────────────────────────────────────────────────────────────┘
```

### Block Header (16 bytes)

```
Offset | Size | Field           | Description
-------|------|-----------------|----------------------------------
0      | 4    | EntryCount      | Number of entries in this block
4      | 4    | UncompressedSize| Size before compression
8      | 4    | CompressedSize  | Size after compression (0 if uncompressed)
12     | 4    | FirstKeyLength  | Length of first key in block
```

### Entry Format

Each entry within a data block follows this format:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Entry Header                               │  (variable)
├─────────────────────────────────────────────────────────────────┤
│                      Vector Data                                │  (dimension * 4 bytes)
├─────────────────────────────────────────────────────────────────┤
│                      Metadata                                   │  (variable, optional)
└─────────────────────────────────────────────────────────────────┘
```

#### Entry Header

```
Offset | Size | Field           | Description
-------|------|-----------------|----------------------------------
0      | 2    | KeyLength       | Length of entry ID string
2      | 2    | Flags           | Entry flags (see below)
4      | 4    | VectorOffset    | Offset to vector data within entry
8      | 4    | MetadataOffset  | Offset to metadata (0 if none)
12     | 4    | MetadataLength  | Length of metadata (0 if none)
```

#### Entry Flags (16 bits)
```
Bit | Description
----|------------------------------------------
0   | Has Metadata (1 if metadata is present)
1   | Quantized Vector (1 if vector is quantized)
2   | Deleted (1 if entry is tombstone)
3-15| Reserved (must be 0)
```

#### Vector Data Format

**Unquantized Vectors:**
```
┌─────────────────────────────────────────────────────────────────┐
│  float32[0]  │  float32[1]  │  ...  │  float32[dimension-1]   │
└─────────────────────────────────────────────────────────────────┘
```

**Quantized Vectors:**
```
┌─────────────────────────────────────────────────────────────────┐
│ Quantization │  Quantized   │  Quantized   │  ...  │ Quantized │
│   Header     │   Subvector  │   Subvector  │       │ Subvector │
│  (8 bytes)   │      1       │      2       │       │     N     │
└─────────────────────────────────────────────────────────────────┘
```

##### Quantization Header (8 bytes)
```
Offset | Size | Field           | Description
-------|------|-----------------|----------------------------------
0      | 1    | Type            | Quantization type (1=PQ, 2=SQ)
1      | 1    | Codebooks       | Number of codebooks (PQ only)
2      | 1    | Bits            | Bits per code
3      | 1    | Reserved        | Reserved
4      | 4    | OriginalSize    | Size of original vector in bytes
```

#### Metadata Format

Metadata is stored as MessagePack-encoded key-value pairs:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MessagePack Data                             │
└─────────────────────────────────────────────────────────────────┘
```

### Block Footer (8 bytes)

```
Offset | Size | Field           | Description
-------|------|-----------------|----------------------------------
0      | 4    | BlockChecksum   | CRC32 of entire block (excluding footer)
4      | 4    | Reserved        | Reserved for future use
```

## Index Block

The index block provides fast key lookup within the segment:

### Index Block Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      Index Header                               │  (16 bytes)
├─────────────────────────────────────────────────────────────────┤
│                      Key Index                                  │  (variable)
├─────────────────────────────────────────────────────────────────┤
│                      Block Index                                │  (variable)
└─────────────────────────────────────────────────────────────────┘
```

### Index Header (16 bytes)

```
Offset | Size | Field           | Description
-------|------|-----------------|----------------------------------
0      | 4    | KeyIndexSize    | Size of key index in bytes
4      | 4    | BlockIndexSize  | Size of block index in bytes
8      | 4    | IndexChecksum   | CRC32 of index data
12     | 4    | Reserved        | Reserved
```

### Key Index

The key index maps entry IDs to their locations:

```
┌─────────────────────────────────────────────────────────────────┐
│  Entry Count (4 bytes)                                          │
├─────────────────────────────────────────────────────────────────┤
│  Key Entry 1                                                    │
├─────────────────────────────────────────────────────────────────┤
│  Key Entry 2                                                    │
├─────────────────────────────────────────────────────────────────┤
│  ...                                                            │
├─────────────────────────────────────────────────────────────────┤
│  Key Entry N                                                    │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Entry Format

```
Offset | Size | Field           | Description
-------|------|-----------------|----------------------------------
0      | 2    | KeyLength       | Length of key string
2      | 2    | BlockIndex      | Index of containing data block
4      | 4    | BlockOffset     | Offset within data block
8      | 4    | EntrySize       | Total size of entry
```

### Block Index

The block index provides metadata about each data block:

```
┌─────────────────────────────────────────────────────────────────┐
│  Block Count (4 bytes)                                          │
├─────────────────────────────────────────────────────────────────┤
│  Block Entry 1                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Block Entry 2                                                  │
├─────────────────────────────────────────────────────────────────┤
│  ...                                                            │
├─────────────────────────────────────────────────────────────────┤
│  Block Entry N                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Block Entry Format

```
Offset | Size | Field           | Description
-------|------|-----------------|----------------------------------
0      | 4    | BlockOffset     | Offset of block in file
4      | 4    | BlockSize       | Size of block in bytes
8      | 2    | FirstKeyLength  | Length of first key in block
10     | 2    | LastKeyLength   | Length of last key in block
12     | 4    | EntryCount      | Number of entries in block
```

## Bloom Filter

The bloom filter enables fast negative lookups:

### Bloom Filter Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Bloom Filter Header                          │  (16 bytes)
├─────────────────────────────────────────────────────────────────┤
│                    Bit Array                                    │  (variable)
└─────────────────────────────────────────────────────────────────┘
```

### Bloom Filter Header (16 bytes)

```
Offset | Size | Field           | Description
-------|------|-----------------|----------------------------------
0      | 4    | BitArraySize    | Size of bit array in bytes
4      | 4    | HashFunctions   | Number of hash functions used
8      | 4    | ExpectedItems   | Expected number of items
12     | 4    | FilterChecksum  | CRC32 of bit array
```

### Bit Array

The bit array is stored as a sequence of 64-bit words in little-endian format.

## Metadata Section

The metadata section contains segment-level information:

### Metadata Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Metadata Header                              │  (32 bytes)
├─────────────────────────────────────────────────────────────────┤
│                    Statistics                                   │  (variable)
├─────────────────────────────────────────────────────────────────┤
│                    Schema Information                           │  (variable)
├─────────────────────────────────────────────────────────────────┤
│                    Custom Properties                            │  (variable)
└─────────────────────────────────────────────────────────────────┘
```

### Metadata Header (32 bytes)

```
Offset | Size | Field           | Description
-------|------|-----------------|----------------------------------
0      | 8    | CreationTime    | Unix timestamp (nanoseconds)
8      | 8    | ModificationTime| Unix timestamp (nanoseconds)
16     | 4    | StatisticsSize  | Size of statistics section
20     | 4    | SchemaSize      | Size of schema section
24     | 4    | PropertiesSize  | Size of properties section
28     | 4    | MetadataChecksum| CRC32 of metadata sections
```

### Statistics Section

```
┌─────────────────────────────────────────────────────────────────┐
│  Min Key Length (4 bytes)                                       │
├─────────────────────────────────────────────────────────────────┤
│  Max Key Length (4 bytes)                                       │
├─────────────────────────────────────────────────────────────────┤
│  Total Uncompressed Size (8 bytes)                              │
├─────────────────────────────────────────────────────────────────┤
│  Total Compressed Size (8 bytes)                                │
├─────────────────────────────────────────────────────────────────┤
│  First Key (variable length, null-terminated)                   │
├─────────────────────────────────────────────────────────────────┤
│  Last Key (variable length, null-terminated)                    │
└─────────────────────────────────────────────────────────────────┘
```

### Schema Information

Schema information is stored as JSON:

```json
{
  "version": 1,
  "dimension": 768,
  "distance_metric": "cosine",
  "quantization": {
    "type": "product",
    "codebooks": 8,
    "bits": 8
  },
  "metadata_schema": {
    "title": "string",
    "category": "string",
    "tags": "string_array"
  }
}
```

## Footer (32 bytes)

The footer appears at the end of the file:

```
Offset | Size | Field           | Description
-------|------|-----------------|----------------------------------
0      | 8    | IndexOffset     | Offset to index block
8      | 8    | BloomOffset     | Offset to bloom filter (0 if none)
16     | 8    | MetadataOffset  | Offset to metadata section
24     | 4    | FileChecksum    | CRC32 of entire file (excluding this field)
28     | 4    | FooterMagic     | Magic number: 0x4C565346 ("LVSF")
```

## Checksum Calculation

All checksums use CRC32 (IEEE 802.3 polynomial):

1. **Header Checksum**: CRC32 of header bytes 0-23
2. **Block Checksum**: CRC32 of block data (excluding footer)
3. **Index Checksum**: CRC32 of key index + block index
4. **Filter Checksum**: CRC32 of bloom filter bit array
5. **Metadata Checksum**: CRC32 of all metadata sections
6. **File Checksum**: CRC32 of entire file excluding the checksum field itself

## Compression

When compression is enabled:

1. **Data Blocks**: Each block is compressed independently
2. **Index Block**: The entire index block may be compressed
3. **Metadata**: Metadata sections may be compressed

Compression is applied after serialization but before checksum calculation.

## Endianness

All multi-byte integers are stored in little-endian format.

## Alignment

- All offsets are 4-byte aligned
- Vector data is 16-byte aligned for SIMD optimization
- Blocks are padded to maintain alignment

## Version Compatibility

This specification defines version 1 of the segment format. Future versions will:

1. Increment the version number in the header
2. Maintain backward compatibility for reading
3. Add new features through flag bits
4. Use reserved fields for new functionality

## Example File Layout

```
Offset    | Size  | Section
----------|-------|------------------
0         | 32    | File Header
32        | 4096  | Data Block 1
4128      | 4096  | Data Block 2
8224      | 2048  | Index Block
10272     | 512   | Bloom Filter
10784     | 256   | Metadata
11040     | 32    | Footer
```

## Implementation Notes

1. **Reading**: Always validate checksums before processing data
2. **Writing**: Calculate checksums after all data is written
3. **Corruption**: Any checksum failure should trigger error handling
4. **Performance**: Use memory mapping for large files when possible
5. **Concurrency**: Multiple readers are safe; writers must coordinate

This specification ensures data integrity, efficient access patterns, and extensibility for future enhancements to LibraVDB's storage format.