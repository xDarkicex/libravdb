package pgwire

import "github.com/xDarkicex/libravdb/internal/catalog"

// PostgreSQL wire-protocol type OIDs.
// These are the standard OIDs used in RowDescription messages.
const (
	// Numeric types
	OIDInt2   = 21  // smallint
	OIDInt4   = 23  // integer
	OIDInt8   = 20  // bigint
	OIDFloat4 = 700 // real
	OIDFloat8 = 701 // double precision

	// String types
	OIDChar    = 18   // "char"
	OIDText    = 25   // text
	OIDOID     = 26   // oid
	OIDVarchar = 1043 // varchar
	OIDBPChar  = 1042 // char
	OIDName    = 19   // name (used for identifiers)

	// Boolean
	OIDBool = 16 // boolean

	// Date/time
	OIDTimestamp   = 1114 // timestamp
	OIDTimestamptz = 1184 // timestamptz
	OIDDate        = 1082 // date

	// Array types
	OIDInt2Array   = 1005  // _int2
	OIDFloat4Array = 1021  // _float4
	OIDFloat8Array = 1022  // _float8
	OIDTextArray   = 1009  // _text
	OIDInt4Array   = 1007  // _int4
	OIDInt8Array   = 1016  // _int8
	OIDBoolArray   = 1000  // _bool
	OIDOIDArray    = 1028  // _oid
	OIDJSON        = 114   // json
	OIDJSONB       = 3802  // jsonb
	OIDUUID        = 2950  // uuid
	OIDVector      = 16384 // vector extension type used by the native vector column
)

// PGTypeName returns the PostgreSQL type name for a given OID.
func PGTypeName(oid uint32) string {
	switch oid {
	case OIDInt2:
		return "int2"
	case OIDInt4:
		return "int4"
	case OIDInt8:
		return "int8"
	case OIDFloat4:
		return "float4"
	case OIDFloat8:
		return "float8"
	case OIDChar:
		return "char"
	case OIDText:
		return "text"
	case OIDOID:
		return "oid"
	case OIDName:
		return "name"
	case OIDVarchar:
		return "varchar"
	case OIDBPChar:
		return "bpchar"
	case OIDBool:
		return "bool"
	case OIDTimestamp:
		return "timestamp"
	case OIDTimestamptz:
		return "timestamptz"
	case OIDDate:
		return "date"
	case OIDFloat4Array:
		return "_float4"
	case OIDInt2Array:
		return "_int2"
	case OIDFloat8Array:
		return "_float8"
	case OIDTextArray:
		return "_text"
	case OIDInt4Array:
		return "_int4"
	case OIDInt8Array:
		return "_int8"
	case OIDBoolArray:
		return "_bool"
	case OIDOIDArray:
		return "_oid"
	case OIDJSON:
		return "json"
	case OIDJSONB:
		return "jsonb"
	case OIDUUID:
		return "uuid"
	case OIDVector:
		return "vector"
	default:
		return "text"
	}
}

// pgTypeSize returns the PostgreSQL RowDescription data-type size. A negative
// value means variable-width, as required by the wire protocol.
func pgTypeSize(oid uint32) int16 {
	switch oid {
	case OIDBool, OIDChar:
		return 1
	case OIDInt2:
		return 2
	case OIDInt4, OIDFloat4, OIDOID:
		return 4
	case OIDInt8, OIDFloat8, OIDTimestamp, OIDTimestamptz:
		return 8
	case OIDName:
		return 64
	case OIDDate:
		return 4
	default:
		return -1
	}
}

// catalogTypeToOID maps a catalog column type to its PostgreSQL wire-protocol OID.
func catalogTypeToOID(catType uint16) uint32 {
	switch catType {
	case catalog.TypeInt:
		return OIDInt4
	case catalog.TypeFloat:
		return OIDFloat8
	case catalog.TypeString:
		return OIDText
	case catalog.TypeVector:
		return OIDFloat4Array
	case catalog.TypeBigInt:
		return OIDInt8
	case catalog.TypeOID:
		return OIDOID
	case catalog.TypeName:
		return OIDName
	case catalog.TypeChar:
		return OIDChar
	case catalog.TypeSmallInt:
		return OIDInt2
	case catalog.TypeBool:
		return OIDBool
	case catalog.TypeFloat4:
		return OIDFloat4
	case catalog.TypeJSON:
		return OIDJSON
	case catalog.TypeJSONB:
		return OIDJSONB
	case catalog.TypeUUID:
		return OIDUUID
	case catalog.TypeTimestamp:
		return OIDTimestamptz
	default:
		return OIDText
	}
}
