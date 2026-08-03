package pgwire

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
	OIDText    = 25   // text
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
	OIDFloat4Array = 1021 // _float4
	OIDFloat8Array = 1022 // _float8
	OIDTextArray   = 1009 // _text
	OIDInt4Array   = 1007 // _int4
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
	case OIDText:
		return "text"
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
	default:
		return "text"
	}
}

// catalogTypeToOID maps a catalog column type to its PostgreSQL wire-protocol OID.
func catalogTypeToOID(catType uint16) uint32 {
	switch catType {
	case 1: // TypeInt
		return OIDInt4
	case 2: // TypeFloat
		return OIDFloat8
	case 3: // TypeString
		return OIDText
	case 4: // TypeVector
		return OIDFloat4Array
	default:
		return OIDText
	}
}
