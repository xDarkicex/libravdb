package io.libravdb

import kotlinx.serialization.json.*

class Filter private constructor(private val node: JsonObject) {
    
    companion object {
        fun eq(field: String, value: Any): Filter = createSimple("eq", field, value)
        fun neq(field: String, value: Any): Filter = createSimple("neq", field, value)
        fun gt(field: String, value: Any): Filter = createSimple("gt", field, value)
        
        fun and(filters: List<Filter>): Filter = createLogical("and", filters)
        fun or(filters: List<Filter>): Filter = createLogical("or", filters)

        private fun createSimple(type: String, field: String, value: Any): Filter {
            val jsonValue = when (value) {
                is Number -> JsonPrimitive(value)
                is String -> JsonPrimitive(value)
                is Boolean -> JsonPrimitive(value)
                else -> throw IllegalArgumentException("Unsupported filter value type")
            }
            return Filter(buildJsonObject {
                put("type", type)
                put("field", field)
                put("value", jsonValue)
            })
        }

        private fun createLogical(type: String, filters: List<Filter>): Filter {
            return Filter(buildJsonObject {
                put("type", type)
                put("value", JsonArray(filters.map { it.node }))
            })
        }
    }

    fun toJsonString(): String {
        return node.toString()
    }
}
