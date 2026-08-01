package io.libravdb;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.util.List;

public class Filter {
    private static final ObjectMapper mapper = new ObjectMapper();
    private final ObjectNode node;

    private Filter(ObjectNode node) {
        this.node = node;
    }

    public static Filter eq(String field, Object value) {
        return createSimpleFilter("eq", field, value);
    }

    public static Filter neq(String field, Object value) {
        return createSimpleFilter("neq", field, value);
    }

    public static Filter gt(String field, Object value) {
        return createSimpleFilter("gt", field, value);
    }

    public static Filter and(List<Filter> filters) {
        return createLogicalFilter("and", filters);
    }

    public static Filter or(List<Filter> filters) {
        return createLogicalFilter("or", filters);
    }

    private static Filter createSimpleFilter(String type, String field, Object value) {
        ObjectNode n = mapper.createObjectNode();
        n.put("type", type);
        n.put("field", field);
        n.putPOJO("value", value);
        return new Filter(n);
    }

    private static Filter createLogicalFilter(String type, List<Filter> filters) {
        ObjectNode n = mapper.createObjectNode();
        n.put("type", type);
        ArrayNode arr = mapper.createArrayNode();
        for (Filter f : filters) {
            arr.add(f.node);
        }
        n.set("value", arr);
        return new Filter(n);
    }

    public String toJsonString() {
        return node.toString();
    }
}
