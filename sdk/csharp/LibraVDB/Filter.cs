using System;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace LibraVDB
{
    public class Filter
    {
        private readonly JsonObject _node;

        private Filter(JsonObject node)
        {
            _node = node;
        }

        public string ToJson()
        {
            return _node.ToJsonString();
        }

        public static Filter Eq(string field, object value) => CreateComparison("eq", field, value);
        public static Filter Gt(string field, object value) => CreateComparison("gt", field, value);
        public static Filter Lt(string field, object value) => CreateComparison("lt", field, value);
        public static Filter In(string field, object[] values) => CreateComparisonList("contains_any", field, values);

        public static Filter And(params Filter[] filters) => CreateLogical("and", filters);
        public static Filter Or(params Filter[] filters) => CreateLogical("or", filters);

        private static Filter CreateComparison(string op, string field, object value)
        {
            var node = new JsonObject
            {
                ["type"] = op,
                ["field"] = field,
                ["value"] = JsonSerializer.SerializeToNode(value)
            };
            return new Filter(node);
        }

        private static Filter CreateComparisonList(string op, string field, object[] values)
        {
            var node = new JsonObject
            {
                ["type"] = op,
                ["field"] = field,
                ["values"] = JsonSerializer.SerializeToNode(values)
            };
            return new Filter(node);
        }

        private static Filter CreateLogical(string op, Filter[] filters)
        {
            var array = new JsonArray();
            foreach (var filter in filters)
            {
                array.Add(filter._node.DeepClone());
            }

            var node = new JsonObject
            {
                ["type"] = op,
                ["filters"] = array
            };

            return new Filter(node);
        }
    }
}
