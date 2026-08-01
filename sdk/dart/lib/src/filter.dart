import 'dart:convert';

class Filter {
  final Map<String, dynamic> _node;

  Filter._(this._node);

  static Filter eq(String field, dynamic value) {
    return Filter._({'type': 'eq', 'field': field, 'value': value});
  }

  static Filter neq(String field, dynamic value) {
    return Filter._({'type': 'neq', 'field': field, 'value': value});
  }

  static Filter gt(String field, dynamic value) {
    return Filter._({'type': 'gt', 'field': field, 'value': value});
  }

  static Filter and(List<Filter> filters) {
    return Filter._({'type': 'and', 'value': filters.map((f) => f._node).toList()});
  }

  static Filter or(List<Filter> filters) {
    return Filter._({'type': 'or', 'value': filters.map((f) => f._node).toList()});
  }

  String toJsonString() {
    return jsonEncode(_node);
  }
}
