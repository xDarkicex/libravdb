import Foundation

public struct Filter: Encodable {
    public let type: String
    public let field: String?
    public let value: AnyEncodable?
    public let values: [AnyEncodable]?
    public let filters: [Filter]?

    private init(type: String, field: String? = nil, value: AnyEncodable? = nil, values: [AnyEncodable]? = nil, filters: [Filter]? = nil) {
        self.type = type
        self.field = field
        self.value = value
        self.values = values
        self.filters = filters
    }

    public func toJson() -> String {
        let encoder = JSONEncoder()
        guard let data = try? encoder.encode(self) else { return "{}" }
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    public static func eq(_ field: String, _ value: Encodable) -> Filter {
        return Filter(type: "eq", field: field, value: AnyEncodable(value))
    }

    public static func gt(_ field: String, _ value: Encodable) -> Filter {
        return Filter(type: "gt", field: field, value: AnyEncodable(value))
    }

    public static func lt(_ field: String, _ value: Encodable) -> Filter {
        return Filter(type: "lt", field: field, value: AnyEncodable(value))
    }

    public static func `in`(_ field: String, _ values: [Encodable]) -> Filter {
        return Filter(type: "contains_any", field: field, values: values.map { AnyEncodable($0) })
    }

    public static func and(_ filters: Filter...) -> Filter {
        return Filter(type: "and", filters: filters)
    }

    public static func or(_ filters: Filter...) -> Filter {
        return Filter(type: "or", filters: filters)
    }
}

public struct AnyEncodable: Encodable {
    public let value: Encodable

    public init(_ value: Encodable) {
        self.value = value
    }

    public func encode(to encoder: Encoder) throws {
        try value.encode(to: encoder)
    }
}
