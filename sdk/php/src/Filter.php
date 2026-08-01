<?php
namespace LibraVDB;

class Filter {
    public static function eq(string $field, $value): array {
        return ["type" => "eq", "field" => $field, "value" => $value];
    }

    public static function gt(string $field, $value): array {
        return ["type" => "gt", "field" => $field, "value" => $value];
    }

    public static function lt(string $field, $value): array {
        return ["type" => "lt", "field" => $field, "value" => $value];
    }

    public static function in(string $field, array $values): array {
        return ["type" => "contains_any", "field" => $field, "values" => $values];
    }

    public static function and(array ...$filters): array {
        return ["type" => "and", "filters" => $filters];
    }

    public static function or(array ...$filters): array {
        return ["type" => "or", "filters" => $filters];
    }

    public static function toJson(array $filter): string {
        return json_encode($filter);
    }
}
