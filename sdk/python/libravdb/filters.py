from typing import List, Any, Optional

class Filter:
    def to_json(self) -> dict:
        raise NotImplementedError()

class Eq(Filter):
    def __init__(self, field: str, value: Any):
        self.field = field
        self.value = value
        
    def to_json(self) -> dict:
        return {"type": "eq", "field": self.field, "value": self.value}

class Gt(Filter):
    def __init__(self, field: str, value: Any):
        self.field = field
        self.value = value
        
    def to_json(self) -> dict:
        return {"type": "gt", "field": self.field, "value": self.value}

class Lt(Filter):
    def __init__(self, field: str, value: Any):
        self.field = field
        self.value = value
        
    def to_json(self) -> dict:
        return {"type": "lt", "field": self.field, "value": self.value}

class Between(Filter):
    def __init__(self, field: str, min_val: Any, max_val: Any):
        self.field = field
        self.min_val = min_val
        self.max_val = max_val
        
    def to_json(self) -> dict:
        return {"type": "between", "field": self.field, "value": [self.min_val, self.max_val]}

class ContainsAny(Filter):
    def __init__(self, field: str, values: List[Any]):
        self.field = field
        self.values = values
        
    def to_json(self) -> dict:
        return {"type": "contains_any", "field": self.field, "values": self.values}

class ContainsAll(Filter):
    def __init__(self, field: str, values: List[Any]):
        self.field = field
        self.values = values
        
    def to_json(self) -> dict:
        return {"type": "contains_all", "field": self.field, "values": self.values}

class ExactMatch(Filter):
    def __init__(self, field: str, values: List[Any]):
        self.field = field
        self.values = values
        
    def to_json(self) -> dict:
        return {"type": "exact_match", "field": self.field, "values": self.values}

class And(Filter):
    def __init__(self, *filters: Filter):
        self.filters = filters
        
    def to_json(self) -> dict:
        return {"type": "and", "filters": [f.to_json() for f in self.filters]}

class Or(Filter):
    def __init__(self, *filters: Filter):
        self.filters = filters
        
    def to_json(self) -> dict:
        return {"type": "or", "filters": [f.to_json() for f in self.filters]}

class Not(Filter):
    def __init__(self, filter: Filter):
        self.filter = filter
        
    def to_json(self) -> dict:
        return {"type": "not", "filter": self.filter.to_json()}
