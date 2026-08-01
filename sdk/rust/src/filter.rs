use serde::Serialize;
use serde_json::Value;

#[derive(Serialize)]
pub struct Filter {
    #[serde(rename = "type")]
    pub filter_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
}

impl Filter {
    pub fn eq<V: Into<Value>>(field: &str, value: V) -> Self {
        Self {
            filter_type: "eq".to_string(),
            field: Some(field.to_string()),
            value: Some(value.into()),
        }
    }

    pub fn neq<V: Into<Value>>(field: &str, value: V) -> Self {
        Self {
            filter_type: "neq".to_string(),
            field: Some(field.to_string()),
            value: Some(value.into()),
        }
    }

    pub fn gt<V: Into<Value>>(field: &str, value: V) -> Self {
        Self {
            filter_type: "gt".to_string(),
            field: Some(field.to_string()),
            value: Some(value.into()),
        }
    }

    pub fn and(filters: Vec<Filter>) -> Self {
        Self {
            filter_type: "and".to_string(),
            field: None,
            value: Some(serde_json::to_value(filters).unwrap()),
        }
    }

    pub fn or(filters: Vec<Filter>) -> Self {
        Self {
            filter_type: "or".to_string(),
            field: None,
            value: Some(serde_json::to_value(filters).unwrap()),
        }
    }
}
