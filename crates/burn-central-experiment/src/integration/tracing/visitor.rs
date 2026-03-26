use tracing::field::{Field, Visit};

use crate::ExperimentId;

#[derive(Debug, Default)]
pub struct EventFieldVisitor {
    pub message: Option<String>,
    pub fields: Vec<(String, String)>,
}

impl EventFieldVisitor {
    fn push(&mut self, field: &Field, value: String) {
        let key = field.name();
        match key {
            "message" => self.message = Some(value),
            _ => self.fields.push((key.to_string(), value)),
        }
    }
}

impl Visit for EventFieldVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.push(field, value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.push(field, format!("{value:?}"));
    }
}

#[derive(Debug, Default)]
pub struct ExperimentIdVisitor {
    experiment_id: Option<ExperimentId>,
}

impl ExperimentIdVisitor {
    fn push(&mut self, field: &Field, value: String) {
        if field.name() == "experiment_id" {
            self.experiment_id = Some(ExperimentId::new(value));
        }
    }
}

impl Visit for ExperimentIdVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.push(field, value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.push(field, format!("{value:?}"));
    }
}

#[derive(Debug, Clone, Default)]
pub struct SpanFields {
    pub experiment_id: Option<ExperimentId>,
}

impl SpanFields {
    pub fn merge(&mut self, other: Self) {
        if other.experiment_id.is_some() {
            self.experiment_id = other.experiment_id;
        }
    }

    pub fn from_attributes(attrs: &tracing::span::Attributes<'_>) -> Self {
        let mut visitor = ExperimentIdVisitor::default();
        attrs.record(&mut visitor);
        Self {
            experiment_id: visitor.experiment_id,
        }
    }

    pub fn from_record(record: &tracing::span::Record<'_>) -> Self {
        let mut visitor = ExperimentIdVisitor::default();
        record.record(&mut visitor);
        Self {
            experiment_id: visitor.experiment_id,
        }
    }
}
