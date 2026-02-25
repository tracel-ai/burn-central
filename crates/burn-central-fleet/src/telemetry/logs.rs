use serde::{Deserialize, Serialize};
use tracing::field::{Field, Visit};
use tracing_subscriber::registry::LookupSpan;

use super::{dispatch_log_record, unix_time_ms};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl From<&tracing::Level> for LogLevel {
    fn from(level: &tracing::Level) -> Self {
        match *level {
            tracing::Level::TRACE => Self::Trace,
            tracing::Level::DEBUG => Self::Debug,
            tracing::Level::INFO => Self::Info,
            tracing::Level::WARN => Self::Warn,
            tracing::Level::ERROR => Self::Error,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogField {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRecord {
    pub timestamp_unix_ms: u64,
    pub fleet_key: String,
    pub level: LogLevel,
    pub target: String,
    pub message: String,
    pub fields: Vec<LogField>,
}

impl LogRecord {
    pub fn new(
        fleet_key: String,
        level: LogLevel,
        target: impl Into<String>,
        message: impl Into<String>,
        fields: Vec<LogField>,
    ) -> Self {
        Self {
            timestamp_unix_ms: unix_time_ms(),
            fleet_key,
            level,
            target: target.into(),
            message: message.into(),
            fields,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogBatch {
    pub entries: Vec<LogRecord>,
}

#[derive(Default)]
struct EventFieldVisitor {
    message: Option<String>,
    fleet_key: Option<String>,
    fields: Vec<LogField>,
}

impl EventFieldVisitor {
    fn push(&mut self, field: &Field, value: String) {
        let key = field.name().to_string();
        if key == "message" {
            self.message = Some(value.clone());
        } else if key == "fleet_key" {
            self.fleet_key = Some(value.clone());
        }

        self.fields.push(LogField { key, value });
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

#[derive(Debug, Clone, Default)]
struct SpanFields {
    fleet_key: Option<String>,
    fields: Vec<LogField>,
}

impl SpanFields {
    fn merge(&mut self, other: SpanFields) {
        if other.fleet_key.is_some() {
            self.fleet_key = other.fleet_key;
        }

        for incoming in other.fields {
            if let Some(existing) = self.fields.iter_mut().find(|f| f.key == incoming.key) {
                existing.value = incoming.value;
            } else {
                self.fields.push(incoming);
            }
        }
    }

    fn from_attributes(attrs: &tracing::span::Attributes<'_>) -> Self {
        let mut visitor = EventFieldVisitor::default();
        attrs.record(&mut visitor);
        Self {
            fleet_key: visitor.fleet_key,
            fields: visitor.fields,
        }
    }

    fn from_record(record: &tracing::span::Record<'_>) -> Self {
        let mut visitor = EventFieldVisitor::default();
        record.record(&mut visitor);
        Self {
            fleet_key: visitor.fleet_key,
            fields: visitor.fields,
        }
    }
}

#[derive(Debug, Default)]
pub struct TelemetryLogLayer;

impl<S> tracing_subscriber::Layer<S> for TelemetryLogLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let Some(span) = ctx.span(id) else {
            return;
        };

        span.extensions_mut()
            .insert(SpanFields::from_attributes(attrs));
    }

    fn on_record(
        &self,
        id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let Some(span) = ctx.span(id) else {
            return;
        };

        let mut extensions = span.extensions_mut();
        let updates = SpanFields::from_record(values);
        if let Some(existing) = extensions.get_mut::<SpanFields>() {
            existing.merge(updates);
        } else {
            extensions.insert(updates);
        }
    }

    fn on_event(&self, event: &tracing::Event<'_>, ctx: tracing_subscriber::layer::Context<'_, S>) {
        let mut visitor = EventFieldVisitor::default();
        event.record(&mut visitor);

        let mut inherited_fields = Vec::new();
        let mut inherited_fleet_key = None;
        if let Some(scope) = ctx.event_scope(event) {
            for span in scope.from_root() {
                let span_name = span.name();
                if let Some(span_fields) = span.extensions().get::<SpanFields>() {
                    if let Some(fleet_key) = span_fields.fleet_key.as_ref() {
                        inherited_fleet_key = Some(fleet_key.clone());
                    }

                    inherited_fields.extend(span_fields.fields.iter().cloned().map(|field| {
                        LogField {
                            key: format!("span.{span_name}.{}", field.key),
                            value: field.value,
                        }
                    }));
                }
            }
        }

        if visitor.fleet_key.is_none() {
            visitor.fleet_key = inherited_fleet_key;
        }

        let metadata = event.metadata();
        let message = visitor
            .message
            .clone()
            .unwrap_or_else(|| metadata.name().to_string());
        inherited_fields.extend(visitor.fields);

        let Some(fleet_key) = visitor.fleet_key else {
            // If no fleet key is found in the event or its parent spans, we skip logging
            return;
        };

        dispatch_log_record(LogRecord::new(
            fleet_key,
            LogLevel::from(metadata.level()),
            metadata.target().to_string(),
            message,
            inherited_fields,
        ));
    }
}
