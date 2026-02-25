use serde::{Deserialize, Serialize};
use tracing::field::{Field, Visit};
use tracing_subscriber::registry::LookupSpan;

#[cfg(test)]
use once_cell::sync::Lazy;
#[cfg(test)]
use std::sync::Mutex;

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
    pub message: String,
    pub fields: Vec<LogField>,
}

impl LogRecord {
    pub fn new(
        fleet_key: String,
        level: LogLevel,
        message: impl Into<String>,
        fields: Vec<LogField>,
    ) -> Self {
        Self {
            timestamp_unix_ms: unix_time_ms(),
            fleet_key,
            level,
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

        let Some(fleet_key) = visitor.fleet_key else {
            return;
        };

        let metadata = event.metadata();
        let message = visitor
            .message
            .clone()
            .unwrap_or_else(|| metadata.name().to_string());
        inherited_fields.extend(visitor.fields);

        inherited_fields.push(LogField {
            key: "event.target".to_string(),
            value: metadata.target().to_string(),
        });

        dispatch_log_record(LogRecord::new(
            fleet_key,
            LogLevel::from(metadata.level()),
            message,
            inherited_fields,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;

    static TEST_SERIAL: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    fn run_with_log_layer(test_fn: impl FnOnce()) -> Vec<LogRecord> {
        let _serial_guard = TEST_SERIAL
            .lock()
            .expect("test serial lock should not be poisoned");
        super::super::clear_dispatched_log_records_for_test();

        let subscriber = tracing_subscriber::registry().with(TelemetryLogLayer);
        tracing::subscriber::with_default(subscriber, test_fn);

        super::super::take_dispatched_log_records_for_test()
    }

    fn field_value<'a>(record: &'a LogRecord, key: &str) -> Option<&'a str> {
        record
            .fields
            .iter()
            .find(|field| field.key == key)
            .map(|field| field.value.as_str())
    }

    #[test]
    fn log_layer_records_event_with_inherited_span_fields() {
        let records = run_with_log_layer(|| {
            let span = tracing::info_span!(
                "request",
                fleet_key = "fleet-a",
                request_id = 7u64,
                model = tracing::field::Empty
            );
            span.record("model", &"resnet50");
            let _guard = span.enter();

            tracing::info!(attempt = 2u64, "inference finished");
        });

        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.fleet_key, "fleet-a");
        assert!(matches!(record.level, LogLevel::Info));
        assert_eq!(record.message, "inference finished");
        assert!(record.timestamp_unix_ms > 0);
        assert_eq!(
            field_value(record, "span.request.request_id"),
            Some("7"),
            "request_id should be inherited from span fields",
        );
        assert_eq!(
            field_value(record, "span.request.model"),
            Some("resnet50"),
            "recorded span fields should appear in log fields",
        );
        assert_eq!(field_value(record, "attempt"), Some("2"));
        assert!(
            field_value(record, "event.target").is_some(),
            "event target should be included",
        );
    }

    #[test]
    fn log_layer_drops_event_without_fleet_key() {
        let records = run_with_log_layer(|| {
            tracing::warn!(attempt = 1u64, "fleet key missing");
        });

        assert!(
            records.is_empty(),
            "events without fleet key should not be dispatched",
        );
    }

    #[test]
    fn log_layer_prefers_event_fleet_key_over_span_fleet_key() {
        let records = run_with_log_layer(|| {
            let span = tracing::info_span!("request", fleet_key = "span-fleet");
            let _guard = span.enter();

            tracing::info!(fleet_key = "event-fleet", "override fleet key");
        });

        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.fleet_key, "event-fleet");
        assert_eq!(field_value(record, "fleet_key"), Some("event-fleet"));
        assert_eq!(
            field_value(record, "span.request.fleet_key"),
            Some("span-fleet")
        );
    }
}
