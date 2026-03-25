use tracing_subscriber::registry::LookupSpan;

use crate::{
    current_experiment,
    integration::tracing::{
        registry::TracingRegistry,
        visitor::{EventFieldVisitor, SpanFields},
    },
};

#[derive(Debug, Default)]
pub struct ExperimentTracingLogLayer;

impl<S> tracing_subscriber::Layer<S> for ExperimentTracingLogLayer
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
        let metadata = event.metadata();
        if metadata.target().starts_with("wgpu") && *metadata.level() == tracing::Level::INFO {
            return;
        }

        let mut visitor = EventFieldVisitor::default();
        event.record(&mut visitor);

        let experiment_id = if let Some(scope) = ctx.event_scope(event) {
            let mut experiment_id = None;
            for span in scope.from_root() {
                if let Some(span_fields) = span.extensions().get::<SpanFields>() {
                    if let Some(span_experiment_id) = span_fields.experiment_id.as_ref() {
                        experiment_id = Some(span_experiment_id.clone());
                    }
                }
            }
            experiment_id
        } else {
            None
        };

        let handle = match experiment_id {
            Some(experiment_id) => match TracingRegistry::global().get_handle(&experiment_id) {
                Some(handle) => handle,
                None => return,
            },
            None => match current_experiment() {
                Some(handle) => handle,
                None => return,
            },
        };

        let _ = handle.log_info(format_event(metadata, visitor.message, visitor.fields));
    }
}

fn format_event(
    metadata: &tracing::Metadata<'_>,
    message: Option<String>,
    fields: Vec<(String, String)>,
) -> String {
    let mut rendered = format!(
        "[{}] {}",
        metadata.level(),
        message.unwrap_or_else(|| metadata.name().to_string())
    );

    if !fields.is_empty() {
        rendered.push(' ');
        rendered.push_str(
            &fields
                .into_iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(" "),
        );
    }

    rendered.push('\n');

    rendered
}
