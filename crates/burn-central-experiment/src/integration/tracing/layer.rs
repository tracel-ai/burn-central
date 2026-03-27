use std::fmt::Write as _;

use tracing_subscriber::{
    fmt::format::{DefaultFields, FormatFields, Writer},
    registry::LookupSpan,
};

use crate::{
    ExperimentRun,
    context::ExperimentGlobalExt,
    integration::tracing::{registry::TracingRegistry, visitor::SpanFields},
};

/// `tracing_subscriber` layer that forwards events into experiment logs.
///
/// The layer resolves the destination experiment in two steps:
/// 1. a span-bound experiment id created by
///    [`super::ExperimentTracingExt::tracing_span`]
/// 2. the current ambient experiment from [`crate::ExperimentGlobalExt`]
///
/// Construct it directly or use [`super::tracing_log_layer`] for a named helper function.
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
            None => match ExperimentRun::current() {
                Some(handle) => handle,
                None => return,
            },
        };

        let rendered = match format_event(event) {
            Some(rendered) => rendered,
            None => return,
        };

        let _ = handle.log_info(rendered);
    }
}

fn format_event(event: &tracing::Event<'_>) -> Option<String> {
    let metadata = event.metadata();
    let mut rendered = String::new();
    write!(&mut rendered, "[{}] ", metadata.level()).ok()?;
    DefaultFields::new()
        .format_fields(Writer::new(&mut rendered), event)
        .ok()?;
    rendered.push('\n');
    Some(rendered)
}
