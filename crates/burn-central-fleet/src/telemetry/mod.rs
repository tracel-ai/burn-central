mod envelope;
mod logs;
mod metrics;
mod pipeline;
mod request;

use metrics_util::layers::Layer;
use once_cell::sync::{Lazy, OnceCell};
pub use request::{InferenceMetadata, InferenceWriterTelemetryObserver};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;

use std::collections::HashMap;
use std::sync::{Mutex, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use logs::LogRecord;

use crate::telemetry::logs::TelemetryLogLayer;
use crate::telemetry::metrics::{InMemoryMetricsRecorder, RecorderHandle};

pub use pipeline::{TelemetryPipeline, TelemetryPipelineError};

fn dispatch_log_record(record: LogRecord) {
    let mut hubs_guard = HUBS.lock().unwrap();
    let Some(weak_pipeline) = hubs_guard.get(&record.fleet_key) else {
        return;
    };

    if let Some(pipeline) = weak_pipeline.upgrade() {
        pipeline.enqueue_log(record);
    } else {
        hubs_guard.remove(&record.fleet_key);
    }
}

fn unix_time_ms() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis() as u64,
        Err(_) => 0,
    }
}

static GLOBAL_ONCE: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));
static GLOBAL_RECORDER: OnceCell<InMemoryMetricsRecorder> = OnceCell::new();
static HUBS: once_cell::sync::Lazy<Mutex<HashMap<String, Weak<TelemetryPipeline>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

/// Creates a tracing layer that injects metrics context from the current span.
/// Required for metrics to inherit tracing labels.
pub fn tracing_metrics_layer<S>() -> impl tracing_subscriber::Layer<S>
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    metrics_tracing_context::MetricsLayer::new()
}

/// Creates a metrics recorder that inherits tracing context from the current span.
/// Required for metrics to inherit tracing labels.
pub fn metrics_recorder_with_tracing_context() -> impl ::metrics::Recorder {
    let global_recorder = GLOBAL_RECORDER.get_or_init(InMemoryMetricsRecorder::new);
    metrics_tracing_context::TracingContextLayer::all().layer(global_recorder.clone())
}

/// Creates a tracing layer that captures tracing events into fleet telemetry logs.
/// Required for any logs emitted via tracing to be captured into fleet telemetry.
pub fn tracing_log_layer<S>() -> impl tracing_subscriber::Layer<S>
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    TelemetryLogLayer
}

/// Initializes global telemetry state.
///
/// This setup is best-effort:
/// - If a tracing subscriber is already installed, we keep using it.
/// - If a metrics recorder is already installed, we keep using it.
pub fn global_init() -> Result<(), &'static str> {
    let mut once_guard = GLOBAL_ONCE.lock().unwrap();
    if *once_guard {
        return Ok(());
    }

    let _ = tracing_subscriber::registry::Registry::default()
        .with(tracing_metrics_layer())
        .with(tracing_log_layer())
        .try_init();

    let recorder = metrics_recorder_with_tracing_context();
    let _ = ::metrics::set_global_recorder(recorder);

    *once_guard = true;
    Ok(())
}

pub fn global_recorder_handle() -> RecorderHandle {
    let global_recorder = GLOBAL_RECORDER.get_or_init(InMemoryMetricsRecorder::new);
    global_recorder.handle()
}
