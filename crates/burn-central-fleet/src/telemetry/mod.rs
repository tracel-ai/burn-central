mod event;
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
use std::sync::{Arc, Mutex, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use logs::LogRecord;

use crate::telemetry::logs::TelemetryLogLayer;
use crate::telemetry::metrics::{InMemoryMetricsRecorder, RecorderHandle};

pub use pipeline::{TelemetryPipeline, TelemetryPipelineError};

/// Global lock for telemetry initialization. Ensures that global telemetry state is only initialized once.
static GLOBAL_ONCE: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));
/// Global metrics recorder instance. Used to support instrumentation via the metrics crate.
static GLOBAL_RECORDER: OnceCell<InMemoryMetricsRecorder> = OnceCell::new();
/// Global registry of telemetry pipelines identified by fleet key. Used to route log records to the correct pipeline.
/// This is managed by [`TelemetryPipeline`] and used by the tracing log layer to dispatch log records.
static PIPELINES: Lazy<PipelineRegistry> = Lazy::new(PipelineRegistry::default);
#[cfg(test)]
static DISPATCHED_LOG_RECORDS_FOR_TEST: Lazy<Mutex<Vec<LogRecord>>> =
    Lazy::new(|| Mutex::new(Vec::new()));

/// Registry for telemetry pipelines, keyed by fleet key. Used to route telemetry to the correct pipeline.
#[derive(Debug, Default)]
struct PipelineRegistry {
    hubs: Mutex<HashMap<String, Weak<TelemetryPipeline>>>,
}

impl PipelineRegistry {
    fn add_pipeline(&self, fleet_key: String, pipeline: &Arc<TelemetryPipeline>) {
        let mut hubs_guard = self.hubs.lock().unwrap();
        hubs_guard.insert(fleet_key, Arc::downgrade(pipeline));
    }

    fn get_pipeline(&self, fleet_key: &str) -> Option<Arc<TelemetryPipeline>> {
        let mut hubs_guard = self.hubs.lock().unwrap();
        if let Some(weak_pipeline) = hubs_guard.get(fleet_key) {
            if let Some(pipeline) = weak_pipeline.upgrade() {
                return Some(pipeline);
            } else {
                hubs_guard.remove(fleet_key);
            }
        }
        None
    }

    fn remove_pipeline(&self, fleet_key: &str) {
        let mut hubs_guard = self.hubs.lock().unwrap();
        hubs_guard.remove(fleet_key);
    }
}

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
pub fn metrics_recorder() -> impl ::metrics::Recorder {
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
fn global_init() -> Result<(), &'static str> {
    let mut once_guard = GLOBAL_ONCE.lock().unwrap();
    if *once_guard {
        return Ok(());
    }

    let _ = tracing_subscriber::registry::Registry::default()
        .with(tracing_metrics_layer())
        .with(tracing_log_layer())
        .try_init();

    let recorder = metrics_recorder();
    let _ = ::metrics::set_global_recorder(recorder);

    *once_guard = true;
    Ok(())
}

fn global_recorder_handle() -> RecorderHandle {
    let global_recorder = GLOBAL_RECORDER.get_or_init(InMemoryMetricsRecorder::new);
    global_recorder.handle()
}

fn dispatch_log_record(record: LogRecord) {
    #[cfg(test)]
    {
        let mut records_guard = DISPATCHED_LOG_RECORDS_FOR_TEST.lock().unwrap();
        records_guard.push(record.clone());
    }

    let Some(pipeline) = PIPELINES.get_pipeline(&record.fleet_key) else {
        return;
    };

    pipeline.enqueue_log(record);
}

#[cfg(test)]
fn clear_dispatched_log_records_for_test() {
    let mut records_guard = DISPATCHED_LOG_RECORDS_FOR_TEST.lock().unwrap();
    records_guard.clear();
}

#[cfg(test)]
fn take_dispatched_log_records_for_test() -> Vec<LogRecord> {
    let mut records_guard = DISPATCHED_LOG_RECORDS_FOR_TEST.lock().unwrap();
    std::mem::take(&mut *records_guard)
}

fn unix_time_ms() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis() as u64,
        Err(_) => 0,
    }
}
