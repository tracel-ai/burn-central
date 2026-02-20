use crate::inference::{InferenceWriterObserver, InferenceWriterStats};
use opentelemetry::metrics::{Counter, Histogram, Meter};
use opentelemetry::{KeyValue, global};

use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

/// Runtime-owned facts about a completed inference request.
#[derive(Debug, Clone)]
pub struct RequestTelemetry {
    pub inference_name: String,
    pub model_name: String,
    pub model_version: String,
    pub duration: Duration,
    pub outputs: usize,
    pub errors: usize,
    pub cancelled: bool,
}

/// Runtime metadata attached to each inference request.
#[derive(Debug, Clone)]
pub struct InferenceMetadata {
    pub inference_name: String,
    pub model_name: String,
    pub model_version: String,
}

impl InferenceMetadata {
    pub fn new(
        inference_name: impl Into<String>,
        model_name: impl Into<String>,
        model_version: impl Into<String>,
    ) -> Self {
        Self {
            inference_name: inference_name.into(),
            model_name: model_name.into(),
            model_version: model_version.into(),
        }
    }
}

impl Default for InferenceMetadata {
    fn default() -> Self {
        Self::new("unknown", "unknown", "unknown")
    }
}

/// Telemetry sink for completed inference requests.
pub trait Telemetry: Send + Sync + 'static {
    fn record_request(&self, data: RequestTelemetry);
}

/// Default no-op telemetry implementation.
#[derive(Debug, Default)]
pub struct NoopTelemetry;

impl Telemetry for NoopTelemetry {
    fn record_request(&self, _data: RequestTelemetry) {}
}

/// OpenTelemetry metrics backend for inference telemetry.
#[derive(Clone)]
pub struct OTelTelemetry {
    request_counter: Counter<u64>,
    error_counter: Counter<u64>,
    output_counter: Counter<u64>,
    cancelled_counter: Counter<u64>,
    duration_histogram_ms: Histogram<f64>,
}

impl OTelTelemetry {
    pub fn new(meter: Meter) -> Self {
        let request_counter = meter
            .u64_counter("inference.requests")
            .with_description("Total number of inference requests")
            .build();

        let error_counter = meter
            .u64_counter("inference.errors")
            .with_description("Total number of inference errors")
            .build();

        let output_counter = meter
            .u64_counter("inference.outputs")
            .with_description("Total number of inference outputs written")
            .build();

        let cancelled_counter = meter
            .u64_counter("inference.cancelled")
            .with_description("Total number of cancelled inference requests")
            .build();

        let duration_histogram_ms = meter
            .f64_histogram("inference.duration_ms")
            .with_description("Inference duration in milliseconds")
            .build();

        Self {
            request_counter,
            error_counter,
            output_counter,
            cancelled_counter,
            duration_histogram_ms,
        }
    }

    pub fn from_global_meter() -> Self {
        Self::new(global::meter("burn-central-runtime.inference"))
    }
}

impl Telemetry for OTelTelemetry {
    fn record_request(&self, data: RequestTelemetry) {
        let status = if data.cancelled {
            "cancelled"
        } else if data.errors > 0 {
            "error"
        } else {
            "ok"
        };

        let attrs = vec![
            KeyValue::new("inference.name", data.inference_name),
            KeyValue::new("model.name", data.model_name),
            KeyValue::new("model.version", data.model_version),
            KeyValue::new("status", status),
        ];

        self.request_counter.add(1, &attrs);

        if data.errors > 0 {
            self.error_counter.add(data.errors as u64, &attrs);
        }

        if data.outputs > 0 {
            self.output_counter.add(data.outputs as u64, &attrs);
        }

        if data.cancelled {
            self.cancelled_counter.add(1, &attrs);
        }

        self.duration_histogram_ms
            .record(data.duration.as_secs_f64() * 1000.0, &attrs);
    }
}

static GLOBAL_TELEMETRY: OnceLock<RwLock<Arc<dyn Telemetry>>> = OnceLock::new();

fn telemetry_store() -> &'static RwLock<Arc<dyn Telemetry>> {
    GLOBAL_TELEMETRY.get_or_init(|| RwLock::new(Arc::new(NoopTelemetry)))
}

/// Set the global inference telemetry backend.
pub fn set_telemetry(telemetry: Arc<dyn Telemetry>) {
    *telemetry_store().write().unwrap() = telemetry;
}

/// Get the global inference telemetry backend.
pub fn telemetry() -> Arc<dyn Telemetry> {
    telemetry_store().read().unwrap().clone()
}

/// Writer observer that reports per-request telemetry on inference completion.
pub struct InferenceWriterTelemetryObserver {
    telemetry: Arc<dyn Telemetry>,
    metadata: InferenceMetadata,
}

impl InferenceWriterTelemetryObserver {
    pub fn new(metadata: InferenceMetadata) -> Self {
        Self {
            telemetry: telemetry(),
            metadata,
        }
    }
}

impl InferenceWriterObserver for InferenceWriterTelemetryObserver {
    fn on_finish(&self, stats: &InferenceWriterStats) {
        self.telemetry.record_request(RequestTelemetry {
            inference_name: self.metadata.inference_name.clone(),
            model_name: self.metadata.model_name.clone(),
            model_version: self.metadata.model_version.clone(),
            duration: stats.duration,
            outputs: stats.outputs,
            errors: stats.errors,
            cancelled: stats.cancelled,
        });
    }
}

/// Install an OpenTelemetry-backed global telemetry sink with a custom meter.
pub fn set_otel_telemetry(meter: Meter) {
    set_telemetry(Arc::new(OTelTelemetry::new(meter)));
}

/// Install an OpenTelemetry-backed global telemetry sink using the global meter provider.
pub fn set_otel_telemetry_from_global_meter() {
    set_otel_telemetry(global::meter("burn-central-runtime.inference"));
}

// /// Inference wrapper that opens a request span around the user inference call.
// pub struct InstrumentedInference<T> {
//     inner: T,
//     metadata: InferenceMetadata,
// }

// impl<T> InstrumentedInference<T> {
//     pub fn new(inner: T, metadata: InferenceMetadata) -> Self {
//         Self { inner, metadata }
//     }
// }

// impl<T> Inference for InstrumentedInference<T>
// where
//     T: Inference,
// {
//     type Input = T::Input;
//     type Output = T::Output;

//     fn infer(&self, input: Self::Input, writer: InferenceWriter<Self::Output>) {
//         let span = tracing::info_span!(
//             "inference",
//             inference.name = %self.metadata.inference_name,
//             model.name = %self.metadata.model_name,
//             model.version = %self.metadata.model_version,
//         );

//         let _guard = span.enter();
//         self.inner
//             .infer(input, writer.with_metadata(self.metadata.clone()));
//     }
// }
