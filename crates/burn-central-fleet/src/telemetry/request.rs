use burn_central_inference::observer::{InferenceWriterObserver, InferenceWriterStats};

use std::time::Duration;

/// Fleet-owned facts about a completed inference request.
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

/// Fleet metadata attached to each inference request.
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

pub fn record_request(data: RequestTelemetry) {
    let cancelled = data.cancelled.to_string();
    let duration_ms = data.duration.as_secs_f64() * 1_000.0;

    ::metrics::counter!(
        "burn_central_fleet_inference_requests_total",
        "inference_name" => data.inference_name.clone(),
        "model_name" => data.model_name.clone(),
        "model_version" => data.model_version.clone(),
        "cancelled" => cancelled.clone(),
    )
    .increment(1);

    ::metrics::counter!(
        "burn_central_fleet_inference_outputs_total",
        "inference_name" => data.inference_name.clone(),
        "model_name" => data.model_name.clone(),
        "model_version" => data.model_version.clone(),
    )
    .increment(data.outputs as u64);

    ::metrics::counter!(
        "burn_central_fleet_inference_errors_total",
        "inference_name" => data.inference_name.clone(),
        "model_name" => data.model_name.clone(),
        "model_version" => data.model_version.clone(),
    )
    .increment(data.errors as u64);

    ::metrics::histogram!(
        "burn_central_fleet_inference_duration_ms",
        "inference_name" => data.inference_name.clone(),
        "model_name" => data.model_name.clone(),
        "model_version" => data.model_version.clone(),
        "cancelled" => cancelled,
    )
    .record(duration_ms);

    if data.errors > 0 {
        tracing::warn!(
            inference_name = data.inference_name.as_str(),
            model_name = data.model_name.as_str(),
            model_version = data.model_version.as_str(),
            errors = data.errors,
            "inference finished with writer errors"
        );
    }
}

/// Writer observer that reports per-request telemetry on inference completion.
pub struct InferenceWriterTelemetryObserver {
    metadata: InferenceMetadata,
}

impl InferenceWriterTelemetryObserver {
    pub fn with_telemetry(metadata: InferenceMetadata) -> Self {
        Self { metadata }
    }
}

impl InferenceWriterObserver for InferenceWriterTelemetryObserver {
    fn on_finish(&self, stats: &InferenceWriterStats) {
        record_request(RequestTelemetry {
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
