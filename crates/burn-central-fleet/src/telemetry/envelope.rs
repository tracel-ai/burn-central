use serde::{Deserialize, Serialize};

use super::{logs::LogBatch, metrics::MetricBatch, unix_time_ms};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEnvelope {
    pub created_at_unix_ms: u64,
    pub data: TelemetryData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "signal", content = "payload")]
pub enum TelemetryData {
    Metrics(MetricBatch),
    Logs(LogBatch),
}

impl TelemetryEnvelope {
    fn new(data: TelemetryData) -> Self {
        Self {
            created_at_unix_ms: unix_time_ms(),
            data,
        }
    }

    pub(crate) fn metrics(payload: MetricBatch) -> Self {
        Self::new(TelemetryData::Metrics(payload))
    }

    pub(crate) fn logs(payload: LogBatch) -> Self {
        Self::new(TelemetryData::Logs(payload))
    }
}
