use crossbeam_queue::SegQueue;

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use crate::telemetry::{HUBS, global_init, global_recorder_handle};

use super::envelope::TelemetryEnvelope;
use super::logs::LogRecord;
use super::metrics::RecorderHandle;
use outbox::InMemoryOutbox;

mod batcher;
mod outbox;
mod shipper;

#[derive(Debug, thiserror::Error)]
pub enum TelemetryPipelineError {
    #[error("failed to initialize telemetry pipeline for fleet '{0}': {1}")]
    InitializationFailed(String, String),
}

pub type OutboxId = i64;

pub trait Outbox: Send + Sync {
    fn enqueue(&self, data: TelemetryEnvelope) -> Result<(), String>;
    fn claim(&self, count: usize) -> Result<Vec<(OutboxId, TelemetryEnvelope)>, String>;
    fn complete(&self, id: OutboxId) -> Result<(), String>;
    fn fail(&self, id: OutboxId, error: &str) -> Result<(), String>;
}

const LOG_INGRESS_CAPACITY: usize = 4_096;

#[derive(Debug, Default)]

struct LogIngress {
    queue: SegQueue<LogRecord>,
    depth: AtomicUsize,
}

impl LogIngress {
    fn push(&self, record: LogRecord) -> bool {
        let previous = self.depth.fetch_add(1, Ordering::AcqRel);
        if previous >= LOG_INGRESS_CAPACITY {
            self.depth.fetch_sub(1, Ordering::AcqRel);
            return false;
        }

        self.queue.push(record);
        true
    }

    fn pop_batch(&self, max_entries: usize) -> Vec<LogRecord> {
        let mut entries = Vec::with_capacity(max_entries);
        for _ in 0..max_entries {
            let Some(record) = self.queue.pop() else {
                break;
            };
            self.depth.fetch_sub(1, Ordering::AcqRel);
            entries.push(record);
        }
        entries
    }
}

pub struct TelemetryPipeline {
    fleet_key: String,
    log_ingress: Arc<LogIngress>,
    batcher_handles: Vec<batcher::BatcherHandle>,
    shipper_handle: shipper::ShipperHandle,
}

impl TelemetryPipeline {
    pub fn get_or_init(fleet_key: String) -> Result<Arc<Self>, TelemetryPipelineError> {
        global_init().map_err(|e| {
            TelemetryPipelineError::InitializationFailed(fleet_key.clone(), e.to_string())
        })?;

        let mut hubs_guard = HUBS.lock().unwrap();
        if let Some(weak_pipeline) = hubs_guard.get(&fleet_key) {
            if let Some(pipeline) = weak_pipeline.upgrade() {
                return Ok(pipeline);
            }
        }

        let recorder = global_recorder_handle();
        let pipeline = Arc::new(Self::start(fleet_key.clone(), recorder)?);
        hubs_guard.insert(fleet_key, Arc::downgrade(&pipeline));
        Ok(pipeline)
    }

    pub(crate) fn enqueue_log(&self, record: LogRecord) {
        let _ = self.log_ingress.push(record);
    }

    fn start(fleet_key: String, recorder: RecorderHandle) -> Result<Self, TelemetryPipelineError> {
        let outbox = Arc::new(InMemoryOutbox::default());
        let log_ingress = Arc::new(LogIngress::default());

        let batcher_handles = vec![
            batcher::start(
                Arc::new(batcher::MetricsEnvelopeBatcher::new(recorder)),
                outbox.clone(),
                Duration::from_secs(5),
                "telemetry-batcher-metrics",
            ),
            batcher::start(
                Arc::new(batcher::LogsBatcher::new(log_ingress.clone(), 256)),
                outbox.clone(),
                Duration::from_secs(2),
                "telemetry-batcher-logs",
            ),
        ];

        let shipper_handle = shipper::start(
            outbox,
            Arc::new(shipper::NoopShipperTransport),
            Duration::from_secs(5),
        );

        Ok(Self {
            fleet_key,
            log_ingress,
            batcher_handles,
            shipper_handle,
        })
    }
}

impl Drop for TelemetryPipeline {
    fn drop(&mut self) {
        {
            let mut hubs_guard = HUBS.lock().unwrap();
            hubs_guard.remove(&self.fleet_key);
        }

        for handle in &mut self.batcher_handles {
            handle.shutdown();
        }
        self.shipper_handle.shutdown();
    }
}
