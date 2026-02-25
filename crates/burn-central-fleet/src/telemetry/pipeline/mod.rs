use crossbeam_queue::SegQueue;

use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use crate::telemetry::{PIPELINES, global_init, global_recorder_handle};

use super::event::TelemetryEvent;
use super::logs::LogRecord;
use super::metrics::RecorderHandle;
use outbox::WalOutbox;

mod collector;
mod outbox;
mod shipper;

#[derive(Debug, thiserror::Error)]
pub enum TelemetryPipelineError {
    #[error("failed to initialize telemetry pipeline for fleet '{0}': {1}")]
    InitializationFailed(String, String),
}

pub type OutboxId = i64;

pub trait Outbox: Send + Sync {
    fn enqueue(&self, data: TelemetryEvent) -> Result<(), String>;
    fn claim(&self, count: usize) -> Result<Vec<(OutboxId, TelemetryEvent)>, String>;
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
    batcher_handles: Vec<collector::CollectorHandle>,
    shipper_handle: shipper::ShipperHandle,
}

impl TelemetryPipeline {
    pub fn get_or_init(
        fleet_key: String,
        root_dir: PathBuf,
    ) -> Result<Arc<Self>, TelemetryPipelineError> {
        global_init().map_err(|e| {
            TelemetryPipelineError::InitializationFailed(fleet_key.clone(), e.to_string())
        })?;

        if let Some(pipeline) = PIPELINES.get_pipeline(&fleet_key) {
            return Ok(pipeline);
        }

        let recorder = global_recorder_handle();
        let pipeline = Arc::new(Self::start(fleet_key.clone(), recorder, root_dir)?);
        PIPELINES.add_pipeline(fleet_key, &pipeline);
        Ok(pipeline)
    }

    pub(crate) fn enqueue_log(&self, record: LogRecord) {
        let _ = self.log_ingress.push(record);
    }

    fn start(
        fleet_key: String,
        recorder: RecorderHandle,
        root_dir: PathBuf,
    ) -> Result<Self, TelemetryPipelineError> {
        let outbox_path = telemetry_outbox_path(&root_dir, &fleet_key);
        let outbox = Arc::new(WalOutbox::new(outbox_path.clone()).map_err(|e| {
            TelemetryPipelineError::InitializationFailed(
                fleet_key.clone(),
                format!(
                    "failed to initialize wal outbox '{}': {e}",
                    outbox_path.display()
                ),
            )
        })?);
        let log_ingress = Arc::new(LogIngress::default());

        let batcher_handles = vec![
            collector::start(
                "telemetry-batcher-metrics",
                Arc::new(collector::MetricsEventCollector::new(recorder)),
                outbox.clone(),
                Duration::from_secs(5),
            ),
            collector::start(
                "telemetry-batcher-logs",
                Arc::new(collector::LogsCollector::new(log_ingress.clone(), 256)),
                outbox.clone(),
                Duration::from_secs(2),
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

fn telemetry_outbox_path(root_dir: &Path, fleet_key: &str) -> PathBuf {
    root_dir
        .join("telemetry")
        .join("outbox")
        .join(format!("{fleet_key}.wal"))
}

impl Drop for TelemetryPipeline {
    fn drop(&mut self) {
        PIPELINES.remove_pipeline(&self.fleet_key);

        for handle in &mut self.batcher_handles {
            handle.shutdown();
        }
        self.shipper_handle.shutdown();
    }
}
