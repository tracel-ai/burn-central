use burn_central_client::FleetClient;

use std::{
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
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
    fn claim(&self, count: usize) -> Result<Option<Vec<(OutboxId, TelemetryEvent)>>, String>;
    fn complete(&self, id: OutboxId) -> Result<(), String>;
    fn fail(&self, id: OutboxId, error: &str) -> Result<(), String>;
}

const LOG_INGRESS_CAPACITY: usize = 4_096;
const METRICS_EMIT_INTERVAL: Duration = Duration::from_secs(60);
const LOG_FLUSH_INTERVAL: Duration = Duration::from_secs(60);
const LOG_BATCH_MAX_ENTRIES: usize = 256;
const SHIPPER_POLL_INTERVAL: Duration = Duration::from_secs(5);

pub struct TelemetryPipeline {
    fleet_key: String,
    log_ingress: collector::LogIngress,
    collector_handles: Vec<collector::CollectorHandle>,
    shipper_handle: shipper::ShipperHandle,
}

impl TelemetryPipeline {
    pub fn get_or_init(
        fleet_key: String,
        auth_token: Arc<RwLock<Option<String>>>,
        client: FleetClient,
        root_dir: PathBuf,
    ) -> Result<Arc<Self>, TelemetryPipelineError> {
        global_init().map_err(|e| {
            TelemetryPipelineError::InitializationFailed(fleet_key.clone(), e.to_string())
        })?;

        if let Some(pipeline) = PIPELINES.get_pipeline(&fleet_key) {
            return Ok(pipeline);
        }

        let recorder = global_recorder_handle();
        let pipeline = Arc::new(Self::start(
            fleet_key.clone(),
            auth_token,
            client,
            recorder,
            root_dir,
        )?);
        PIPELINES.add_pipeline(fleet_key, &pipeline);
        Ok(pipeline)
    }

    pub(crate) fn enqueue_log(&self, record: LogRecord) {
        self.log_ingress.push(record);
    }

    fn start(
        fleet_key: String,
        auth_token: Arc<RwLock<Option<String>>>,
        client: FleetClient,
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
        let (log_ingress, logs_collector_handle) = collector::LogsCollector::spawn(
            "telemetry-collector-logs",
            outbox.clone(),
            LOG_INGRESS_CAPACITY,
            LOG_BATCH_MAX_ENTRIES,
            LOG_FLUSH_INTERVAL,
        );

        let collector_handles = vec![
            collector::MetricsEventCollector::new(
                fleet_key.clone(),
                recorder,
                METRICS_EMIT_INTERVAL,
            )
            .start("telemetry-collector-metrics", outbox.clone()),
            logs_collector_handle,
        ];

        let shipper_handle = shipper::start(
            outbox,
            Arc::new(shipper::BurnCentralFleetShipperTransport::new(
                auth_token, client,
            )),
            SHIPPER_POLL_INTERVAL,
        );

        Ok(Self {
            fleet_key,
            log_ingress,
            collector_handles,
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

        for handle in &mut self.collector_handles {
            handle.shutdown();
        }
        self.shipper_handle.shutdown();
    }
}
