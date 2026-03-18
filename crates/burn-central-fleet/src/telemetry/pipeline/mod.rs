use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use crate::auth_client::AuthenticatedFleetClient;
use crate::telemetry::{
    PIPELINES, global_init, global_recorder_handle,
    pipeline::shipper::{ShipperConfig, ShipperWaker},
};

use super::logs::LogRecord;
use super::metrics::RecorderHandle;

mod collector;
mod outbox;
mod shipper;

#[derive(Debug, thiserror::Error)]
pub enum TelemetryPipelineError {
    #[error("failed to initialize telemetry pipeline for fleet '{0}': {1}")]
    InitializationFailed(String, String),
}

const METRICS_EMIT_INTERVAL: Duration = Duration::from_secs(60);
const LOG_FLUSH_INTERVAL: Duration = Duration::from_secs(60);
const LOG_BATCH_MAX_ENTRIES: usize = 256;
const SHIPPER_IDLE_SWEEP_INTERVAL: Duration = Duration::from_secs(3 * 60);

pub struct TelemetryPipeline {
    fleet_key: String,
    client: AuthenticatedFleetClient,
    log_ingress: collector::LogIngress,
    collector_handles: Vec<collector::CollectorHandle>,
    shipper_handle: shipper::ShipperHandle,
}

impl TelemetryPipeline {
    pub fn get(fleet_key: &str) -> Option<Arc<Self>> {
        PIPELINES.get_pipeline(fleet_key)
    }

    pub fn create(
        fleet_key: String,
        client: AuthenticatedFleetClient,
        root_dir: PathBuf,
    ) -> Result<Arc<Self>, TelemetryPipelineError> {
        global_init().map_err(|e| {
            TelemetryPipelineError::InitializationFailed(fleet_key.clone(), e.to_string())
        })?;

        if let Some(pipeline) = PIPELINES.get_pipeline(&fleet_key) {
            return Ok(pipeline);
        }

        tracing::info!("initializing telemetry pipeline for fleet '{}'", fleet_key);

        let recorder = global_recorder_handle();
        let pipeline = Arc::new(Self::start(fleet_key.clone(), client, recorder, root_dir)?);
        PIPELINES.add_pipeline(fleet_key, &pipeline);
        Ok(pipeline)
    }

    pub(crate) fn enqueue_log(&self, record: LogRecord) {
        self.log_ingress.push(record);
    }

    pub(crate) fn client(&self) -> AuthenticatedFleetClient {
        self.client.clone()
    }

    fn start(
        fleet_key: String,
        client: AuthenticatedFleetClient,
        recorder: RecorderHandle,
        root_dir: PathBuf,
    ) -> Result<Self, TelemetryPipelineError> {
        let outbox_path = telemetry_outbox_path(&root_dir, &fleet_key);
        let (shipper_waker, shipper_wake_rx) = ShipperWaker::new();
        let outbox = Arc::new(outbox::NotifyingOutbox::new(
            outbox::wal::WalOutbox::new(outbox_path.clone()).map_err(|e| {
                TelemetryPipelineError::InitializationFailed(
                    fleet_key.clone(),
                    format!(
                        "failed to initialize wal outbox '{}': {e}",
                        outbox_path.display()
                    ),
                )
            })?,
            Box::new({
                let shipper_waker = shipper_waker.clone();
                move || shipper_waker.wake()
            }),
        ));

        let (log_ingress, logs_collector_handle) = collector::LogsCollector::spawn(
            &format!("telemetry-collector-logs-{fleet_key}"),
            outbox.clone(),
            LOG_BATCH_MAX_ENTRIES,
            LOG_FLUSH_INTERVAL,
        );

        let collector_handles = vec![
            collector::MetricsEventCollector::new(
                fleet_key.clone(),
                recorder,
                METRICS_EMIT_INTERVAL,
            )
            .start(
                &format!("telemetry-collector-metrics-{fleet_key}"),
                outbox.clone(),
            ),
            logs_collector_handle,
        ];

        let shipper_handle = shipper::start(
            &format!("telemetry-shipper-{fleet_key}"),
            outbox,
            shipper_wake_rx,
            Arc::new(shipper::BurnCentralFleetShipperTransport::new(client.clone())),
            ShipperConfig {
                idle_sweep_interval: SHIPPER_IDLE_SWEEP_INTERVAL,
                min_retry_interval: Duration::from_secs(5),
                max_retry_interval: Duration::from_secs(30 * 60),
                max_batch_size: 50,
            },
        );

        Ok(Self {
            fleet_key,
            client,
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
        tracing::debug!(
            "shutting down telemetry pipeline for fleet '{}'",
            self.fleet_key
        );
        PIPELINES.remove_pipeline(&self.fleet_key);

        for handle in &mut self.collector_handles {
            handle.shutdown();
        }
        self.shipper_handle.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn_central_client::{Env, FleetClient};

    use crate::state::FleetState;

    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "burn-central-fleet-telemetry-pipeline-{name}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ))
    }

    fn remove_dir(path: &Path) {
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn pipeline_exposes_shared_auth_client() {
        let fleet_key = format!(
            "fleet-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let root_dir = temp_dir("shared-client");
        let mut state = FleetState::default();
        state.set_auth_token("token".to_string(), 60);

        let client = AuthenticatedFleetClient::new(
            FleetClient::new(Env::Development),
            state.auth_token().cloned(),
        );
        let pipeline = TelemetryPipeline::create(fleet_key, client.clone(), root_dir.clone())
            .expect("telemetry pipeline should initialize");

        let pipeline_client = pipeline.client();
        client
            .clear_auth_token()
            .expect("shared client auth should clear");

        assert!(
            !pipeline_client.is_ready(),
            "pipeline should keep using the shared auth client state"
        );

        drop(pipeline);
        remove_dir(&root_dir);
    }
}
