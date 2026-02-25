use std::{path::PathBuf, sync::Arc};

use burn_central_client::{Env, FleetClient};
use directories::{BaseDirs, ProjectDirs};

use crate::{
    DeviceMetadata, FleetRegistrationToken,
    error::FleetError,
    model::{self, ModelSource},
    state,
    telemetry::TelemetryPipeline,
};

pub struct FleetDeviceSession {
    registration_token: FleetRegistrationToken,
    identity_key: String,
    state: state::FleetState,
    client: FleetClient,
    fleet_key: String,
    store: state::FleetLocalStateStore,
    pending_bootstrap_metadata: Option<DeviceMetadata>,
    _telemetry: Arc<TelemetryPipeline>,
}

impl FleetDeviceSession {
    pub fn init(
        token: impl Into<FleetRegistrationToken>,
        metadata: DeviceMetadata,
        env: &Env,
    ) -> Result<Self, FleetError> {
        let root_dir = default_data_dir(env)?;
        let registration_token = token.into();
        let fleet_key = state::fleet_key_from_registration_token(&registration_token);

        tracing::info!(
            fleet_key = %fleet_key,
            "registering fleet device session with fleet management service"
        );

        let client = FleetClient::new(env.clone());
        let store = state::FleetLocalStateStore::new(root_dir);
        let telemetry = TelemetryPipeline::get_or_init(fleet_key.clone())?;

        let identity_key = store.load_or_create_machine_identity_key()?;
        let state = store.load_fleet_state(&fleet_key)?.unwrap_or_default();
        let fleet_device = FleetDeviceSession {
            registration_token,
            identity_key,
            state,
            client,
            fleet_key,
            store,
            pending_bootstrap_metadata: Some(metadata),
            _telemetry: telemetry,
        };

        Ok(fleet_device)
    }

    pub fn active_model_version_id(&self) -> &str {
        self.state.active_model_version_id()
    }

    pub fn model_source(&self) -> Result<ModelSource, FleetError> {
        model::load_cached_model_source(
            &self.store.models_dir(&self.fleet_key),
            self.state.active_model_version_id(),
        )
        .map_err(FleetError::from)
    }

    pub fn runtime_config(&self) -> &serde_json::Value {
        self.state.runtime_config()
    }

    pub fn sync_for_reconcile(&mut self) -> Result<(), FleetError> {
        let metadata = self.pending_bootstrap_metadata.clone();
        match self.sync(metadata) {
            Ok(()) => {
                self.pending_bootstrap_metadata = None;
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    fn sync(&mut self, metadata: Option<DeviceMetadata>) -> Result<(), FleetError> {
        tracing::info!(
            ?metadata,
            "syncing fleet device with fleet management service"
        );
        let snapshot = self
            .client
            .sync(
                self.registration_token.clone(),
                self.identity_key.clone(),
                metadata,
            )
            .map_err(|e| FleetError::SyncFailed(e.to_string()))?;

        let download = self
            .client
            .model_download(self.registration_token.clone(), self.identity_key.clone())
            .map_err(|e| FleetError::DownloadFailed(e.to_string()))?;

        model::ensure_cached_model(
            &self.store.models_dir(&self.fleet_key),
            &snapshot.model_version_id,
            &download,
        )?;

        self.state
            .update(snapshot.model_version_id, snapshot.runtime_config);

        self.persist_state()
    }

    fn persist_state(&self) -> Result<(), FleetError> {
        self.store
            .save_fleet_state(&self.fleet_key, &self.state)
            .map_err(FleetError::from)
    }
}

/// Get the default cache directory for a given environment.
fn default_data_dir(env: &Env) -> Result<PathBuf, FleetError> {
    let fleets_subdir = match env {
        Env::Production => "fleets".to_string(),
        Env::Staging(version) => format!("fleets-staging-{version}"),
        Env::Development => "fleets-dev".to_string(),
    };
    if let Some(project) = ProjectDirs::from("ai", "tracel", "burn-central") {
        return Ok(project.data_dir().join(fleets_subdir));
    }
    if let Some(base) = BaseDirs::new() {
        return Ok(base.data_dir().join("burn-central").join(fleets_subdir));
    }
    Err(FleetError::CacheDirUnavailable)
}
